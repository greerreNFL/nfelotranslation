'''
KeyModelFitter — seasonal trainer for the ratio-based KeyModel.

Concrete SeasonalFitter that replicates the prototype training loop:
for each season, compute per-game baseline PMFs, count hits at each
integer 1-40, and update the KeyModel's NumberOutcome trackers.

Output file convention:
    key_model_{season+1}.json = trained through {season}, valid for {season+1}
'''

## built-ins ##
import os
import pathlib
from typing import Any, Dict, List, Optional

## external ##
import numpy
import pandas
import nfelotranslation
from scipy.stats import norm as scipy_norm

## local ##
from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Distribution.Base import BaseDistribution
from nfelotranslation.Distribution.Key.KeyModel import KeyModel
from nfelotranslation.Utilities.JsonIo import ConfigMetadata
from nfelotranslation.Utilities.ValidationTypes import ValidationReport
from nfelotranslation.SpreadMap.SpreadMapper import load_mapper_pair
from training.Seasonal import SeasonalDiagnostics, SeasonalFitter

from nfelotranslation.Distribution.Key import KEY_MODEL_PARAMS as _DEFAULT_PARAMS


## ==================== Defaults ==================== ##

_MODULE_DIR = str(pathlib.Path(nfelotranslation.__file__).resolve().parent / 'Distribution' / 'Key')

_DEFAULT_OUTPUT_DIR = str(
    pathlib.Path(nfelotranslation.__file__).resolve().parent / 'Distribution' / 'Key' / 'configs'
)

## margin range used for baseline PMF computation ##
_MARGINS = numpy.arange(-75, 76)

## all tracked integers ##
_ALL_NUMBERS = list(range(1, 41))


class KeyModelFitter(SeasonalFitter):
    '''
    Seasonal trainer for the KeyModel (40 credibility-weighted trackers).

    For each season:
    1. Compute per-game baseline PMFs from (spread, wp) → σ → normal
    2. Aggregate to season-level baseline rates for each ±k
    3. Count actual hits at each ±k
    4. Score out-of-sample (before training)
    5. Update all 40 NumberOutcome trackers
    6. Save model state as key_model_{season+1}.json

    Parameters:
    * games: optional DataFrame (auto-loads via DataLoader if omitted)
    * params: config dict for NumberOutcome (forgetting_rate, threshold,
              initial_prior_size)
    * output_path: directory for output config files
    '''

    def __init__(
        self,
        games: pandas.DataFrame = None,
        params: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ):
        super().__init__(games=games, output_path=output_path, pipeline_id=pipeline_id)
        self._params = params or _DEFAULT_PARAMS
        self._output_path = output_path or _DEFAULT_OUTPUT_DIR
        self._model: Optional[KeyModel] = None

    ## ==================== Abstract Implementation ==================== ##

    @property
    def model_name(self) -> str:
        return 'key_model'

    @property
    def output_dir(self) -> str:
        return self._output_path

    def prepare_data(self) -> pandas.DataFrame:
        '''Filter to games with result, spread, and recalibrated win prob.'''
        games = self.games
        mask = (
            games['result'].notna()
            & games['spread_line'].notna()
            & games['ml_wp_cal'].notna()
        )
        df = games[mask].copy()
        df['margin'] = df['result'].round().astype(int)
        return df

    def initialize_model(self) -> None:
        '''Create fresh KeyModel with 40 zero-state trackers.'''
        self._model = KeyModel.from_initial(self._params)

    def train_season(self, season_data: pandas.DataFrame, season: int) -> None:
        '''
        Ingest one season: count hits at each ±k, update all trackers.

        Parameters:
        * season_data: games for this season
        * season: season identifier
        '''
        n_games = len(season_data)
        margins = season_data['margin'].values
        baselines = self._compute_baselines(season_data)
        for k in _ALL_NUMBERS:
            hits = int((margins == k).sum() + (margins == -k).sum())
            baseline_rate = baselines.get(k, 0.01)
            self._model.outcomes[k].update(hits, n_games, baseline_rate, int(season))

    def save_season(self, season: int) -> None:
        '''
        Save model state as key_model_{season+1}.json, stamped with the
        current training-run metadata.
        '''
        season = int(season)
        filepath = os.path.join(self._output_path, f'{self.model_name}_{season + 1}.json')
        self._model.metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        self._model.to_file(filepath, include_history=False)

    def score_season(self, season_data: pandas.DataFrame, season: int) -> SeasonalDiagnostics:
        '''
        Evaluate out-of-sample: compare predicted excess to actual excess.

        Called BEFORE train_season, so the model has not seen this season.

        Also computes a baseline RMSE — the error you'd get predicting 0 excess
        for everything (i.e. just using the normal distribution with no key
        number adjustment). The model should beat this baseline.

        Parameters:
        * season_data: games for this season
        * season: season identifier

        Returns:
        * SeasonalDiagnostics with model and baseline RMSE/MAE, per-number detail
        '''
        n_games = len(season_data)
        margins = season_data['margin'].values
        baselines = self._compute_baselines(season_data)
        sq_errors = []
        abs_errors = []
        baseline_sq_errors = []
        per_number = {}
        for k in _ALL_NUMBERS:
            ## actual excess this season ##
            hits = int((margins == k).sum() + (margins == -k).sum())
            baseline_rate = baselines.get(k, 0.01)
            ## model's pre-update prediction (ratio → excess for comparison) ##
            ratio = self._model.outcomes[k].get_ratio()
            pred_excess = (ratio - 1.0) * baseline_rate
            emp_rate = hits / n_games if n_games > 0 else 0.0
            actual_excess = emp_rate - baseline_rate
            ## model error ##
            error = pred_excess - actual_excess
            sq_errors.append(error ** 2)
            abs_errors.append(abs(error))
            ## baseline error (predicting 0 excess = just using the normal) ##
            baseline_sq_errors.append(actual_excess ** 2)
            ## per-number detail ##
            per_number[k] = {
                'pred_excess': pred_excess,
                'actual_excess': actual_excess,
                'error': error,
                'hits': hits,
            }
        rmse_pp = float(numpy.sqrt(numpy.mean(sq_errors))) * 100.0
        mae_pp = float(numpy.mean(abs_errors)) * 100.0
        baseline_rmse_pp = float(numpy.sqrt(numpy.mean(baseline_sq_errors))) * 100.0
        return SeasonalDiagnostics(
            season=season,
            metrics={
                'rmse_excess_pp': rmse_pp,
                'mean_abs_excess_pp': mae_pp,
                'baseline_rmse_pp': baseline_rmse_pp,
            },
            metadata={
                'n_games': n_games,
                'per_number': per_number,
            },
        )

    ## ==================== Template Method Override ==================== ##

    def fit(self, seasons: Optional[List[int]] = None):
        '''
        Verify upstream dependencies, run the seasonal loop, then persist
        the canonical root config.

        If self.pipeline_id is set, both the Recalibrator and SpreadMapper
        configs on disk must carry a matching pipeline_id — this ensures
        the ml_wp_cal and spread coordinates used to build baseline PMFs
        came from the same training run.
        '''
        self._verify_upstream()
        result = super().fit(seasons=seasons)
        self.save_config()
        return result

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Validate the KeyModel using OOS diagnostics from the seasonal fit.

        Delegates to KeyModelValidator for the actual validation logic.

        Returns:
        * ValidationReport with checks and metrics
        '''
        self._ensure_fitted()
        from training.Validation import KeyModelValidator
        validator = KeyModelValidator(
            games=self.games,
            seasonal_result=self._fitted_model,
        )
        return validator.validate()

    ## ==================== Config Persistence ==================== ##

    def save_config(self) -> str:
        '''
        Persist the fitted KeyModel to the root key_model.json envelope,
        stamped with a ConfigMetadata carrying self.pipeline_id.

        Returns:
        * path to the written root config
        '''
        self._ensure_fitted()
        if self._model is None:
            raise RuntimeError('No fitted KeyModel — call fit() first')
        self._model.metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        root_path = os.path.join(_MODULE_DIR, 'key_model.json')
        self._model.to_file(root_path, include_history=False)
        return root_path

    def _verify_upstream(self) -> None:
        '''
        If a pipeline_id was supplied, verify both the Recalibrator and
        SpreadMapper configs on disk carry the same id.  This catches
        stale upstream state — e.g. ml_wp_cal values from a DataLoader
        that was not reset between training phases.
        '''
        if self._pipeline_id is None:
            return
        rec = Recalibrator.from_file()
        if rec.metadata.pipeline_id != self._pipeline_id:
            raise RuntimeError(
                f'Upstream Recalibrator pipeline_id mismatch: '
                f'expected {self._pipeline_id!r}, got {rec.metadata.pipeline_id!r}. '
                f'Run RecalibratorFitter(pipeline_id={self._pipeline_id!r}).fit() '
                f'and reset DataLoader before fitting the KeyModel.'
            )
        _, _, sm_metadata = load_mapper_pair()
        if sm_metadata.pipeline_id != self._pipeline_id:
            raise RuntimeError(
                f'Upstream SpreadMapper pipeline_id mismatch: '
                f'expected {self._pipeline_id!r}, got {sm_metadata.pipeline_id!r}. '
                f'Run SpreadMapperFitter(pipeline_id={self._pipeline_id!r}).fit() '
                f'before fitting the KeyModel.'
            )

    ## ==================== Private ==================== ##

    @staticmethod
    def _compute_baselines(season_data: pandas.DataFrame) -> Dict[int, float]:
        '''
        Compute season-aggregate baseline rates for each ±k.

        For each game, derive sigma from (spread, wp), evaluate the normal
        PMF at all integers in [-75, 75], normalize to sum 1, and accumulate.
        Returns the average baseline rate for P(margin=+k) + P(margin=-k).

        Parameters:
        * season_data: games for this season

        Returns:
        * dict mapping k (1-40) → combined baseline rate for ±k
        '''
        agg = numpy.zeros(len(_MARGINS), dtype=float)
        n_valid = 0
        for _, row in season_data.iterrows():
            mu = float(row['spread_line'])
            wp = float(row['ml_wp_cal'])
            wp = numpy.clip(wp, 0.01, 0.99)
            ## inline sigma derivation — compute ppf first to avoid 0/0 ##
            z = float(scipy_norm.ppf(wp))
            if abs(z) < 1e-9:
                sigma = 13.2
            else:
                sigma = mu / z
            if not numpy.isfinite(sigma) or sigma <= 0:
                continue
            raw = scipy_norm.pdf(_MARGINS.astype(float), loc=mu, scale=sigma)
            Z = raw.sum()
            if Z < 1e-15:
                continue
            agg += raw / Z
            n_valid += 1
        if n_valid == 0:
            return {k: 0.01 for k in _ALL_NUMBERS}
        avg_pmf = agg / n_valid
        ## extract combined baseline rate for each ±k ##
        baselines = {}
        for k in _ALL_NUMBERS:
            ## _MARGINS goes from -75 to 75, so index for value v is v + 75 ##
            baselines[k] = float(avg_pmf[k + 75]) + float(avg_pmf[-k + 75])
        return baselines
