'''
SpreadMapperFitter — seasonal trainer for linear-in-logit spread mappings.

Extends SeasonalFitter with exponential decay weighting.  Both model and market
mappers are re-fitted each season using all accumulated data, with more recent
seasons weighted more heavily.

Sign convention — positive = favorite throughout the entire package.
    DataLoader provides spread_line in positive = home-favorite convention
    (matching nflfastR's native convention), so this fitter reads it directly
    with no sign adjustment.

Output file convention:
    spread_map_params_{season+1}.json = trained through {season}, valid for {season+1}
'''

## built-ins ##
import os
import pathlib
from typing import Any, Dict, List, Optional

## external ##
import numpy
import pandas
import nfelotranslation
from scipy.optimize import minimize

## local ##
from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Utilities.JsonIo import ConfigMetadata
from nfelotranslation.Utilities.ValidationTypes import ValidationReport
from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper, save_mapper_pair
from nfelotranslation.SpreadMap.Types import MapType, LinearMapParams, SpreadMapResult
from nfelotranslation.Utilities.MathUtils import logit
from training.Seasonal import SeasonalDiagnostics, SeasonalFitter


## ==================== Constants ==================== ##

_PKG_SM = pathlib.Path(nfelotranslation.__file__).resolve().parent / 'SpreadMap'
_MODULE_DIR = str(_PKG_SM)
_DEFAULT_OUTPUT_DIR = str(_PKG_SM / 'configs')
_DEFAULT_PARAMS: Dict[str, Any] = {
    'decay_model': 0.0,
    'decay_market': 0.15,
}


class SpreadMapperFitter(SeasonalFitter):
    '''
    Seasonal trainer for both MODEL and MARKET SpreadMappers.

    For each season:
    1. Score out-of-sample (before training) — model MAE + bisection, market MAE
    2. Accumulate this season's data
    3. Re-fit both mappers on all accumulated data with exponential decay weights
    4. Save params as spread_map_params_{season+1}.json

    The model mapper trains on recalibrated win probabilities (ml_wp_cal)
    because the WP→margin relationship should use "true" probabilities.
    The market mapper trains on raw market win probabilities (ml_wp_close)
    because the market sets spreads based on its own implied probabilities.

    Separate decay rates: model uses flat weighting (decay=0) because the
    WP→margin relationship is structural; market uses decay=0.15 because
    the market evolves over time.

    Instance usage (auto-loads data):
        fitter = SpreadMapperFitter()
        result = fitter.fit()
        print(result.summary())

    Static usage (bring your own arrays):
        mapper = SpreadMapperFitter.fit_from_arrays(win_probs, targets)
        result = SpreadMapperFitter.diagnostics_from_arrays(mapper, win_probs, margins)

    Parameters:
    * games: optional DataFrame (auto-loads via DataLoader if omitted)
    * params: config dict with decay_model and decay_market rates
    * output_path: directory for per-season config files
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
        ## accumulated per-season data ##
        self._season_data_list: List[Dict[str, Any]] = []
        ## current fitted mappers ##
        self._model_mapper: Optional[SpreadMapper] = None
        self._market_mapper: Optional[SpreadMapper] = None

    ## ==================== Abstract Implementation ==================== ##

    @property
    def model_name(self) -> str:
        return 'spread_mapper'

    @property
    def output_dir(self) -> str:
        return self._output_path

    def prepare_data(self) -> pandas.DataFrame:
        '''
        Filter to games with both market and recalibrated win probs,
        spreads, and results.

        Returns a DataFrame with 'season' column intact for the
        seasonal loop.  Array extraction and sign flipping happen
        in train_season() and score_season().
        '''
        games = self.games
        mask = (
            games['ml_wp_close'].notna()
            & games['ml_wp_cal'].notna()
            & games['spread_line'].notna()
            & games['result'].notna()
        )
        return games[mask].copy()

    def initialize_model(self) -> None:
        '''Clear accumulated data and mappers.'''
        self._season_data_list = []
        self._model_mapper = None
        self._market_mapper = None

    def train_season(self, season_data: pandas.DataFrame, season: int) -> None:
        '''
        Accumulate this season's data and re-fit both mappers with decay weights.

        Model mapper fits on recalibrated WPs (ml_wp_cal) vs actual margins.
        Market mapper fits on raw market WPs (ml_wp_close) vs market spreads.
        Model uses decay_model (default 0 — flat weighting) because the
        WP→margin relationship is structural.  Market uses decay_market
        (default 0.15) because the market evolves over time.

        Parameters:
        * season_data: games for this season
        * season: season identifier
        '''
        ## extract arrays — spread_line already in positive = favorite convention ##
        wps_cal = season_data['ml_wp_cal'].values.astype(float)
        wps_close = season_data['ml_wp_close'].values.astype(float)
        margins = season_data['margin'].values.astype(float)
        markets = season_data['spread_line'].values.astype(float)
        self._season_data_list.append({
            'season': season,
            'win_probs_cal': wps_cal,
            'win_probs_close': wps_close,
            'margins': margins,
            'market_spreads': markets,
        })
        ## build weighted arrays — separate decay rates ##
        all_wps_cal, all_wps_close, all_margins, all_markets = self._build_concat_arrays()
        model_weights = self._build_weights(season, self._params['decay_model'])
        market_weights = self._build_weights(season, self._params['decay_market'])
        ## fit both mappers — model on cal WPs, market on close WPs ##
        ## model mapper: force intercept=0 so WP=0.50 ↔ spread=0 exactly ##
        self._model_mapper = self.fit_from_arrays(
            all_wps_cal, all_margins,
            weights=model_weights if self._params['decay_model'] > 0 else None,
            force_zero_intercept=True,
        )
        self._market_mapper = self.fit_from_arrays(all_wps_close, all_markets, weights=market_weights)

    def save_season(self, season: int) -> None:
        '''
        Save both mappers to configs/spread_map_params_{season+1}.json.

        The per-season envelope carries the same pipeline_id as the root
        config, so backtests can trace which training run produced them.
        The root spread_map_params.json is updated separately by
        save_config() after the full seasonal loop completes.
        '''
        season = int(season)
        filepath = os.path.join(self._output_path, f'spread_map_params_{season + 1}.json')
        metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        save_mapper_pair(self._model_mapper, self._market_mapper, metadata, filepath=filepath)

    def score_season(self, season_data: pandas.DataFrame, season: int) -> SeasonalDiagnostics:
        '''
        Evaluate out-of-sample quality BEFORE training on this season.

        First season returns NaN metrics (no prior model exists).

        Parameters:
        * season_data: games for this season (held out from model)
        * season: season identifier

        Returns:
        * SeasonalDiagnostics with model MAE + bisection, market MAE
        '''
        n_games = len(season_data)
        if self._model_mapper is None:
            ## first season — no model to score with ##
            return SeasonalDiagnostics(
                season=season,
                metrics={
                    'model_mae': float('nan'),
                    'model_bisection': float('nan'),
                    'market_mae': float('nan'),
                    'market_r2': float('nan'),
                },
                metadata={'n_games': n_games, 'first_season': True},
            )
        ## extract arrays — spread_line already in positive = favorite convention ##
        wps_cal = season_data['ml_wp_cal'].values.astype(float)
        wps_close = season_data['ml_wp_close'].values.astype(float)
        margins = season_data['margin'].values.astype(float)
        markets = season_data['spread_line'].values.astype(float)
        ## model mapper: MAE + bisection vs actual margins (uses cal WPs) ##
        model_result = self.diagnostics_from_arrays(self._model_mapper, wps_cal, margins)
        ## market mapper: MAE and R² vs market-posted lines (uses close WPs) ##
        market_pred = self._market_mapper.win_prob_to_spread(wps_close).continuous
        market_mae = float(numpy.mean(numpy.abs(markets - market_pred)))
        ss_res = float(numpy.sum((markets - market_pred) ** 2))
        ss_tot = float(numpy.sum((markets - numpy.mean(markets)) ** 2))
        market_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        ## spread gap: per-game using each mapper's native WP, binned by cal WP ##
        model_pred_all = self._model_mapper.win_prob_to_spread(wps_cal).continuous
        market_pred_all = self._market_mapper.win_prob_to_spread(wps_close).continuous
        per_game_gap = market_pred_all - model_pred_all
        spread_gap = {}
        for wp_val in [0.50, 0.60, 0.70, 0.80]:
            bin_mask = numpy.abs(wps_cal - wp_val) < 0.05
            if bin_mask.sum() > 0:
                spread_gap[f'wp_{int(wp_val * 100)}'] = float(numpy.mean(per_game_gap[bin_mask]))
            else:
                spread_gap[f'wp_{int(wp_val * 100)}'] = float('nan')
        return SeasonalDiagnostics(
            season=season,
            metrics={
                'model_mae': model_result.mae,
                'model_bisection': model_result.bisection_rate,
                'market_mae': market_mae,
                'market_r2': market_r2,
            },
            metadata={'n_games': n_games, 'spread_gap': spread_gap},
        )

    ## ==================== Aggregation ==================== ##

    def _compute_aggregate(self, per_season: List[SeasonalDiagnostics]) -> Dict[str, float]:
        '''
        Mean of each metric across seasons, skipping NaN (first season).
        '''
        if not per_season:
            return {}
        metric_names = list(per_season[0].metrics.keys())
        aggregate = {}
        for name in metric_names:
            values = [
                d.metrics[name] for d in per_season
                if name in d.metrics and numpy.isfinite(d.metrics[name])
            ]
            aggregate[name] = sum(values) / len(values) if values else float('nan')
        return aggregate

    ## ==================== Template Method Override ==================== ##

    def fit(self, seasons: Optional[List[int]] = None):
        '''
        Verify upstream dependencies, run the seasonal loop, then persist
        the canonical root config.

        If self.pipeline_id is set, the Recalibrator config on disk must
        carry a matching pipeline_id — this ensures the ml_wp_cal values
        fed through DataLoader came from the same training run.
        '''
        self._verify_upstream()
        result = super().fit(seasons=seasons)
        self.save_config()
        return result

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Validate using OOS diagnostics from the seasonal fit.

        Delegates to SpreadMapperValidator for the actual validation logic.

        Returns:
        * ValidationReport with checks and metrics
        '''
        self._ensure_fitted()
        from training.Validation import SpreadMapperValidator
        validator = SpreadMapperValidator(
            games=self.games,
            seasonal_result=self._fitted_model,
            model_mapper=self._model_mapper,
            market_mapper=self._market_mapper,
        )
        return validator.validate()

    ## ==================== Config Persistence ==================== ##

    def save_config(self) -> str:
        '''
        Persist both mappers to the root spread_map_params.json envelope,
        stamped with a ConfigMetadata carrying self.pipeline_id.

        Returns:
        * path to the written root config
        '''
        self._ensure_fitted()
        if self._model_mapper is None or self._market_mapper is None:
            raise RuntimeError('No fitted mappers — call fit() first')
        metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        root_path = os.path.join(_MODULE_DIR, 'spread_map_params.json')
        return save_mapper_pair(
            self._model_mapper, self._market_mapper, metadata, filepath=root_path,
        )

    def _verify_upstream(self) -> None:
        '''
        If a pipeline_id was supplied, verify the Recalibrator config on
        disk carries the same id.  This catches stale ml_wp_cal values
        from a DataLoader that was not reset between training phases.
        '''
        if self._pipeline_id is None:
            return
        rec = Recalibrator.from_file()
        if rec.metadata.pipeline_id != self._pipeline_id:
            raise RuntimeError(
                f'Upstream Recalibrator pipeline_id mismatch: '
                f'expected {self._pipeline_id!r}, got {rec.metadata.pipeline_id!r}. '
                f'Run RecalibratorFitter(pipeline_id={self._pipeline_id!r}).fit() '
                f'and reset DataLoader before fitting the SpreadMapper.'
            )

    ## ==================== Private ==================== ##

    def _build_concat_arrays(self):
        '''
        Concatenate all accumulated data into flat arrays.

        Returns:
        * (win_probs_cal, win_probs_close, margins, market_spreads) arrays
        '''
        return (
            numpy.concatenate([e['win_probs_cal'] for e in self._season_data_list]),
            numpy.concatenate([e['win_probs_close'] for e in self._season_data_list]),
            numpy.concatenate([e['margins'] for e in self._season_data_list]),
            numpy.concatenate([e['market_spreads'] for e in self._season_data_list]),
        )

    def _build_weights(self, current_season: int, decay: float) -> numpy.ndarray:
        '''
        Build per-game exponential decay weights.

        Weight for each game in season s:
            w = (1 - decay) ^ (current_season - s)

        Parameters:
        * current_season: the season just ingested (most recent)
        * decay: decay rate per season

        Returns:
        * weights array (same length as concatenated data)
        '''
        parts = []
        for entry in self._season_data_list:
            n = len(entry['win_probs_cal'])
            w = (1 - decay) ** (current_season - entry['season'])
            parts.append(numpy.full(n, w))
        return numpy.concatenate(parts)

    ## ==================== Static Methods ==================== ##

    @staticmethod
    def fit_from_arrays(
        win_probs: numpy.ndarray,
        targets: numpy.ndarray,
        weights: numpy.ndarray = None,
        force_zero_intercept: bool = False,
    ) -> SpreadMapper:
        '''
        Fit a SpreadMapper from arrays via Nelder-Mead MAE optimization.

        Parameters:
        * win_probs: win probabilities in (0, 1) ## positive-favorite convention
        * targets: margins or spreads ## positive-favorite convention
        * weights: optional per-observation weights for weighted MAE
        * force_zero_intercept: if True, fix intercept=0 (only optimize slope)

        Returns:
        * fitted SpreadMapper
        '''
        params = _fit(
            numpy.asarray(win_probs, dtype=float),
            numpy.asarray(targets, dtype=float),
            weights=numpy.asarray(weights, dtype=float) if weights is not None else None,
            force_zero_intercept=force_zero_intercept,
        )
        return SpreadMapper(params)

    @staticmethod
    def diagnostics_from_arrays(
        mapper: SpreadMapper,
        win_probs: numpy.ndarray,
        margins: numpy.ndarray,
    ) -> SpreadMapResult:
        '''
        Evaluate mapping quality on arrays.

        Parameters:
        * mapper: fitted SpreadMapper instance
        * win_probs: win probabilities ## >0.5 = favorite
        * margins: actual game margins ## positive = favorite won

        Returns:
        * SpreadMapResult with MAE and bisection rate
        '''
        wp = numpy.asarray(win_probs, dtype=float)
        mg = numpy.asarray(margins, dtype=float)
        ## predicted spreads ##
        predicted = mapper.win_prob_to_spread(wp).continuous
        ## MAE ##
        mae = float(numpy.mean(numpy.abs(mg - predicted)))
        ## bisection rate: fraction where margin > spread ≈ 0.5 ##
        bisection_rate = float(numpy.mean(mg > predicted))
        return SpreadMapResult(
            params=mapper.params,
            n_games=len(wp),
            mae=mae,
            bisection_rate=bisection_rate,
        )


## ==================== Private ==================== ##

def _fit(
    win_probs: numpy.ndarray,
    targets: numpy.ndarray,
    weights: numpy.ndarray = None,
    force_zero_intercept: bool = False,
) -> LinearMapParams:
    '''
    Fit linear-in-logit spread-mapping parameters via Nelder-Mead MAE.

    A spread is a median — the number where 50% of outcomes land above and
    50% below.  MAE targets the median of the conditional distribution,
    making it the correct loss for both model and market mappers.

    When weights are provided, uses numpy.average (weighted MAE) so that
    more recent seasons contribute more to the fit.

    When force_zero_intercept is True, only slope is optimized and
    intercept is fixed at 0.  This enforces WP=0.50 ↔ spread=0,
    which is the principled constraint for the model mapper.

    Parameters:
    * win_probs: favorite win probabilities
    * targets: actual margins (model mapper) or market spreads (market mapper)
    * weights: optional per-observation weights for weighted MAE
    * force_zero_intercept: if True, fix intercept=0 (only optimize slope)

    Returns:
    * fitted LinearMapParams
    '''
    z = logit(win_probs)
    ## MAE objective — targets the median ##
    if force_zero_intercept:
        if weights is not None:
            def objective(params: numpy.ndarray) -> float:
                predicted = params[0] * z
                return float(numpy.average(numpy.abs(targets - predicted), weights=weights))
        else:
            def objective(params: numpy.ndarray) -> float:
                predicted = params[0] * z
                return float(numpy.mean(numpy.abs(targets - predicted)))
        result = minimize(
            objective,
            x0=numpy.array([6.0]),
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
        )
        if not result.success:
            raise RuntimeError(
                f'SpreadMapper optimization did not converge: {result.message}'
            )
        return LinearMapParams(slope=float(result.x[0]), intercept=0.0)
    else:
        if weights is not None:
            def objective(params: numpy.ndarray) -> float:
                predicted = params[0] * z + params[1]
                return float(numpy.average(numpy.abs(targets - predicted), weights=weights))
        else:
            def objective(params: numpy.ndarray) -> float:
                predicted = params[0] * z + params[1]
                return float(numpy.mean(numpy.abs(targets - predicted)))
        result = minimize(
            objective,
            x0=numpy.array([6.0, 0.5]),
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
        )
        if not result.success:
            raise RuntimeError(
                f'SpreadMapper optimization did not converge: {result.message}'
            )
        return LinearMapParams(slope=float(result.x[0]), intercept=float(result.x[1]))
