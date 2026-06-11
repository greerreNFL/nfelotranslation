'''
SpreadMapperFitter — seasonal trainer for linear-in-logit spread mapping.

Extends SeasonalFitter with exponential decay weighting.  A single mapper is
re-fitted each season on recalibrated win probabilities vs actual margins.

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
from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper
from nfelotranslation.SpreadMap.Types import LinearMapParams, SpreadMapResult
from nfelotranslation.Utilities.MathUtils import logit
from training.Seasonal import SeasonalDiagnostics, SeasonalFitter


## ==================== Constants ==================== ##

_PKG_SM = pathlib.Path(nfelotranslation.__file__).resolve().parent / 'SpreadMap'
_MODULE_DIR = str(_PKG_SM)
_DEFAULT_OUTPUT_DIR = str(_PKG_SM / 'configs')
_DEFAULT_PARAMS: Dict[str, Any] = {'decay': 0.0}


class SpreadMapperFitter(SeasonalFitter):
    '''
    Seasonal trainer for the SpreadMapper.

    For each season:
    1. Score out-of-sample (before training) — MAE + bisection
    2. Accumulate this season's data
    3. Re-fit the mapper on all accumulated data with optional decay weights
    4. Save params as spread_map_params_{season+1}.json

    The mapper trains on recalibrated win probabilities (ml_wp_cal) because
    the WP→margin relationship should use training labels in true WP space.

    Instance usage (auto-loads data):
        fitter = SpreadMapperFitter()
        result = fitter.fit()
        print(result.summary())

    Static usage (bring your own arrays):
        mapper = SpreadMapperFitter.fit_from_arrays(win_probs, targets)
        result = SpreadMapperFitter.diagnostics_from_arrays(mapper, win_probs, margins)

    Parameters:
    * games: optional DataFrame (auto-loads via DataLoader if omitted)
    * params: config dict with decay rate (default 0 — flat weighting)
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
        self._season_data_list: List[Dict[str, Any]] = []
        self._mapper: Optional[SpreadMapper] = None

    @property
    def model_name(self) -> str:
        return 'spread_mapper'

    @property
    def output_dir(self) -> str:
        return self._output_path

    def prepare_data(self) -> pandas.DataFrame:
        games = self.games
        mask = (
            games['ml_wp_close'].notna()
            & games['ml_wp_cal'].notna()
            & games['spread_line'].notna()
            & games['result'].notna()
        )
        return games[mask].copy()

    def initialize_model(self) -> None:
        self._season_data_list = []
        self._mapper = None

    def train_season(self, season_data: pandas.DataFrame, season: int) -> None:
        wps_cal = season_data['ml_wp_cal'].values.astype(float)
        margins = season_data['margin'].values.astype(float)
        self._season_data_list.append({
            'season': season,
            'win_probs_cal': wps_cal,
            'margins': margins,
        })
        all_wps_cal, all_margins = self._build_concat_arrays()
        decay = self._params.get('decay', 0.0)
        weights = self._build_weights(season, decay) if decay > 0 else None
        self._mapper = self.fit_from_arrays(
            all_wps_cal,
            all_margins,
            weights=weights,
            force_zero_intercept=True,
        )

    def save_season(self, season: int) -> None:
        season = int(season)
        filepath = os.path.join(self._output_path, f'spread_map_params_{season + 1}.json')
        metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        self._mapper.metadata = metadata
        self._mapper.to_file(filepath)

    def score_season(self, season_data: pandas.DataFrame, season: int) -> SeasonalDiagnostics:
        n_games = len(season_data)
        if self._mapper is None:
            return SeasonalDiagnostics(
                season=season,
                metrics={'mae': float('nan'), 'bisection': float('nan')},
                metadata={'n_games': n_games, 'first_season': True},
            )
        wps_cal = season_data['ml_wp_cal'].values.astype(float)
        margins = season_data['margin'].values.astype(float)
        result = self.diagnostics_from_arrays(self._mapper, wps_cal, margins)
        return SeasonalDiagnostics(
            season=season,
            metrics={'mae': result.mae, 'bisection': result.bisection_rate},
            metadata={'n_games': n_games},
        )

    def _compute_aggregate(self, per_season: List[SeasonalDiagnostics]) -> Dict[str, float]:
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

    def fit(self, seasons: Optional[List[int]] = None):
        self._verify_upstream()
        result = super().fit(seasons=seasons)
        self.save_config()
        return result

    def validate(self) -> ValidationReport:
        self._ensure_fitted()
        from training.Validation import SpreadMapperValidator
        validator = SpreadMapperValidator(
            games=self.games,
            seasonal_result=self._fitted_model,
            mapper=self._mapper,
        )
        return validator.validate()

    def save_config(self) -> str:
        self._ensure_fitted()
        if self._mapper is None:
            raise RuntimeError('No fitted mapper — call fit() first')
        metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        self._mapper.metadata = metadata
        root_path = os.path.join(_MODULE_DIR, 'spread_map_params.json')
        self._mapper.to_file(root_path)
        return root_path

    def _verify_upstream(self) -> None:
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

    def _build_concat_arrays(self):
        return (
            numpy.concatenate([e['win_probs_cal'] for e in self._season_data_list]),
            numpy.concatenate([e['margins'] for e in self._season_data_list]),
        )

    def _build_weights(self, current_season: int, decay: float) -> numpy.ndarray:
        parts = []
        for entry in self._season_data_list:
            n = len(entry['win_probs_cal'])
            w = (1 - decay) ** (current_season - entry['season'])
            parts.append(numpy.full(n, w))
        return numpy.concatenate(parts)

    @staticmethod
    def fit_from_arrays(
        win_probs: numpy.ndarray,
        targets: numpy.ndarray,
        weights: numpy.ndarray = None,
        force_zero_intercept: bool = False,
    ) -> SpreadMapper:
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
        wp = numpy.asarray(win_probs, dtype=float)
        mg = numpy.asarray(margins, dtype=float)
        predicted = mapper.win_prob_to_spread(wp).continuous
        mae = float(numpy.mean(numpy.abs(mg - predicted)))
        bisection_rate = float(numpy.mean(mg > predicted))
        return SpreadMapResult(
            params=mapper.params,
            n_games=len(wp),
            mae=mae,
            bisection_rate=bisection_rate,
        )


def _fit(
    win_probs: numpy.ndarray,
    targets: numpy.ndarray,
    weights: numpy.ndarray = None,
    force_zero_intercept: bool = False,
) -> LinearMapParams:
    z = logit(win_probs)
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
    if force_zero_intercept:
        return LinearMapParams(slope=float(result.x[0]), intercept=0.0)
    return LinearMapParams(slope=float(result.x[0]), intercept=float(result.x[1]))
