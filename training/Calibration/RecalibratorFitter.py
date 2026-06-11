'''
RecalibratorFitter — seasonal split Platt recalibration for training labels.

Fits omniscient split slopes on the full sample, then per-season intercepts
using a centered window (default width 5, edge-padded).
'''

## built-ins ##
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

## external ##
import numpy
import pandas
from scipy.optimize import minimize

## local ##
import nfelotranslation
from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Calibration.Types import CalibrationResult, SplitPlattParams
from nfelotranslation.Utilities.JsonIo import ConfigMetadata
from nfelotranslation.Utilities.ValidationTypes import ValidationReport
from nfelotranslation.Utilities.MathUtils import brier_score, clip_prob, expit, log_loss, logit
from training.Seasonal import SeasonalDiagnostics, SeasonalFitter


_MODULE_DIR = str(pathlib.Path(nfelotranslation.__file__).resolve().parent / 'Calibration')
_DEFAULT_OUTPUT_DIR = str(pathlib.Path(_MODULE_DIR) / 'configs')
_DEFAULT_WINDOW_WIDTH = 5


def centered_window_seasons(target_season: int, seasons: List[int], width: int) -> List[int]:
    seasons = sorted(seasons)
    idx = seasons.index(target_season)
    n = len(seasons)
    width = min(width, n)
    left = (width - 1) // 2
    right = width - 1 - left
    start = idx - left
    end = idx + right
    if start < 0:
        end = min(n - 1, end + (-start))
        start = 0
    if end >= n:
        start = max(0, start - (end - (n - 1)))
        end = n - 1
    if end - start + 1 < width:
        if start == 0:
            end = min(n - 1, width - 1)
        elif end == n - 1:
            start = max(0, n - width)
    return seasons[start:end + 1]


class RecalibratorFitter(SeasonalFitter):
    '''
    Seasonal trainer for split Platt recalibration.

    1. Fit omniscient split slopes (a_home, a_away) on the full sample.
    2. For each season S, fit intercepts on games in the centered window
       around S (width configurable, default 5).
    3. Save ``configs/platt_params_{S}.json`` for labeling season S games.

    Parameters:
    * window_width: centered window size in seasons (default 5)
    '''

    def __init__(
        self,
        games: pandas.DataFrame = None,
        pipeline_id: Optional[str] = None,
        output_path: Optional[str] = None,
        window_width: int = _DEFAULT_WINDOW_WIDTH,
    ):
        super().__init__(games=games, pipeline_id=pipeline_id, output_path=output_path)
        self._output_path = output_path or _DEFAULT_OUTPUT_DIR
        self._window_width = window_width
        self._fixed_slopes: Dict[str, float] = {}
        self._full_df: Optional[pandas.DataFrame] = None
        self._latest_rec: Optional[Recalibrator] = None

    @property
    def model_name(self) -> str:
        return 'recalibrator'

    @property
    def output_dir(self) -> str:
        return self._output_path

    def prepare_data(self) -> pandas.DataFrame:
        games = self.games
        mask = games['ml_wp_close'].notna() & games['result'].notna()
        df = games[mask].copy()
        df = df[df['result'] != 0]
        df['home_is_fav'] = (df['ml_wp_close'] >= 0.5).astype(int)
        df['fav_ml_wp'] = numpy.where(
            df['home_is_fav'],
            df['ml_wp_close'].astype(float),
            1.0 - df['ml_wp_close'].astype(float),
        )
        df['fav_win'] = numpy.where(
            df['home_is_fav'],
            (df['result'] > 0).astype(int),
            (df['result'] < 0).astype(int),
        )
        df['logit_wp'] = logit(clip_prob(df['fav_ml_wp'].values))
        return df

    def initialize_model(self) -> None:
        self._full_df = self.prepare_data()
        self._fixed_slopes = self._fit_split_slopes(self._full_df)

    def train_season(self, season_data: pandas.DataFrame, season: int) -> None:
        pass

    def save_season(self, season: int) -> None:
        season = int(season)
        seasons = sorted(self._full_df['season'].unique())
        window = centered_window_seasons(season, seasons, self._window_width)
        train = self._full_df[self._full_df['season'].isin(window)]
        intercepts = self._fit_split_intercepts(train, self._fixed_slopes)
        fit_meta = {
            'window_type': 'centered',
            'window_width': self._window_width,
            'seasons_used': window,
            'label_season': season,
        }
        params = SplitPlattParams(
            slopes=dict(self._fixed_slopes),
            intercepts=intercepts,
            fit=fit_meta,
        )
        rec = Recalibrator(params, metadata=ConfigMetadata.new(pipeline_id=self._pipeline_id))
        self._latest_rec = rec
        filepath = os.path.join(self._output_path, f'platt_params_{season}.json')
        rec.to_file(filepath)

    def score_season(self, season_data: pandas.DataFrame, season: int) -> SeasonalDiagnostics:
        seasons = sorted(self._full_df['season'].unique())
        window = centered_window_seasons(int(season), seasons, self._window_width)
        train = self._full_df[self._full_df['season'].isin(window)]
        intercepts = self._fit_split_intercepts(train, self._fixed_slopes)
        rec = Recalibrator(
            SplitPlattParams(
                slopes=dict(self._fixed_slopes),
                intercepts=intercepts,
            )
        )
        wp = season_data['fav_ml_wp'].values.astype(float)
        out = season_data['fav_win'].values.astype(float)
        is_home = season_data['home_is_fav'].values.astype(bool)
        cal = rec.calibrate(wp, is_home_fav=is_home)
        return SeasonalDiagnostics(
            season=int(season),
            metrics={
                'log_loss_before': log_loss(out, wp),
                'log_loss_after': log_loss(out, cal),
                'brier_before': brier_score(out, wp),
                'brier_after': brier_score(out, cal),
            },
            metadata={'n_games': len(season_data), 'window': window},
        )

    def fit(self, seasons: Optional[List[int]] = None):
        result = super().fit(seasons=seasons)
        self.save_config()
        return result

    def save_config(self) -> str:
        if self._latest_rec is None:
            raise RuntimeError('No fitted Recalibrator — call fit() first')
        self._latest_rec.metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        root = os.path.join(_MODULE_DIR, 'platt_params.json')
        self._latest_rec.to_file(root)
        return root

    def validate(self) -> ValidationReport:
        self._ensure_fitted()
        from training.Validation import RecalibratorValidator
        validator = RecalibratorValidator(
            games=self.games,
            fitted_model=self._latest_rec,
            full_df=self._full_df,
            fixed_slopes=self._fixed_slopes,
            window_width=self._window_width,
        )
        return validator.validate()

    def diagnostics(self, recalibrator: Optional[Recalibrator] = None) -> CalibrationResult:
        rec = recalibrator or self._latest_rec
        if rec is None:
            raise RuntimeError('No fitted Recalibrator')
        df = self._full_df
        wp = df['fav_ml_wp'].values
        out = df['fav_win'].values
        is_home = df['home_is_fav'].values.astype(bool)
        cal = rec.calibrate(wp, is_home_fav=is_home)
        seasons = sorted(df['season'].unique().tolist())
        return CalibrationResult(
            params=rec.params,
            n_games=len(df),
            log_loss_before=log_loss(out, wp),
            log_loss_after=log_loss(out, cal),
            brier_before=brier_score(out, wp),
            brier_after=brier_score(out, cal),
            seasons=seasons,
        )

    @staticmethod
    def fit_from_arrays(
        win_probs: numpy.ndarray,
        outcomes: numpy.ndarray,
        is_home_fav: numpy.ndarray,
    ) -> Recalibrator:
        df = pandas.DataFrame({
            'fav_ml_wp': win_probs,
            'fav_win': outcomes,
            'home_is_fav': is_home_fav.astype(int),
            'season': 0,
        })
        df['logit_wp'] = logit(clip_prob(win_probs))
        slopes = RecalibratorFitter._fit_split_slopes(df)
        intercepts = RecalibratorFitter._fit_split_intercepts(df, slopes)
        return Recalibrator(
            SplitPlattParams(slopes=slopes, intercepts=intercepts)
        )

    @staticmethod
    def _fit_split_slopes(df: pandas.DataFrame) -> Dict[str, float]:
        slopes = {}
        for is_home, key in [(1, 'home'), (0, 'away')]:
            sub = df[df['home_is_fav'] == is_home]
            slopes[key] = _fit_platt_full(
                sub['logit_wp'].values,
                sub['fav_win'].values,
            )[0]
        return slopes

    @staticmethod
    def _fit_split_intercepts(
        df: pandas.DataFrame,
        slopes: Dict[str, float],
    ) -> Dict[str, float]:
        intercepts = {}
        for is_home, key in [(1, 'home'), (0, 'away')]:
            sub = df[df['home_is_fav'] == is_home]
            if len(sub) < 20:
                intercepts[key] = 0.0
            else:
                intercepts[key] = _fit_intercept_fixed_slope(
                    sub['logit_wp'].values,
                    sub['fav_win'].values,
                    slopes[key],
                )
        return intercepts


def _fit_platt_full(logit_wp: numpy.ndarray, y: numpy.ndarray) -> Tuple[float, float]:
    def nll(params: numpy.ndarray) -> float:
        p = clip_prob(expit(params[0] * logit_wp + params[1]))
        return float(-numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))

    result = minimize(
        nll,
        x0=numpy.array([1.0, 0.0]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    if not result.success:
        raise RuntimeError(f'Platt optimization did not converge: {result.message}')
    return float(result.x[0]), float(result.x[1])


def _fit_intercept_fixed_slope(
    logit_wp: numpy.ndarray,
    y: numpy.ndarray,
    slope: float,
) -> float:
    def nll(b: numpy.ndarray) -> float:
        p = clip_prob(expit(slope * logit_wp + b[0]))
        return float(-numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))

    result = minimize(
        nll,
        x0=numpy.array([0.0]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    if not result.success:
        raise RuntimeError(f'Intercept optimization did not converge: {result.message}')
    return float(result.x[0])
