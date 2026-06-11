'''
RecalibratorValidator — validates split Platt recalibration.

Runs LOSO cross-validation and stationarity checks on a fitted
Recalibrator.  Decoupled from the fitter: takes games and the
fitted model directly.
'''

## external ##
import numpy
import pandas
from scipy.stats import linregress

## local ##
from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Utilities.ValidationTypes import (
    TrackedMetric,
    ValidationCheck,
    ValidationReport,
)
from nfelotranslation.Utilities.MathUtils import log_loss, brier_score
from .Validator import Validator


## ==================== Helpers ==================== ##

def _fold_to_favorite(games: pandas.DataFrame):
    '''
    Extract (win_probs, outcomes, is_home_fav) folded to favorite perspective.

    Every game contributes one observation with win_prob >= 0.5.
    '''
    mask = games['ml_wp_close'].notna() & games['result'].notna()
    df = games[mask].copy()
    df = df[df['result'] != 0]
    home_wp = df['ml_wp_close'].values.astype(float)
    home_won = (df['result'] > 0).values.astype(float)
    is_home_fav = home_wp >= 0.5
    win_probs = numpy.where(is_home_fav, home_wp, 1.0 - home_wp)
    outcomes = numpy.where(is_home_fav, home_won, 1.0 - home_won)
    return win_probs, outcomes, is_home_fav


## ==================== Constants ==================== ##

_MIN_FOLD_GAMES = 50


class RecalibratorValidator(Validator):
    '''
    Validator for the Recalibrator.

    Gated checks:
    * home slope > 1.05 (market must hedge toward 50%)
    * in-sample log loss improves after recalibration

    Tracked metrics:
    * in-sample log loss and Brier improvements
    * LOSO pooled OOS log loss and Brier improvements
    * per-season home/away intercept trends
    '''

    def __init__(
        self,
        games: pandas.DataFrame,
        fitted_model: Recalibrator,
        full_df: pandas.DataFrame = None,
        fixed_slopes: dict = None,
        window_width: int = 5,
    ):
        self._games = games
        self._fitted_model = fitted_model
        self._full_df = full_df
        self._fixed_slopes = fixed_slopes
        self._window_width = window_width

    @property
    def model_name(self) -> str:
        return 'recalibrator'

    def validate(self) -> ValidationReport:
        insample = self._run_insample()
        checks = []
        metrics = []
        home_slope = self._fitted_model.params.slopes['home']
        checks.append(ValidationCheck(
            name='home_slope_positive',
            value=home_slope,
            threshold=1.05,
            passed=home_slope > 1.05,
            detail=f'home_slope={home_slope:.4f}; market must hedge toward 50%',
        ))
        checks.append(ValidationCheck(
            name='insample_log_loss_improvement',
            value=insample['log_loss_improvement'],
            threshold=0.0,
            passed=insample['log_loss_improvement'] > 0,
            detail=f'{insample["log_loss_before"]:.6f} -> {insample["log_loss_after"]:.6f}',
        ))
        metrics.append(TrackedMetric(
            name='insample_log_loss_improvement',
            value=insample['log_loss_improvement'],
        ))
        metrics.append(TrackedMetric(
            name='insample_brier_improvement',
            value=insample['brier_improvement'],
        ))
        loso_results = self._run_loso()
        if loso_results['n_folds'] > 0:
            loso_ll_imp = loso_results['pooled_ll_before'] - loso_results['pooled_ll_after']
            metrics.append(TrackedMetric(
                name='loso_oos_log_loss_improvement',
                value=loso_ll_imp,
                detail=f'{loso_results["pooled_ll_before"]:.6f} -> {loso_results["pooled_ll_after"]:.6f}',
            ))
            metrics.append(TrackedMetric(
                name='loso_oos_brier_improvement',
                value=loso_results['pooled_brier_before'] - loso_results['pooled_brier_after'],
            ))
        stationarity = self._run_stationarity()
        if stationarity['n_seasons'] > 2:
            metrics.append(TrackedMetric(
                name='home_intercept_trend',
                value=stationarity['home_intercept_trend'],
                detail=f'{stationarity["home_intercept_trend"]:+.5f}/yr (p={stationarity["home_intercept_p"]:.3f})',
            ))
            metrics.append(TrackedMetric(
                name='away_intercept_trend',
                value=stationarity['away_intercept_trend'],
                detail=f'{stationarity["away_intercept_trend"]:+.5f}/yr (p={stationarity["away_intercept_p"]:.3f})',
            ))
        data_through = int(self._games['season'].max())
        return ValidationReport(
            model_name=self.model_name,
            data_through=data_through,
            checks=checks,
            metrics=metrics,
        )

    def _prepare_data(self) -> tuple:
        return _fold_to_favorite(self._games)

    def _run_insample(self) -> dict:
        win_probs, outcomes, is_home_fav = self._prepare_data()
        cal = self._fitted_model.calibrate(win_probs, is_home_fav=is_home_fav)
        ll_before = log_loss(outcomes, win_probs)
        ll_after = log_loss(outcomes, cal)
        brier_before = brier_score(outcomes, win_probs)
        brier_after = brier_score(outcomes, cal)
        return {
            'log_loss_before': ll_before,
            'log_loss_after': ll_after,
            'log_loss_improvement': ll_before - ll_after,
            'brier_before': brier_before,
            'brier_after': brier_after,
            'brier_improvement': brier_before - brier_after,
        }

    def _loso_folds(self):
        seasons = sorted(self._games['season'].unique())
        for season in seasons:
            train = self._games[self._games['season'] != season]
            test = self._games[self._games['season'] == season]
            yield season, train, test

    def _fit_from_games(self, games: pandas.DataFrame) -> Recalibrator:
        win_probs, outcomes, is_home_fav = _fold_to_favorite(games)
        if len(win_probs) < _MIN_FOLD_GAMES:
            return None
        from training.Calibration.RecalibratorFitter import RecalibratorFitter
        return RecalibratorFitter.fit_from_arrays(win_probs, outcomes, is_home_fav)

    def _run_loso(self) -> dict:
        total_n = 0
        sum_ll_before = 0.0
        sum_ll_after = 0.0
        sum_brier_before = 0.0
        sum_brier_after = 0.0
        n_folds = 0
        for season, train_games, test_games in self._loso_folds():
            fold_rec = self._fit_from_games(train_games)
            if fold_rec is None:
                continue
            test_wp, test_out, test_home = _fold_to_favorite(test_games)
            if len(test_wp) < _MIN_FOLD_GAMES:
                continue
            cal = fold_rec.calibrate(test_wp, is_home_fav=test_home)
            n = len(test_wp)
            sum_ll_before += log_loss(test_out, test_wp) * n
            sum_ll_after += log_loss(test_out, cal) * n
            sum_brier_before += brier_score(test_out, test_wp) * n
            sum_brier_after += brier_score(test_out, cal) * n
            total_n += n
            n_folds += 1
        if total_n == 0:
            return {'n_folds': 0}
        return {
            'n_folds': n_folds,
            'total_games': total_n,
            'pooled_ll_before': sum_ll_before / total_n,
            'pooled_ll_after': sum_ll_after / total_n,
            'pooled_brier_before': sum_brier_before / total_n,
            'pooled_brier_after': sum_brier_after / total_n,
        }

    def _run_stationarity(self) -> dict:
        years, home_ints, away_ints = [], [], []
        for season in sorted(self._games['season'].unique()):
            season_games = self._games[self._games['season'] == season]
            fold_rec = self._fit_from_games(season_games)
            if fold_rec is None:
                continue
            years.append(int(season))
            home_ints.append(fold_rec.params.intercepts['home'])
            away_ints.append(fold_rec.params.intercepts['away'])
        if len(years) < 3:
            return {'n_seasons': len(years)}
        years_arr = numpy.array(years, dtype=float)
        home_reg = linregress(years_arr, numpy.array(home_ints))
        away_reg = linregress(years_arr, numpy.array(away_ints))
        return {
            'n_seasons': len(years),
            'home_intercept_trend': float(home_reg.slope),
            'home_intercept_p': float(home_reg.pvalue),
            'away_intercept_trend': float(away_reg.slope),
            'away_intercept_p': float(away_reg.pvalue),
        }
