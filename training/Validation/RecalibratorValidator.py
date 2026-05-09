'''
RecalibratorValidator — validates Platt / logit-linear recalibration.

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
from nfelotranslation.Utilities.MathUtils import logit, expit, clip_prob, log_loss, brier_score
from .Validator import Validator


## ==================== Helpers ==================== ##

def _fold_to_favorite(games: pandas.DataFrame):
    '''
    Extract (win_probs, outcomes) folded to favorite perspective.

    Every game contributes one observation with win_prob >= 0.5.
    This isolates the market's compression-toward-50% bias.
    '''
    mask = games['ml_wp_close'].notna() & games['result'].notna()
    df = games[mask].copy()
    df = df[df['result'] != 0]
    home_wp = df['ml_wp_close'].values.astype(float)
    home_won = (df['result'] > 0).values.astype(float)
    is_home_fav = home_wp >= 0.5
    win_probs = numpy.where(is_home_fav, home_wp, 1.0 - home_wp)
    outcomes = numpy.where(is_home_fav, home_won, 1.0 - home_won)
    return win_probs, outcomes

## ==================== Constants ==================== ##

_MIN_FOLD_GAMES = 50


class RecalibratorValidator(Validator):
    '''
    Validator for the Recalibrator.

    Gated checks:
    * slope > 1.05 (market must hedge toward 50%; historical mean ~1.16)
    * in-sample log loss improves after recalibration

    Tracked metrics:
    * in-sample log loss and Brier improvements
    * LOSO pooled OOS log loss and Brier improvements
    * per-season Platt slope trend and CV
    * per-season Platt intercept trend

    Parameters:
    * games: full games DataFrame
    * fitted_model: the fitted Recalibrator to validate
    '''

    def __init__(self, games: pandas.DataFrame, fitted_model: Recalibrator):
        self._games = games
        self._fitted_model = fitted_model

    @property
    def model_name(self) -> str:
        return 'recalibrator'

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Run LOSO cross-validation and stationarity checks.

        Returns:
        * ValidationReport with checks and metrics
        '''
        insample = self._run_insample()
        checks = []
        metrics = []
        ## --- gated: slope > 1.05 (historical mean ~1.16) --- ##
        slope = self._fitted_model.params.slope
        checks.append(ValidationCheck(
            name='slope_positive',
            value=slope,
            threshold=1.05,
            passed=slope > 1.05,
            detail=f'slope={slope:.4f}; market must hedge toward 50% (historical mean ~1.16)',
        ))
        ## --- gated: in-sample log loss improvement --- ##
        checks.append(ValidationCheck(
            name='insample_log_loss_improvement',
            value=insample['log_loss_improvement'],
            threshold=0.0,
            passed=insample['log_loss_improvement'] > 0,
            detail=f'{insample["log_loss_before"]:.6f} -> {insample["log_loss_after"]:.6f}',
        ))
        ## --- tracked: in-sample metrics --- ##
        metrics.append(TrackedMetric(
            name='insample_log_loss_improvement',
            value=insample['log_loss_improvement'],
        ))
        metrics.append(TrackedMetric(
            name='insample_brier_improvement',
            value=insample['brier_improvement'],
        ))
        ## --- LOSO cross-validation --- ##
        loso_results = self._run_loso()
        if loso_results['n_folds'] > 0:
            loso_ll_imp = loso_results['pooled_ll_before'] - loso_results['pooled_ll_after']
            ## tracked, not gated — recalibrator corrects bias for training ##
            ## data, so OOS predictive improvement is not the goal ##
            metrics.append(TrackedMetric(
                name='loso_oos_log_loss_improvement',
                value=loso_ll_imp,
                detail=f'{loso_results["pooled_ll_before"]:.6f} -> {loso_results["pooled_ll_after"]:.6f}',
            ))
            metrics.append(TrackedMetric(
                name='loso_oos_brier_improvement',
                value=loso_results['pooled_brier_before'] - loso_results['pooled_brier_after'],
            ))
        ## --- stationarity: per-season parameter trends --- ##
        stationarity = self._run_stationarity()
        if stationarity['n_seasons'] > 2:
            metrics.append(TrackedMetric(
                name='slope_trend',
                value=stationarity['slope_trend'],
                detail=f'{stationarity["slope_trend"]:+.5f}/yr (p={stationarity["slope_p"]:.3f})',
            ))
            metrics.append(TrackedMetric(
                name='intercept_trend',
                value=stationarity['intercept_trend'],
                detail=f'{stationarity["intercept_trend"]:+.5f}/yr (p={stationarity["intercept_p"]:.3f})',
            ))
            metrics.append(TrackedMetric(
                name='slope_cv_pct',
                value=stationarity['slope_cv'],
                detail=f'mean={stationarity["slope_mean"]:.4f} std={stationarity["slope_std"]:.4f}',
            ))
        ## build report ##
        data_through = int(self._games['season'].max())
        return ValidationReport(
            model_name=self.model_name,
            data_through=data_through,
            checks=checks,
            metrics=metrics,
        )

    ## ==================== Private ==================== ##

    def _prepare_data(self) -> tuple:
        '''Extract (win_probs, outcomes) in favorite perspective.'''
        return _fold_to_favorite(self._games)

    def _run_insample(self) -> dict:
        '''Compute in-sample calibration metrics.'''
        win_probs, outcomes = self._prepare_data()
        cal = self._fitted_model.calibrate(win_probs)
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
        '''Yield leave-one-season-out folds from games.'''
        seasons = sorted(self._games['season'].unique())
        for season in seasons:
            train = self._games[self._games['season'] != season]
            test = self._games[self._games['season'] == season]
            yield season, train, test

    def _fit_from_games(self, games: pandas.DataFrame) -> Recalibrator:
        '''Fit a Recalibrator from a games DataFrame (favorite perspective).'''
        win_probs, outcomes = _fold_to_favorite(games)
        if len(win_probs) < _MIN_FOLD_GAMES:
            return None
        from training.Calibration.RecalibratorFitter import RecalibratorFitter
        return RecalibratorFitter.fit_from_arrays(win_probs, outcomes)

    def _run_loso(self) -> dict:
        '''Run LOSO cross-validation, return pooled metrics.'''
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
            ## prepare test data (favorite perspective) ##
            test_wp, test_out = _fold_to_favorite(test_games)
            if len(test_wp) < _MIN_FOLD_GAMES:
                continue
            cal = fold_rec.calibrate(test_wp)
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
        '''
        Fit the Platt model on each season in isolation and report
        trend statistics across the resulting per-season parameters.
        '''
        years, slopes, intercepts = [], [], []
        for season in sorted(self._games['season'].unique()):
            season_games = self._games[self._games['season'] == season]
            fold_rec = self._fit_from_games(season_games)
            if fold_rec is None:
                continue
            years.append(int(season))
            slopes.append(fold_rec.params.slope)
            intercepts.append(fold_rec.params.intercept)
        if len(years) < 3:
            return {'n_seasons': len(years)}
        slopes_arr = numpy.array(slopes)
        intercepts_arr = numpy.array(intercepts)
        years_arr = numpy.array(years, dtype=float)
        slope_reg = linregress(years_arr, slopes_arr)
        int_reg = linregress(years_arr, intercepts_arr)
        return {
            'n_seasons': len(years),
            'slope_mean': float(slopes_arr.mean()),
            'slope_std': float(slopes_arr.std()),
            'slope_cv': float(slopes_arr.std() / slopes_arr.mean() * 100) if slopes_arr.mean() != 0 else 0.0,
            'slope_trend': float(slope_reg.slope),
            'slope_p': float(slope_reg.pvalue),
            'intercept_mean': float(intercepts_arr.mean()),
            'intercept_std': float(intercepts_arr.std()),
            'intercept_trend': float(int_reg.slope),
            'intercept_p': float(int_reg.pvalue),
        }
