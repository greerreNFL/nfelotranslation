'''
KeyModelValidator — validates the credibility-weighted KeyModel.

Evaluates OOS excess prediction RMSE/MAE, model vs baseline ratio,
per-number analysis, and stationarity.  Decoupled from the fitter:
takes games and the seasonal result directly.
'''

## external ##
import numpy
import pandas
from scipy.stats import linregress

## local ##
from nfelotranslation.Utilities.ValidationTypes import (
    TrackedMetric,
    ValidationCheck,
    ValidationReport,
)
from ..Seasonal.Types import SeasonalResult
from .Validator import Validator

## ==================== Constants ==================== ##

_ALL_NUMBERS = list(range(1, 41))


class KeyModelValidator(Validator):
    '''
    Validator for the KeyModel.

    Gated checks:
    * model beats baseline (model RMSE < baseline RMSE)
    * aggregate OOS RMSE excess < 1.5 pp
    * aggregate OOS MAE excess < 1.2 pp
    * worst individual number OOS RMSE < 5.0 pp

    Tracked metrics:
    * aggregate OOS RMSE, MAE, and baseline RMSE
    * model vs baseline RMSE ratio
    * per-season OOS RMSE trend (stationarity)
    * worst number detail
    * top 10 most-landed-on numbers detail

    Parameters:
    * games: full games DataFrame
    * seasonal_result: SeasonalResult from the seasonal fit
    '''

    def __init__(self, games: pandas.DataFrame, seasonal_result: SeasonalResult):
        self._games = games
        self._seasonal_result = seasonal_result

    @property
    def model_name(self) -> str:
        return 'key_model'

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Validate the KeyModel using OOS diagnostics from the seasonal fit.

        Returns:
        * ValidationReport with checks and metrics
        '''
        result = self._seasonal_result
        checks = []
        metrics = []
        ## --- aggregate metrics from per-season results --- ##
        rmse = result.aggregate_metrics.get('rmse_excess_pp', float('inf'))
        mae = result.aggregate_metrics.get('mean_abs_excess_pp', float('inf'))
        baseline_rmse = result.aggregate_metrics.get('baseline_rmse_pp', float('inf'))
        ## --- gated: model beats baseline --- ##
        if baseline_rmse > 0:
            ratio = rmse / baseline_rmse
        else:
            ratio = 0.0
        checks.append(ValidationCheck(
            name='model_beats_baseline',
            value=ratio,
            threshold=1.0,
            passed=ratio < 1.0,
            detail=f'model RMSE={rmse:.4f}pp vs baseline={baseline_rmse:.4f}pp (ratio={ratio:.3f})',
        ))
        ## --- gated: aggregate OOS RMSE --- ##
        checks.append(ValidationCheck(
            name='oos_rmse_excess',
            value=rmse,
            threshold=1.5,
            passed=rmse < 1.5,
            detail=f'aggregate OOS RMSE excess = {rmse:.4f} pp',
        ))
        ## --- gated: aggregate OOS MAE --- ##
        checks.append(ValidationCheck(
            name='oos_mae_excess',
            value=mae,
            threshold=1.2,
            passed=mae < 1.2,
            detail=f'aggregate OOS MAE excess = {mae:.4f} pp',
        ))
        ## --- per-number aggregation across seasons --- ##
        per_number_sq_errors = {k: [] for k in _ALL_NUMBERS}
        per_number_total_hits = {k: 0 for k in _ALL_NUMBERS}
        for diag in result.per_season:
            per_number = diag.metadata.get('per_number', {})
            for k in _ALL_NUMBERS:
                if k in per_number:
                    per_number_sq_errors[k].append(per_number[k]['error'] ** 2)
                    per_number_total_hits[k] += per_number[k]['hits']
        ## per-number OOS RMSE ##
        per_number_rmse = {}
        for k in _ALL_NUMBERS:
            errs = per_number_sq_errors[k]
            if errs:
                per_number_rmse[k] = float(numpy.sqrt(numpy.mean(errs))) * 100.0
        ## --- gated: worst individual number --- ##
        if per_number_rmse:
            worst_k = max(per_number_rmse, key=per_number_rmse.get)
            worst_rmse = per_number_rmse[worst_k]
            checks.append(ValidationCheck(
                name='worst_number_rmse',
                value=worst_rmse,
                threshold=5.0,
                passed=worst_rmse < 5.0,
                detail=f'number ±{worst_k}: OOS RMSE = {worst_rmse:.4f} pp',
            ))
        ## --- tracked: aggregate metrics --- ##
        per_season_rmse = self._extract_per_season('rmse_excess_pp')
        per_season_mae = self._extract_per_season('mean_abs_excess_pp')
        per_season_baseline = self._extract_per_season('baseline_rmse_pp')
        metrics.append(TrackedMetric(
            name='oos_rmse_excess_pp',
            value=rmse,
            per_season=per_season_rmse,
        ))
        metrics.append(TrackedMetric(
            name='oos_mae_excess_pp',
            value=mae,
            per_season=per_season_mae,
        ))
        metrics.append(TrackedMetric(
            name='baseline_rmse_pp',
            value=baseline_rmse,
            detail='RMSE if we predicted 0 excess (no key number adjustment)',
            per_season=per_season_baseline,
        ))
        metrics.append(TrackedMetric(
            name='model_vs_baseline_ratio',
            value=ratio,
            detail=f'{ratio:.3f} (lower = model adds more value vs baseline)',
        ))
        ## --- tracked: top 10 most-landed-on numbers --- ##
        top_10 = sorted(per_number_total_hits.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_detail = ', '.join(
            f'±{k}({per_number_rmse.get(k, 0):.2f}pp)'
            for k, _ in top_10
        )
        metrics.append(TrackedMetric(
            name='top_10_numbers_rmse',
            value=float(numpy.mean([per_number_rmse.get(k, 0) for k, _ in top_10])),
            detail=top_10_detail,
        ))
        ## --- tracked: worst number detail --- ##
        if per_number_rmse:
            metrics.append(TrackedMetric(
                name='worst_number_detail',
                value=worst_rmse,
                detail=f'±{worst_k}: RMSE={worst_rmse:.4f}pp, total_hits={per_number_total_hits[worst_k]}',
            ))
        ## --- stationarity: per-season OOS metric trends --- ##
        if len(result.per_season) > 2:
            seasons = numpy.array([d.season for d in result.per_season], dtype=float)
            rmses = numpy.array([d.metrics.get('rmse_excess_pp', 0.0) for d in result.per_season])
            maes = numpy.array([d.metrics.get('mean_abs_excess_pp', 0.0) for d in result.per_season])
            rmse_reg = linregress(seasons, rmses)
            mae_reg = linregress(seasons, maes)
            metrics.append(TrackedMetric(
                name='rmse_trend',
                value=float(rmse_reg.slope),
                detail=f'{rmse_reg.slope:+.5f}/yr (p={rmse_reg.pvalue:.3f})',
            ))
            metrics.append(TrackedMetric(
                name='mae_trend',
                value=float(mae_reg.slope),
                detail=f'{mae_reg.slope:+.5f}/yr (p={mae_reg.pvalue:.3f})',
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

    def _extract_per_season(self, metric_name: str) -> dict:
        '''Extract {season: value} for a metric across per-season diagnostics.'''
        per_season = {}
        for d in self._seasonal_result.per_season:
            val = d.metrics.get(metric_name, float('nan'))
            if numpy.isfinite(val):
                per_season[int(d.season)] = val
        return per_season
