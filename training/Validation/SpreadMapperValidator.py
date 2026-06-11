'''
SpreadMapperValidator — validates the SpreadMapper seasonal fit.

Evaluates OOS diagnostics, stationarity, and binned median R².
Decoupled from the fitter: takes games, the seasonal result, and
the mapper directly.
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
from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper
from ..Seasonal.Types import SeasonalResult
from .Validator import Validator


class SpreadMapperValidator(Validator):
    '''
    Validator for the SpreadMapper.

    Gated checks:
    * slope > 0
    * aggregate OOS bisection rate near 0.5

    Tracked metrics:
    * aggregate OOS MAE, bisection rate
    * binned median R² and MAE
    * final mapper params
    * per-season MAE trend
    '''

    def __init__(
        self,
        games: pandas.DataFrame,
        seasonal_result: SeasonalResult,
        mapper: SpreadMapper,
    ):
        self._games = games
        self._seasonal_result = seasonal_result
        self._mapper = mapper

    @property
    def model_name(self) -> str:
        return 'spread_mapper'

    def validate(self) -> ValidationReport:
        result = self._seasonal_result
        checks = []
        metrics = []
        slope = self._mapper.params.slope
        checks.append(ValidationCheck(
            name='slope_positive',
            value=slope,
            threshold=0.0,
            passed=slope > 0,
            detail='logit-WP and spread must be positively related',
        ))
        agg_bisection = result.aggregate_metrics.get('bisection', float('nan'))
        bisection_dev = abs(agg_bisection - 0.5)
        checks.append(ValidationCheck(
            name='bisection_rate_centered',
            value=agg_bisection,
            threshold=0.05,
            passed=bisection_dev < 0.05,
            detail=f'|{agg_bisection:.4f} - 0.5| = {bisection_dev:.4f}',
        ))
        per_season_mae = self._extract_per_season('mae')
        per_season_bisection = self._extract_per_season('bisection')
        metrics.append(TrackedMetric(
            name='mae',
            value=result.aggregate_metrics.get('mae', float('nan')),
            per_season=per_season_mae,
        ))
        metrics.append(TrackedMetric(
            name='bisection',
            value=agg_bisection,
            per_season=per_season_bisection,
        ))
        prepared = self._prepare_data()
        cal_wps = prepared['ml_wp_cal'].values.astype(float)
        actual_margins = prepared['margin'].values.astype(float)
        bin_width = 0.005
        bins = numpy.round(cal_wps / bin_width) * bin_width
        unique_bins = numpy.unique(bins)
        pooled_medians = {}
        for b in unique_bins:
            b_mask = bins == b
            if b_mask.sum() >= 20:
                pooled_medians[b] = float(numpy.median(actual_margins[b_mask]))
        bin_mids_list, bin_med_list, bin_pred_list = [], [], []
        for b in sorted(pooled_medians.keys()):
            bin_mids_list.append(b)
            bin_med_list.append(pooled_medians[b])
            bin_pred_list.append(float(self._mapper.win_prob_to_spread(b).continuous))
        bin_med_arr = numpy.array(bin_med_list)
        bin_pred_arr = numpy.array(bin_pred_list)
        bin_errors = bin_pred_arr - bin_med_arr
        binned_mae = float(numpy.mean(numpy.abs(bin_errors)))
        ss_res_b = float(numpy.sum(bin_errors ** 2))
        ss_tot_b = float(numpy.sum((bin_med_arr - numpy.mean(bin_med_arr)) ** 2))
        binned_r2 = 1.0 - ss_res_b / ss_tot_b if ss_tot_b > 0 else float('nan')
        metrics.append(TrackedMetric(
            name='binned_r2',
            value=binned_r2,
            detail=f'{len(bin_mids_list)} bins with ≥20 games',
        ))
        metrics.append(TrackedMetric(
            name='binned_mae',
            value=binned_mae,
            detail='parametric form vs empirical bin medians',
        ))
        metrics.append(TrackedMetric(
            name='slope',
            value=self._mapper.params.slope,
            detail=f'intercept={self._mapper.params.intercept:.4f}',
        ))
        valid_seasons = [
            d for d in result.per_season
            if numpy.isfinite(d.metrics.get('mae', float('nan')))
        ]
        if len(valid_seasons) > 2:
            seasons_arr = numpy.array([d.season for d in valid_seasons], dtype=float)
            maes = numpy.array([d.metrics['mae'] for d in valid_seasons])
            reg = linregress(seasons_arr, maes)
            metrics.append(TrackedMetric(
                name='mae_trend',
                value=float(reg.slope),
                detail=f'{reg.slope:+.5f}/yr (p={reg.pvalue:.3f})',
            ))
        data_through = int(self._games['season'].max())
        return ValidationReport(
            model_name=self.model_name,
            data_through=data_through,
            checks=checks,
            metrics=metrics,
        )

    def _prepare_data(self) -> pandas.DataFrame:
        games = self._games
        mask = (
            games['ml_wp_close'].notna()
            & games['ml_wp_cal'].notna()
            & games['spread_line'].notna()
            & games['result'].notna()
        )
        return games[mask].copy()

    def _extract_per_season(self, metric_name: str) -> dict:
        per_season = {}
        for d in self._seasonal_result.per_season:
            val = d.metrics.get(metric_name, float('nan'))
            if numpy.isfinite(val):
                per_season[int(d.season)] = val
        return per_season
