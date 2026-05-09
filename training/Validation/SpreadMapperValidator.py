'''
SpreadMapperValidator — validates the SpreadMapper seasonal fit.

Evaluates OOS diagnostics, market round-trip accuracy, spread gaps,
stationarity, and binned median R².  Decoupled from the fitter:
takes games, the seasonal result, and both mappers directly.
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
    * model slope > 0
    * aggregate OOS model bisection rate near 0.5
    * aggregate OOS market round-trip R² > 0.98

    Tracked metrics:
    * aggregate OOS model MAE, bisection rate
    * aggregate OOS market MAE, R²
    * market round-trip baseline ratio
    * binned median R² and MAE
    * spread gap at WP=70% and its trend
    * final model and market params
    * per-season model and market MAE trends

    Parameters:
    * games: full games DataFrame
    * seasonal_result: SeasonalResult from the seasonal fit
    * model_mapper: fitted model SpreadMapper
    * market_mapper: fitted market SpreadMapper
    '''

    def __init__(
        self,
        games: pandas.DataFrame,
        seasonal_result: SeasonalResult,
        model_mapper: SpreadMapper,
        market_mapper: SpreadMapper,
    ):
        self._games = games
        self._seasonal_result = seasonal_result
        self._model_mapper = model_mapper
        self._market_mapper = market_mapper

    @property
    def model_name(self) -> str:
        return 'spread_mapper'

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Validate using OOS diagnostics from the seasonal fit.

        Returns:
        * ValidationReport with checks and metrics
        '''
        result = self._seasonal_result
        checks = []
        metrics = []
        ## --- gated: model slope positive --- ##
        slope = self._model_mapper.params.slope
        checks.append(ValidationCheck(
            name='slope_positive',
            value=slope,
            threshold=0.0,
            passed=slope > 0,
            detail='logit-WP and spread must be positively related',
        ))
        ## --- gated: bisection rate near 0.5 --- ##
        agg_bisection = result.aggregate_metrics.get('model_bisection', float('nan'))
        bisection_dev = abs(agg_bisection - 0.5)
        checks.append(ValidationCheck(
            name='bisection_rate_centered',
            value=agg_bisection,
            threshold=0.05,
            passed=bisection_dev < 0.05,
            detail=f'|{agg_bisection:.4f} - 0.5| = {bisection_dev:.4f}',
        ))
        ## --- gated: market round-trip R² --- ##
        agg_market_r2 = result.aggregate_metrics.get('market_r2', float('nan'))
        checks.append(ValidationCheck(
            name='market_roundtrip_r2',
            value=agg_market_r2,
            threshold=0.98,
            passed=agg_market_r2 > 0.98,
            detail=f'aggregate OOS market round-trip R² = {agg_market_r2:.5f}',
        ))
        ## --- tracked: aggregate OOS metrics --- ##
        agg_market_mae = result.aggregate_metrics.get('market_mae', float('nan'))
        per_season_model_mae = self._extract_per_season('model_mae')
        per_season_bisection = self._extract_per_season('model_bisection')
        per_season_market_mae = self._extract_per_season('market_mae')
        per_season_market_r2 = self._extract_per_season('market_r2')
        metrics.append(TrackedMetric(
            name='model_mae',
            value=result.aggregate_metrics.get('model_mae', float('nan')),
            per_season=per_season_model_mae,
        ))
        metrics.append(TrackedMetric(
            name='model_bisection',
            value=agg_bisection,
            per_season=per_season_bisection,
        ))
        metrics.append(TrackedMetric(
            name='market_mae',
            value=agg_market_mae,
            per_season=per_season_market_mae,
        ))
        metrics.append(TrackedMetric(
            name='market_r2',
            value=agg_market_r2,
            per_season=per_season_market_r2,
        ))
        ## --- tracked: market round-trip baseline ratio --- ##
        ## market mapper uses ml_wp_close, so bin by market WPs ##
        prepared = self._prepare_data()
        market_wps = prepared['ml_wp_close'].values.astype(float)
        actual_spreads = prepared['spread_line'].values.astype(float)
        bin_width = 0.005
        bins = numpy.round(market_wps / bin_width) * bin_width
        unique_bins = numpy.unique(bins)
        pooled_medians = {}
        for b in unique_bins:
            b_mask = bins == b
            if b_mask.sum() >= 20:
                pooled_medians[b] = float(numpy.median(actual_spreads[b_mask]))
        valid_mask = numpy.array([b in pooled_medians for b in bins])
        baseline_preds = numpy.array([pooled_medians.get(b, 0.0) for b in bins])
        baseline_mae = float(numpy.mean(
            numpy.abs(actual_spreads[valid_mask] - baseline_preds[valid_mask])
        ))
        baseline_ratio = agg_market_mae / baseline_mae if baseline_mae > 0 else float('nan')
        metrics.append(TrackedMetric(
            name='market_baseline_ratio',
            value=baseline_ratio,
            detail=f'model={agg_market_mae:.3f} / baseline={baseline_mae:.3f} (1.0=noise floor)',
        ))
        ## --- tracked: binned median R² and MAE --- ##
        bin_mids_list, bin_med_list, bin_pred_list = [], [], []
        for b in sorted(pooled_medians.keys()):
            bin_mids_list.append(b)
            bin_med_list.append(pooled_medians[b])
            bin_pred_list.append(float(self._market_mapper.win_prob_to_spread(b).continuous))
        bin_med_arr = numpy.array(bin_med_list)
        bin_pred_arr = numpy.array(bin_pred_list)
        bin_errors = bin_pred_arr - bin_med_arr
        binned_mae = float(numpy.mean(numpy.abs(bin_errors)))
        ss_res_b = float(numpy.sum(bin_errors ** 2))
        ss_tot_b = float(numpy.sum((bin_med_arr - numpy.mean(bin_med_arr)) ** 2))
        binned_r2 = 1.0 - ss_res_b / ss_tot_b if ss_tot_b > 0 else float('nan')
        metrics.append(TrackedMetric(
            name='market_binned_r2',
            value=binned_r2,
            detail=f'{len(bin_mids_list)} bins with ≥20 games',
        ))
        metrics.append(TrackedMetric(
            name='market_binned_mae',
            value=binned_mae,
            detail=f'parametric form vs empirical bin medians',
        ))
        ## --- tracked: final params --- ##
        metrics.append(TrackedMetric(
            name='model_slope',
            value=self._model_mapper.params.slope,
            detail=f'intercept={self._model_mapper.params.intercept:.4f}',
        ))
        metrics.append(TrackedMetric(
            name='market_slope',
            value=self._market_mapper.params.slope,
            detail=f'intercept={self._market_mapper.params.intercept:.4f}',
        ))
        ## --- stationarity: per-season metric trends --- ##
        valid_seasons = [
            d for d in result.per_season
            if numpy.isfinite(d.metrics.get('model_mae', float('nan')))
        ]
        if len(valid_seasons) > 2:
            seasons_arr = numpy.array([d.season for d in valid_seasons], dtype=float)
            model_maes = numpy.array([d.metrics['model_mae'] for d in valid_seasons])
            market_maes = numpy.array([d.metrics['market_mae'] for d in valid_seasons])
            model_reg = linregress(seasons_arr, model_maes)
            market_reg = linregress(seasons_arr, market_maes)
            metrics.append(TrackedMetric(
                name='model_mae_trend',
                value=float(model_reg.slope),
                detail=f'{model_reg.slope:+.5f}/yr (p={model_reg.pvalue:.3f})',
            ))
            metrics.append(TrackedMetric(
                name='market_mae_trend',
                value=float(market_reg.slope),
                detail=f'{market_reg.slope:+.5f}/yr (p={market_reg.pvalue:.3f})',
            ))
        ## --- tracked: spread gap at WP=70% --- ##
        if valid_seasons:
            latest_gap = valid_seasons[-1].metadata.get('spread_gap', {})
            gap_70 = latest_gap.get('wp_70', float('nan'))
            metrics.append(TrackedMetric(
                name='spread_gap_wp70',
                value=gap_70,
                detail='market_spread - model_spread at WP=70% (latest season)',
            ))
            ## spread gap trend ##
            gap_70_seasons = []
            gap_70_values = []
            for d in valid_seasons:
                g = d.metadata.get('spread_gap', {}).get('wp_70', float('nan'))
                if numpy.isfinite(g):
                    gap_70_seasons.append(d.season)
                    gap_70_values.append(g)
            if len(gap_70_seasons) > 2:
                gap_reg = linregress(
                    numpy.array(gap_70_seasons, dtype=float),
                    numpy.array(gap_70_values),
                )
                metrics.append(TrackedMetric(
                    name='spread_gap_wp70_trend',
                    value=float(gap_reg.slope),
                    detail=f'{gap_reg.slope:+.4f} pts/yr (p={gap_reg.pvalue:.3f})',
                ))
        ## --- table: spread gap pivot (season × WP) --- ##
        tables = {}
        if valid_seasons:
            gap_wps = [0.50, 0.60, 0.70, 0.80]
            gap_labels = ['WP=50%', 'WP=60%', 'WP=70%', 'WP=80%']
            header = f'{"Season":>8}  ' + ''.join(f'{l:>10}' for l in gap_labels)
            sep = '-' * (10 + 10 * len(gap_labels))
            table_lines = [header, sep]
            for d in valid_seasons:
                gap = d.metadata.get('spread_gap', {})
                row = f'{int(d.season):>8}  '
                row += ''.join(f'{gap.get(f"wp_{int(w*100)}", float("nan")):>+10.3f}' for w in gap_wps)
                table_lines.append(row)
            table_lines.append(sep)
            ## trend per WP ##
            for wp_val in gap_wps:
                col_key = f'wp_{int(wp_val * 100)}'
                vals = []
                szns = []
                for d in valid_seasons:
                    g = d.metadata.get('spread_gap', {}).get(col_key, float('nan'))
                    if numpy.isfinite(g):
                        vals.append(g)
                        szns.append(d.season)
                if len(szns) > 2:
                    reg = linregress(numpy.array(szns, dtype=float), numpy.array(vals))
                    sig = '*' if reg.pvalue < 0.05 else ''
                    table_lines.append(
                        f'  WP={int(wp_val*100)}%: {reg.slope:+.4f} pts/yr (p={reg.pvalue:.3f}{sig})'
                    )
            tables['Spread Gap Pivot (market - model)'] = '\n'.join(table_lines)
        ## build report ##
        data_through = int(self._games['season'].max())
        return ValidationReport(
            model_name=self.model_name,
            data_through=data_through,
            checks=checks,
            metrics=metrics,
            tables=tables,
        )

    ## ==================== Private ==================== ##

    def _prepare_data(self) -> pandas.DataFrame:
        '''Filter to games with both market and recalibrated win probs, spreads, and results.'''
        games = self._games
        mask = (
            games['ml_wp_close'].notna()
            & games['ml_wp_cal'].notna()
            & games['spread_line'].notna()
            & games['result'].notna()
        )
        return games[mask].copy()

    def _extract_per_season(self, metric_name: str) -> dict:
        '''Extract {season: value} for a metric across valid per-season diagnostics.'''
        per_season = {}
        for d in self._seasonal_result.per_season:
            val = d.metrics.get(metric_name, float('nan'))
            if numpy.isfinite(val):
                per_season[int(d.season)] = val
        return per_season
