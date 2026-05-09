'''
MarginDistributionValidator — validates the full margin distribution pipeline
(Base + Key + Normalizer) using out-of-sample aggregate fit metrics.

This is a system-level validator that runs after all components (Recalibrator,
SpreadMapper, KeyModel) are trained.  It uses per-season key model configs
so each game is predicted with only data available before that season.

Evaluates: aggregate SAE, regional bias, key number accuracy, tail mass,
and per-season stationarity.
'''

## built-ins ##
import pathlib
from typing import Dict, Optional

## external ##
import numpy
import pandas
import nfelotranslation
from scipy.stats import linregress

## local ##
from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Distribution.Key import KEY_MODEL_PARAMS, KeyModel
from nfelotranslation.Distribution.MarginDistributionModel import MarginDistributionModel
from nfelotranslation.Utilities.JsonIo import find_config_path
from nfelotranslation.Utilities.ValidationTypes import TrackedMetric, ValidationCheck, ValidationReport
from nfelotranslation.SpreadMap.SpreadMapper import load_mapper_pair
from .Validator import Validator

## ==================== Constants ==================== ##

## first OOS season — hold out earlier years while model learns ##
_MIN_SEASON = 2010

_PKG_ROOT = pathlib.Path(nfelotranslation.__file__).resolve().parent

## config directory for per-season key model files ##
_CONFIG_DIR = str(_PKG_ROOT / 'Distribution' / 'Key' / 'configs')

## output directory for validation reports ##
_OUTPUT_DIR = str(_PKG_ROOT / 'Distribution')

## key numbers to report individually ##
_KEY_NUMBERS = [3, 7, 10, 14, 17, 21]


class MarginDistributionValidator(Validator):
    '''
    System-level validator for the full margin distribution pipeline.

    Uses per-season key model configs for OOS evaluation: each game is
    predicted using only data available before that season.  Seasons
    before min_season are held out to let the model learn.

    Gated checks:
    * OOS SAE % < 15% (aggregate fit quality)
    * |tail bias| < 10% (tail mass accuracy)
    * |close region bias| < 5% (close-game accuracy)

    Tracked metrics:
    * aggregate SAE (total and %)
    * regional breakdown (close, mid, tail) — bias and SAE
    * per key number predicted/actual ratios
    * tail mass ratio
    * per-season SAE trend (stationarity)

    Parameters:
    * games: full games DataFrame
    * min_season: first OOS season (default: 2010)
    '''

    def __init__(
        self,
        games: pandas.DataFrame,
        min_season: int = _MIN_SEASON,
    ):
        self._games = games
        self._min_season = min_season

    @property
    def model_name(self) -> str:
        return 'margin_distribution'

    @property
    def output_dir(self) -> str:
        return _OUTPUT_DIR

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Run OOS aggregate fit validation.

        Returns:
        * ValidationReport with checks and metrics
        '''
        ## load fixed components ##
        rec = Recalibrator.from_file()
        model_mapper, market_mapper, _ = load_mapper_pair()
        ## load per-season margin models ##
        oos_games = self._games[self._games['season'] >= self._min_season].copy()
        seasons = sorted(oos_games['season'].unique())
        mm_cache = {}
        for season in seasons:
            mm_cache[int(season)] = self._load_margin_model(int(season))
        ## compute aggregate predicted vs actual counts ##
        model_counts = numpy.zeros(151)
        actual_counts = numpy.zeros(151)
        per_season_sae = {}
        for season in seasons:
            season_games = oos_games[oos_games['season'] == season]
            mm = mm_cache[int(season)]
            season_model = numpy.zeros(151)
            season_actual = numpy.zeros(151)
            for spread_val, group in season_games.groupby('spread_line'):
                n = len(group)
                market_wp = float(market_mapper.spread_to_win_prob(float(spread_val)))
                cal_wp = float(rec.calibrate(numpy.array([market_wp]))[0])
                model_spread = model_mapper.win_prob_to_spread(cal_wp)
                dist = mm.predict(float(model_spread.continuous), cal_wp)
                season_model += dist.pmf * n
                for margin in group['result'].values:
                    m_int = int(round(margin))
                    if -75 <= m_int <= 75:
                        season_actual[m_int + 75] += 1
            model_counts += season_model
            actual_counts += season_actual
            ## per-season SAE as percentage ##
            n_season = len(season_games)
            if n_season > 0:
                season_sae = float(numpy.abs(season_model - season_actual).sum())
                per_season_sae[int(season)] = season_sae / n_season * 100
        ## fold to absolute margins for regional analysis ##
        f_pred = self._fold(model_counts)
        f_act = self._fold(actual_counts)
        f_resid = f_pred - f_act
        ## aggregate metrics ##
        n_total = len(oos_games)
        sae_total = float(numpy.abs(model_counts - actual_counts).sum())
        sae_pct = sae_total / n_total * 100
        ## regional metrics ##
        regions = self._compute_regions(f_pred, f_act, f_resid)
        ## build checks and metrics ##
        checks = self._build_checks(sae_pct, sae_total, n_total, regions)
        metrics = self._build_metrics(
            sae_total, sae_pct, n_total, regions,
            f_pred, f_act, per_season_sae,
        )
        tables = self._build_tables(regions, f_pred, f_act)
        ## build report ##
        data_through = int(self._games['season'].max())
        return ValidationReport(
            model_name=self.model_name,
            data_through=data_through,
            checks=checks,
            metrics=metrics,
            tables=tables,
        )

    def save(self, report: ValidationReport) -> str:
        '''
        Save the validation report to Distribution/validation/.

        Returns:
        * path to the saved file
        '''
        val_dir = pathlib.Path(self.output_dir) / 'validation'
        filepath = str(val_dir / f'{self.model_name}.json')
        report.to_file(filepath)
        return filepath

    ## ==================== Checks ==================== ##

    def _build_checks(self, sae_pct, sae_total, n_total, regions):
        '''Build gated validation checks.'''
        checks = []
        ## SAE percentage ##
        checks.append(ValidationCheck(
            name='oos_sae_pct',
            value=sae_pct,
            threshold=15.0,
            passed=sae_pct < 15.0,
            detail=f'SAE={sae_total:.0f} games ({sae_pct:.1f}%) across {n_total:,} OOS games',
        ))
        ## tail bias ##
        tail = regions['tail']
        checks.append(ValidationCheck(
            name='tail_bias_pct',
            value=abs(tail['bias_pct']),
            threshold=10.0,
            passed=abs(tail['bias_pct']) < 10.0,
            detail=(
                f'tail (|m|>=17): pred={tail["pred"]:.0f} '
                f'act={tail["act"]:.0f} ({tail["bias_pct"]:+.1f}%)'
            ),
        ))
        ## close region bias ##
        close = regions['close']
        checks.append(ValidationCheck(
            name='close_bias_pct',
            value=abs(close['bias_pct']),
            threshold=5.0,
            passed=abs(close['bias_pct']) < 5.0,
            detail=(
                f'close (1-6): pred={close["pred"]:.0f} '
                f'act={close["act"]:.0f} ({close["bias_pct"]:+.1f}%)'
            ),
        ))
        return checks

    ## ==================== Metrics ==================== ##

    def _build_metrics(
        self, sae_total, sae_pct, n_total, regions,
        f_pred, f_act, per_season_sae,
    ):
        '''Build tracked metrics.'''
        metrics = []
        ## aggregate SAE ##
        metrics.append(TrackedMetric(
            name='oos_sae_total',
            value=sae_total,
            detail=f'{sae_total:.0f} games ({sae_pct:.1f}%)',
        ))
        metrics.append(TrackedMetric(
            name='oos_sae_pct',
            value=sae_pct,
            detail=f'{sae_pct:.1f}% of {n_total:,} OOS games',
            per_season=per_season_sae,
        ))
        metrics.append(TrackedMetric(
            name='oos_n_games',
            value=float(n_total),
            detail=f'{n_total:,} games (seasons {self._min_season}+)',
        ))
        ## regional bias ##
        for name, region in regions.items():
            metrics.append(TrackedMetric(
                name=f'{name}_region_bias_pct',
                value=region['bias_pct'],
                detail=(
                    f'{region["label"]}: pred={region["pred"]:.0f} '
                    f'act={region["act"]:.0f} SAE={region["sae"]:.0f}'
                ),
            ))
        ## key number ratios ##
        for k in _KEY_NUMBERS:
            p, a = f_pred[k], f_act[k]
            ratio = p / a if a > 0 else float('inf')
            metrics.append(TrackedMetric(
                name=f'key_{k}_ratio',
                value=ratio,
                detail=f'+-{k}: pred={p:.0f} act={a:.0f} resid={p - a:+.0f}',
            ))
        ## tail mass ratio ##
        tail = regions['tail']
        tail_ratio = tail['pred'] / tail['act'] if tail['act'] > 0 else float('inf')
        metrics.append(TrackedMetric(
            name='tail_mass_ratio',
            value=tail_ratio,
            detail=f'pred={tail["pred"]:.0f} act={tail["act"]:.0f} ({(tail_ratio - 1) * 100:+.1f}%)',
        ))
        ## stationarity: per-season SAE trend ##
        if len(per_season_sae) > 2:
            season_arr = numpy.array(sorted(per_season_sae.keys()), dtype=float)
            sae_arr = numpy.array([per_season_sae[int(s)] for s in season_arr])
            reg = linregress(season_arr, sae_arr)
            metrics.append(TrackedMetric(
                name='sae_trend',
                value=float(reg.slope),
                detail=f'{reg.slope:+.4f}%/yr (p={reg.pvalue:.3f})',
            ))
        return metrics

    ## ==================== Tables ==================== ##

    def _build_tables(self, regions, f_pred, f_act):
        '''Build formatted output tables.'''
        tables = {}
        ## regional breakdown ##
        lines = []
        lines.append(f'{"Range":<16} {"Pred":>8} {"Actual":>8} {"Bias":>9} {"Bias%":>8} {"SAE":>8}')
        for region in regions.values():
            bias = region['pred'] - region['act']
            lines.append(
                f'{region["label"]:<16} {region["pred"]:>8.1f} {region["act"]:>8.0f} '
                f'{bias:>+9.1f} {region["bias_pct"]:>+7.1f}% {region["sae"]:>8.1f}'
            )
        tables['Regional Breakdown'] = '\n'.join(lines)
        ## key number detail ##
        lines = []
        lines.append(f'{"k":>3} {"Pred":>8} {"Actual":>8} {"Ratio":>7} {"Resid":>9}')
        for k in _KEY_NUMBERS:
            p, a = f_pred[k], f_act[k]
            r = p / a if a > 0 else float('inf')
            lines.append(f'{k:>3} {p:>8.1f} {a:>8.0f} {r:>7.3f} {p - a:>+9.1f}')
        tables['Key Number Detail'] = '\n'.join(lines)
        return tables

    ## ==================== Private ==================== ##

    @staticmethod
    def _load_margin_model(season: int) -> MarginDistributionModel:
        '''Load MarginDistributionModel using per-season KeyModel config.'''
        config_path = find_config_path('key_model', season, _CONFIG_DIR)
        if config_path is not None:
            key_model = KeyModel.from_file(filepath=config_path, params=KEY_MODEL_PARAMS)
        else:
            ## first-season warmup: no per-season config exists yet.
            ## Fresh, no-key-signal model is the deliberate choice here
            ## (different from Translator, which uses the latest fit). ##
            key_model = KeyModel.from_initial(KEY_MODEL_PARAMS)
        return MarginDistributionModel(key_model)

    @staticmethod
    def _fold(arr: numpy.ndarray) -> numpy.ndarray:
        '''Fold 151-element array to 76 elements by combining +-k.'''
        f = numpy.zeros(76)
        f[0] = arr[75]
        for k in range(1, 76):
            f[k] = arr[75 + k] + arr[75 - k]
        return f

    @staticmethod
    def _compute_regions(f_pred, f_act, f_resid) -> Dict[str, dict]:
        '''Compute regional breakdown metrics.'''
        regions = {}
        for name, label, lo, hi in [
            ('close', 'Close (1-6)', 1, 6),
            ('mid', 'Mid (7-16)', 7, 16),
            ('tail', 'Tail (17-40)', 17, 40),
        ]:
            pred = float(f_pred[lo:hi + 1].sum())
            act = float(f_act[lo:hi + 1].sum())
            bias_pct = (pred - act) / act * 100 if act > 0 else 0.0
            sae = float(numpy.abs(f_resid[lo:hi + 1]).sum())
            regions[name] = {
                'label': label,
                'pred': pred,
                'act': act,
                'bias_pct': bias_pct,
                'sae': sae,
            }
        return regions
