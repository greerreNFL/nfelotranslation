'''
HyperparamOptimizer — optimizes KeyModel hyperparameters against the
aggregate margin distribution across all historical games.

The objective function minimizes the sum of absolute errors between
the model's predicted margin frequencies and actual margin frequencies
when aggregated across all games in the dataset.

Fixed components (loaded once, independent of hyperparams):
    - SpreadMapper
    - Actual outcome histogram
    - Per-spread (model_spread, cal_wp, count) groups
    - Per-season training data (hits, baselines, n_games)

Variable components (rebuilt each iteration):
    - KeyModel with candidate hyperparams
    - MarginDistributionModel
'''

## built-ins ##
from typing import Any, Dict, List, Optional, Tuple

## external ##
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm

## local ##
from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper
from nfelotranslation.Distribution import MARGIN_HYPERPARAMS
from nfelotranslation.Distribution.Key import KeyModel, KEY_MODEL_PARAMS
from nfelotranslation.Distribution.MarginDistributionModel import MarginDistributionModel
from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper
from training.Data import DataLoader


## ==================== Constants ==================== ##

_MARGINS = numpy.arange(-75, 76)
_ALL_NUMBERS = list(range(1, 41))

## OOS cutoff — seasons >= _OOS_MIN_SEASON are held out for SAE evaluation;
## seasons < _OOS_MIN_SEASON are used only to warm up the KeyModel.  Matches
## MarginDistributionValidator's convention so this objective scores on the
## same dataset (and using per-season snapshot predictions, the same way). ##
_OOS_MIN_SEASON = 2010

## live snapshot of shipped hyperparams — single source of truth ##
DEFAULT_PARAMS: Dict[str, Any] = {
    'forgetting_rate': float(KEY_MODEL_PARAMS['forgetting_rate']),
    'threshold': float(KEY_MODEL_PARAMS['threshold']),
    'initial_prior_size': float(KEY_MODEL_PARAMS['initial_prior_size']),
    'beta': float(MARGIN_HYPERPARAMS['beta']),
}


class HyperparamOptimizer:
    '''
    Optimizes KeyModel hyperparameters against the aggregate margin
    distribution.

    Pre-computes all fixed components on init so that the inner loop
    only rebuilds the KeyModel and MarginDistributionModel.

    Parameters:
    * games: optional DataFrame (auto-loads via DataLoader if omitted)
    '''

    def __init__(self, games=None):
        ## load all games (including pre-OOS for warm-up training) ##
        self._games = self._load_games(games)
        ## pre-compute fixed components ##
        self._per_season_actual = self._compute_per_season_actual_counts()
        ## aggregate over OOS seasons — useful for diagnostics, NOT the objective ##
        self._actual_counts = sum(self._per_season_actual.values(), numpy.zeros(151))
        self._mapper = self._load_mapper()
        self._per_season_spread_groups = self._compute_per_season_spread_groups()
        self._season_data = self._compute_season_data()
        ## iteration tracking ##
        self._n_evals = 0
        self._best_sae = float('inf')

    ## ==================== Public Interface ==================== ##

    def objective(self, x: numpy.ndarray, optimize_params: List[str], fixed_params: Dict[str, Any]) -> float:
        '''
        Compute SAE for candidate hyperparameters.

        Parameters:
        * x:               array of candidate values for optimize_params
        * optimize_params:  list of param names being optimized
        * fixed_params:     dict of params held constant

        Returns:
        * sum of absolute errors (scalar)
        '''
        ## build full params dict ##
        params = fixed_params.copy()
        for name, val in zip(optimize_params, x):
            params[name] = val
        ## split key model params from distribution params ##
        key_params = {k: v for k, v in params.items() if k != 'beta'}
        beta = params.get('beta', 2.0)
        ## per-season OOS snapshot prediction, matching the Validator ##
        per_season_pred = self._compute_per_season_predictions(key_params, beta)
        ## mean of per-season SAE — each snapshot scored against the games
        ## it was responsible for, then averaged across seasons.  Prevents
        ## per-season mispredictions from cancelling in aggregate. ##
        season_saes = [
            float(numpy.abs(per_season_pred[s] - self._per_season_actual[s]).sum())
            for s in per_season_pred
        ]
        sae = float(numpy.mean(season_saes))
        ## progress tracking ##
        self._n_evals += 1
        if sae < self._best_sae:
            self._best_sae = sae
        if self._n_evals % 10 == 0:
            print(f'  eval {self._n_evals}: SAE = {sae:.1f}  (best = {self._best_sae:.1f})')
        return sae

    def optimize(
        self,
        optimize_params: List[str],
        bounds: Dict[str, Tuple[float, float]],
        fixed_params: Optional[Dict[str, Any]] = None,
        x0: Optional[List[float]] = None,
        method: str = 'Nelder-Mead',
    ):
        '''
        Run optimization via scipy.optimize.minimize.

        Parameters:
        * optimize_params: list of param names to optimize
        * bounds:          dict mapping param name → (min, max)
        * fixed_params:    dict of params held constant (defaults fill from DEFAULT_PARAMS)
        * x0:              initial guess (defaults to current DEFAULT_PARAMS values)
        * method:          scipy.optimize.minimize method

        Returns:
        * scipy OptimizeResult with .optimized_params dict attached
        '''
        ## build fixed params (defaults for everything not being optimized) ##
        fixed = DEFAULT_PARAMS.copy()
        if fixed_params:
            fixed.update(fixed_params)
        for name in optimize_params:
            fixed.pop(name, None)
        ## initial guess ##
        if x0 is None:
            x0 = [DEFAULT_PARAMS[name] for name in optimize_params]
        ## bounds for scipy ##
        scipy_bounds = [bounds[name] for name in optimize_params]
        ## reset tracking ##
        self._n_evals = 0
        self._best_sae = float('inf')
        ## initial evaluation ##
        initial_sae = self.objective(x0, optimize_params, fixed)
        n_seasons = len(self._per_season_actual)
        avg_season_size = int(self._actual_counts.sum() / n_seasons)
        print(f'Initial mean per-season SAE: {initial_sae:.1f} games '
              f'(avg season size {avg_season_size}, {n_seasons} OOS seasons)')
        print(f'Optimizing: {optimize_params}')
        print(f'x0: {dict(zip(optimize_params, x0))}')
        print()
        ## minimize ##
        result = minimize(
            self.objective,
            x0=x0,
            args=(optimize_params, fixed),
            method=method,
            bounds=scipy_bounds,
            options={'disp': True},
        )
        ## attach readable params ##
        result.optimized_params = {
            name: val for name, val in zip(optimize_params, result.x)
        }
        ## print results ##
        print()
        print(f'Evaluations: {self._n_evals}')
        print(f'Initial mean per-season SAE: {initial_sae:.1f} games')
        print(f'Final   mean per-season SAE: {result.fun:.1f} games')
        print(f'Improvement:                 {initial_sae - result.fun:.1f} games')
        print()
        print('Optimized params:')
        for name, val in result.optimized_params.items():
            print(f'  {name}: {DEFAULT_PARAMS[name]} -> {val:.6f}')
        return result

    ## ==================== Pre-computation ==================== ##

    def _load_games(self, games):
        '''Load ALL seasons with result, spread, and calibrated WP.

        Pre-OOS seasons are kept so they can warm up the KeyModel during
        training; the SAE objective only scores seasons >= _OOS_MIN_SEASON.
        '''
        if games is None:
            games = DataLoader.get().games.copy()
        mask = (
            games['result'].notna()
            & games['spread_line'].notna()
            & games['ml_wp_cal'].notna()
        )
        df = games[mask].copy()
        df['margin'] = df['result'].round().astype(int)
        return df

    def _compute_per_season_actual_counts(self):
        '''dict: OOS season -> 151-element margin count histogram.

        Per-season separation lets the objective score each snapshot against
        the games it was actually responsible for, rather than pooling errors
        across seasons (which allows per-season mispredictions to cancel).'''
        result = {}
        oos = self._games[self._games['season'] >= _OOS_MIN_SEASON]
        for season, season_games in oos.groupby('season'):
            counts = numpy.zeros(151)
            for margin in season_games['margin'].values:
                if -75 <= margin <= 75:
                    counts[margin + 75] += 1
            result[int(season)] = counts
        return result

    def _load_mapper(self):
        '''Load root SpreadMapper config.'''
        return SpreadMapper.from_file()

    def _compute_per_season_spread_groups(self):
        '''
        For each season, list of (model_spread, cal_wp, count) groups.
        Uses training labels (ml_wp_cal) and the fitted SpreadMapper.
        '''
        per_season = {}
        for season, season_games in self._games.groupby('season'):
            groups = {}
            for _, row in season_games.iterrows():
                if pandas.isna(row.get('ml_wp_cal')):
                    continue
                cal_wp = float(row['ml_wp_cal'])
                model_spread = self._mapper.win_prob_to_spread(cal_wp)
                key = (float(model_spread.continuous), cal_wp)
                groups[key] = groups.get(key, 0) + 1
            per_season[int(season)] = [
                {'model_spread': ms, 'cal_wp': wp, 'count': n}
                for (ms, wp), n in groups.items()
            ]
        return per_season

    def _compute_season_data(self):
        '''
        Pre-compute per-season training inputs: hits at each ±k,
        n_games, and baseline rates.  All independent of hyperparams.
        '''
        seasons = sorted(self._games['season'].unique())
        result = []
        for season in seasons:
            season_games = self._games[self._games['season'] == season]
            n_games = len(season_games)
            margins = season_games['margin'].values
            ## hits at each ±k ##
            hits = {}
            for k in _ALL_NUMBERS:
                hits[k] = int((margins == k).sum() + (margins == -k).sum())
            ## baseline rates ##
            baselines = _compute_baselines(season_games)
            result.append({
                'season': int(season),
                'n_games': n_games,
                'hits': hits,
                'baselines': baselines,
            })
        return result

    ## ==================== Objective Helpers ==================== ##

    def _compute_per_season_predictions(self, key_params, beta):
        '''
        Walk seasons in order; for each OOS season, predict using the
        KeyModel trained through the prior season (matching
        MarginDistributionValidator's per-season-snapshot pattern).
        Pre-OOS seasons are used only to warm up the model.

        Returns dict: OOS season -> 151-element predicted margin histogram.
        Per-season separation is what lets the objective avoid the
        pathology where per-season mispredictions cancel in aggregate.
        '''
        key_model = KeyModel.from_initial(key_params)
        per_season = {}
        for sd in self._season_data:
            season = sd['season']
            if season >= _OOS_MIN_SEASON:
                ## predict this OOS season using the current snapshot
                ## (trained through season-1) ##
                margin_model = MarginDistributionModel(key_model, beta=beta)
                agg = numpy.zeros(151)
                for g in self._per_season_spread_groups[season]:
                    dist = margin_model.predict(g['model_spread'], g['cal_wp'])
                    agg += dist.pmf * g['count']
                per_season[season] = agg
            ## then fold this season's data into the model ##
            for k in _ALL_NUMBERS:
                key_model.outcomes[k].update(
                    sd['hits'][k], sd['n_games'], sd['baselines'][k], season,
                )
        return per_season

    def _compute_aggregate_counts_oos(self, key_params, beta):
        '''
        Diagnostic helper — returns the pooled predicted histogram across
        OOS seasons.  NOT used by the objective (see
        _compute_per_season_predictions).
        '''
        per_season = self._compute_per_season_predictions(key_params, beta)
        return sum(per_season.values(), numpy.zeros(151))

    def _train_key_model(self, params):
        '''Train a KeyModel through all seasons (used by diagnostics /
        external callers who want the final in-sample state).'''
        model = KeyModel.from_initial(params)
        for sd in self._season_data:
            for k in _ALL_NUMBERS:
                model.outcomes[k].update(
                    sd['hits'][k], sd['n_games'], sd['baselines'][k], sd['season'],
                )
        return model


## ==================== Module-Level Helpers ==================== ##

def _compute_baselines(season_data):
    '''
    Compute season-aggregate baseline rates for each ±k.

    For each game, derive sigma from (spread, wp), evaluate the normal
    PMF at all integers in [-75, 75], normalize to sum 1, and accumulate.

    Parameters:
    * season_data: DataFrame for one season

    Returns:
    * dict mapping k (1-40) → combined baseline rate for ±k
    '''
    agg = numpy.zeros(len(_MARGINS), dtype=float)
    n_valid = 0
    for _, row in season_data.iterrows():
        mu = float(row['spread_line'])
        wp = float(row['ml_wp_cal'])
        wp = numpy.clip(wp, 0.01, 0.99)
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
    baselines = {}
    for k in _ALL_NUMBERS:
        baselines[k] = float(avg_pmf[k + 75]) + float(avg_pmf[-k + 75])
    return baselines
