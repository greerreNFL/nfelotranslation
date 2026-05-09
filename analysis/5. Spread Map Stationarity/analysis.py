'''
Spread Map Stationarity — refits the model and market mappers on each
season in isolation and tests the resulting per-season parameters for a
trend across time. A stationary parameter set supports a single static
fit; a drifting parameter set would call for decay weighting or a
re-estimation cadence.

The model mapper is fit with intercept fixed at zero (matching the
shipped form). The market mapper is fit with a free intercept (matching
the shipped form).
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize
from scipy.stats import linregress

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import setup_style, C_EMPIRICAL, C_MARKET, C_NEUTRAL


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

setup_style()


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[
    df['fav_wp_cal'].notna()
    & df['fav_ml_wp'].notna()
    & df['fav_margin'].notna()
    & df['fav_spread'].notna()
    & (df['fav_spread'] > 0)
].copy()

EPS = 1e-6
df['logit_fav_wp_cal'] = numpy.log(
    numpy.clip(df['fav_wp_cal'].values, EPS, 1.0 - EPS)
    / (1.0 - numpy.clip(df['fav_wp_cal'].values, EPS, 1.0 - EPS))
)
df['logit_fav_ml_wp'] = numpy.log(
    numpy.clip(df['fav_ml_wp'].values, EPS, 1.0 - EPS)
    / (1.0 - numpy.clip(df['fav_ml_wp'].values, EPS, 1.0 - EPS))
)

print(f'Games: {len(df):,}, '
      f'seasons {int(df["season"].min())}–{int(df["season"].max())}')


## ==================== Fitters ==================== ##

def fit_zero(z: numpy.ndarray, target: numpy.ndarray) -> float:
    '''MAE fit of slope only; intercept fixed at 0.'''
    def obj(p):
        return float(numpy.mean(numpy.abs(p[0] * z - target)))
    res = sp_minimize(
        obj, x0=numpy.array([6.0]), method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    return float(res.x[0])


def fit_free(z: numpy.ndarray, target: numpy.ndarray) -> tuple:
    '''MAE fit of slope and intercept jointly.'''
    def obj(p):
        return float(numpy.mean(numpy.abs(p[0] * z + p[1] - target)))
    res = sp_minimize(
        obj, x0=numpy.array([6.0, 0.0]), method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    return float(res.x[0]), float(res.x[1])


## ==================== Per-Season Fits ==================== ##

rows = []
for season, sub in df.groupby('season'):
    if len(sub) < 100:
        continue
    z_cal = sub['logit_fav_wp_cal'].values
    z_close = sub['logit_fav_ml_wp'].values
    margin = sub['fav_margin'].values.astype(float)
    spread = sub['fav_spread'].values.astype(float)
    model_slope = fit_zero(z_cal, margin)
    market_slope, market_intercept = fit_free(z_close, spread)
    rows.append({
        'season': int(season),
        'n': int(len(sub)),
        'model_slope': model_slope,
        'market_slope': market_slope,
        'market_intercept': market_intercept,
    })

per_season = pandas.DataFrame(rows)
per_season.to_csv(HERE / 'output.csv', index=False)
print(f'\nWrote: {HERE / "output.csv"}')


## ==================== Trend Tests ==================== ##

def trend(name: str, values: numpy.ndarray) -> dict:
    seasons = per_season['season'].values
    res = linregress(seasons, values)
    return {
        'name': name,
        'slope_per_year': float(res.slope),
        'intercept': float(res.intercept),
        'r_value': float(res.rvalue),
        'p_value': float(res.pvalue),
        'std_err': float(res.stderr),
        'mean': float(numpy.mean(values)),
        'std': float(numpy.std(values, ddof=1)),
    }


trend_rows = [
    trend('model_slope', per_season['model_slope'].values),
    trend('market_slope', per_season['market_slope'].values),
    trend('market_intercept', per_season['market_intercept'].values),
]
trend_df = pandas.DataFrame(trend_rows)
trend_df.to_csv(HERE / 'output_trend.csv', index=False)
print(f'Wrote: {HERE / "output_trend.csv"}')

print('\n=== Per-season Means and Trends ===')
for r in trend_rows:
    print(
        f'  {r["name"]:<18s}  mean={r["mean"]:.4f}  std={r["std"]:.4f}  '
        f'trend/yr={r["slope_per_year"]:+.4f}  p={r["p_value"]:.4f}'
    )


## ==================== Chart ==================== ##

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

def plot_series(ax, label: str, values: numpy.ndarray, color: str, t: dict) -> None:
    ax.plot(per_season['season'], values, marker='o', color=color, lw=1.6)
    ax.axhline(t['mean'], color=C_NEUTRAL, lw=1, ls='--',
               label=f'mean={t["mean"]:.3f}')
    fit_line = t['intercept'] + t['slope_per_year'] * per_season['season']
    ax.plot(per_season['season'], fit_line, color=C_NEUTRAL, lw=1, ls=':',
            label=f'trend/yr={t["slope_per_year"]:+.3f}, p={t["p_value"]:.3f}')
    ax.set_xlabel('Season')
    ax.set_ylabel(label)
    ax.legend(fontsize=9)


plot_series(axes[0, 0], 'Model slope (intercept=0)',
            per_season['model_slope'].values, C_EMPIRICAL, trend_rows[0])
axes[0, 0].set_title('Model mapper slope by season')

plot_series(axes[0, 1], 'Market slope',
            per_season['market_slope'].values, C_MARKET, trend_rows[1])
axes[0, 1].set_title('Market mapper slope by season')

plot_series(axes[1, 0], 'Market intercept',
            per_season['market_intercept'].values, C_MARKET, trend_rows[2])
axes[1, 0].set_title('Market mapper intercept by season')

## n per season for context ##
ax = axes[1, 1]
ax.bar(per_season['season'], per_season['n'], color=C_NEUTRAL, alpha=0.7,
       edgecolor='white')
ax.set_xlabel('Season')
ax.set_ylabel('Games per season')
ax.set_title('Sample size by season')

fig.suptitle('Analysis 5 — Spread Map Stationarity', fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {HERE / "chart.png"}')
