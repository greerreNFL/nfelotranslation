'''
Calibration Stationarity — tests whether the market ML calibration bias is
stable across seasons or drifts over time. Looks at per-season Platt
parameters, calibration error shape across eras, and weighted MAE per season.
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
from scipy.special import logit, expit
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import setup_style, C_EMPIRICAL, C_MARKET, C_NEUTRAL


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

setup_style()


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[df['ml_wp_close'].notna() & df['result'].notna() & (df['result'] != 0)].copy()
print(f'Games (no ties): {len(df):,}')
print(f'Seasons: {df["season"].min()} - {df["season"].max()}')

## fold to favorite perspective ##
df['home_is_fav'] = (df['ml_wp_close'] >= 0.50).astype(int)
df['fav_ml_wp'] = numpy.where(
    df['home_is_fav'], df['ml_wp_close'], 1.0 - df['ml_wp_close']
)
df['fav_win'] = numpy.where(
    df['home_is_fav'],
    (df['result'] > 0).astype(int),
    (df['result'] < 0).astype(int),
)
df['logit_fav_wp'] = logit(numpy.clip(df['fav_ml_wp'], 1e-6, 1.0 - 1e-6))


## ==================== Per-Season Platt Fit ==================== ##

def _fit_platt(logit_wp: numpy.ndarray, y: numpy.ndarray) -> tuple:
    '''Fit logit-linear recalibration; returns (slope, intercept).'''
    def neg_ll(params):
        p = expit(params[0] * logit_wp + params[1])
        p = numpy.clip(p, 1e-10, 1.0 - 1e-10)
        return -numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p))
    res = minimize(neg_ll, [1.0, 0.0], method='Nelder-Mead')
    return float(res.x[0]), float(res.x[1])


platt_rows = []
for season in sorted(df['season'].unique()):
    sub = df[df['season'] == season]
    if len(sub) < 50:
        continue
    slope, intercept = _fit_platt(
        sub['logit_fav_wp'].values, sub['fav_win'].values
    )
    platt_rows.append({
        'season': int(season),
        'n': len(sub),
        'slope': slope,
        'intercept': intercept,
    })
platt = pandas.DataFrame(platt_rows)

slope_reg = linregress(platt['season'], platt['slope'])
int_reg = linregress(platt['season'], platt['intercept'])

print('\n=== Per-Season Platt Fit ===')
print(platt[['season', 'n', 'slope', 'intercept']].round(4).to_string(index=False))
print(f'\nSlope: mean={platt["slope"].mean():.4f}, '
      f'std={platt["slope"].std():.4f}, '
      f'cv={platt["slope"].std() / platt["slope"].mean() * 100:.2f}%')
print(f'Slope trend: {slope_reg.slope:+.5f}/yr (p={slope_reg.pvalue:.3f})')
print(f'Intercept: mean={platt["intercept"].mean():.4f}, '
      f'std={platt["intercept"].std():.4f}')
print(f'Intercept trend: {int_reg.slope:+.5f}/yr (p={int_reg.pvalue:.3f})')


## ==================== Calibration Error by Era ==================== ##

ERAS = [
    (2006, 2010, 'Early (06-10)'),
    (2011, 2015, 'Mid (11-15)'),
    (2016, 2020, 'Late (16-20)'),
    (2021, 2025, 'Recent (21-25)'),
]
era_bin_edges = [0.50, 0.575, 0.65, 0.725, 0.80, 0.90, 1.00]

era_frames = []
for start, end, label in ERAS:
    sub = df[(df['season'] >= start) & (df['season'] <= end)].copy()
    sub['ml_bin'] = pandas.cut(sub['fav_ml_wp'], bins=era_bin_edges, include_lowest=True)
    agg = sub.groupby('ml_bin', observed=True).agg(
        n=('fav_win', 'count'),
        wins=('fav_win', 'sum'),
        mean_ml_wp=('fav_ml_wp', 'mean'),
    ).reset_index()
    agg['actual_wr'] = agg['wins'] / agg['n']
    agg['error'] = agg['actual_wr'] - agg['mean_ml_wp']
    agg['era'] = label
    era_frames.append(agg)
era = pandas.concat(era_frames, ignore_index=True)

print('\n=== Calibration Error by Era ===')
pivot = era.pivot_table(index='ml_bin', columns='era', values='error', observed=True)
print(pivot.round(4).to_string())


## ==================== WMAE per Season ==================== ##

annual_bin_edges = [0.50, 0.60, 0.70, 0.80, 1.00]
annual_rows = []
for season in sorted(df['season'].unique()):
    sub = df[df['season'] == season].copy()
    sub['ml_bin'] = pandas.cut(sub['fav_ml_wp'], bins=annual_bin_edges, include_lowest=True)
    agg = sub.groupby('ml_bin', observed=True).agg(
        n=('fav_win', 'count'),
        wins=('fav_win', 'sum'),
        mean_ml_wp=('fav_ml_wp', 'mean'),
    ).reset_index()
    agg['actual_wr'] = agg['wins'] / agg['n']
    agg['abs_error'] = numpy.abs(agg['actual_wr'] - agg['mean_ml_wp'])
    wmae = float((agg['abs_error'] * agg['n']).sum() / agg['n'].sum())
    annual_rows.append({'season': int(season), 'n': len(sub), 'wmae': wmae})
annual = pandas.DataFrame(annual_rows)
wmae_reg = linregress(annual['season'], annual['wmae'])

print('\n=== WMAE per Season ===')
print(annual.round(4).to_string(index=False))
print(f'\nWMAE trend: {wmae_reg.slope:+.5f}/yr (p={wmae_reg.pvalue:.3f})')


## ==================== Output ==================== ##

platt.to_csv(HERE / 'output.csv', index=False)
era_out = era[['era', 'ml_bin', 'n', 'mean_ml_wp', 'actual_wr', 'error']].copy()
era_out['ml_bin'] = era_out['ml_bin'].astype(str)
era_out.to_csv(HERE / 'output_era.csv', index=False)
annual.to_csv(HERE / 'output_annual.csv', index=False)
print(f'\nSaved: {HERE / "output.csv"}')


## ==================== Chart ==================== ##

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_fit = platt['season'].values

## top-left — per-season slope ##
ax = axes[0, 0]
ax.plot(platt['season'], platt['slope'], 'o-', color=C_MARKET, markersize=5, lw=1.2)
ax.axhline(platt['slope'].mean(), color=C_NEUTRAL, ls='--', lw=1,
           label=f'Mean = {platt["slope"].mean():.3f}')
ax.fill_between(
    platt['season'],
    platt['slope'].mean() - platt['slope'].std(),
    platt['slope'].mean() + platt['slope'].std(),
    alpha=0.15, color=C_MARKET,
)
ax.plot(x_fit, slope_reg.intercept + slope_reg.slope * x_fit,
        '--', color=C_EMPIRICAL, lw=1,
        label=f'Trend: {slope_reg.slope:+.4f}/yr (p={slope_reg.pvalue:.2f})')
ax.set_xlabel('Season')
ax.set_ylabel('Platt slope')
ax.set_title('Platt slope by season')
ax.legend(fontsize=8)

## top-right — per-season intercept ##
ax = axes[0, 1]
ax.plot(platt['season'], platt['intercept'], 'o-', color=C_EMPIRICAL, markersize=5, lw=1.2)
ax.axhline(platt['intercept'].mean(), color=C_NEUTRAL, ls='--', lw=1,
           label=f'Mean = {platt["intercept"].mean():.3f}')
ax.axhline(0, color='black', lw=0.5)
ax.fill_between(
    platt['season'],
    platt['intercept'].mean() - platt['intercept'].std(),
    platt['intercept'].mean() + platt['intercept'].std(),
    alpha=0.15, color=C_EMPIRICAL,
)
ax.plot(x_fit, int_reg.intercept + int_reg.slope * x_fit,
        '--', color=C_MARKET, lw=1,
        label=f'Trend: {int_reg.slope:+.4f}/yr (p={int_reg.pvalue:.2f})')
ax.set_xlabel('Season')
ax.set_ylabel('Platt intercept')
ax.set_title('Platt intercept by season')
ax.legend(fontsize=8)

## bottom-left — calibration error shape by era ##
ax = axes[1, 0]
era_colors = ['#2E86AB', '#3BB273', '#E84855', '#F49D37']
ax.plot([0.5, 1.0], [0.0, 0.0], '--', color=C_NEUTRAL, lw=0.8)
for (start, end, label), color in zip(ERAS, era_colors):
    sub = era[era['era'] == label]
    ax.plot(sub['mean_ml_wp'], sub['error'], 'o-', color=color,
            markersize=5, lw=1.2, label=f'{label} (n={sub["n"].sum():,})', alpha=0.8)
ax.set_xlabel('Favorite ML implied win probability')
ax.set_ylabel('Error (observed - ML implied)')
ax.set_title('Calibration error shape by era')
ax.legend(fontsize=8)
ax.set_xlim(0.5, 1.0)

## bottom-right — WMAE per season ##
ax = axes[1, 1]
ax.bar(annual['season'], annual['wmae'], color=C_MARKET, alpha=0.6, edgecolor='white')
ax.axhline(annual['wmae'].mean(), color=C_NEUTRAL, ls='--', lw=1,
           label=f'Mean = {annual["wmae"].mean():.4f}')
ax.plot(x_fit, wmae_reg.intercept + wmae_reg.slope * x_fit,
        '--', color=C_EMPIRICAL, lw=1.2,
        label=f'Trend: {wmae_reg.slope:+.4f}/yr (p={wmae_reg.pvalue:.2f})')
ax.set_xlabel('Season')
ax.set_ylabel('WMAE (probability units)')
ax.set_title('Weighted MAE per season')
ax.legend(fontsize=8)

fig.suptitle('ML calibration bias: stability over time', y=1.01, fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {HERE / "chart.png"}')
