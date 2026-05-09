'''
Spread Mapping Form — compares parametric (linear-in-logit) and lookup-table
forms for the win-prob → margin mapping, and contrasts a free-intercept fit
with the shipped intercept-forced-to-zero fit.

All work is done in favorite-perspective coordinates: positive WP > 0.5 means
the favorite, positive margin means the favorite covered or won by that many.
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import (
    setup_style,
    C_EMPIRICAL,
    C_MARKET,
    C_FORMULA,
    C_NEUTRAL,
)


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

setup_style()


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[
    df['fav_wp_cal'].notna()
    & df['fav_margin'].notna()
    & df['fav_spread'].notna()
    & (df['fav_spread'] > 0)
].copy()
print(f'Games with fav_wp_cal + fav_margin + fav_spread > 0: {len(df):,}')

## logit of recalibrated favorite WP — fed into the linear-in-logit fits ##
EPS = 1e-6
fav_wp = numpy.clip(df['fav_wp_cal'].values, EPS, 1.0 - EPS)
df['logit_fav_wp_cal'] = numpy.log(fav_wp / (1.0 - fav_wp))


## ==================== Fitters ==================== ##

def fit_linear_free(z: numpy.ndarray, target: numpy.ndarray) -> dict:
    '''Fit ``target = slope * z + intercept`` under MAE loss.'''
    def obj(p):
        return float(numpy.mean(numpy.abs(p[0] * z + p[1] - target)))
    res = sp_minimize(
        obj,
        x0=numpy.array([6.0, 0.5]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    return {'slope': float(res.x[0]), 'intercept': float(res.x[1])}


def fit_linear_zero(z: numpy.ndarray, target: numpy.ndarray) -> dict:
    '''Fit ``target = slope * z`` under MAE loss; intercept fixed at 0.'''
    def obj(p):
        return float(numpy.mean(numpy.abs(p[0] * z - target)))
    res = sp_minimize(
        obj,
        x0=numpy.array([6.0]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    return {'slope': float(res.x[0]), 'intercept': 0.0}


def fit_lookup(wp: numpy.ndarray, target: numpy.ndarray, bin_width: float = 0.01) -> dict:
    '''Per-bin median lookup table; returns sorted bin centers and medians.'''
    edges = numpy.arange(0.50, 1.0 + bin_width, bin_width)
    centers = []
    medians = []
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (wp >= lo) & (wp < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append((lo + hi) / 2.0)
        medians.append(float(numpy.median(target[mask])))
        counts.append(n)
    return {
        'bin_centers': numpy.array(centers),
        'medians': numpy.array(medians),
        'counts': numpy.array(counts),
    }


## ==================== Predictors ==================== ##

def predict_linear(z: numpy.ndarray, params: dict) -> numpy.ndarray:
    return params['slope'] * z + params['intercept']


def predict_lookup(wp: numpy.ndarray, table: dict) -> numpy.ndarray:
    '''
    Linear interpolation between bin centers; flat extrapolation at the tails
    so unseen WPs fall back to the nearest known bin median.
    '''
    if len(table['bin_centers']) == 0:
        return numpy.full_like(wp, numpy.nan)
    return numpy.interp(
        numpy.clip(wp, table['bin_centers'].min(), table['bin_centers'].max()),
        table['bin_centers'],
        table['medians'],
    )


## ==================== Metric Helpers ==================== ##

def metrics(predicted: numpy.ndarray, actual: numpy.ndarray) -> dict:
    '''MAE and bisection rate (fraction with actual > predicted).'''
    pred = predicted[~numpy.isnan(predicted)]
    act = actual[~numpy.isnan(predicted)]
    return {
        'n': int(len(pred)),
        'mae': float(numpy.mean(numpy.abs(act - pred))),
        'bisection_rate': float(numpy.mean(act > pred)),
    }


## ==================== In-Sample Fit ==================== ##

z_all = df['logit_fav_wp_cal'].values
margin_all = df['fav_margin'].values.astype(float)
wp_all = df['fav_wp_cal'].values

free_params = fit_linear_free(z_all, margin_all)
zero_params = fit_linear_zero(z_all, margin_all)
lookup_table = fit_lookup(wp_all, margin_all)

is_metrics = {
    'linear_free': metrics(predict_linear(z_all, free_params), margin_all),
    'linear_zero': metrics(predict_linear(z_all, zero_params), margin_all),
    'lookup': metrics(predict_lookup(wp_all, lookup_table), margin_all),
}

print('\n=== In-Sample Fits ===')
print(f'  Linear (free intercept):  slope={free_params["slope"]:.4f}, '
      f'intercept={free_params["intercept"]:.4f}')
print(f'  Linear (intercept=0):     slope={zero_params["slope"]:.4f}, '
      f'intercept=0.0000')
print(f'  Lookup table:             {len(lookup_table["bin_centers"])} bins, '
      f'min n={lookup_table["counts"].min()}, max n={lookup_table["counts"].max()}')


## ==================== Leave-One-Season-Out ==================== ##

print('\n=== Leave-One-Season-Out ===')
oos_records = {'linear_free': [], 'linear_zero': [], 'lookup': []}
seasons = sorted(df['season'].unique())
for held in seasons:
    train = df[df['season'] != held]
    test = df[df['season'] == held]
    if len(train) < 200 or len(test) < 50:
        continue
    z_tr = train['logit_fav_wp_cal'].values
    m_tr = train['fav_margin'].values.astype(float)
    wp_tr = train['fav_wp_cal'].values
    z_te = test['logit_fav_wp_cal'].values
    m_te = test['fav_margin'].values.astype(float)
    wp_te = test['fav_wp_cal'].values
    free_p = fit_linear_free(z_tr, m_tr)
    zero_p = fit_linear_zero(z_tr, m_tr)
    table = fit_lookup(wp_tr, m_tr)
    oos_records['linear_free'].append(metrics(predict_linear(z_te, free_p), m_te))
    oos_records['linear_zero'].append(metrics(predict_linear(z_te, zero_p), m_te))
    oos_records['lookup'].append(metrics(predict_lookup(wp_te, table), m_te))

oos_summary = {}
for form, records in oos_records.items():
    if not records:
        oos_summary[form] = {'mae': float('nan'), 'bisection_rate': float('nan')}
        continue
    weights = numpy.array([r['n'] for r in records], dtype=float)
    mae = numpy.array([r['mae'] for r in records])
    bis = numpy.array([r['bisection_rate'] for r in records])
    oos_summary[form] = {
        'mae': float(numpy.average(mae, weights=weights)),
        'bisection_rate': float(numpy.average(bis, weights=weights)),
    }

for form in ['linear_free', 'linear_zero', 'lookup']:
    s = oos_summary[form]
    print(f'  {form:<14s}  LOSO mae={s["mae"]:.4f}, bisection={s["bisection_rate"]:.4f}')


## ==================== Outputs ==================== ##

summary_rows = []
for form, label in [
    ('linear_free', 'Linear (free intercept)'),
    ('linear_zero', 'Linear (intercept=0)'),
    ('lookup', 'Lookup (1% bin median)'),
]:
    is_m = is_metrics[form]
    oos_m = oos_summary[form]
    if form == 'linear_free':
        slope, intercept = free_params['slope'], free_params['intercept']
    elif form == 'linear_zero':
        slope, intercept = zero_params['slope'], 0.0
    else:
        slope, intercept = float('nan'), float('nan')
    summary_rows.append({
        'form': form,
        'label': label,
        'slope': slope,
        'intercept': intercept,
        'in_sample_mae': is_m['mae'],
        'in_sample_bisection': is_m['bisection_rate'],
        'loso_mae': oos_m['mae'],
        'loso_bisection': oos_m['bisection_rate'],
    })
summary_df = pandas.DataFrame(summary_rows)
summary_df.to_csv(HERE / 'output.csv', index=False)
print(f'\nWrote: {HERE / "output.csv"}')

## per-bin predictions for the curve plot ##
bin_rows = []
for center, median, count in zip(
    lookup_table['bin_centers'], lookup_table['medians'], lookup_table['counts']
):
    z_center = numpy.log(center / (1.0 - center))
    bin_rows.append({
        'wp_center': center,
        'n': int(count),
        'empirical_median_margin': median,
        'pred_linear_free': predict_linear(numpy.array([z_center]), free_params)[0],
        'pred_linear_zero': predict_linear(numpy.array([z_center]), zero_params)[0],
    })
bins_df = pandas.DataFrame(bin_rows)
bins_df.to_csv(HERE / 'output_bins.csv', index=False)
print(f'Wrote: {HERE / "output_bins.csv"}')


## ==================== Chart ==================== ##

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

## TOP-LEFT — fitted curves vs empirical bin medians ##
ax = axes[0, 0]
ax.scatter(
    bins_df['wp_center'],
    bins_df['empirical_median_margin'],
    s=numpy.sqrt(bins_df['n']) * 3,
    color=C_EMPIRICAL,
    alpha=0.55,
    edgecolor='white',
    label='Empirical bin median',
    zorder=3,
)
wp_grid = numpy.linspace(0.50, 0.99, 200)
z_grid = numpy.log(wp_grid / (1.0 - wp_grid))
ax.plot(wp_grid, predict_linear(z_grid, free_params), color=C_MARKET, lw=2,
        label=f'Linear free (slope={free_params["slope"]:.2f}, b={free_params["intercept"]:.2f})')
ax.plot(wp_grid, predict_linear(z_grid, zero_params), color=C_FORMULA, lw=2, ls='--',
        label=f'Linear b=0 (slope={zero_params["slope"]:.2f})')
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.axvline(0.50, color=C_NEUTRAL, lw=0.8)
ax.set_xlabel('Recalibrated Favorite WP')
ax.set_ylabel('Favorite Margin (median)')
ax.set_title('Fitted curves vs empirical medians')
ax.legend(fontsize=9)

## TOP-RIGHT — residuals at z=0 (mid-range bins) for free vs forced ##
ax = axes[0, 1]
mid = bins_df[(bins_df['wp_center'] >= 0.50) & (bins_df['wp_center'] <= 0.65)]
x = numpy.arange(len(mid))
width = 0.4
ax.bar(
    x - width / 2,
    mid['empirical_median_margin'] - mid['pred_linear_free'],
    width=width,
    color=C_MARKET,
    label='Free intercept',
)
ax.bar(
    x + width / 2,
    mid['empirical_median_margin'] - mid['pred_linear_zero'],
    width=width,
    color=C_FORMULA,
    label='Intercept=0',
)
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'{c:.2f}' for c in mid['wp_center']], rotation=45, fontsize=7)
ax.set_xlabel('WP bin center')
ax.set_ylabel('Residual (empirical − model)')
ax.set_title('Mid-range residuals — bias the forced fit absorbs')
ax.legend(fontsize=9)

## BOTTOM-LEFT — per-bin sample size ##
ax = axes[1, 0]
ax.bar(
    bins_df['wp_center'],
    bins_df['n'],
    width=0.008,
    color=C_NEUTRAL,
    alpha=0.7,
    edgecolor='white',
)
ax.set_xlabel('Recalibrated Favorite WP')
ax.set_ylabel('Sample size per 1% bin')
ax.set_title('Lookup-table density by WP bin')
ax.set_yscale('log')

## BOTTOM-RIGHT — in-sample vs LOSO MAE per form ##
ax = axes[1, 1]
labels = ['Linear\nfree', 'Linear\nb=0', 'Lookup\n(1% bin)']
is_mae = [is_metrics[k]['mae'] for k in ['linear_free', 'linear_zero', 'lookup']]
oos_mae = [oos_summary[k]['mae'] for k in ['linear_free', 'linear_zero', 'lookup']]
x = numpy.arange(len(labels))
ax.bar(x - 0.2, is_mae, width=0.4, color=C_EMPIRICAL, label='In-sample')
ax.bar(x + 0.2, oos_mae, width=0.4, color=C_MARKET, label='LOSO')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('MAE (points)')
ax.set_title('In-sample vs leave-one-season-out MAE')
ax.legend(fontsize=9)
for i, (a, b) in enumerate(zip(is_mae, oos_mae)):
    ax.text(i - 0.2, a + 0.05, f'{a:.3f}', ha='center', fontsize=8)
    ax.text(i + 0.2, b + 0.05, f'{b:.3f}', ha='center', fontsize=8)

fig.suptitle('Analysis 4 — Spread Mapping Form', fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {HERE / "chart.png"}')
