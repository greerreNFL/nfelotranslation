'''
Key Number Distance Dependence — for every integer k in 1..40, measures how
per-bucket excess (empirical hit rate minus baseline hit rate) at signed margin
+/- k varies with the distance between the favorite-perspective spread and the
signed margin.

Each integer is treated as one concept: both `+k` and `-k` margins are
occurrences of "number k", with per-occurrence distance `|fav_spread - signed_margin|`.

The trainer's variable-scale Gaussian baseline `N(loc=spread, scale=spread / norm.ppf(wp))`
is precomputed once per spread bucket as the average per-game baseline PMF, so any
integer's bucket-level baseline rate is an index lookup.

A weighted linear OLS is fit per integer:
    excess = intercept + slope * distance
with bucket sample size as the weight.  Slope sign separates persistent excess
(positive slope, excess grows with distance) from proximity-dependent excess
(negative slope, excess fades with distance).
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

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


## ==================== Constants ==================== ##

## integer domain for the baseline PMF lookup ##
MARGINS = numpy.arange(-75, 76)
IDX_OFFSET: int = 75

## tracked margins (matches the 40 NumberOutcome trackers in KeyModel) ##
ALL_NUMBERS = list(range(1, 41))

## minimum games per spread bucket to be included in OLS ##
MIN_N_SPREAD: int = 30

## upper bound on |fav_spread - signed_margin| for OLS inclusion ##
MAX_DIST: float = 35.0

## fallback scale for pickem games — same as BaseDistribution._FALLBACK_SCALE ##
FALLBACK_SCALE: float = 13.2

## numerical guard for the (spread ~ 0, wp ~ 0.5) degenerate case ##
DEGEN_EPS: float = 1e-6


## ==================== Baseline Helpers ==================== ##

def trainer_sigma(spread: float, win_prob: float) -> float:
    '''Variable-scale Gaussian sigma used by KeyModelFitter._compute_baselines.'''
    if abs(spread) < DEGEN_EPS or abs(win_prob - 0.5) < DEGEN_EPS:
        return FALLBACK_SCALE
    return spread / scipy_norm.ppf(win_prob)


def game_baseline_pmf(spread: float, win_prob: float) -> numpy.ndarray:
    '''Return the per-game variable-scale Gaussian PMF over MARGINS, normalized to 1.'''
    sigma = trainer_sigma(spread, win_prob)
    if not numpy.isfinite(sigma) or sigma <= 0:
        return None
    raw = scipy_norm.pdf(MARGINS.astype(float), loc=spread, scale=sigma)
    total = raw.sum()
    if total < 1e-15:
        return None
    return raw / total


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[
    df['fav_margin'].notna()
    & df['fav_spread'].notna()
    & df['fav_wp_cal'].notna()
    & (df['fav_spread'] >= 0)
    & (df['fav_wp_cal'] >= 0.5)
].copy()
df['fav_margin_int'] = df['fav_margin'].astype(int)
print(f'Games with fav_margin + fav_spread + fav_wp_cal: {len(df):,}')


## ==================== Precompute Per-Bucket Baselines ==================== ##

print('Precomputing per-bucket baseline PMFs...')
bucket_baselines = {}
bucket_n = {}
for spread_val, grp in df.groupby('fav_spread'):
    n = len(grp)
    if n < MIN_N_SPREAD:
        continue
    pmf_sum = numpy.zeros(len(MARGINS))
    n_valid = 0
    for _, row in grp.iterrows():
        pmf = game_baseline_pmf(float(row['fav_spread']), float(row['fav_wp_cal']))
        if pmf is None:
            continue
        pmf_sum += pmf
        n_valid += 1
    if n_valid > 0:
        bucket_baselines[spread_val] = pmf_sum / n_valid
        bucket_n[spread_val] = n
print(f'  {len(bucket_baselines)} spread buckets with >={MIN_N_SPREAD} games')


## ==================== Per-Bucket Excess Points ==================== ##

print('Computing per-bucket excess for all integers in 1..40...')
all_points = []
for k in ALL_NUMBERS:
    for signed_margin in (+k, -k):
        idx = signed_margin + IDX_OFFSET
        for spread_val, n in bucket_n.items():
            dist = abs(float(spread_val) - signed_margin)
            if dist > MAX_DIST:
                continue
            grp = df[df['fav_spread'] == spread_val]
            emp_rate = float((grp['fav_margin_int'] == signed_margin).sum()) / n
            base_rate = float(bucket_baselines[spread_val][idx])
            excess = emp_rate - base_rate
            all_points.append({
                'number': int(k),
                'signed_margin': int(signed_margin),
                'fav_spread': float(spread_val),
                'distance': float(dist),
                'n_bucket': int(n),
                'empirical_rate': emp_rate,
                'baseline_rate': base_rate,
                'excess_rate': excess,
            })
points_df = pandas.DataFrame(all_points)
points_df.to_csv(HERE / 'output_raw_points.csv', index=False)
print(f'  Total data points: {len(points_df):,}')
print(f'Wrote: {HERE / "output_raw_points.csv"}')


## ==================== Per-Integer Weighted OLS ==================== ##

print('Fitting weighted OLS per integer...')
fit_rows = []
for k in ALL_NUMBERS:
    kdf = points_df[points_df['number'] == k]
    if len(kdf) < 6:
        fit_rows.append({
            'number': int(k),
            'n_points': int(len(kdf)),
            'mean_excess_rate': 0.0,
            'weighted_mean_excess_rate': 0.0,
            'intercept': 0.0,
            'slope_per_pt': 0.0,
            'r2_weighted': 0.0,
        })
        continue
    xs = kdf['distance'].values
    ys = kdf['excess_rate'].values
    ws = kdf['n_bucket'].values.astype(float)
    sw = ws.sum()
    swx = (ws * xs).sum()
    swx2 = (ws * xs ** 2).sum()
    swy = (ws * ys).sum()
    swxy = (ws * xs * ys).sum()
    denom = sw * swx2 - swx ** 2
    if abs(denom) > 1e-12:
        slope = (sw * swxy - swx * swy) / denom
        intercept = (swy - slope * swx) / sw
        y_wm = swy / sw
        ss_tot = (ws * (ys - y_wm) ** 2).sum()
        ss_res = (ws * (ys - (intercept + slope * xs)) ** 2).sum()
        r2 = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
    else:
        slope, intercept, r2 = 0.0, 0.0, 0.0
    fit_rows.append({
        'number': int(k),
        'n_points': int(len(kdf)),
        'mean_excess_rate': float(ys.mean()),
        'weighted_mean_excess_rate': float(swy / sw),
        'intercept': float(intercept),
        'slope_per_pt': float(slope),
        'r2_weighted': float(r2),
    })
fit_df = pandas.DataFrame(fit_rows)
fit_df.to_csv(HERE / 'output.csv', index=False)
print(f'Wrote: {HERE / "output.csv"}')


## ==================== Console Summary ==================== ##

print('\n=== Per-integer weighted OLS fits (excess_rate ~ distance) ===')
print('  k    w_mean      intercept   slope/pt     R²')
for _, r in fit_df.iterrows():
    print(
        f'  {int(r["number"]):>2d}   {r["weighted_mean_excess_rate"]:+.4f}    '
        f'{r["intercept"]:+.4f}    {r["slope_per_pt"]:+.5f}    {r["r2_weighted"]:.3f}'
    )

print('\n=== Excess integers (weighted mean > +0.003), sorted by mean ===')
ex = fit_df[fit_df['weighted_mean_excess_rate'] > 0.003].sort_values(
    'weighted_mean_excess_rate', ascending=False
)
for _, r in ex.iterrows():
    sl = r['slope_per_pt']
    if sl > 0.0002:
        cat = 'persistent (excess grows with distance)'
    elif sl < -0.0002:
        cat = 'proximity-dependent (excess fades with distance)'
    else:
        cat = 'constant'
    print(
        f'  k={int(r["number"]):>2d}  mean={r["weighted_mean_excess_rate"]:+.4f}  '
        f'slope={sl:+.5f}/pt  R²={r["r2_weighted"]:.3f}  ({cat})'
    )

print('\n=== Deficit integers (weighted mean < -0.003), sorted by mean ===')
defc = fit_df[fit_df['weighted_mean_excess_rate'] < -0.003].sort_values(
    'weighted_mean_excess_rate'
)
for _, r in defc.iterrows():
    sl = r['slope_per_pt']
    if sl > 0.0002:
        cat = 'proximity-dependent (deficit fades with distance)'
    elif sl < -0.0002:
        cat = 'persistent (deficit grows with distance)'
    else:
        cat = 'constant'
    print(
        f'  k={int(r["number"]):>2d}  mean={r["weighted_mean_excess_rate"]:+.4f}  '
        f'slope={sl:+.5f}/pt  R²={r["r2_weighted"]:.3f}  ({cat})'
    )


## ==================== Chart ==================== ##

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

## LEFT — weighted mean excess landscape across all 40 integers ##
ax = axes[0]
xs_bar = fit_df['number'].values
ys_bar = fit_df['weighted_mean_excess_rate'].values
colors = [C_FORMULA if y > 0 else C_EMPIRICAL for y in ys_bar]
ax.bar(xs_bar, ys_bar, color=colors, alpha=0.85, edgecolor='white', linewidth=0.3)
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xlabel('Margin number k')
ax.set_ylabel('Weighted mean excess rate')
ax.set_title('Weighted mean excess at each integer')
ax.set_xticks(range(1, 41, 2))
ax.tick_params(labelsize=8)

## CENTER — slope per point across all 40 integers ##
ax = axes[1]
slope_vals = fit_df['slope_per_pt'].values
slope_colors = [C_FORMULA if s > 0 else C_EMPIRICAL for s in slope_vals]
ax.bar(xs_bar, slope_vals, color=slope_colors, alpha=0.85, edgecolor='white', linewidth=0.3)
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xlabel('Margin number k')
ax.set_ylabel('Slope of excess vs distance (per point)')
ax.set_title('Distance slope at each integer')
ax.set_xticks(range(1, 41, 2))
ax.tick_params(labelsize=8)

## RIGHT — slope vs mean excess scatter (persistence quadrant map) ##
ax = axes[2]
me = fit_df['weighted_mean_excess_rate'].values
sl = fit_df['slope_per_pt'].values
nums = fit_df['number'].values.astype(int)
sizes = numpy.clip(numpy.abs(me) * 800, 20, 220)
scatter_colors = [C_FORMULA if m > 0.003 else (C_EMPIRICAL if m < -0.003 else C_NEUTRAL)
                  for m in me]
ax.scatter(me, sl, s=sizes, c=scatter_colors, alpha=0.7, edgecolor='#333', linewidth=0.5)
for i, k in enumerate(nums):
    if abs(me[i]) > 0.005 or abs(sl[i]) > 0.001:
        ax.annotate(str(k), (me[i], sl[i]), textcoords='offset points',
                    xytext=(6, 4), fontsize=7,
                    fontweight='bold' if abs(me[i]) > 0.01 else 'normal')
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.axvline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xlabel('Weighted mean excess rate')
ax.set_ylabel('Slope of excess vs distance (per point)')
ax.set_title('Mean excess vs distance slope')
ax.tick_params(labelsize=9)

fig.suptitle('Analysis 9 — Key Number Distance Dependence', fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nWrote: {HERE / "chart.png"}')
