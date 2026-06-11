'''
Key Number Excess Magnitude — measures observed minus baseline hit rates at
each integer margin |k| in 1..40, characterizes per-season variance, and
compares pre-2015 vs 2015+ eras.  Reports two baselines:

    1. Variable-scale Gaussian:   N(loc=spread, scale=spread / norm.ppf(wp))
    2. Generalized normal:        gennorm(beta) with the same scale derivation

The Gaussian is the baseline the Key model trainer uses to compute expected
hits.  The gennorm is the baseline the inference path applies the trained
ratios against.  Both are reported so the Key README's magnitude and stability
tables are reproducible against either side of the train/inference split.
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from scipy.stats import gennorm as scipy_gennorm

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

## shipped Base shape parameter (margin_hyperparams.json).  Hard-coded here so the
## analysis tracks whatever was shipped at the time it was last run; the value is
## also reported in the output so the snapshot is self-describing. ##
SHIPPED_BETA: float = 1.24

## fallback scale for pickem games — same as BaseDistribution._FALLBACK_SCALE ##
FALLBACK_SCALE: float = 13.2

## numerical guard for the (spread ~ 0, wp ~ 0.5) degenerate case ##
DEGEN_EPS: float = 1e-6

## integers k for which we report excess.  Range matches the Key module's tracked
## margins (NumberOutcome trackers run for k in 1..40). ##
K_RANGE = list(range(1, 41))

## era boundary — 2015 PAT rule change (PAT moved from 2-yard to 15-yard line) ##
ERA_BOUNDARY: int = 2015

## headline integers for the per-season panel (not used to filter the table —
## just the line series drawn on the right-hand chart) ##
HEADLINE = [3, 7, 9, 10, 12, 14]


## ==================== Baseline Helpers ==================== ##

def _scale(spread: float, win_prob: float, beta: float, dist: str) -> float:
    '''Return the analytic scale used by the Gaussian or gennorm baseline.'''
    if abs(spread) < DEGEN_EPS or abs(win_prob - 0.5) < DEGEN_EPS:
        return FALLBACK_SCALE
    if dist == 'gaussian':
        return spread / scipy_norm.ppf(win_prob)
    return spread / scipy_gennorm.ppf(win_prob, beta)


def baseline_prob_at_k(
    spread: float, win_prob: float, k: int, beta: float, dist: str
) -> tuple:
    '''
    Per-bin baseline probability mass at +k and -k under the chosen distribution.

    Parameters:
    * spread:    favorite-perspective spread (positive = favorite)
    * win_prob:  favorite win probability in (0.5, 1)
    * k:         integer margin (positive)
    * beta:      gennorm shape (ignored if dist == 'gaussian')
    * dist:      'gaussian' or 'gennorm'

    Returns:
    * (P(margin in [+k - 0.5, +k + 0.5]), P(margin in [-k - 0.5, -k + 0.5]))
    '''
    scale = _scale(spread, win_prob, beta, dist)
    if dist == 'gaussian':
        cdf = lambda x: scipy_norm.cdf(x, loc=spread, scale=scale)
    else:
        cdf = lambda x: scipy_gennorm.cdf(x, beta, loc=spread, scale=scale)
    p_pos = float(cdf(k + 0.5) - cdf(k - 0.5))
    p_neg = float(cdf(-k + 0.5) - cdf(-k - 0.5))
    return p_pos, p_neg


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
df['abs_margin'] = df['fav_margin_int'].abs()
print(f'Games with fav_margin + fav_spread + fav_wp_cal: {len(df):,}')


## ==================== Per-Game Baselines ==================== ##

n_games = len(df)
spreads = df['fav_spread'].values.astype(float)
wps = df['fav_wp_cal'].values.astype(float)
margins = df['fav_margin_int'].values.astype(int)
seasons = df['season'].values.astype(int)

baseline_pos = {
    'gaussian': numpy.zeros((n_games, len(K_RANGE))),
    'gennorm':  numpy.zeros((n_games, len(K_RANGE))),
}
baseline_neg = {
    'gaussian': numpy.zeros((n_games, len(K_RANGE))),
    'gennorm':  numpy.zeros((n_games, len(K_RANGE))),
}

for i in range(n_games):
    for j, k in enumerate(K_RANGE):
        for dist in ('gaussian', 'gennorm'):
            p_pos, p_neg = baseline_prob_at_k(spreads[i], wps[i], k, SHIPPED_BETA, dist)
            baseline_pos[dist][i, j] = p_pos
            baseline_neg[dist][i, j] = p_neg


## ==================== Aggregate Magnitude Table ==================== ##

rows = []
for j, k in enumerate(K_RANGE):
    obs_pos = int(numpy.sum(margins == k))
    obs_neg = int(numpy.sum(margins == -k))
    n_obs_combined = obs_pos + obs_neg
    for dist in ('gaussian', 'gennorm'):
        exp_pos_total = float(baseline_pos[dist][:, j].sum())
        exp_neg_total = float(baseline_neg[dist][:, j].sum())
        exp_combined = exp_pos_total + exp_neg_total
        excess_combined_rate = (n_obs_combined - exp_combined) / n_games
        ratio_combined = (
            n_obs_combined / exp_combined if exp_combined > 0 else float('nan')
        )
        rows.append({
            'k': int(k),
            'baseline': dist,
            'observed_pos': int(obs_pos),
            'observed_neg': int(obs_neg),
            'observed_combined': int(n_obs_combined),
            'expected_pos': exp_pos_total,
            'expected_neg': exp_neg_total,
            'expected_combined': exp_combined,
            'observed_rate_combined': n_obs_combined / n_games,
            'expected_rate_combined': exp_combined / n_games,
            'excess_rate_combined': excess_combined_rate,
            'ratio_combined': ratio_combined,
        })
mag_df = pandas.DataFrame(rows)
mag_df.to_csv(HERE / 'output.csv', index=False)
print(f'Wrote: {HERE / "output.csv"}')


## ==================== Per-Season Stability ==================== ##

stability_rows = []
season_list = sorted(numpy.unique(seasons).tolist())
for j, k in enumerate(K_RANGE):
    for dist in ('gaussian', 'gennorm'):
        per_season_excess = []
        for s in season_list:
            sel = seasons == s
            if not sel.any():
                continue
            n_s = int(sel.sum())
            obs_combined_s = int(((margins[sel] == k) | (margins[sel] == -k)).sum())
            exp_combined_s = float(
                baseline_pos[dist][sel, j].sum() + baseline_neg[dist][sel, j].sum()
            )
            per_season_excess.append((obs_combined_s - exp_combined_s) / n_s)
        arr = numpy.array(per_season_excess, dtype=float)
        mean_excess = float(arr.mean())
        std_excess = float(arr.std(ddof=1))
        cv = float(abs(std_excess / mean_excess)) if abs(mean_excess) > 1e-9 else float('nan')
        stability_rows.append({
            'k': int(k),
            'baseline': dist,
            'mean_excess_rate': mean_excess,
            'std_excess_rate': std_excess,
            'cv_abs': cv,
            'min_excess_rate': float(arr.min()),
            'max_excess_rate': float(arr.max()),
            'n_seasons': int(arr.size),
        })
stab_df = pandas.DataFrame(stability_rows)
stab_df.to_csv(HERE / 'output_stability.csv', index=False)
print(f'Wrote: {HERE / "output_stability.csv"}')


## ==================== Era Comparison ==================== ##

pre_mask = seasons < ERA_BOUNDARY
post_mask = seasons >= ERA_BOUNDARY
n_pre = int(pre_mask.sum())
n_post = int(post_mask.sum())

era_rows = []
for j, k in enumerate(K_RANGE):
    for dist in ('gaussian', 'gennorm'):
        obs_pre = int(((margins[pre_mask] == k) | (margins[pre_mask] == -k)).sum())
        obs_post = int(((margins[post_mask] == k) | (margins[post_mask] == -k)).sum())
        exp_pre = float(
            baseline_pos[dist][pre_mask, j].sum() + baseline_neg[dist][pre_mask, j].sum()
        )
        exp_post = float(
            baseline_pos[dist][post_mask, j].sum() + baseline_neg[dist][post_mask, j].sum()
        )
        excess_pre = (obs_pre - exp_pre) / n_pre if n_pre > 0 else float('nan')
        excess_post = (obs_post - exp_post) / n_post if n_post > 0 else float('nan')
        era_rows.append({
            'k': int(k),
            'baseline': dist,
            'n_pre': n_pre,
            'n_post': n_post,
            'observed_pre': obs_pre,
            'observed_post': obs_post,
            'expected_pre': exp_pre,
            'expected_post': exp_post,
            'excess_rate_pre': excess_pre,
            'excess_rate_post': excess_post,
            'delta_excess_rate': excess_post - excess_pre,
        })
era_df = pandas.DataFrame(era_rows)
era_df.to_csv(HERE / 'output_era.csv', index=False)
print(f'Wrote: {HERE / "output_era.csv"}')


## ==================== Console Summary ==================== ##

print('\n=== Combined excess rate at |m| = k (gennorm baseline) ===')
sub = mag_df[mag_df['baseline'] == 'gennorm'].sort_values('k')
print('  k    obs   exp     excess_rate  ratio')
for _, r in sub.iterrows():
    print(
        f'  {int(r["k"]):>2d}  {int(r["observed_combined"]):>4d}  '
        f'{r["expected_combined"]:>6.1f}  {r["excess_rate_combined"]:+.4f}      '
        f'{r["ratio_combined"]:.3f}'
    )

print('\n=== Per-season stability (gennorm baseline) ===')
sub = stab_df[stab_df['baseline'] == 'gennorm'].sort_values('k')
print('  k    mean         std       cv      min        max     n')
for _, r in sub.iterrows():
    cv_str = 'n/a' if numpy.isnan(r['cv_abs']) else f'{r["cv_abs"]:.2f}'
    print(
        f'  {int(r["k"]):>2d}  {r["mean_excess_rate"]:+.4f}      '
        f'{r["std_excess_rate"]:.4f}    {cv_str:<5s}   '
        f'{r["min_excess_rate"]:+.4f}  {r["max_excess_rate"]:+.4f}    '
        f'{int(r["n_seasons"]):>2d}'
    )

print(f'\n=== Era comparison: pre-{ERA_BOUNDARY} vs {ERA_BOUNDARY}+ (gennorm baseline) ===')
print(f'  n_pre  = {n_pre:>4d}    n_post = {n_post:>4d}')
sub = era_df[era_df['baseline'] == 'gennorm'].sort_values('k')
print('  k    pre       post      delta')
for _, r in sub.iterrows():
    print(
        f'  {int(r["k"]):>2d}  {r["excess_rate_pre"]:+.4f}   '
        f'{r["excess_rate_post"]:+.4f}   {r["delta_excess_rate"]:+.4f}'
    )


## ==================== Chart ==================== ##

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

## LEFT — aggregate excess rate at each |k| under both baselines ##
ax = axes[0]
gauss = mag_df[mag_df['baseline'] == 'gaussian'].sort_values('k')
genn = mag_df[mag_df['baseline'] == 'gennorm'].sort_values('k')
x = numpy.array(K_RANGE, dtype=float)
width = 0.4
ax.bar(x - width / 2, gauss['excess_rate_combined'], width=width,
       color=C_EMPIRICAL, alpha=0.75, label='vs variable-scale Gaussian')
ax.bar(x + width / 2, genn['excess_rate_combined'], width=width,
       color=C_MARKET, alpha=0.75, label=f'vs gennorm (beta={SHIPPED_BETA})')
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xticks(K_RANGE[::2])
ax.set_xlabel('|margin|  (k)')
ax.set_ylabel('Excess rate (observed - baseline), 0-1 fraction')
ax.set_title('Aggregate excess rate at |margin| = k')
ax.legend(fontsize=9)

## CENTER — per-season excess at headline integers (gennorm baseline) ##
ax = axes[1]
season_arr = numpy.array(season_list)
for k in HEADLINE:
    j = K_RANGE.index(k)
    series = []
    for s in season_list:
        sel = seasons == s
        n_s = int(sel.sum())
        obs_c = int(((margins[sel] == k) | (margins[sel] == -k)).sum())
        exp_c = float(
            baseline_pos['gennorm'][sel, j].sum() + baseline_neg['gennorm'][sel, j].sum()
        )
        series.append((obs_c - exp_c) / n_s)
    ax.plot(season_arr, series, marker='o', ms=3.5, lw=1.2, label=f'k={k}')
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xlabel('Season')
ax.set_ylabel('Per-season excess rate (gennorm baseline)')
ax.set_title('Per-season excess at headline integers')
ax.legend(fontsize=8, ncol=2)

## RIGHT — era comparison at headline integers (gennorm baseline) ##
ax = axes[2]
era_g = era_df[era_df['baseline'] == 'gennorm'].set_index('k')
xs = numpy.arange(len(HEADLINE))
pre_vals = [era_g.loc[k, 'excess_rate_pre'] for k in HEADLINE]
post_vals = [era_g.loc[k, 'excess_rate_post'] for k in HEADLINE]
width = 0.4
ax.bar(xs - width / 2, pre_vals, width=width, color=C_NEUTRAL, alpha=0.85,
       label=f'pre-{ERA_BOUNDARY}  (n={n_pre})')
ax.bar(xs + width / 2, post_vals, width=width, color=C_FORMULA, alpha=0.85,
       label=f'{ERA_BOUNDARY}+  (n={n_post})')
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([f'k={k}' for k in HEADLINE])
ax.set_xlabel('Headline integer')
ax.set_ylabel('Excess rate (gennorm baseline)')
ax.set_title(f'Era comparison: pre-{ERA_BOUNDARY} vs {ERA_BOUNDARY}+')
ax.legend(fontsize=9)

fig.suptitle('Analysis 8 — Key Number Excess Magnitude', fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nWrote: {HERE / "chart.png"}')
