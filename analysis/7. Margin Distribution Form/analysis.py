'''
Margin Distribution Form — fits a generalized normal and a Gaussian to
empirical NFL margin distributions, bucketed by absolute spread, to back
the Base module's choice of a generalized normal as the parametric form.

All work is on favorite perspective (positive margin = favorite covered or
won by that many).
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.stats import gennorm as scipy_gennorm
from scipy.stats import norm as scipy_norm
from scipy.stats import kstest

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


## ==================== Buckets ==================== ##

## Buckets are defined on |fav_spread|.  Edges chosen to give reasonable
## sample size at each level while still resolving how the empirical shape
## changes as the favorite's edge grows. ##
BUCKETS = [
    ('Pickem to 3',     0.0,   3.5),
    ('3.5 to 7',        3.5,   7.5),
    ('7.5 to 10',       7.5,  10.5),
    ('10.5 to 14',     10.5,  14.5),
    ('14.5+',          14.5,  100.0),
]


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[
    df['fav_margin'].notna()
    & df['fav_spread'].notna()
    & (df['fav_spread'] >= 0)
].copy()
print(f'Games with fav_margin + fav_spread: {len(df):,}')


## ==================== Per-Bucket Fits ==================== ##

def fit_and_score(margins: numpy.ndarray) -> dict:
    '''
    Fit a generalized normal (loc, scale, beta) and a Gaussian (loc, scale)
    to ``margins`` by maximum likelihood, and report log-likelihood and
    one-sample KS distance against each fitted model.
    '''
    n = int(len(margins))
    ## scipy parameterization: gennorm(beta, loc, scale)
    beta, loc_g, scale_g = scipy_gennorm.fit(margins)
    ll_gennorm = float(scipy_gennorm.logpdf(margins, beta, loc=loc_g, scale=scale_g).sum())
    ks_gennorm = float(kstest(margins, 'gennorm', args=(beta, loc_g, scale_g)).statistic)
    loc_n, scale_n = scipy_norm.fit(margins)
    ll_norm = float(scipy_norm.logpdf(margins, loc=loc_n, scale=scale_n).sum())
    ks_norm = float(kstest(margins, 'norm', args=(loc_n, scale_n)).statistic)
    return {
        'n': n,
        'fitted_loc_gennorm': float(loc_g),
        'fitted_scale_gennorm': float(scale_g),
        'fitted_beta_gennorm': float(beta),
        'fitted_loc_gaussian': float(loc_n),
        'fitted_scale_gaussian': float(scale_n),
        'll_gennorm': ll_gennorm,
        'll_gaussian': ll_norm,
        'll_per_game_gennorm': ll_gennorm / max(n, 1),
        'll_per_game_gaussian': ll_norm / max(n, 1),
        'delta_ll_per_game': (ll_gennorm - ll_norm) / max(n, 1),
        'ks_gennorm': ks_gennorm,
        'ks_gaussian': ks_norm,
    }


bucket_rows = []
bucket_data = {}
for label, lo, hi in BUCKETS:
    mask = (df['fav_spread'] >= lo) & (df['fav_spread'] < hi)
    margins = df.loc[mask, 'fav_margin'].values.astype(float)
    if len(margins) == 0:
        continue
    fit = fit_and_score(margins)
    fit['bucket_label'] = label
    fit['spread_lo'] = lo
    fit['spread_hi'] = hi
    fit['mean_abs_spread'] = float(df.loc[mask, 'fav_spread'].mean())
    bucket_rows.append(fit)
    bucket_data[label] = margins

## also fit on the pooled set for a top-line single-shape view ##
all_margins = df['fav_margin'].values.astype(float)
pooled = fit_and_score(all_margins)
pooled.update({
    'bucket_label': 'ALL',
    'spread_lo': 0.0,
    'spread_hi': float('inf'),
    'mean_abs_spread': float(df['fav_spread'].mean()),
})
bucket_rows.append(pooled)
bucket_data['ALL'] = all_margins


## ==================== Output Table ==================== ##

cols_order = [
    'bucket_label', 'spread_lo', 'spread_hi', 'n', 'mean_abs_spread',
    'fitted_loc_gennorm', 'fitted_scale_gennorm', 'fitted_beta_gennorm',
    'fitted_loc_gaussian', 'fitted_scale_gaussian',
    'll_gennorm', 'll_gaussian',
    'll_per_game_gennorm', 'll_per_game_gaussian', 'delta_ll_per_game',
    'ks_gennorm', 'ks_gaussian',
]
out_df = pandas.DataFrame(bucket_rows)[cols_order]
out_df.to_csv(HERE / 'output.csv', index=False)
print(f'\nWrote: {HERE / "output.csv"}')

print('\n=== Per-bucket fits ===')
for r in bucket_rows:
    print(
        f'  {r["bucket_label"]:<14s}  n={r["n"]:>4d}  '
        f'beta={r["fitted_beta_gennorm"]:.3f}  '
        f'loc={r["fitted_loc_gennorm"]:+.2f}  '
        f'scale={r["fitted_scale_gennorm"]:.2f}  '
        f'd_ll/game={r["delta_ll_per_game"]:+.4f}  '
        f'KS gennorm={r["ks_gennorm"]:.4f}  KS norm={r["ks_gaussian"]:.4f}'
    )


## ==================== Chart ==================== ##

panel_buckets = [b for b in BUCKETS]  ## exclude pooled from per-bucket panels
n_panels = len(panel_buckets)
ncols = 3
nrows = int(numpy.ceil(n_panels / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.0 * nrows))
axes = numpy.atleast_2d(axes)

for idx, (label, lo, hi) in enumerate(panel_buckets):
    ax = axes[idx // ncols, idx % ncols]
    margins = bucket_data[label]
    row = next(r for r in bucket_rows if r['bucket_label'] == label)
    lo_x = float(numpy.percentile(margins, 0.5)) - 5
    hi_x = float(numpy.percentile(margins, 99.5)) + 5
    grid = numpy.linspace(lo_x, hi_x, 400)
    ax.hist(
        margins,
        bins=numpy.arange(lo_x, hi_x + 1, 1.0),
        density=True,
        color=C_EMPIRICAL,
        alpha=0.40,
        edgecolor='white',
        label=f'Empirical (n={row["n"]:,})',
    )
    ax.plot(
        grid,
        scipy_gennorm.pdf(
            grid,
            row['fitted_beta_gennorm'],
            loc=row['fitted_loc_gennorm'],
            scale=row['fitted_scale_gennorm'],
        ),
        color=C_MARKET,
        lw=2.0,
        label=f'gennorm (beta={row["fitted_beta_gennorm"]:.2f})',
    )
    ax.plot(
        grid,
        scipy_norm.pdf(
            grid,
            loc=row['fitted_loc_gaussian'],
            scale=row['fitted_scale_gaussian'],
        ),
        color=C_FORMULA,
        lw=1.5,
        ls='--',
        label='Gaussian',
    )
    ax.axvline(row['mean_abs_spread'], color=C_NEUTRAL, lw=0.9, ls=':')
    ax.set_title(f'|spread| {label}  (mean={row["mean_abs_spread"]:.1f})')
    ax.set_xlabel('Favorite margin')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper right')

## hide any empty subplot slots ##
for j in range(n_panels, nrows * ncols):
    axes[j // ncols, j % ncols].axis('off')

fig.suptitle('Analysis 7 — Margin Distribution Form by Spread Bucket', fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {HERE / "chart.png"}')
