'''
Within-Spread Price Signal — tests whether the price (juice) attached to
a posted spread carries a margin signal beyond the spread itself.

For each posted spread level, games are split into terciles by the
no-vig favorite cover probability (derived from the favorite-side and
underdog-side spread prices). If the price signal is informative, the
top tercile should produce a higher median favorite margin than the
bottom tercile. If the price variation is noise around the half-point
spread, the terciles should not separate.
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import (
    setup_style,
    american_to_prob,
    C_EMPIRICAL,
    C_MARKET,
    C_NEUTRAL,
)


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

setup_style()


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[
    df['fav_spread'].notna()
    & (df['fav_spread'] > 0)
    & df['fav_margin'].notna()
    & df['fav_spread_close_price'].notna()
    & df['dog_spread_close_price'].notna()
].copy()

## clamp the favorite spread to the half-point grid actually posted ##
df['fav_spread_grid'] = numpy.round(df['fav_spread'].values * 2.0) / 2.0

## no-vig favorite cover probability from the two spread prices ##
fp_raw = american_to_prob(df['fav_spread_close_price'].values)
dp_raw = american_to_prob(df['dog_spread_close_price'].values)
total = fp_raw + dp_raw
df['fav_cover_prob'] = fp_raw / total
df['cover_prob_excess'] = df['fav_cover_prob'] - 0.50

## drop rows where the no-vig probability is undefined ##
df = df[df['fav_cover_prob'].notna()].copy()

## binary cover & push flags for cover-rate metrics ##
df['fav_covers'] = (df['fav_margin'] > df['fav_spread_grid']).astype(int)
df['is_push'] = (df['fav_margin'] == df['fav_spread_grid']).astype(int)

print(f'Games: {len(df):,}, '
      f'seasons {int(df["season"].min())}–{int(df["season"].max())}')
print(f'Mean fav_cover_prob: {df["fav_cover_prob"].mean():.4f}')
print(f'Std fav_cover_prob:  {df["fav_cover_prob"].std():.4f}')


## ==================== Per-Spread Tercile Analysis ==================== ##

print('\n=== Per-Spread Tercile (cover prob) ===')
tercile_rows = []
for spread, sub in df.groupby('fav_spread_grid'):
    if len(sub) < 200:
        continue
    sub = sub.copy()
    try:
        sub['tercile'] = pandas.qcut(
            sub['fav_cover_prob'], q=3, labels=False, duplicates='drop'
        )
    except ValueError:
        continue
    if sub['tercile'].nunique() < 3:
        continue
    for tercile in sorted(sub['tercile'].unique()):
        bucket = sub[sub['tercile'] == tercile]
        if len(bucket) < 30:
            continue
        no_push = bucket[bucket['is_push'] == 0]
        cover_rate = (
            float(no_push['fav_covers'].mean()) if len(no_push) > 0
            else float('nan')
        )
        tercile_rows.append({
            'spread': float(spread),
            'tercile': int(tercile),
            'tercile_label': {0: 'low', 1: 'mid', 2: 'high'}[int(tercile)],
            'n': int(len(bucket)),
            'mean_cover_prob': float(bucket['fav_cover_prob'].mean()),
            'median_cover_prob': float(bucket['fav_cover_prob'].median()),
            'median_margin': float(bucket['fav_margin'].median()),
            'mean_margin': float(bucket['fav_margin'].mean()),
            'fav_cover_rate': cover_rate,
        })


tercile_df = pandas.DataFrame(tercile_rows)
tercile_df.to_csv(HERE / 'output.csv', index=False)
print(f'Wrote: {HERE / "output.csv"}')


## ==================== High-vs-Low Tercile Differences ==================== ##

diff_rows = []
for spread, sub in tercile_df.groupby('spread'):
    sub = sub.set_index('tercile')
    if 0 not in sub.index or 2 not in sub.index:
        continue
    diff_rows.append({
        'spread': float(spread),
        'n_low': int(sub.loc[0, 'n']),
        'n_high': int(sub.loc[2, 'n']),
        'cp_low': float(sub.loc[0, 'mean_cover_prob']),
        'cp_high': float(sub.loc[2, 'mean_cover_prob']),
        'cp_gap': float(sub.loc[2, 'mean_cover_prob'] - sub.loc[0, 'mean_cover_prob']),
        'median_margin_low': float(sub.loc[0, 'median_margin']),
        'median_margin_high': float(sub.loc[2, 'median_margin']),
        'median_margin_gap': float(
            sub.loc[2, 'median_margin'] - sub.loc[0, 'median_margin']
        ),
        'cover_rate_low': float(sub.loc[0, 'fav_cover_rate']),
        'cover_rate_high': float(sub.loc[2, 'fav_cover_rate']),
        'cover_rate_gap': float(
            sub.loc[2, 'fav_cover_rate'] - sub.loc[0, 'fav_cover_rate']
        ),
    })

diff_df = pandas.DataFrame(diff_rows)
diff_df.to_csv(HERE / 'output_high_vs_low.csv', index=False)
print(f'Wrote: {HERE / "output_high_vs_low.csv"}')

print('\n=== High vs Low Tercile (top minus bottom) ===')
print(diff_df.round(4).to_string(index=False))

if len(diff_df) > 0:
    print(f'\nMean cp gap (high − low):           {diff_df["cp_gap"].mean():+.4f}')
    print(f'Mean median-margin gap:              {diff_df["median_margin_gap"].mean():+.4f}')
    print(f'Mean cover-rate gap:                 {diff_df["cover_rate_gap"].mean():+.4f}')
    print(
        f'Spreads with positive median gap:    '
        f'{int((diff_df["median_margin_gap"] > 0).sum())} / {len(diff_df)}'
    )
    print(
        f'Spreads with positive cover-rate gap:'
        f'{int((diff_df["cover_rate_gap"] > 0).sum())} / {len(diff_df)}'
    )


## ==================== Pooled Demeaned Slope ==================== ##

## within-spread demeaning isolates the price signal from the spread itself ##
df['median_margin_at_spread'] = df.groupby('fav_spread_grid')['fav_margin'].transform('median')
df['mean_cp_at_spread'] = df.groupby('fav_spread_grid')['fav_cover_prob'].transform('mean')
df['margin_demeaned'] = df['fav_margin'] - df['median_margin_at_spread']
df['cp_demeaned'] = df['fav_cover_prob'] - df['mean_cp_at_spread']

eligible = df[df['cp_demeaned'].notna() & df['margin_demeaned'].notna()]

if len(eligible) > 100:
    cov = numpy.cov(eligible['cp_demeaned'].values, eligible['margin_demeaned'].values, ddof=1)
    slope_pooled = float(cov[0, 1] / cov[0, 0])
    corr = float(numpy.corrcoef(
        eligible['cp_demeaned'].values, eligible['margin_demeaned'].values
    )[0, 1])
    print(
        f'\nPooled within-spread regression: '
        f'slope={slope_pooled:.4f} pts per unit cp, corr={corr:.4f}'
    )

## binned demeaned plot data ##
edges = numpy.linspace(-0.04, 0.04, 17)
bin_rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (eligible['cp_demeaned'] >= lo) & (eligible['cp_demeaned'] < hi)
    bucket = eligible[mask]
    if len(bucket) < 50:
        continue
    bin_rows.append({
        'cp_demeaned_lo': float(lo),
        'cp_demeaned_hi': float(hi),
        'cp_demeaned_mid': float((lo + hi) / 2),
        'n': int(len(bucket)),
        'median_margin_demeaned': float(bucket['margin_demeaned'].median()),
        'mean_margin_demeaned': float(bucket['margin_demeaned'].mean()),
    })
binned_df = pandas.DataFrame(bin_rows)
binned_df.to_csv(HERE / 'output_demeaned_bins.csv', index=False)
print(f'Wrote: {HERE / "output_demeaned_bins.csv"}')


## ==================== Chart ==================== ##

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

## TOP-LEFT — median margin by tercile per spread ##
ax = axes[0, 0]
spreads = sorted(diff_df['spread'].unique())
x = numpy.arange(len(spreads))
low_vals = [diff_df.loc[diff_df['spread'] == s, 'median_margin_low'].values[0]
            for s in spreads]
high_vals = [diff_df.loc[diff_df['spread'] == s, 'median_margin_high'].values[0]
             for s in spreads]
ax.bar(x - 0.2, low_vals, width=0.4, color=C_NEUTRAL, label='Bottom tercile')
ax.bar(x + 0.2, high_vals, width=0.4, color=C_MARKET, label='Top tercile')
ax.set_xticks(x)
ax.set_xticklabels([f'{s:.1f}' for s in spreads])
ax.set_xlabel('Posted favorite spread')
ax.set_ylabel('Median fav margin (points)')
ax.set_title('Median fav margin by cover-prob tercile')
ax.legend(fontsize=9)

## TOP-RIGHT — cover-rate gap and median-margin gap per spread ##
ax = axes[0, 1]
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.bar(
    x - 0.2,
    diff_df.set_index('spread').loc[spreads, 'cover_rate_gap'].values,
    width=0.4,
    color=C_EMPIRICAL,
    label='Cover-rate gap',
)
ax.bar(
    x + 0.2,
    diff_df.set_index('spread').loc[spreads, 'median_margin_gap'].values / 10.0,
    width=0.4,
    color=C_MARKET,
    label='Median margin gap (÷10)',
)
ax.set_xticks(x)
ax.set_xticklabels([f'{s:.1f}' for s in spreads])
ax.set_xlabel('Posted favorite spread')
ax.set_ylabel('High − low tercile')
ax.set_title('Top minus bottom tercile per spread')
ax.legend(fontsize=9)

## BOTTOM-LEFT — pooled demeaned scatter ##
ax = axes[1, 0]
if len(binned_df) > 0:
    sizes = numpy.sqrt(binned_df['n'].values) * 4
    ax.scatter(
        binned_df['cp_demeaned_mid'],
        binned_df['median_margin_demeaned'],
        s=sizes,
        color=C_EMPIRICAL,
        alpha=0.7,
        edgecolor='white',
        zorder=3,
    )
    if len(eligible) > 100:
        x_grid = numpy.linspace(
            binned_df['cp_demeaned_mid'].min(),
            binned_df['cp_demeaned_mid'].max(),
            50,
        )
        ax.plot(x_grid, slope_pooled * x_grid, color=C_NEUTRAL, lw=1, ls='--',
                label=f'pooled slope={slope_pooled:.2f}')
        ax.legend(fontsize=9)
ax.axhline(0, color=C_NEUTRAL, lw=0.8)
ax.axvline(0, color=C_NEUTRAL, lw=0.8)
ax.set_xlabel('Cover-prob demeaned (within posted spread)')
ax.set_ylabel('Margin demeaned (median)')
ax.set_title('Within-spread price → margin signal')

## BOTTOM-RIGHT — sample size per spread per tercile ##
ax = axes[1, 1]
pivot = tercile_df.pivot(index='spread', columns='tercile_label', values='n').fillna(0)
ordered_cols = [c for c in ['low', 'mid', 'high'] if c in pivot.columns]
pivot = pivot[ordered_cols]
pivot.plot(kind='bar', ax=ax,
           color=[C_NEUTRAL, C_EMPIRICAL, C_MARKET][:len(ordered_cols)],
           edgecolor='white', alpha=0.85)
ax.set_xlabel('Posted favorite spread')
ax.set_ylabel('Games per tercile')
ax.set_title('Tercile sample sizes')
ax.legend(fontsize=9, title='Tercile')

fig.suptitle('Analysis 6 — Within-Spread Price Signal', fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Wrote: {HERE / "chart.png"}')
