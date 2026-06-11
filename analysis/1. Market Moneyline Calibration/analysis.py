'''
Market Moneyline Calibration — compares market ML implied win probabilities
against observed win rates across favorite-perspective bins, and fits split
Platt parameters by favorite location (home vs away).
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import minimize

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import setup_style, wilson_ci, C_EMPIRICAL, C_MARKET, C_NEUTRAL
from nfelotranslation.Utilities.MathUtils import logit, clip_prob, expit


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

setup_style()


def fit_platt(logit_wp, y):
    def nll(params):
        p = clip_prob(expit(params[0] * logit_wp + params[1]))
        return float(-numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))
    res = minimize(nll, x0=numpy.array([1.0, 0.0]), method='Nelder-Mead')
    return float(res.x[0]), float(res.x[1])


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[df['ml_wp_close'].notna() & df['result'].notna()].copy()
print(f'Games with ML close + result: {len(df):,}')
print(f'Seasons: {df["season"].min()} - {df["season"].max()}')

## fold to favorite perspective: favorite_wp = max(home_wp, 1 - home_wp) ##
df['home_is_fav'] = (df['ml_wp_close'] >= 0.50).astype(int)
df['fav_ml_wp'] = numpy.where(
    df['home_is_fav'], df['ml_wp_close'], 1.0 - df['ml_wp_close']
)
df['fav_win'] = numpy.where(
    df['home_is_fav'],
    (df['result'] > 0).astype(int),
    (df['result'] < 0).astype(int),
)

## exclude ties for the rate calculation ##
df['is_tie'] = (df['result'] == 0).astype(int)
df_no_tie = df[df['is_tie'] == 0].copy()
print(f'Games after excluding ties: {len(df_no_tie):,}')


## ==================== Bin by Favorite ML Win Prob ==================== ##

bin_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
df_no_tie['ml_bin'] = pandas.cut(
    df_no_tie['fav_ml_wp'], bins=bin_edges, include_lowest=True
)
cal = df_no_tie.groupby('ml_bin', observed=True).agg(
    n=('fav_win', 'count'),
    wins=('fav_win', 'sum'),
    mean_ml_wp=('fav_ml_wp', 'mean'),
).reset_index()
cal['actual_wr'] = cal['wins'] / cal['n']
cal['ci_lo'], cal['ci_hi'] = wilson_ci(cal['wins'], cal['n'])
cal['error'] = cal['actual_wr'] - cal['mean_ml_wp']

## weighted mean absolute error across bins (probability units) ##
wmae = float(
    (numpy.abs(cal['actual_wr'] - cal['mean_ml_wp']) * cal['n']).sum()
    / cal['n'].sum()
)
print('\n=== ML Calibration by Bin (Favorite Perspective) ===')
print(cal[['ml_bin', 'n', 'mean_ml_wp', 'actual_wr', 'error']]
      .round(4).to_string(index=False))
print(f'\nWeighted MAE: {wmae:.4f}')


## ==================== Split Platt by Favorite Location ==================== ##

split_rows = []
for is_home, label in [(1, 'home'), (0, 'away')]:
    sub = df_no_tie[df_no_tie['home_is_fav'] == is_home]
    z = logit(clip_prob(sub['fav_ml_wp'].values))
    slope, intercept = fit_platt(z, sub['fav_win'].values)
    split_rows.append({
        'location': label,
        'n': len(sub),
        'slope': slope,
        'intercept': intercept,
    })
split = pandas.DataFrame(split_rows)

print('\n=== Split Platt (Full Sample, by Favorite Location) ===')
print(split.round(4).to_string(index=False))
print('  p_cal = expit(a_loc * logit(p_market) + b_loc)')


## ==================== Output ==================== ##

cal_out = cal[['ml_bin', 'n', 'mean_ml_wp', 'actual_wr',
               'ci_lo', 'ci_hi', 'error']].copy()
cal_out['ml_bin'] = cal_out['ml_bin'].astype(str)
cal_out.to_csv(HERE / 'output.csv', index=False)
split.to_csv(HERE / 'output_split_platt.csv', index=False)
print(f'\nSaved: {HERE / "output.csv"}')
print(f'Saved: {HERE / "output_split_platt.csv"}')


## ==================== Chart ==================== ##

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

## left — calibration curve with Wilson CIs ##
ax = axes[0]
ax.plot([0.5, 1.0], [0.5, 1.0], '--', color=C_NEUTRAL, lw=1,
        label='Perfect calibration')
ax.errorbar(
    cal['mean_ml_wp'], cal['actual_wr'],
    yerr=[cal['actual_wr'] - cal['ci_lo'], cal['ci_hi'] - cal['actual_wr']],
    fmt='o-', color=C_MARKET, markersize=6, capsize=3, lw=1.5,
    label='Market ML vs observed',
)
for _, row in cal.iterrows():
    ax.annotate(
        f'n={int(row["n"])}',
        (row['mean_ml_wp'], row['actual_wr']),
        textcoords='offset points', xytext=(0, 10),
        fontsize=7, ha='center', color='#555555',
    )
ax.set_xlabel('Favorite ML implied win probability')
ax.set_ylabel('Observed favorite win rate')
ax.set_title('Calibration (favorite perspective)')
ax.legend(loc='upper left')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.5, 1.0)

## right — bin-level error ##
ax = axes[1]
colors = [C_MARKET if e > 0 else C_EMPIRICAL for e in cal['error']]
ax.bar(
    range(len(cal)), cal['error'],
    color=colors, alpha=0.7, edgecolor='white', linewidth=0.5,
)
ax.axhline(0, color=C_NEUTRAL, lw=1)
ax.set_xticks(range(len(cal)))
ax.set_xticklabels(
    [f'{row["mean_ml_wp"]:.2f}' for _, row in cal.iterrows()],
    rotation=45, ha='right', fontsize=8,
)
ax.set_xlabel('Mean favorite ML win prob in bin')
ax.set_ylabel('Error (observed - ML implied)')
ax.set_title('Calibration error by bin')

fig.suptitle('Market ML calibration vs observed outcomes', y=1.02)
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {HERE / "chart.png"}')
