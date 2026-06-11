'''
Calibration Stationarity — per-season split Platt parameters, era error shape,
and comparison of static vs centered intercept schemes for training labels.
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import linregress

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import setup_style, C_EMPIRICAL, C_MARKET, C_NEUTRAL
from nfelotranslation.Utilities.MathUtils import logit, clip_prob, expit, log_loss, brier_score


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
WINDOW = 5

setup_style()


def fit_platt(logit_wp, y):
    def nll(params):
        p = clip_prob(expit(params[0] * logit_wp + params[1]))
        return float(-numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))
    res = minimize(nll, x0=numpy.array([1.0, 0.0]), method='Nelder-Mead')
    return float(res.x[0]), float(res.x[1])


def fit_intercept(logit_wp, y, slope):
    def nll(b):
        p = clip_prob(expit(slope * logit_wp + b[0]))
        return float(-numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))
    res = minimize(nll, x0=numpy.array([0.0]), method='Nelder-Mead')
    return float(res.x[0])


def centered_window(target, seasons, width):
    seasons = sorted(seasons)
    idx = seasons.index(target)
    n = len(seasons)
    width = min(width, n)
    left = (width - 1) // 2
    right = width - 1 - left
    start, end = idx - left, idx + right
    if start < 0:
        end = min(n - 1, end + (-start))
        start = 0
    if end >= n:
        start = max(0, start - (end - (n - 1)))
        end = n - 1
    return seasons[start:end + 1]


def apply_split(wp, y_home, slopes, intercepts):
    z = logit(clip_prob(wp))
    a = numpy.where(y_home, slopes['home'], slopes['away'])
    b = numpy.where(y_home, intercepts['home'], intercepts['away'])
    return expit(a * z + b)


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[df['ml_wp_close'].notna() & df['result'].notna()].copy()
print(f'Games with ML close + result: {len(df):,}')

df['home_is_fav'] = (df['ml_wp_close'] >= 0.50).astype(int)
df['fav_ml_wp'] = numpy.where(
    df['home_is_fav'], df['ml_wp_close'], 1.0 - df['ml_wp_close']
)
df['fav_win'] = numpy.where(
    df['home_is_fav'],
    (df['result'] > 0).astype(int),
    (df['result'] < 0).astype(int),
)
df['is_tie'] = (df['result'] == 0).astype(int)
df = df[df['is_tie'] == 0].copy()
df['logit_wp'] = logit(clip_prob(df['fav_ml_wp'].values))
print(f'Games after excluding ties: {len(df):,}')


## ==================== Omniscient Split Slopes ==================== ##

slopes = {}
for is_home, key in [(1, 'home'), (0, 'away')]:
    sub = df[df['home_is_fav'] == is_home]
    slopes[key] = fit_platt(sub['logit_wp'].values, sub['fav_win'].values)[0]

print('\n=== Omniscient Split Slopes (Full Sample) ===')
for key in ('home', 'away'):
    print(f'  {key}: a = {slopes[key]:.4f}')


## ==================== Per-Season Split Platt Fit ==================== ##

platt_rows = []
for season in sorted(df['season'].unique()):
    sub = df[df['season'] == season]
    row = {'season': int(season), 'n': len(sub)}
    for is_home, key in [(1, 'home'), (0, 'away')]:
        loc = sub[sub['home_is_fav'] == is_home]
        if len(loc) < 50:
            continue
        s, b = fit_platt(loc['logit_wp'].values, loc['fav_win'].values)
        row[f'slope_{key}'] = s
        row[f'intercept_{key}'] = b
    platt_rows.append(row)
platt = pandas.DataFrame(platt_rows)

slope_home_reg = linregress(platt['season'], platt['slope_home'])
slope_away_reg = linregress(platt['season'], platt['slope_away'])
int_home_reg = linregress(platt['season'], platt['intercept_home'])
int_away_reg = linregress(platt['season'], platt['intercept_away'])

print('\n=== Per-Season Split Platt Parameters ===')
print(platt.round(4).to_string(index=False))

print('\n=== Summary Statistics ===')
for col, label in [
    ('slope_home', 'Home slope'),
    ('slope_away', 'Away slope'),
    ('intercept_home', 'Home intercept'),
    ('intercept_away', 'Away intercept'),
]:
    print(f'{label}: mean={platt[col].mean():.4f}, std={platt[col].std():.4f}')

print(f'\nHome slope trend: {slope_home_reg.slope:+.4f}/yr (p={slope_home_reg.pvalue:.3f})')
print(f'Away slope trend: {slope_away_reg.slope:+.4f}/yr (p={slope_away_reg.pvalue:.3f})')
print(f'Home intercept trend: {int_home_reg.slope:+.4f}/yr (p={int_home_reg.pvalue:.3f})')
print(f'Away intercept trend: {int_away_reg.slope:+.4f}/yr (p={int_away_reg.pvalue:.3f})')


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


## ==================== Intercept Scheme Comparison ==================== ##

wp = df['fav_ml_wp'].values
y = df['fav_win'].values
home = df['home_is_fav'].values.astype(bool)
seasons = sorted(df['season'].unique())

static_int = {}
for is_home, key in [(1, 'home'), (0, 'away')]:
    sub = df[df['home_is_fav'] == is_home]
    static_int[key] = fit_intercept(sub['logit_wp'].values, sub['fav_win'].values, slopes[key])

raw_ll = log_loss(y, wp)
raw_brier = brier_score(y, wp)
static_cal = apply_split(wp, home, slopes, static_int)
static_ll = log_loss(y, static_cal)
static_brier = brier_score(y, static_cal)

centered_cal = numpy.zeros_like(wp)
for season in seasons:
    window = centered_window(int(season), seasons, WINDOW)
    train = df[df['season'].isin(window)]
    intercepts = {}
    for is_home, key in [(1, 'home'), (0, 'away')]:
        loc = train[train['home_is_fav'] == is_home]
        intercepts[key] = fit_intercept(
            loc['logit_wp'].values, loc['fav_win'].values, slopes[key]
        )
    mask = df['season'].values == season
    centered_cal[mask] = apply_split(wp[mask], home[mask], slopes, intercepts)

compare = pandas.DataFrame([
    {'scheme': 'raw_market', 'log_loss': raw_ll, 'brier': raw_brier},
    {'scheme': 'static_omniscient_intercept', 'log_loss': static_ll, 'brier': static_brier},
    {
        'scheme': 'centered_5yr_intercept',
        'log_loss': log_loss(y, centered_cal),
        'brier': brier_score(y, centered_cal),
    },
])

print('\n=== Intercept Scheme Comparison (Full Sample) ===')
print(compare.round(6).to_string(index=False))


## ==================== Output ==================== ##

platt.to_csv(HERE / 'output.csv', index=False)
era_out = era[['era', 'ml_bin', 'n', 'mean_ml_wp', 'actual_wr', 'error']].copy()
era_out['ml_bin'] = era_out['ml_bin'].astype(str)
era_out.to_csv(HERE / 'output_era.csv', index=False)
annual.to_csv(HERE / 'output_annual.csv', index=False)
compare.to_csv(HERE / 'output_centered_compare.csv', index=False)
print(f'\nSaved: {HERE / "output.csv"}')


## ==================== Chart ==================== ##

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_fit = platt['season'].values

## top-left — per-season home/away slopes ##
ax = axes[0, 0]
ax.plot(platt['season'], platt['slope_home'], 'o-', color=C_MARKET, markersize=5,
        lw=1.2, label='Home')
ax.plot(platt['season'], platt['slope_away'], 'o-', color=C_EMPIRICAL, markersize=5,
        lw=1.2, label='Away')
ax.axhline(slopes['home'], color=C_MARKET, ls='--', lw=0.8, alpha=0.6)
ax.axhline(slopes['away'], color=C_EMPIRICAL, ls='--', lw=0.8, alpha=0.6)
ax.set_xlabel('Season')
ax.set_ylabel('Platt slope')
ax.set_title('Split Platt slope by season')
ax.legend(fontsize=8)

## top-right — per-season home/away intercepts ##
ax = axes[0, 1]
ax.plot(platt['season'], platt['intercept_home'], 'o-', color=C_MARKET, markersize=5,
        lw=1.2, label='Home')
ax.plot(platt['season'], platt['intercept_away'], 'o-', color=C_EMPIRICAL, markersize=5,
        lw=1.2, label='Away')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Season')
ax.set_ylabel('Platt intercept')
ax.set_title('Split Platt intercept by season')
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

fig.suptitle('ML calibration bias: stability over time (split Platt)', y=1.01,
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {HERE / "chart.png"}')
