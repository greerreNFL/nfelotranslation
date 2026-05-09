'''
Recalibration Method Comparison — pools leave-one-season-out predictions
from several recalibration families and ranks them on log loss, Expected
Calibration Error (ECE), and tail-weighted ECE.
'''

## built-ins ##
import pathlib
import sys

## external ##
import numpy
import pandas
from scipy.special import logit, expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

## local ##
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _shared.data import load_data
from _shared.utils import setup_style, wilson_ci, C_NEUTRAL


HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent

setup_style()


## ==================== Load & Prepare ==================== ##

df = load_data()
df = df[df['ml_wp_close'].notna() & df['result'].notna() & (df['result'] != 0)].copy()

df['home_is_fav'] = (df['ml_wp_close'] >= 0.50).astype(int)
df['fav_ml_wp'] = numpy.where(
    df['home_is_fav'], df['ml_wp_close'], 1.0 - df['ml_wp_close']
)
df['fav_win'] = numpy.where(
    df['home_is_fav'],
    (df['result'] > 0).astype(int),
    (df['result'] < 0).astype(int),
)
df['logit_wp'] = logit(numpy.clip(df['fav_ml_wp'], 1e-6, 1.0 - 1e-6))
seasons = sorted(df['season'].unique())
print(f'Games: {len(df):,}, Seasons: {seasons[0]}-{seasons[-1]}')


## ==================== Method Definitions ==================== ##

def _neg_log_loss(p: numpy.ndarray, y: numpy.ndarray) -> float:
    p = numpy.clip(p, 1e-10, 1.0 - 1e-10)
    return float(-numpy.mean(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))


def fit_platt(z, y):
    res = minimize(
        lambda p: _neg_log_loss(expit(p[0] * z + p[1]), y),
        [1.0, 0.0], method='Nelder-Mead',
    )
    return {'a': res.x[0], 'b': res.x[1]}


def predict_platt(z, p):
    return expit(p['a'] * z + p['b'])


def fit_beta(z, y):
    p_raw = expit(z)
    log_p = numpy.log(numpy.clip(p_raw, 1e-10, 1.0))
    log_1mp = numpy.log(numpy.clip(1.0 - p_raw, 1e-10, 1.0))
    res = minimize(
        lambda p: _neg_log_loss(expit(p[0] * log_p + p[1] * log_1mp + p[2]), y),
        [1.0, -1.0, 0.0], method='Nelder-Mead',
    )
    return {'a': res.x[0], 'b': res.x[1], 'c': res.x[2]}


def predict_beta(z, p):
    p_raw = expit(z)
    log_p = numpy.log(numpy.clip(p_raw, 1e-10, 1.0))
    log_1mp = numpy.log(numpy.clip(1.0 - p_raw, 1e-10, 1.0))
    return expit(p['a'] * log_p + p['b'] * log_1mp + p['c'])


def fit_poly2(z, y):
    res = minimize(
        lambda p: _neg_log_loss(expit(p[0] * z + p[1] * z ** 2 + p[2]), y),
        [1.0, 0.0, 0.0], method='Nelder-Mead',
    )
    return {'a': res.x[0], 'b': res.x[1], 'c': res.x[2]}


def predict_poly2(z, p):
    return expit(p['a'] * z + p['b'] * z ** 2 + p['c'])


def fit_platt_relu(z, y):
    ## expit(a*z + b + c * max(z - k, 0)) — extra slope above threshold k ##
    res = minimize(
        lambda p: _neg_log_loss(
            expit(p[0] * z + p[1] + p[2] * numpy.maximum(z - p[3], 0.0)), y
        ),
        [1.1, -0.1, 0.1, 1.1], method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10},
    )
    return {'a': res.x[0], 'b': res.x[1], 'c': res.x[2], 'k': res.x[3]}


def predict_platt_relu(z, p):
    return expit(p['a'] * z + p['b'] + p['c'] * numpy.maximum(z - p['k'], 0.0))


def fit_platt_softplus(z, y):
    ## expit(a*z + b + c * log(1 + exp(z - k))) — smooth tail boost ##
    def obj(p):
        clipped = numpy.clip(z - p[3], -50.0, 50.0)
        lin = p[0] * z + p[1] + p[2] * numpy.log1p(numpy.exp(clipped))
        return _neg_log_loss(expit(lin), y)
    res = minimize(
        obj, [1.1, -0.1, 0.1, 1.1], method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10},
    )
    return {'a': res.x[0], 'b': res.x[1], 'c': res.x[2], 'k': res.x[3]}


def predict_platt_softplus(z, p):
    clipped = numpy.clip(z - p['k'], -50.0, 50.0)
    return expit(p['a'] * z + p['b'] + p['c'] * numpy.log1p(numpy.exp(clipped)))


def fit_piecewise(z, y):
    ## two linear segments in logit space with continuity at breakpoint k ##
    def obj(p):
        a1, b1, a2, k = p
        lin = numpy.where(
            z <= k,
            a1 * z + b1,
            (a1 * k + b1) + a2 * (z - k),
        )
        return _neg_log_loss(expit(lin), y)
    res = minimize(
        obj, [1.1, -0.1, 1.3, 0.85], method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10},
    )
    return {'a1': res.x[0], 'b1': res.x[1], 'a2': res.x[2], 'k': res.x[3]}


def predict_piecewise(z, p):
    lin = numpy.where(
        z <= p['k'],
        p['a1'] * z + p['b1'],
        (p['a1'] * p['k'] + p['b1']) + p['a2'] * (z - p['k']),
    )
    return expit(lin)


METHODS = {
    'Raw ML':       {'fit': None,                'predict': lambda z, _: expit(z),    'n_params': 0},
    'Platt':        {'fit': fit_platt,           'predict': predict_platt,            'n_params': 2},
    'Beta':         {'fit': fit_beta,            'predict': predict_beta,             'n_params': 3},
    'Poly Logit 2': {'fit': fit_poly2,           'predict': predict_poly2,            'n_params': 3},
    'Platt+ReLU':   {'fit': fit_platt_relu,      'predict': predict_platt_relu,       'n_params': 4},
    'Platt+Soft':   {'fit': fit_platt_softplus,  'predict': predict_platt_softplus,   'n_params': 4},
    'Piecewise':    {'fit': fit_piecewise,       'predict': predict_piecewise,        'n_params': 4},
}


## ==================== ECE Computation ==================== ##

def _ece(p: numpy.ndarray, y: numpy.ndarray, edges: numpy.ndarray) -> float:
    '''Expected calibration error: population-weighted bin error in probability units.'''
    bins = numpy.clip(numpy.digitize(p, edges) - 1, 0, len(edges) - 2)
    total = len(y)
    err = 0.0
    for b in range(len(edges) - 1):
        mask = bins == b
        if not mask.any():
            continue
        err += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(err)


BINS_FULL = numpy.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00])
BINS_TAIL = numpy.array([0.70, 0.75, 0.80, 0.85, 0.90, 1.00])
BINS_CORE = numpy.array([0.50, 0.55, 0.60, 0.65, 0.70])


## ==================== Leave-One-Season-Out ==================== ##

print('\n=== Leave-One-Season-Out Cross-Validation ===')
df['idx'] = numpy.arange(len(df))
oos_preds = {name: numpy.full(len(df), numpy.nan) for name in METHODS}
for hold_season in seasons:
    train_mask = df['season'] != hold_season
    test_mask = df['season'] == hold_season
    z_train = df.loc[train_mask, 'logit_wp'].values
    y_train = df.loc[train_mask, 'fav_win'].values
    z_test = df.loc[test_mask, 'logit_wp'].values
    test_idx = df.loc[test_mask, 'idx'].values
    for name, method in METHODS.items():
        params = {} if method['fit'] is None else method['fit'](z_train, y_train)
        oos_preds[name][test_idx] = method['predict'](z_test, params)


## ==================== Pooled Metrics ==================== ##

y_true = df['fav_win'].values
raw_wp = df['fav_ml_wp'].values

rows = []
for name in METHODS:
    p = oos_preds[name]
    rows.append({
        'method': name,
        'n_params': METHODS[name]['n_params'],
        'log_loss': _neg_log_loss(p, y_true),
        'brier': float(numpy.mean((p - y_true) ** 2)),
        'ece_full': _ece(p, y_true, BINS_FULL),
        'ece_tail': _ece(p[raw_wp >= 0.70], y_true[raw_wp >= 0.70], BINS_TAIL),
        'ece_core': _ece(p[raw_wp < 0.70], y_true[raw_wp < 0.70], BINS_CORE),
    })
results = pandas.DataFrame(rows).sort_values('ece_full').reset_index(drop=True)
results['rank_ece'] = range(1, len(results) + 1)
results['rank_ll'] = results['log_loss'].rank().astype(int)

print('\n=== Pooled OOS Results (sorted by ECE) ===')
print(results[['rank_ece', 'rank_ll', 'method', 'n_params',
               'log_loss', 'ece_full', 'ece_core', 'ece_tail']]
      .round(4).to_string(index=False))


## ==================== Per-Bin Residuals ==================== ##

bin_edges = BINS_FULL
bin_rows = []
for name in METHODS:
    p = oos_preds[name]
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (raw_wp >= lo) & (raw_wp < hi)
        else:
            mask = (raw_wp >= lo) & (raw_wp <= hi)
        if not mask.any():
            continue
        bin_rows.append({
            'method': name,
            'bin_lo': float(lo),
            'bin_hi': float(hi),
            'bin_mid': float((lo + hi) / 2.0),
            'n': int(mask.sum()),
            'actual_wr': float(y_true[mask].mean()),
            'mean_pred': float(p[mask].mean()),
            'error': float(y_true[mask].mean() - p[mask].mean()),
        })
bins = pandas.DataFrame(bin_rows)

print('\n=== Per-Bin Residual Error — OOS, binned by raw ML ===')
pivot = bins.pivot_table(index=['bin_lo', 'bin_hi'], columns='method', values='error')
pivot = pivot[list(METHODS.keys())]
print(pivot.round(4).to_string())


## ==================== Full-Sample Fit for Curves ==================== ##

z_all = df['logit_wp'].values
y_all = df['fav_win'].values
full_params = {}
for name, method in METHODS.items():
    full_params[name] = {} if method['fit'] is None else method['fit'](z_all, y_all)

wp_grid = numpy.linspace(0.50, 0.99, 200)
z_grid = logit(wp_grid)
curves = {name: method['predict'](z_grid, full_params[name])
          for name, method in METHODS.items()}

df['ml_bin'] = pandas.cut(df['fav_ml_wp'], bins=BINS_FULL, include_lowest=True)
bin_cal = df.groupby('ml_bin', observed=True).agg(
    n=('fav_win', 'count'),
    wins=('fav_win', 'sum'),
    mean_wp=('fav_ml_wp', 'mean'),
).reset_index()
bin_cal['actual_wr'] = bin_cal['wins'] / bin_cal['n']
bin_cal['ci_lo'], bin_cal['ci_hi'] = wilson_ci(bin_cal['wins'], bin_cal['n'])


## ==================== Output ==================== ##

results.to_csv(HERE / 'output.csv', index=False)
bins.to_csv(HERE / 'output_bins.csv', index=False)
print(f'\nSaved: {HERE / "output.csv"}')


## ==================== Chart ==================== ##

method_colors = {
    'Raw ML':       '#888888',
    'Platt':        '#E84855',
    'Beta':         '#3BB273',
    'Poly Logit 2': '#2E86AB',
    'Platt+ReLU':   '#F49D37',
    'Platt+Soft':   '#8B5CF6',
    'Piecewise':    '#D946EF',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

## top-left — ECE comparison ##
ax = axes[0, 0]
ranked = results.sort_values('ece_full')
x = numpy.arange(len(ranked))
w = 0.25
ax.barh(x - w, ranked['ece_core'], w, label='Core (0.50-0.70)',
        color=[method_colors[m] for m in ranked['method']], alpha=0.5, edgecolor='white')
ax.barh(x, ranked['ece_full'], w, label='Full (0.50-1.00)',
        color=[method_colors[m] for m in ranked['method']], alpha=0.75, edgecolor='white')
ax.barh(x + w, ranked['ece_tail'], w, label='Tail (0.70-1.00)',
        color=[method_colors[m] for m in ranked['method']], alpha=1.0, edgecolor='white')
ax.set_yticks(x)
ax.set_yticklabels([f'{row["method"]} ({row["n_params"]}p)'
                    for _, row in ranked.iterrows()], fontsize=9)
ax.set_xlabel('ECE (lower = better, probability units)')
ax.set_title('Expected calibration error by method')
ax.legend(fontsize=8, loc='lower right')
ax.invert_yaxis()
for i, (_, row) in enumerate(ranked.iterrows()):
    ax.text(row['ece_full'] + 0.0005, i, f'{row["ece_full"]:.4f}',
            va='center', fontsize=8)

## top-right — calibration curves ##
ax = axes[0, 1]
ax.plot([0.5, 1.0], [0.5, 1.0], '--', color=C_NEUTRAL, lw=1, zorder=0)
ax.errorbar(
    bin_cal['mean_wp'], bin_cal['actual_wr'],
    yerr=[bin_cal['actual_wr'] - bin_cal['ci_lo'], bin_cal['ci_hi'] - bin_cal['actual_wr']],
    fmt='ko', markersize=6, capsize=3, lw=1, zorder=10, label='Empirical',
)
for name in METHODS:
    lw = 2.0 if name != 'Raw ML' else 1.2
    ls = '--' if name == 'Raw ML' else '-'
    ax.plot(wp_grid, curves[name], color=method_colors[name], lw=lw, ls=ls, label=name)
ax.set_xlabel('ML implied win probability')
ax.set_ylabel('Calibrated win probability')
ax.set_title('Recalibration curves')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.5, 1.0)

## bottom-left — per-bin residuals ##
ax = axes[1, 0]
ax.axhline(0, color=C_NEUTRAL, lw=1)
for name in METHODS:
    sub = bins[bins['method'] == name]
    ax.plot(sub['bin_mid'], sub['error'], 'o-', color=method_colors[name],
            markersize=5, lw=1.5, label=name)
ax.set_xlabel('ML win probability bin')
ax.set_ylabel('Residual error (observed - predicted)')
ax.set_title('OOS per-bin residual error')
ax.legend(fontsize=8)
ax.set_xlim(0.5, 1.0)
raw_bins = bins[bins['method'] == 'Raw ML']
for _, row in raw_bins.iterrows():
    ax.annotate(f'n={row["n"]}', (row['bin_mid'], -0.005),
                fontsize=7, ha='center', color='#999999', style='italic')

## bottom-right — tail zoom ##
ax = axes[1, 1]
ax.plot([0.7, 1.0], [0.7, 1.0], '--', color=C_NEUTRAL, lw=1, zorder=0)
tail = bin_cal[bin_cal['mean_wp'] >= 0.70]
ax.errorbar(
    tail['mean_wp'], tail['actual_wr'],
    yerr=[tail['actual_wr'] - tail['ci_lo'], tail['ci_hi'] - tail['actual_wr']],
    fmt='ko', markersize=6, capsize=3, lw=1, zorder=10, label='Empirical',
)
for name in METHODS:
    lw = 2.0 if name != 'Raw ML' else 1.2
    ls = '--' if name == 'Raw ML' else '-'
    mask = wp_grid >= 0.70
    ax.plot(wp_grid[mask], curves[name][mask],
            color=method_colors[name], lw=lw, ls=ls, label=name)
for _, row in tail.iterrows():
    ax.annotate(f'n={int(row["n"])}', (row['mean_wp'], row['actual_wr']),
                textcoords='offset points', xytext=(0, 10), fontsize=7,
                ha='center', color='#555555')
ax.set_xlabel('ML implied win probability')
ax.set_ylabel('Calibrated win probability')
ax.set_title('Tail zoom (0.70-1.00)')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(0.70, 1.0)
ax.set_ylim(0.70, 1.0)

fig.suptitle('Recalibration method comparison', y=1.01, fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(HERE / 'chart.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {HERE / "chart.png"}')
