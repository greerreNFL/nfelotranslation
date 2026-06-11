# Training Guide

This document explains how to retrain the nfelotranslation pipeline. Training
happens once per year after the NFL season ends. The main package (`nfelotranslation`) installed via pip
is used for inference only, which creates path complexity when training must occur

## Why a separate setup

The `training/` directory is not part of the distributed package.
`pip install nfelotranslation`, provides an inference only package, which
is the code in `src/nfelotranslation/`.

Training lives at the repo root and requires
additional dependencies (`nfelodcm` for historical game data) plus access
to the source tree.

If `nfelotranslation` is already installed via pip in the env (which it is likely to be),
 **training should not occur in that environment** as the editable pointer install can/will
clash with the alrady installed version, OR, if the editable pointer is not installed, the
training may expect a different version of the code when it imports nfelotranslation.

A seperate environment ensures these issues are avoided. This is the recommended setup.

## One-time setup

### 1. Create a dedicated dev environment

```bash
conda create -n nfelotranslation-dev python=3.12
conda activate nfelotranslation-dev
```

### 2. Install the inference package in editable mode

From the **repo root** (the directory containing this file):

```bash
pip install -e ".[dev]"
```

This does two things:
- Installs `nfelotranslation` as a pointer to `src/nfelotranslation/` — any
  edits to source files are immediately live, no reinstall needed.
- Installs dev dependencies: `pytest`, `nfelodcm`, `build`, `twine`.

### 3. Verify

```bash
python -c "from nfelotranslation import Translator; print('OK')"
```

This setup survives until the package is uninstalled or the repo is moved to a different path.


## Running the training pipeline

The `training/` package is not installed — it lives at the repo root and
needs `PYTHONPATH=.` so Python can find it. This is the only command needed
to run the training pipeline (again with the cwd set to the repo root).

```bash
PYTHONPATH=. python -m training.Scripts
```

This runs `training/Scripts/__init__.py:train_all()`, which executes the
full pipeline in dependency order:

1. **Recalibrator** — fits split Platt params (omniscient home/away slopes, per-season intercepts)
2. **SpreadMapper** — fits linear-in-logit wp-to-spread mapping on recalibrated win probabilities vs actual margins
3. **KeyModel** — updates all 40 credibility-weighted ratio trackers per season
4. **Distribution validation** — system-level integration check (SAE, bias)

Each phase writes its config to `src/nfelotranslation/` (the live source).
A shared `pipeline_id` links all configs from the same run. The DataLoader
is reset between phases so downstream fitters see upstream outputs.

### What PYTHONPATH=. does

It temporarily adds the current directory (repo root) to Python's module
search path for this one command. This is what makes `import training` work.
It does not modify your environment permanently — the next command you run
is unaffected.

### Training individual components

If you only need to retrain one component (e.g., after a parameter change):

```bash
# Just the KeyModel
PYTHONPATH=. python -m training.Scripts.train_key_model

# Or interactively
PYTHONPATH=. python
>>> from training.Distribution.Key.KeyModelFitter import KeyModelFitter
>>> fitter = KeyModelFitter()
>>> result = fitter.fit()
>>> print(result.summary())
```

### Validating without retraining

```bash
PYTHONPATH=. python -c "
from training.Validation import MarginDistributionValidator
from training.Data.DataLoader import DataLoader
validator = MarginDistributionValidator(games=DataLoader.get().games)
report = validator.validate()
print(report.summary())
validator.save(report)
"
```

## Running tests

Tests do not need the `PYTHONPATH` prefix. The `pyproject.toml` configures
pytest to add `src` and `.` to the path automatically:

```bash
pytest
```

This works both locally and in GitHub Actions CI. The CI workflow installs
`pip install -e ".[dev]"` and runs `pytest` — same as local.

## Output locations

Training writes configs directly into the source tree so the editable
install picks them up immediately:

| Component | Root config | Per-season configs |
|-----------|------------|-------------------|
| Recalibrator | `src/nfelotranslation/Calibration/platt_params.json` | `Calibration/configs/` |
| SpreadMapper | `src/nfelotranslation/SpreadMap/spread_map_params.json` | `SpreadMap/configs/` |
| KeyModel | `src/nfelotranslation/Distribution/Key/key_model.json` | `Distribution/Key/configs/` |

Validation reports are written to `validation/` subdirectories within each
component's folder.

## Per-component notes

### SpreadMap

#### When to retrain

Each offseason, retrain on all available data. The mapper is fit
seasonally with flat weighting (`decay = 0.0`) because the WP → margin
relationship is structural. It fits on recalibrated win probabilities
(`ml_wp_cal`) against actual game margins, with intercept fixed at zero.

#### How to retrain

```python
from training.SpreadMap.SpreadMapperFitter import SpreadMapperFitter

fitter = SpreadMapperFitter()
result = fitter.fit()
print(result.summary())

report = fitter.validate()
print(report.summary())
fitter.save_validation(report)
fitter.save_config()  ## update spread_map_params.json for runtime use
```

Per-season configs are saved automatically to
`SpreadMap/configs/spread_map_params_{season+1}.json` during training.
These point-in-time snapshots support backtesting of downstream consumers.

#### Interpreting validation output

Validation answers two core questions:

1. Does the mapper produce a true median spread?
2. Does the parametric form track empirical bin medians of actual margins?

**Mapper quality.** The SpreadMapper converts a win probability
into a predicted spread by fitting against actual game margins under MAE
loss. A "good spread" is a median: 50% of margins land above it. The
bisection rate measures this directly. The MAE (~10.2 points)
looks large but reflects NFL margin unpredictability (σ ≈ 13 points) —
the model is finding the distribution center, not predicting individual
games.

**Binned fit quality.** `binned_r2` and `binned_mae` compare the mapper
to pooled empirical medians of actual margins within fine WP bins. This
measures parametric misfit against realized outcomes, not posted market
lines.

#### Gated checks

| Check                        | Threshold        | What it measures |
|------------------------------|------------------|------------------|
| `slope_positive`             | > 0              | Higher win probability must map to a larger predicted margin. |
| `bisection_rate_centered`    | \|rate − 0.5\| < 0.05 | Aggregate OOS fraction of game margins above the predicted spread. |

#### Tracked metrics

| Metric                        | What it measures                                                              | What to expect            |
|-------------------------------|-------------------------------------------------------------------------------|---------------------------|
| `mae`                         | OOS MAE: predicted spread vs actual margin. Reflects margin unpredictability.  | ~10.2 pts                 |
| `bisection`                   | OOS bisection rate. Confirms MAE produces a true median.                      | ~0.502                    |
| `binned_r2`                   | R² of mapper predictions vs empirical bin medians of actual margins (pooled). | ~0.81                     |
| `binned_mae`                  | MAE of mapper predictions vs bin medians. Parametric misfit.                  | ~2.0 pts                  |
| `slope`                       | Final slope (intercept fixed at 0).                                             | ~6.52                     |
| `mae_trend`                   | Linear trend in per-season MAE. Flat = stable.                                  | Slightly negative         |

#### What to watch for

- **Bisection rate drifting from 0.5** — the single most important check.
  If the MAE-fitted model stops producing a true median, the loss
  function or the data filter may need attention.
- **Binned R² collapsing** — would indicate the parametric form no longer
  tracks empirical margin medians across the WP range.
- **Slope trending** — would indicate the WP → spread relationship
  is shifting over time (e.g. from scoring-era changes).

### Key

#### When to retrain

Each offseason, after the season completes. The seasonal fitter processes
all seasons chronologically, so retraining is a full replay from scratch
with the new season appended. Per-season configs (one snapshot per
season trained) support point-in-time backtesting of downstream consumers.

#### How to retrain

```python
from training.Distribution.Key.KeyModelFitter import KeyModelFitter

fitter = KeyModelFitter()
result = fitter.fit()
print(result.summary())

report = fitter.validate()
print(report.summary())
fitter.save_validation(report)
```

Per-season configs are written to
`src/nfelotranslation/Distribution/Key/configs/key_model_{season+1}.json`
during training.

#### Interpreting validation output

Validation answers two questions: does tracking per-integer key number
excess improve the margin distribution, and how well does the model
predict next season's key number patterns?

The base margin distribution is a generalized normal — smooth, symmetric,
and silent on integer effects. Empirically, margin = 3 occurs roughly
0.097 above the baseline rate, margin = 7 occurs roughly 0.042 above,
and margin = 9 occurs roughly 0.026 below (analysis 8). The KeyModel
tracks 40 integers (margins 1–40) using credibility-weighted ratio
trackers. Each season the model predicts a ratio for each integer
*before* seeing that season's data; validation compares those predictions
to what actually happened.

Excess is reported in percentage points (pp) inside the validator's
output for readability. A 1 pp prediction error at a given integer
corresponds to roughly 2.8 games per ~280-game season.

Without the KeyModel, the margin distribution predicts ratio = 1.0
(zero excess) at every integer. The validator's `baseline_rmse`
measures how wrong that assumption is; the model's `oos_rmse` measures
how much residual error remains after the model's corrections. The
ratio of the two is the headline value-add measurement: if it ever
exceeds 1.0, the key number adjustments make the distribution worse
than ignoring them.

#### Gated checks

| Check                  | Threshold     | What it measures |
|------------------------|---------------|------------------|
| `model_beats_baseline` | ratio < 1.0   | model RMSE / baseline RMSE. If above 1.0, the model is harmful. |
| `oos_rmse_excess`      | < 1.5 pp      | Root-mean-square of OOS prediction errors across all 40 numbers, averaged across seasons. |
| `oos_mae_excess`       | < 1.2 pp      | Same concept as RMSE but using mean absolute error. Less sensitive to the largest-excess outliers (such as `+/-3`) that dominate RMSE. |
| `worst_number_rmse`    | < 5.0 pp      | OOS RMSE for the single worst-predicted number. Typically `+/-3` because of its large excess and per-season variance. The threshold ensures no single number is wildly mispredicted. |

#### Tracked metrics

| Metric                    | What it measures |
|---------------------------|------------------|
| `oos_rmse_excess_pp`      | Same value as the gated RMSE check, with per-season breakdown. |
| `oos_mae_excess_pp`       | Same value as the gated MAE check, with per-season breakdown. |
| `baseline_rmse_pp`        | Error from predicting ratio = 1.0 at every integer. Falling means key number effects are weakening. |
| `model_vs_baseline_ratio` | model RMSE / baseline RMSE. Lower means more value added. |
| `top_10_numbers_rmse`     | Average RMSE across the 10 most-landed-on margins. |
| `worst_number_detail`     | Identity and exposure (hits across all seasons) of the worst OOS number. |
| `rmse_trend`              | Linear trend in per-season OOS RMSE. Negative = improving; positive = patterns drifting faster than the model adapts. |
| `mae_trend`               | Same for MAE. |

#### What to watch for

- **`model_beats_baseline` ratio approaching 1.0** — the model's value
  is eroding. If it crosses 1.0, the key number adjustments make the
  distribution worse and the credibility parameters likely need
  retuning.
- **RMSE/MAE spiking for a single season** — check the per-season
  diagnostics. A spike can indicate a structural rule change (the 2015
  PAT change is the canonical example) shifting which numbers are key.
  The exponential decay will adapt over multiple seasons, but a spike
  is a signal to investigate.
- **Worst-number RMSE growing year-over-year** — the worst number's
  behavior may have changed structurally.
- **`baseline_rmse` dropping below ~1.5 pp** — key number effects in
  general are weakening. The KeyModel adjustment may not be worth its
  complexity at that point.

#### Per-season diagnostics

```python
result = fitter.fit()
for d in result.per_season:
    print(f'{d.season}: model={d.metrics["rmse_excess_pp"]:.2f}pp  '
          f'baseline={d.metrics["baseline_rmse_pp"]:.2f}pp  '
          f'n={d.metadata["n_games"]}')
```

#### Distribution-level validation

After all pipeline components are trained (Recalibrator, SpreadMapper,
KeyModel), a system-level `MarginDistributionValidator` evaluates the
full composed distribution against historical outcomes using per-season
key model configs for true OOS evaluation:

```python
from training.Validation import MarginDistributionValidator
from training.Data.DataLoader import DataLoader

validator = MarginDistributionValidator(games=DataLoader.get().games)
report = validator.validate()
print(report.summary())
validator.save(report)
```

This checks aggregate SAE (sum of absolute errors between predicted and
actual margin frequencies), regional bias (close, mid, tail),
per-key-number accuracy, and stationarity trends. It is the end-to-end
integration test for the margin distribution pipeline.

## Troubleshooting

**`ModuleNotFoundError: No module named 'training'`**
You forgot `PYTHONPATH=.` or you're not in the repo root directory.

**`ModuleNotFoundError: No module named 'nfelotranslation'`**
You haven't run `pip install -e .` in this environment, or the editable
install is pointing to an old location. Re-run `pip install -e ".[dev]"`
from the repo root.

**`ModuleNotFoundError: No module named 'nfelodcm'`**
Install with `pip install nfelodcm`. This is the NFL game data source
required for training. It is included in `pip install -e ".[dev]"`.

**Training output doesn't affect my other project**
Correct. The editable install only affects the `nfelo-dev` environment.
Your other projects using `pip install nfelotranslation` from PyPI are
unchanged until you publish a new version.
