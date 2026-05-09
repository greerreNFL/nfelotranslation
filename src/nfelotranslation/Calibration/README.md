# Calibration

Platt / logit-linear recalibration of market ML win probabilities.

## Why it exists

NFL moneylines exhibit a systematic, time-persistent bias at the tails: they overstate the odds of slight favorites (roughly 0.55 to 0.65 implied) and understate the odds of heavy favorites (above 0.85 implied).

Pairwise comparison models typically optimize for observed outcomes, not implied moneyline win probabilities. A translation model that will be consumed by a pairwise model therefore needs to be built from a "true" win probability, not a moneyline implied one. Recalibration of moneyline implied win probabilities is necessary to create a training dataset of true expected win probabilities.

The Recalibrator takes moneyline implied win probabilities and applies a Platt / logit-linear transformation to correct their bias.

This bias is observed across decades, which points towards it being structural in nature and appropriate for a stationary, static fit.

## Empirical evidence

### The bias is real and measurable

Across 5,281 games (2006 to 2025, excluding ties), the observed favorite win rate diverges from the implied probability in the mid and upper bins. Full per-bin results in `analysis/1. Market Moneyline Calibration/`:

| Bin | n | Implied | Observed | Error |
|-----|---:|---:|---:|---:|
| 0.55 - 0.60 | 918 | 0.5748 | 0.5599 | -0.0149 |
| 0.60 - 0.65 | 965 | 0.6252 | 0.5959 | -0.0294 |
| 0.85 - 0.90 | 219 | 0.8719 | 0.8995 | +0.0276 |
| 0.90 - 1.00 | 49 | 0.9155 | 0.9592 | +0.0437 |

The crossover occurs around 0.65 to 0.70 where the implied probability tracks the observed rate. Weighted MAE across all bins: 0.0129.

### The bias is stationary

Full per-season results in `analysis/2. Calibration Stationarity/`. Per-season Platt slopes have a mean of 1.1637 with linear trend p=0.795; per-season intercepts have a mean of -0.1331 with trend p=0.495; per-season weighted MAE has trend p=0.659. None of the three reject zero at conventional thresholds over the 2006 to 2025 window, and the per-era error shape retains the same sign pattern in the upper and mid-range bins across all four eras. A single static fit over the full sample is appropriate.

### Platt scaling is the best method

Seven recalibration methods were compared via leave-one-season-out CV. Full results in `analysis/3. Recalibration Method Comparison/`:

| Rank | Method | Params | ECE full | ECE tail |
|---:|---|---:|---:|---:|
| 1 | Platt | 2 | 0.0107 | 0.0095 |
| 2 | Platt+Softplus | 4 | 0.0112 | 0.0144 |
| 3 | Poly Logit 2 | 3 | 0.0114 | 0.0137 |
| 4 | Beta | 3 | 0.0116 | 0.0167 |
| 7 | Platt+ReLU | 4 | 0.0125 | 0.0120 |

Log loss does not discriminate between methods (all within 0.0005), but ECE separates them. Every method with more than two parameters has a higher ECE tail than Platt; the four-parameter tail extensions (ReLU, Softplus, Piecewise) reduce the 0.85 to 1.00 residuals relative to Platt but enlarge the 0.80 to 0.85 residual, and the 0.80 to 0.85 bin population dominates the ECE tail calculation.

A residual in the 0.60 to 0.65 bin (roughly -0.0144 to -0.0201 across fitted methods) persists across all methods, folds, and eras and is not removed by any of the tested functional forms.

## The math

```
z     = logit(p_market)           # log-odds of market probability
z_cal = slope * z + intercept     # linear correction on logit scale
p_cal = expit(z_cal)              # back to probability
```

- **slope > 1** — the market hedges toward 50%; recalibration stretches probabilities outward toward the extremes.
- **intercept ≈ 0** — no systematic directional bias.

## Pipeline position

The Recalibrator only exists as a translation layer between the moneyline implied win probabilities and the true expected win probabilities. Its main use is in creating a dataset to model downstream primitives from a stronger foundation. Its inclusion also offers functionality for the translator to produce true distributions from biased moneylines, or produce biased moneylines from true distributions.

```
raw market WP  →  [Recalibrator]  →  recalibrated WP  →  downstream primitives
```

## Modules

### `Recalibrator`

The runtime primitive. Init + calibrate + serialize — no fitting dependencies.

- `__init__(params)` — construct from PlattParams
- `calibrate(win_prob)` — apply recalibration to an array of probabilities
- `to_file(filepath)` / `from_file(filepath)` — JSON persistence
- `from_params(slope, intercept)` — direct construction

### Types

- `PlattParams` — `@dataclass(slope, intercept)` with JSON serialization
- `CalibrationResult` — diagnostic output with before/after metrics and `summary()` display

## Retraining

Fitting and validation live in the repository's `training/` package, which is not part of the installed distribution. See `training/TRAINING.md` for how to run the pipeline.

## `platt_params.json`

| Field | Type | Description |
|-------|------|-------------|
| `slope` | float | Logit-linear slope. Values > 1 mean the market compresses toward 50%; recalibration stretches probabilities outward. |
| `intercept` | float | Logit-linear intercept. Near 0 means no systematic directional bias. |
