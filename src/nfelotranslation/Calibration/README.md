# Calibration

Split Platt / logit-linear recalibration of market ML win probabilities.

## Why it exists

NFL moneylines exhibit a systematic, time-persistent bias: they compress toward 50%, overstating mild favorites and understating (or inconsistently pricing) strong favorites depending on location. Home favorites and away favorites do not share the same miscalibration shape — see `analysis/1. Market Moneyline Calibration/`.

Pairwise comparison models and downstream fitters (SpreadMapper, KeyModel) need training labels in **model win-probability space**, not raw market implied probabilities. The Recalibrator derives those labels from closing moneylines.

Recalibration is a **training and research primitive only**. It is not composed into the inference `Translator`. At prediction time callers pass model win probabilities or model spreads; applying recalibration in production would inject information not knowable at that point in time (see `analysis/2. Calibration Stationarity/`).

## Empirical evidence

### The bias is real and measurable

Across 5,281 games (2006 to 2025, ties excluded), favorite-perspective implied probabilities diverge from observed win rates. Full per-bin results in `analysis/1. Market Moneyline Calibration/`:

| Bin | n | Implied | Observed | Error |
|-----|---:|---:|---:|---:|
| 0.55 - 0.60 | 918 | 0.5748 | 0.5599 | -0.0149 |
| 0.60 - 0.65 | 965 | 0.6252 | 0.5959 | -0.0294 |
| 0.85 - 0.90 | 219 | 0.8719 | 0.8995 | +0.0276 |
| 0.90 - 1.00 | 49 | 0.9155 | 0.9592 | +0.0437 |

Weighted MAE across bins: 0.0129. Home and away favorites require separate slopes (full-sample: `a_home ≈ 1.21`, `a_away ≈ 0.97`).

### Slopes are structural; intercepts drift

`analysis/2. Calibration Stationarity/` fits split Platt parameters by season. Per-location **slopes** are stable enough to treat as omniscient (full-sample fit, held fixed across seasons). **Intercepts** vary over calendar time — especially for home favorites — consistent with slow-moving home-field-advantage mispricing.

The shipped training scheme holds slopes fixed and refits **intercepts** on a centered 5-season window (edge-padded) for each label season. Walk-forward intercepts improve training-label calibration vs a static omniscient intercept; they are not used at inference.

### Platt scaling is the best method

Seven recalibration methods were compared via leave-one-season-out CV. Full results in `analysis/3. Recalibration Method Comparison/`:

| Rank | Method | Params | ECE full | ECE tail |
|---:|---|---:|---:|---:|
| 1 | Platt | 2 | 0.0107 | 0.0095 |
| 2 | Platt+Softplus | 4 | 0.0112 | 0.0144 |
| 3 | Poly Logit 2 | 3 | 0.0114 | 0.0137 |
| 4 | Beta | 3 | 0.0116 | 0.0167 |
| 7 | Platt+ReLU | 4 | 0.0125 | 0.0120 |

Log loss does not discriminate between methods (all within 0.0005), but ECE separates them. Every method with more than two parameters has a higher ECE tail than Platt.

## The math

Favorite-perspective split Platt:

```
z       = logit(p_market)                         # log-odds of market probability
z_cal   = a_loc * z + b_loc                       # separate slope/intercept by location
p_cal   = expit(z_cal)                            # back to probability
```

- **`a_home > a_away`** — home favorites need more de-compression; away favorites are closer to identity.
- **`b_loc`** — captures level shift (HFA / market pricing drift); varies by season in training configs.

## Pipeline position

```
Training:  market ML  →  [Recalibrator]  →  ml_wp_cal  →  SpreadMapper / KeyModel fitters
Inference: model WP or spread  →  [Translator]  (no Recalibrator)
```

End users may import `Recalibrator` directly for historical labeling or research.

## Modules

### `Recalibrator`

Runtime primitive. Init + calibrate + serialize — no fitting dependencies.

- `__init__(params)` — construct from `SplitPlattParams`
- `calibrate(win_prob, is_home_fav=None)` — apply recalibration
- `to_file(filepath)` / `from_file(filepath, season=None)` — JSON persistence with optional per-season config
- `from_params(slopes, intercepts)` — direct construction

### Types

- `SplitPlattParams` — `slopes` and `intercepts` dicts (`home` / `away`), optional `fit` metadata
- `CalibrationResult` — diagnostic output with before/after metrics and `summary()` display

## Retraining

Fitting and validation live in the repository's `training/` package, which is not part of the installed distribution. See `training/TRAINING.md` for how to run the pipeline.

## Config files

Root envelope `platt_params.json` holds the latest shipped parameters. Per-season training labels live in `configs/platt_params_{season}.json`.

| Field | Type | Description |
|-------|------|-------------|
| `slopes.home`, `slopes.away` | float | Omniscient split slopes |
| `intercepts.home`, `intercepts.away` | float | Intercepts for the label season |
| `fit.window_type` | str | `centered` for the shipped intercept scheme |
| `fit.window_width` | int | Seasons in the intercept window (shipped: 5) |
| `fit.seasons_used` | list | Seasons contributing to the intercept fit |
