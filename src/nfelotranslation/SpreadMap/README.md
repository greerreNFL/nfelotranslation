# SpreadMap

Bidirectional win probability ↔ spread mapping via a linear-in-logit model.

## Why it exists

As a translation package, the model must be able to translate win probabilities to spreads, and vice versa. It is a fundamental primitive. The main purpose of the spread map is to translate a single win probability into a spread, or a spread into a probability. It forms the center of the margin distribution, but does not provide a distribution itself.

Training labels require mapping **recalibrated** market win probabilities (`ml_wp_cal`) to **actual game margins** — the real outcomes. Given a WP, the mapper answers: "what margin number would games be equally likely to exceed or fall short of?" At inference, callers pass model win probabilities or model spreads in the same coordinate system; no market mapper is needed.

## Why MAE and intercept = 0

The shipped mapper uses MAE (mean absolute error) loss, which finds the **median** of the target distribution. A spread is a median by definition: the number where 50% of outcomes land above and 50% below.

The mapper is fit with `intercept = 0`, leaving only the slope free. This preserves the empirical anchor `WP = 0.50 ↔ spread = 0`. Training data uses recalibrated market WPs as proxies for true win probabilities; constraining the intercept guarantees at least this anchor point is correct.

## Empirical evidence

### Parametric vs lookup, free vs forced intercept

Three forms were fit on the same `(fav_wp_cal, fav_margin)` data and evaluated both in-sample and via leave-one-season-out (LOSO). Full per-bin results in `analysis/4. Spread Mapping Form/`:

| Form                  | slope  | intercept | in-sample MAE | LOSO MAE | LOSO bisection |
|-----------------------|-------:|----------:|--------------:|---------:|---------------:|
| Linear (free)         | 6.0754 | 0.4754    | 10.2217       | 10.2285  | 0.4999         |
| Linear (intercept=0)  | 6.5500 | 0.0000    | 10.2252       | 10.2272  | 0.5071         |
| Lookup (1% bin)       | —      | —         | 10.1600       | 10.2624  | 0.4965         |

The two linear forms generalize equally well; the lookup wins on in-sample MAE but loses on LOSO MAE, the gap consistent with overfit at sparsely populated bins above WP ≈ 0.90. The forced-zero form pays 0.0013 LOSO MAE for a clean WP=0.50 anchor.

### Stationarity

Per-season fits across 2006 to 2025 with the mapper refit independently per season. Full results in `analysis/5. Spread Map Stationarity/`:

| Parameter | Mean   | Std    | Trend / yr | p-value |
|-----------|-------:|-------:|-----------:|--------:|
| Slope     | 6.5627 | 1.1230 | +0.0823    | 0.0560  |

The slope is stationary at conventional thresholds. A single static fit is shipped. Season-indexed parameters are not warranted at present; revisit as additional seasons accumulate or if the per-season trend strengthens.

### Within-spread price signal

Posted spreads are quoted on the half-point grid, with American prices on either side. Treating the prices as a fine-grained spread signal would complicate the mapping; the question is whether they carry margin information at all. Full results in `analysis/6. Within-Spread Price Signal/`:

| Spread | Top-bottom mean cp gap | Top-bottom median margin gap | Top-bottom cover rate gap |
|-------:|----------------------:|-----------------------------:|--------------------------:|
| 2.5    | +0.0292               | -2.0000                      | -0.1132                   |
| 3.0    | +0.0386               | +0.0000                      | -0.0670                   |
| 3.5    | +0.0272               | +0.0000                      | -0.0384                   |
| 7.0    | +0.0239               | -1.0000                      | +0.0032                   |

Across the four posted spreads with sufficient sample (n ≥ 200) for a tercile split, none produce a positive median-margin gap and one produces a positive cover-rate gap. The pooled within-spread regression of demeaned margin on demeaned cover probability yields a correlation of 0.0170 across 3,406 games. The mapping therefore operates on posted spread values only, not on price sub-structure.

## The math

```
spread = slope * logit(wp) + intercept      # WP → spread
wp     = expit((spread - intercept) / slope) # spread → WP (exact inverse)
```

- **slope** — points of spread per unit of logit. With `slope ≈ 6.5`, each unit of logit (≈ 10pp of WP at the center) corresponds to about 6 to 7 points of spread.
- **intercept** — the spread at 50% win probability. Shipped value is `0` by construction.

## The Spread type

`Spread(posted, continuous)`:
- **posted** — clamped to the nearest 0.5 grid point, for display and comparison with market lines.
- **continuous** — raw formula output, for lossless round-trip precision when feeding back into `spread_to_win_prob()`.

## Pipeline position

Maps between win probability and spread spaces. Used at inference by the `Translator` and at training time to derive model spreads from recalibrated market WPs for the KeyModel fitter.

```
Training:  ml_wp_cal  →  [SpreadMapper]  →  model spread  →  KeyModel
Inference: model WP or spread  ↔  [SpreadMapper]  ↔  MarginDistributionModel
```

Recalibration lives upstream of training only; it is not composed into the inference `Translator`.

## Modules

### `SpreadMapper`

The runtime primitive. Init + map + serialize — no fitting dependencies.

- `__init__(params)` — construct from `LinearMapParams`
- `win_prob_to_spread(win_prob) -> Spread` — forward mapping
- `spread_to_win_prob(spread) -> float` — inverse mapping
- `to_file(filepath)` / `from_file(filepath, season=None)` — JSON persistence with optional per-season config
- `from_params(slope, intercept)` — direct construction

### Types

- `LinearMapParams` — `@dataclass(slope, intercept)` with dict serialization
- `Spread` — `@dataclass(posted, continuous)`
- `SpreadMapResult` — diagnostic output with MAE, bisection rate, and `summary()` display

## Retraining

Fitting and validation live in the repository's `training/` package, which is not part of the installed distribution. See `training/TRAINING.md` for how to run the pipeline.

## `spread_map_params.json`

| Field | Type | Description |
|-------|------|-------------|
| `slope` | float | Points of spread per unit of logit. ~6 to 7 means each logit unit (roughly 10pp of WP at the center) corresponds to about 6 to 7 points of spread. |
| `intercept` | float | Spread at 50% win probability. Constrained to 0 at fit time. |

Per-season configs live in `configs/spread_map_params_{season}.json`.
