# SpreadMap

Bidirectional win probability ↔ spread mapping via a linear-in-logit model.

## Why it exists

As a translation package, the model must be able to translate win probabilities to spreads, and vice versa. It is a fundamental primitive. The main purpose of the spread map is to translate a single win probability into a spread, or a spread into a probability. It forms the center of the margin distribution, but does not provide a distribution itself.

## Why two instances

The package ships with two SpreadMappers. MODEL, which fits calibrated win probabilities to margin outcomes, and MARKET, which fits moneyline implied win probabilities to spreads. Model exists to provide the most accurate translation of true win probabilities to spreads as possible, hence why it is fitted to actual outcomes and is constrained to have an intercept of 0 (so WP=0.50 ↔ spread=0). However, since the starting point of translation is not always a "true" win probability or "true" spread, a second model is needed to map market variants. For instance if a market spread is used to init the translator, it needs to first be converted into a market implied win probability before it can then be calibrated into a "true" win probability to form the rest of the translators properties. If the same mapper was used for both, the win probability derived from the market spread would not be reflective of the actual win probability the market was implying.

### What each mapper fits to

The MODEL mapper fits win probability against **actual game margins** — the real outcomes. Given a WP, it answers: "what margin number would games be equally likely to exceed or fall short of?"

The MARKET mapper fits win probability against **market-posted lines** — what the market actually set the line at. Given a WP, it answers: "what spread would the market post?"

These are different target variables. A model-derived spread from WP corresponds to the center of the realized outcome distribution. A market spread is an observed value set by the market, which may or may not perfectly center the outcome distribution.

### Why MAE for both

Both mappers use MAE (mean absolute error) loss, which finds the **median** of the target distribution. A spread is a median by definition: the number where 50% of outcomes land above and 50% below. This holds regardless of whether the target variable is actual game margins or market-posted lines.

### Why the model intercept is forced to zero

The MODEL mapper is fit with `intercept = 0`, leaving only the slope free. The MARKET mapper is fit with both parameters free. The model is attempting to translate true win probabilities into true spreads, but since our training data of "true" spreads is actually just market implied win probabilities, the model is baised. "True" win probabilities are not actually known! There is only one point in the "true" map that is knowable: WP=0.50 ↔ spread=0. To get as close as possible to the true map, the model therefore constrains the intercept to be 0 so the resulting map is gauranteed to at least get this point correct.

The MARKET mapper on the other hand is not trying to translate true win probabilities into true spreads. It's job is to most accurately reflect a translation between moneyline win probabilities and spreads. Thus, it is allowed to have a free intercept.

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

Per-season fits across 2006 to 2025 with both mappers refit independently per season. Full results in `analysis/5. Spread Map Stationarity/`:

| Parameter         | Mean   | Std    | Trend / yr | p-value |
|-------------------|-------:|-------:|-----------:|--------:|
| Model slope       | 6.5627 | 1.1230 | +0.0823    | 0.0560  |
| Market slope      | 6.7595 | 0.3934 | -0.0108    | 0.4947  |
| Market intercept  | 0.2830 | 0.2767 | +0.0251    | 0.0146  |

Both slopes are stationary at conventional thresholds. The market intercept shows a small but statistically detectable upward drift of +0.025 points per year, accumulating to roughly half a point of offset at `WP = 0.50` across the 20-year window — under one half-point grid step.

A single static fit is shipped for each mapper. The observed drift in the market intercept is potentially real (the market gets smarter over time!) but small enough that season-indexed parameters are not warranted at present. This trade-off should be revisited as additional seasons accumulate or if the per-season trend strengthens.

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

- **slope** — points of spread per unit of logit. With `slope ≈ 6` to `7`, each unit of logit (≈ 10pp of WP at the center) corresponds to about 6 to 7 points of spread.
- **intercept** — the spread at 50% win probability. The MODEL mapper has `intercept = 0` by construction. The MARKET mapper has a fitted intercept that is empirically near zero.

## The Spread type

`Spread(posted, continuous)`:
- **posted** — clamped to the nearest 0.5 grid point, for display and comparison with market lines.
- **continuous** — raw formula output, for lossless round-trip precision when feeding back into `spread_to_win_prob()`.

## Pipeline position

Maps between win probability and spread spaces; downstream the translator composes this with the Recalibrator and margin distribution.

```
win_prob  ↔  [SpreadMapper]  ↔  spread
```

## Modules

### `SpreadMapper`

The runtime primitive. Init + map + serialize — no fitting dependencies.

- `__init__(params)` — construct from `LinearMapParams`
- `win_prob_to_spread(win_prob) -> Spread` — forward mapping
- `spread_to_win_prob(spread) -> float` — inverse mapping
- `to_file(map_type, filepath)` / `from_file(map_type, filepath)` — JSON persistence (merges MODEL/MARKET into one file)
- `from_params(slope, intercept)` — direct construction

### Types

- `MapType` — enum: `MODEL` or `MARKET`
- `LinearMapParams` — `@dataclass(slope, intercept)` with dict serialization
- `Spread` — `@dataclass(posted, continuous)`
- `SpreadMapResult` — diagnostic output with MAE, bisection rate, and `summary()` display

## Retraining

Fitting and validation live in the repository's `training/` package, which is not part of the installed distribution. See `training/TRAINING.md` for how to run the pipeline.

## `spread_map_params.json`

Contains two keyed objects, one per mapper instance:

| Key | Description |
|-----|-------------|
| `model` | Params for the MODEL mapper (fitted against actual margins). |
| `market` | Params for the MARKET mapper (fitted against market-posted lines). |

Each object has:

| Field | Type | Description |
|-------|------|-------------|
| `slope` | float | Points of spread per unit of logit. ~6 to 7 means each logit unit (roughly 10pp of WP at the center) corresponds to about 6 to 7 points of spread. |
| `intercept` | float | Spread at 50% win probability. The MODEL mapper is constrained to 0 by design; the MARKET mapper is fit freely and is empirically near zero. |
