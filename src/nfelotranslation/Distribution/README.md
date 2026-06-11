# Distribution

Discrete margin distributions over integer outcomes, anchored to a spread and a win probability.

## Why it exists

The package's primary job is translating between win probabilities and spreads — `SpreadMap` handles that direction. A spread, however, only describes the bisection point of the margin distribution; it says nothing about the rest of the integer outcomes, which are needed for common use cases like:

- Pricing alternative spreads — given a market spread of `-3`, what is the probability of a `-7` or better?
- Expected value calculations — what is `E[margin]` and how does it compare to a posted line?
- Probabilities of integer outcomes — the chance of a push at any number, or the cumulative probability through a series of integer thresholds.

The Distribution module produces that distribution. It takes `(spread, win_prob)` and returns a `MarginDistribution` whose discrete PMF over integer margins `-75..+75` integrates to the same win probability the spread map ties to that spread, while assigning realistic, key aware mass to every integer in between.

## How a margin distribution is formed

The distribution is built in three stages, each owned by a sub-module:

### Base — the smooth shape

NFL margin outcomes are roughly normal around the spread, with slightly heavier tails than a Gaussian. The `Base/` sub-module provides a continuous generalized normal centered at the spread, with the scale parameter derived analytically from `(spread, win_prob)` so that `P(margin > 0) = win_prob` holds by construction. The shape parameter `beta` is a fitted hyperparameter (shipped value: `1.24`). Empirical backing for the form lives in Analysis 7 and is summarized in `Base/README.md`.

### Key — per-integer adjustments

A smooth distribution alone misses NFL's integer-level structure: certain margins (`±3`, `±7`) occur substantially more often than a smooth shape predicts, while others (`±9`, `±12`) occur less often. The `Key/` sub-module tracks a credibility-weighted observed-to-expected ratio for every integer `1..40` and applies it multiplicatively to the baseline PMF: `bin += (ratio - 1) * baseline`. Empirical backing lives in Analysis 8 and is summarized in `Key/README.md`.

### Normalizer — preserve the core translation

Discretizing a continuous distribution into integer bins introduces rounding, and the Key adjustments add and subtract mass at specific integers. Either step can break the constraints that the rest of the package depends on:

- `P(margin > 0)` must equal `win_prob`.
- `P(margin = 0)` must equal `tie_prob`.
- `P(margin < 0)` must equal `1 - win_prob - tie_prob`.
- The spread must remain the bisection point of the discrete PMF.

The `Normalizer/` sub-module restores these by partitioning the margin space into labelled regions around zero and the spread, then per-region scaling so each constraint holds. Details in `Normalizer/README.md`.

## Pipeline

```
                          spread, win_prob
                                 ↓
                       [BaseDistribution]
                                 ↓ continuous gennorm(loc=spread, scale, beta)
                  [Normalizer.discretize()]
                                 ↓ baseline PMF (151 integer bins)
                   [KeyModel.get_all_excess()]
                                 ↓ (ratio - 1) * baseline added at ±k for k in 1..40
                                 ↓ negatives clamped to 0
                   [Normalizer.normalize()]
                                 ↓ A/T/B/C/D region scaling to enforce constraints
                       [MarginDistribution]
                                 ↓
                  cover_prob, push_prob, expected_margin, ...
```

## Modules

### `MarginDistributionModel`

The top-level composer. Holds a `KeyModel` and the shipped hyperparameters; turns a `(spread, win_prob)` pair into a `MarginDistribution`.

- `__init__(key_model, tie_prob, beta)` — `tie_prob` and `beta` default to the values in `margin_hyperparams.json`.
- `predict(spread, win_prob)` → `MarginDistribution` — runs the full pipeline above.

### Types

- `MarginDistribution` — result container holding `spread`, `win_prob`, `tie_prob`, and the discrete `pmf` (ndarray of shape `(151,)`). Provides:
  - `cover_prob(line)` — `P(margin > line)`, with integer-line push handling.
  - `push_prob(line)` — `P(margin == line)`; zero for non-integer lines.
  - `win_prob_from_pmf()`, `loss_prob_from_pmf()`, `tie_prob_from_pmf()` — recompute the constraint quantities directly from the PMF.
  - `expected_margin()` — `E[margin]` from the discrete PMF.

### Sub-modules

- `Base/` — continuous generalized normal baseline. See `Base/README.md`.
- `Key/` — per-integer credibility-weighted excess trackers. See `Key/README.md`.
- `Normalizer/` — discretization and region-based normalization. See `Normalizer/README.md`.

## `margin_hyperparams.json`

Loaded once at module import as `MARGIN_HYPERPARAMS`, the single source of truth for the shipped values used by `MarginDistributionModel`.

| Field | Type | Description |
|-------|------|-------------|
| `beta` | float | Generalized-normal shape parameter for the base distribution. Shipped value `1.24` (between Gaussian at `2.0` and Laplace at `1.0`). |
| `tie_prob` | float | Discrete tie probability carved out at margin `0` by the Normalizer. Shipped value `0.002`. |

## Retraining

Selection of `beta`, fitting of the `KeyModel`, and system-level distribution validation live in the repository's `training/` package, which is not part of the installed distribution. See `training/TRAINING.md`.
