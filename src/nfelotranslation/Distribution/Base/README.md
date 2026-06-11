# Base

Continuous generalized normal distribution parameterized by `(spread, win_prob, beta)`.

## Why it exists

The margin distribution pipeline needs a smooth continuous foundation from which the discrete PMF is derived. A generalized normal distribution is a natural choice: it is parameterized by location, scale, and shape (`beta`), and the inputs available at inference time — spread, win probability, and a fixed shape parameter — uniquely determine all three.

Setting `loc = spread` makes the spread the center of the distribution (its mean and median). The scale is then solved analytically from the win-probability constraint `P(margin > 0) = win_prob`, giving:

```
scale = spread / gennorm.ppf(win_prob, beta)
```

No fitting or optimization is required at runtime — the distribution is fully determined by the inputs and one equation.

The Base module creates a distribution that satisfies the two most crritical constraints of the translator -- win probability must be the sum of outcomes above 0, and the spread must bisect the distribution of outcomes.

From this, the distribution can be manipulatd in any way (ie add spikes for key numbers, descritize to integer bins), so long as the resulting distribution is always normalized back to the shape and characteristics of Base.

## Why a generalized normal

A standard Gaussian (`beta = 2.0`) produces tails that are too thin for NFL margins. The symmetric Gaussian under-predicts blowout games (`|margin| >= 17`) and over-concentrates mass near the spread. Setting `beta < 2.0` redistributes mass from the center into the tails and produces a much closer regional fit.

The empirical evidence for this choice is in `analysis/7. Margin Distribution Form/`. Fitting a free generalized normal directly on bucketed empirical margins (2006–2025, favorite perspective) returns `beta = 1.60` on the pooled sample, with all buckets that have `n > 800` falling in `[1.53, 1.80]`. KS distance and log-likelihood favor gennorm over Gaussian in every bucket. The shape is consistent enough across spread levels to support a single fixed `beta` rather than a spread-dependent one.



### Choice of `beta`

The shipped value is `beta = 1.24`, loaded from `src/nfelotranslation/Distribution/margin_hyperparams.json`. This value for beta does not minimize prediction error, but rather is hand picked to make the model more balanced and generalizable across values.

The selected value sits inside the feasible window where all three regional checks pass with headroom to spare. The current validation snapshot (`src/nfelotranslation/Distribution/validation/margin_distribution.json`) reports close-region bias of `+0.3%` against a `+/-5%` threshold and tail-region bias of `-5.5%` against a `+/-10%` threshold at `beta = 1.24`. Said more plainly, it is possible to create a model that performs better on an error basis, but fits tails poorly because there are so few observations at higher spreads. In an effort to make nfelotranslation generalizable, a less-than-optimal beta is hand picked based on the tradeoff (pure error minimization vs generalizability).

It's also worth noting that the shipped value is lower than the marginal MLE (`beta = 1.60`, see analysis 7). Becaude key numbers impact the ultimate end distribution, and is optimized with the baseline, best beta need not be the marginal MLE beta.

### Skew was investigated and set aside

NFL margin distributions are mildly asymmetric across spread — favorites carry a slightly longer blowout-win tail than a fully symmetric distribution would predict. A skew-normal alternative was evaluated, and the marginal improvement in log-likelihood was small enough that it did not justify the additional parameter or the loss of analytic invertibility that the symmetric form provides. The remaining asymmetry that matters for derived quantities (cover probability, push probability, expected value) is absorbed downstream by `Key` and `Normalizer`.

## The math

```
scale = spread / gennorm.ppf(win_prob, beta)
```

This follows directly from the win-probability constraint:

```
P(margin > 0) = gennorm.sf(0, beta, loc=spread, scale=scale) = win_prob
```

Solving for `scale` gives the closed form above.

### Degenerate case

When `spread ~ 0` and `win_prob ~ 0.5` (pick'em), the formula produces `0 / 0`. A fallback `scale = 13.2` is used instead — the empirical standard deviation of NFL margins across the modern era.

### Spread clamping

On construction, `spread` is rounded to the nearest 0.5 grid point. NFL spreads are posted in half-point increments, and the downstream `Normalizer`'s region logic assumes grid-aligned spreads.

## Pipeline position

Base sits at the bottom of the Distribution pipeline. It provides the continuous density that the `Normalizer` discretizes and the CDF/SF values that the `Normalizer` uses to compute region targets.

```
(spread, wp, beta) -> [BaseDistribution] -> continuous gennorm(spread, scale, beta)
                                                      |
                                                [Normalizer] -> discrete PMF
```

## Modules

### `BaseDistribution`

Stateful dataclass. Init plus continuous evaluators — no discrete operations.

- `__init__(spread, win_prob, beta=<shipped default>)` — clamp `spread` to the 0.5 grid, derive `scale`, and compute region masses. The default `beta` is loaded at import time from `margin_hyperparams.json`.
- `pdf(x)` — continuous generalized normal density.
- `cdf(x)` — continuous CDF.
- `sf(x)` — survival function `P(X > x)`.

Derived fields (set in `__post_init__`):

- `scale` — scale parameter of the generalized normal.
- `loss_mass` — `P(margin < 0)` from the continuous CDF.
- `win_mass` — `P(margin > 0)` from the continuous SF (equals `win_prob` outside the degenerate case).

Class constants:

- `_FALLBACK_SCALE = 13.2` — empirical scale used in the degenerate `(spread ~ 0, win_prob ~ 0.5)` case.
- `_DEGEN_EPS = 1e-6` — threshold for detecting that case.
