# 7. Margin Distribution Form

## Hypothesis

A symmetric generalized normal centered at the spread is an appropriate parametric form for the empirical NFL margin distribution across spread levels. Specifically:

1. The empirical shape has heavier tails than a Gaussian, so a shape parameter `beta < 2` should fit better than the Gaussian special case (`beta = 2`).
2. The shape is sufficiently consistent across spread levels that a single fixed `beta` can serve all games rather than a spread-dependent shape.
3. The improvement of generalized normal over Gaussian is large enough to justify the extra shape parameter that the Base module carries.

## Method

Source data: merged games and market frame produced by `analysis/_shared/data.load_data` for seasons 2006 to 2025, restricted to games with valid `fav_margin` and `fav_spread`. All margins are folded to the favorite's perspective so that positive values represent the favorite winning by that many points.

Games are bucketed by absolute spread on favorite perspective:

| Bucket | Range (`|fav_spread|`) |
|---|---|
| Pickem to 3 | `[0, 3.5)` |
| 3.5 to 7 | `[3.5, 7.5)` |
| 7.5 to 10 | `[7.5, 10.5)` |
| 10.5 to 14 | `[10.5, 14.5)` |
| 14.5+ | `[14.5, infinity)` |

Within each bucket, two distributions are fit by maximum likelihood directly on the empirical margins:

- **gennorm**: `scipy.stats.gennorm.fit(margins)`, three free parameters `(loc, scale, beta)`.
- **Gaussian**: `scipy.stats.norm.fit(margins)`, two free parameters `(loc, scale)`.

For each fit, log-likelihood, log-likelihood per game, and one-sample Kolmogorov-Smirnov distance against the fitted CDF are reported. The same fit is run on the pooled sample (`ALL`) for a single-shape view.

The objective here is to characterize the empirical margin shape and contrast it with the Gaussian special case. It is deliberately not a selection of the shipped `beta`. The shipped value is chosen by `MarginDistributionFitter` under a pipeline-aware objective that includes the Key module.

## Findings

Per-bucket fits:

| Bucket | n | beta | loc | scale | KS gennorm | KS Gaussian | delta LL/game |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pickem to 3 | 2,197 | 1.533 | +1.43 | 15.90 | 0.0581 | 0.0686 | +0.0075 |
| 3.5 to 7 | 2,344 | 1.588 | +5.55 | 16.00 | 0.0574 | 0.0714 | +0.0047 |
| 7.5 to 10 | 803 | 1.795 | +8.91 | 17.65 | 0.0533 | 0.0542 | +0.0012 |
| 10.5 to 14 | 453 | 2.072 | +12.64 | 19.21 | 0.0722 | 0.0733 | +0.0001 |
| 14.5+ | 102 | 1.946 | +16.46 | 17.33 | 0.0716 | 0.0741 | +0.0001 |
| ALL | 5,899 | 1.603 | +5.07 | 16.93 | 0.0539 | 0.0673 | +0.0043 |

1. The pooled fit gives `beta = 1.60`. Tails are clearly heavier than Gaussian.
2. In the three buckets with `n > 800`, fitted `beta` lies in the range `[1.53, 1.80]`. The two highest buckets (`n = 453` and `n = 102`) return `beta` near 2, but those samples are small enough that the estimates are noisy.
3. KS distance under gennorm is lower than Gaussian in every bucket. The pooled gain is `0.054` vs `0.067`.
4. Log-likelihood gain per game is small in absolute terms but consistently positive, with the largest gains in the well-sampled close-spread buckets where heavy tails matter most.
5. Fitted `loc` tracks the bucket's mean spread closely (e.g., `loc = +5.55` for the 3.5–7 bucket; `loc = +12.64` for the 10.5–14 bucket), supporting the choice of `loc = spread` in Base.

## Conclusion

The empirical margin distribution is well-characterized by a generalized normal with `beta < 2` across the bulk of the data. The Gaussian special case is dominated by gennorm in every bucket on KS distance and log-likelihood, and the fitted `beta` in well-sampled buckets falls in a tight range that supports using a single shipped shape parameter rather than a spread-dependent one. The high-spread buckets where the fit drifts toward `beta = 2` are sample-limited and do not provide meaningful evidence against a single shape.

This establishes the parametric form for the Base module. The specific shipped value of `beta` is selected separately by `MarginDistributionFitter` under a pipeline-aware objective and is not the MLE value reported here.
