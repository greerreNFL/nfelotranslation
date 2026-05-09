# 4. Spread Mapping Form

## Hypothesis

A linear-in-logit mapping `margin = slope * logit(wp) + intercept` recovers
the win-prob → margin relationship as accurately as a per-bin lookup table,
while remaining smooth, analytically invertible, and resistant to tail
overfit. Forcing the intercept to zero costs negligible out-of-sample MAE
relative to a free-intercept fit and preserves the empirical anchor
`WP=0.50 ↔ spread=0`.

## Method

* Source data: 5,249 NFL regular and post-season games with valid
  recalibrated favorite-side win probability, valid favorite margin, and a
  posted spread that agrees with the favorite assignment (`fav_spread > 0`).
* All work in favorite-perspective coordinates: `fav_wp_cal > 0.5`,
  `fav_margin > 0` means the favorite covered or won.
* Three mapping forms fit on `(fav_wp_cal, fav_margin)`:
  1. **Linear (free intercept)** — `margin = slope * z + intercept`,
     MAE loss, both parameters optimized.
  2. **Linear (intercept = 0)** — same form, `intercept` fixed at zero;
     this is the form shipped for the model mapper.
  3. **Lookup (1% bin median)** — per-bin median of `fav_margin` across
     `fav_wp_cal` bins of width 0.01; predictions interpolate linearly
     between adjacent bin centers.
* Reported metrics: in-sample MAE and bisection rate, plus
  leave-one-season-out (LOSO) MAE and bisection rate, weighted by
  held-out season size.

## Findings

Fitted parameters and metrics (in-sample, then LOSO across 27 seasons):

| Form                  | slope  | intercept | in-sample MAE | in-sample bisection | LOSO MAE | LOSO bisection |
|-----------------------|--------|-----------|---------------|---------------------|----------|----------------|
| Linear (free)         | 6.0754 | 0.4754    | 10.2217       | 0.4999              | 10.2285  | 0.4999         |
| Linear (intercept=0)  | 6.5500 | 0.0000    | 10.2252       | 0.5066              | 10.2272  | 0.5071         |
| Lookup (1% bin)       | —      | —         | 10.1600       | 0.4969              | 10.2624  | 0.4965         |

Per-bin sample sizes for the lookup form range from 2 to 193 across the
47 occupied bins, with the sparser bins concentrated above WP ≈ 0.90.

## Conclusion

* The linear forms generalize as well as the lookup table at the center of
  the WP range and clearly better in the tails: the lookup wins on
  in-sample MAE by 0.0617 points but loses on LOSO MAE by 0.0339–0.0352
  points relative to the linear forms. The gap is consistent with tail
  overfit — bins above WP ≈ 0.90 contain too few games for stable medians.
* The free-intercept linear fit absorbs 0.4754 points of upward bias at
  `z = 0`. This bias is the residual of two effects: (i) Recalibrator
  miscalibration that survives the Platt fit and (ii) the right-skew of
  the margin distribution under a median-targeting MAE loss.
* Forcing the intercept to zero raises the slope from 6.0754 to 6.5500
  and shifts the bisection rate from 0.4999 to 0.5071 — favorites cover
  the predicted spread 0.7 pp more often than the median condition. The
  LOSO MAE penalty is 0.0013 points (10.2285 → 10.2272, in fact slightly
  lower under LOSO because the bias term is itself noisy across seasons).
* Given a negligible MAE cost, the constrained form is preferred for the
  shipped model mapper: it preserves the empirical anchor
  `WP = 0.50 ↔ spread = 0` and isolates the fitted slope from any bias
  introduced upstream by the Recalibrator.
