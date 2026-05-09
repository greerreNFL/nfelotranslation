# Market Moneyline Calibration

## Hypothesis

Closing moneyline implied win probabilities are not perfectly calibrated to observed outcomes across the probability range. Specifically, the favorite-perspective implied probability tends to understate strong favorites and overstate mild favorites, producing a systematic error profile that a monotone recalibration can correct.

## Method

1. Load completed regular-season and post-season games from `nfelodcm` with a valid closing home/away moneyline and a non-missing result. Ties are excluded from rate calculations.
2. Convert American odds to raw implied probabilities, then normalize the home/away pair to sum to one to remove the book's hold.
3. Fold to favorite perspective: relabel each game so the favorite's implied probability is in `[0.5, 1.0]` and the target is "favorite won" in `{0, 1}`.
4. Bin favorite implied probability using edges `[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]` and compute the observed favorite win rate per bin with a 95% Wilson confidence interval.
5. Summarize bias as the weighted mean absolute error across bins, weighting by bin count.

Outputs (probabilities as floats in `[0, 1]`, four decimal places):

- `output.csv` — per-bin counts, mean implied probability, observed rate, Wilson CI bounds, and signed error.
- `chart.png` — calibration curve with Wilson CIs (left) and signed bin-level error (right).

## Findings

Sample: 5,281 games across seasons 2006 to 2025 (ties excluded). Per-bin results:

| Bin | n | Mean implied | Observed rate | Error (observed - implied) |
|-----|---:|---:|---:|---:|
| 0.50 - 0.55 | 665 | 0.5261 | 0.5218 | -0.0043 |
| 0.55 - 0.60 | 918 | 0.5748 | 0.5599 | -0.0149 |
| 0.60 - 0.65 | 965 | 0.6252 | 0.5959 | -0.0294 |
| 0.65 - 0.70 | 789 | 0.6756 | 0.6755 | -0.0001 |
| 0.70 - 0.75 | 752 | 0.7256 | 0.7354 | +0.0098 |
| 0.75 - 0.80 | 590 | 0.7748 | 0.7847 | +0.0099 |
| 0.80 - 0.85 | 334 | 0.8223 | 0.8174 | -0.0050 |
| 0.85 - 0.90 | 219 | 0.8719 | 0.8995 | +0.0276 |
| 0.90 - 1.00 | 49 | 0.9155 | 0.9592 | +0.0437 |

Weighted MAE across bins: 0.0129.

- Mid bins from 0.55 to 0.65 show negative error, with the largest gap at 0.60 to 0.65 (-0.0294).
- Upper bins at 0.85 and above show positive error, with the largest gap at 0.90 to 1.00 (+0.0437) on a small sample (n=49).
- The error changes sign near 0.65 to 0.75, where the implied probability tracks the observed rate closely.

The direction of the error (understating the upper tail, overstating the mid-range) is consistent with a logit-linear slope greater than one applied to the raw implied probability.

## Conclusion

Closing moneyline implied probabilities are miscalibrated relative to observed outcomes, with a monotone shape that is well described by a two-parameter logit-linear transform. This is the empirical basis for including a Platt-scaling `Recalibrator` in the shipping package. Stationarity of this shape over time is examined in `2. Calibration Stationarity/`; candidate recalibration functional forms are compared in `3. Recalibration Method Comparison/`.
