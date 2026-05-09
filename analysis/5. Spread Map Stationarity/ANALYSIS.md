# 5. Spread Map Stationarity

## Hypothesis

The slope of the model mapper (`fav_wp_cal → fav_margin`, intercept fixed
at zero) and the slope and intercept of the market mapper
(`fav_ml_wp → fav_spread`, free intercept) are stationary across seasons.
A stationary parameter set supports a single static fit; a drifting
parameter set would require either a re-estimation cadence or
decay-weighted training.

## Method

- Source data: 5,249 games across seasons 2006–2025 with valid recalibrated
favorite-side win probability, valid raw favorite-side market WP,
valid favorite margin, and a posted spread that agrees with the
favorite assignment (`fav_spread > 0`).
- For each season independently:
  - Refit the model mapper on `(fav_wp_cal, fav_margin)` with intercept
  fixed at zero (matching the shipped form).
  - Refit the market mapper on `(fav_ml_wp, fav_spread)` with both
  parameters free (matching the shipped form).
- Each parameter series is regressed on season; the per-year slope and
the OLS p-value summarize the trend.

## Findings

Per-season parameters and trend statistics (20 seasons):


| Parameter        | Mean   | Std    | Trend / yr | p-value |
| ---------------- | ------ | ------ | ---------- | ------- |
| Model slope      | 6.5627 | 1.1230 | +0.0823    | 0.0560  |
| Market slope     | 6.7595 | 0.3934 | -0.0108    | 0.4947  |
| Market intercept | 0.2830 | 0.2767 | +0.0251    | 0.0146  |


Per-season values are written to `output.csv`; trend statistics are in
`output_trend.csv`.

## Conclusion

- The market slope is stationary: a standard deviation of 0.39 around a
mean of 6.76 is consistent with sampling noise, and the linear trend is
indistinguishable from zero (p = 0.4947).
- The model slope's per-season variation (std = 1.12) is dominated by
small-sample noise at n ≈ 200–290 per season. The fitted trend of
+0.0823 per year sits at p = 0.0560 — outside conventional rejection
but not strong enough to claim a real drift on this sample.
- The market intercept shows a small but statistically detectable
upward drift of +0.025 points per year (p = 0.0146), accumulating to
roughly half a point of offset at `WP = 0.5` across the 20-year window
— under one half-point grid step.

The shipping decision is to fit each mapper as a single static set of
parameters. The market intercept drift is the only parameter that
exceeds a stationarity test, and its magnitude is small enough that the
added complexity of season-indexed parameters is not justified at
present. This position is contingent on the observed drift remaining
small; it should be revisited as additional seasons accumulate or if
the per-season trend strengthens.

