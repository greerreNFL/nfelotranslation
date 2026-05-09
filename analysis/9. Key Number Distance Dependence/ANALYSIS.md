# 9. Key Number Distance Dependence

## Hypothesis

Per-integer excess at margin `k` varies linearly with the distance `|fav_spread - signed_margin|`. The sign of that slope separates two distinct regimes: positive slope means excess grows with distance (the integer pulls games toward it from far away), negative slope means excess fades with distance (the integer is "key" only when the spread is near it). The integer-by-integer slope landscape determines whether the model needs an explicit per-integer distance term or whether the form `(ratio - 1) * baseline_pmf[k]` handles distance dependence implicitly.

## Method

Source data: merged games and market frame produced by `analysis/_shared/data.load_data` for seasons 2006 to 2025, restricted to games with valid `fav_margin`, `fav_spread`, and `fav_wp_cal` (`n = 5,011`).

Each integer `k` in `1..40` is treated as one concept. Both `+k` and `-k` margins count as occurrences of "number `k`", with per-occurrence distance `|fav_spread - signed_margin|`.

Spread buckets are formed on the `fav_spread` half-point grid, retaining buckets with at least 30 games (25 buckets in this sample). For each bucket, the per-game variable-scale Gaussian baseline `N(loc=fav_spread, scale=fav_spread / norm.ppf(fav_wp_cal))` is computed and averaged into a single bucket-level baseline PMF. Any integer's bucket-level baseline rate is then an index lookup. Bucket-level excess at signed margin `m` is `(observed_rate - baseline_rate)`.

For each integer `k`, points within `|distance| <= 35` are pooled across both `+k` and `-k` and across all qualifying buckets, and a weighted least-squares regression of `excess_rate ~ distance` is fit with bucket sample size as the weight. The fit reports an intercept (excess rate at `distance = 0`), a slope (per-point change in excess rate), and a weighted `R^2`.

Per-integer slopes are then classified by sign and magnitude: slope `> +0.0002` per point is "persistent," slope `< -0.0002` per point is "proximity-dependent," and the band in between is "constant." Combined with the sign of the weighted mean excess, this produces a two-axis classification: persistent excess, proximity-dependent excess, persistent deficit, proximity-dependent deficit, or noise.

## Findings

### Per-integer fits (k = 1..21)

| k | Weighted mean | Intercept | Slope per point | Weighted R² | Classification |
|---:|---:|---:|---:|---:|:---|
| 1  | -0.0049 | +0.0023 | -0.00135 | 0.151 | persistent |
| 2  | -0.0053 | -0.0005 | -0.00089 | 0.046 | persistent |
| 3  | +0.0471 | +0.0620 | -0.00270 | 0.336 | proximity-dependent |
| 4  | -0.0018 | +0.0003 | -0.00036 | 0.022 | constant |
| 5  | -0.0057 | -0.0053 | -0.00007 | 0.001 | constant |
| 6  | +0.0081 | +0.0092 | -0.00015 | 0.003 | constant |
| 7  | +0.0205 | +0.0285 | -0.00105 | 0.131 | proximity-dependent |
| 8  | -0.0038 | -0.0042 | +0.00005 | 0.001 | constant |
| 9  | -0.0147 | -0.0259 | +0.00121 | 0.496 | proximity-dependent |
| 10 | +0.0057 | +0.0096 | -0.00038 | 0.028 | proximity-dependent |
| 11 | -0.0086 | -0.0179 | +0.00084 | 0.255 | proximity-dependent |
| 12 | -0.0105 | -0.0219 | +0.00095 | 0.382 | proximity-dependent |
| 13 | -0.0056 | -0.0148 | +0.00070 | 0.221 | proximity-dependent |
| 14 | +0.0083 | +0.0170 | -0.00062 | 0.114 | proximity-dependent |
| 15 | -0.0075 | -0.0240 | +0.00110 | 0.414 | proximity-dependent |
| 16 | -0.0030 | -0.0105 | +0.00046 | 0.168 | proximity-dependent |
| 17 | +0.0031 | +0.0012 | +0.00011 | 0.006 | constant |
| 18 | +0.0001 | -0.0031 | +0.00018 | 0.014 | constant |
| 19 | -0.0058 | -0.0199 | +0.00074 | 0.412 | proximity-dependent |
| 20 | +0.0007 | -0.0004 | +0.00006 | 0.002 | constant |
| 21 | +0.0038 | +0.0133 | -0.00045 | 0.126 | proximity-dependent |

Full per-integer fit table (all `k` in `1..40`) is in `output.csv`; per-bucket per-integer points are in `output_raw_points.csv`.

1. The "persistent excess" classification is empty. No integer with positive weighted mean excess has a slope above `+0.0002` per point. Every integer with positive mean excess either fades with distance or is flat.
2. The strongest positive excess (`k = 3`) and the strongest dead zones (`k = 9`, `12`, `15`, `19`) have the largest absolute slopes (`|slope| >= 0.0007 per point`) and the highest `R²` (`0.22`–`0.50`). Distance explains a meaningful share of the per-bucket variance at these integers.
3. At canonical key numbers `k = 7` and `k = 14`, the slope sign is the same (negative), but the `R²` is lower (`0.11`–`0.13`) and the slope magnitude is smaller. The proximity effect is real but a smaller share of the bucket-level variance.
4. The two smallest positive integers (`k = 1`, `k = 2`) classify as persistent deficits: their mean excess is small and negative, and the slope is also negative, so the deficit grows slightly with distance. Both are below the `R²` threshold where distance is a strong explanatory variable.
5. `k = 6` shows a clearly positive weighted mean (`+0.008`) with a near-zero slope and `R² = 0.003`. The integer carries a structural positive excess that does not vary materially with distance from the spread within the sample.
6. Above `k = 20` there are several integers (`k = 24`, `28`, `31`, `34`, `35`) with non-trivial slopes and `R²`, but their weighted mean excess is small (`|mean| < 0.005`). The slope picks up bucket-to-bucket variation that does not aggregate into a strong integer-level signal. Full results in `output.csv`.

## Conclusion

The "persistent excess" regime is empty across all 40 tracked integers. Every integer with material positive excess fades with distance from the spread; every integer with material negative excess (with the small `k = 1`, `k = 2` exceptions) softens with distance from the spread. This is the empirical basis for not using a separate distance-decay parameter in the Key module.

The shipping `(ratio - 1) * baseline_pmf[k]` form provides distance-dependent moderation implicitly: `baseline_pmf[k]` is small at integers far from the spread, so the absolute size of the per-game correction is naturally bounded at distant integers without an additional learned parameter. The form is consistent with the slope sign at every integer that shows a real signal.
