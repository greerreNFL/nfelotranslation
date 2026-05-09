# 8. Key Number Excess Magnitude

## Hypothesis

Integer-level excess in NFL margins is large enough at certain integers to be worth modeling, varies in magnitude across the integer range, and shifts measurably across eras at some integers. The combination motivates the Key module's design: per-integer trackers for all 40 integers (track-all rather than prescribe-keys), credibility weighting on each tracker (low-frequency integers self-regulate), and exponential decay across seasons (the model can follow real drift).

## Method

Source data: merged games and market frame produced by `analysis/_shared/data.load_data` for seasons 2006 to 2025, restricted to games with valid `fav_margin`, `fav_spread`, and `fav_wp_cal`. Favorite perspective throughout (`n = 5,011`).

Two baselines are evaluated, paralleling the shipped pipeline's behavior:

1. **Variable-scale Gaussian** — `N(loc=spread, scale=spread / norm.ppf(wp))`. Used by `KeyModelFitter._compute_baselines` to derive expected hits during training.
2. **Generalized normal at the shipped shape** — `gennorm(beta=1.35, loc=spread, scale=spread / gennorm.ppf(wp, 1.35))`. Constructed by `MarginDistributionModel.predict` at inference time and used as the surface that the trained ratios are applied against.

Both baselines satisfy `P(margin > 0) = wp` by construction; they differ only in how they distribute mass across integer margins.

For each integer `k` in `1..40` (matching the 40 `NumberOutcome` trackers in `KeyModel`), the analysis computes:

- Aggregate observed combined hit count at `+k` and `-k`.
- Aggregate expected combined hit count under each baseline.
- Combined excess as a 0–1 fraction: `(observed - expected) / n_games`.
- Raw observed-to-expected ratio.
- Per-season excess rate at the same integer; across-season mean, standard deviation, and absolute coefficient of variation (`|std / mean|`).
- Era-conditioned excess rate: pre-2015 (`n = 2,156`) vs 2015+ (`n = 2,855`). The 2015 boundary is the PAT rule change (extra-point distance moved from 2-yard to 15-yard line), the largest scoring-rule discontinuity in the sample.

Combined `+k`/`-k` framing matches how `KeyModelFitter` aggregates hits and how `NumberOutcome` stores its ratio.

## Findings

### Aggregate magnitude (n = 5,011 games, gennorm baseline)

| k | Observed | Expected | Excess rate | Ratio |
|---:|---:|---:|---:|---:|
| 1  | 216 | 245.8 | -0.0059 | 0.879 |
| 2  | 213 | 245.5 | -0.0065 | 0.868 |
| 3  | 731 | 244.7 | +0.0971 | 2.988 |
| 4  | 239 | 242.2 | -0.0006 | 0.987 |
| 5  | 192 | 238.1 | -0.0092 | 0.806 |
| 6  | 322 | 232.7 | +0.0178 | 1.384 |
| 7  | 435 | 225.9 | +0.0417 | 1.925 |
| 8  | 188 | 217.6 | -0.0059 | 0.864 |
| 9  | 76  | 208.3 | -0.0264 | 0.365 |
| 10 | 266 | 198.4 | +0.0135 | 1.341 |
| 11 | 110 | 187.8 | -0.0155 | 0.586 |
| 12 | 86  | 177.0 | -0.0182 | 0.486 |
| 13 | 121 | 166.1 | -0.0090 | 0.728 |
| 14 | 249 | 155.4 | +0.0187 | 1.603 |
| 15 | 80  | 144.7 | -0.0129 | 0.553 |
| 16 | 115 | 134.3 | -0.0038 | 0.856 |
| 17 | 166 | 124.3 | +0.0083 | 1.335 |
| 18 | 121 | 114.8 | +0.0012 | 1.054 |
| 19 | 54  | 105.8 | -0.0103 | 0.510 |
| 20 | 110 | 97.3  | +0.0025 | 1.130 |
| 21 | 134 | 89.4  | +0.0089 | 1.499 |

Full per-integer data (all `k` in `1..40`) under both baselines is in `output.csv`.

1. `k = 3` dominates: an excess rate of `+0.097` and an observed-to-expected ratio of `2.99`.
2. `k = 7` is the second-largest positive integer (`+0.042`, ratio `1.93`); `k = 6`, `k = 14`, `k = 10`, and `k = 17` form a second tier of positive integers in the `+0.008` to `+0.019` band.
3. The strongest dead zones are `k = 9` (`-0.026`, ratio `0.37`) and `k = 12` (`-0.018`, ratio `0.49`); `k = 11`, `k = 15`, and `k = 19` form a second tier in the `-0.010` to `-0.016` band.
4. Above `k = 20`, aggregate excess rates fall to `|excess| < 0.01` for almost all integers, with several reversals in sign across the range.
5. Switching the baseline from Gaussian to gennorm at `beta = 1.35` lowers expected hits at every integer in `1..40` by a few percent, which raises the implied ratio uniformly without shifting the locations of keys or dead zones.

### Per-season variance (gennorm baseline, 20 seasons)

| k | Mean excess | Std | CV (\|std/mean\|) |
|---:|---:|---:|---:|
| 1  | -0.0062 | 0.0130 | 2.11 |
| 2  | -0.0068 | 0.0119 | 1.76 |
| 3  | +0.0964 | 0.0253 | 0.26 |
| 4  | -0.0001 | 0.0145 | 100.23 |
| 5  | -0.0096 | 0.0160 | 1.67 |
| 6  | +0.0176 | 0.0188 | 1.07 |
| 7  | +0.0422 | 0.0185 | 0.44 |
| 8  | -0.0065 | 0.0107 | 1.64 |
| 9  | -0.0262 | 0.0107 | 0.41 |
| 10 | +0.0139 | 0.0124 | 0.89 |
| 11 | -0.0155 | 0.0108 | 0.70 |
| 12 | -0.0183 | 0.0072 | 0.39 |
| 13 | -0.0088 | 0.0102 | 1.15 |
| 14 | +0.0186 | 0.0159 | 0.86 |
| 15 | -0.0131 | 0.0097 | 0.74 |
| 16 | -0.0040 | 0.0100 | 2.48 |
| 17 | +0.0081 | 0.0124 | 1.52 |
| 18 | +0.0011 | 0.0104 | 9.03 |
| 19 | -0.0103 | 0.0066 | 0.64 |
| 20 | +0.0030 | 0.0120 | 4.01 |
| 21 | +0.0091 | 0.0136 | 1.50 |

Full per-`k` stability table (`k` in `1..40`) is in `output_stability.csv`.

1. Four integers — `3`, `7`, `9`, `12` — have CVs in the `0.26`–`0.44` band, meaning across-season variance is well below the magnitude of the mean signal.
2. `11`, `15`, and `19` form a second band of stable signals (`CV` between `0.6` and `0.8`), each with a negative mean excess.
3. `10` and `14` have CVs near `0.9`, where across-season variance is comparable to the mean signal at those integers.
4. `k = 6` has a clearly non-zero mean (`+0.018`) but `CV = 1.07`, meaning a small number of seasons contribute most of that mean (max season excess `+0.063`, min `-0.009`). The era table below resolves this further.
5. Integers with mean excess near zero (`k = 4`, `18`, `20`, and most integers above `k = 20`) report very high CVs as a numerical consequence of dividing by a near-zero mean — sampling noise dominates structural signal.

### Era comparison (gennorm baseline, pre-2015 vs 2015+)

| k | Pre-2015 | 2015+ | Delta |
|---:|---:|---:|---:|
| 1  | -0.0116 | -0.0017 | +0.0099 |
| 2  | -0.0133 | -0.0013 | +0.0120 |
| 3  | +0.0940 | +0.0993 | +0.0053 |
| 4  | +0.0041 | -0.0042 | -0.0084 |
| 5  | -0.0200 | -0.0010 | +0.0190 |
| 6  | +0.0076 | +0.0256 | +0.0180 |
| 7  | +0.0424 | +0.0412 | -0.0012 |
| 8  | -0.0130 | -0.0006 | +0.0124 |
| 9  | -0.0287 | -0.0247 | +0.0041 |
| 10 | +0.0206 | +0.0081 | -0.0125 |
| 11 | -0.0148 | -0.0161 | -0.0014 |
| 12 | -0.0208 | -0.0161 | +0.0047 |
| 13 | -0.0051 | -0.0120 | -0.0068 |
| 14 | +0.0153 | +0.0213 | +0.0060 |
| 15 | -0.0145 | -0.0117 | +0.0028 |
| 16 | -0.0067 | -0.0017 | +0.0051 |
| 17 | +0.0070 | +0.0094 | +0.0024 |
| 18 | +0.0025 | +0.0003 | -0.0021 |
| 19 | -0.0114 | -0.0095 | +0.0019 |
| 20 | +0.0033 | +0.0019 | -0.0014 |
| 21 | +0.0166 | +0.0031 | -0.0135 |

Full per-`k` era table (`k` in `1..40`) is in `output_era.csv`.

1. The two largest signal integers (`k = 3`, `k = 7`) shift by `|delta| <= 0.006` between eras.
2. The two strongest dead zones (`k = 9`, `k = 12`) each soften by roughly `+0.004` post-2015.
3. Several integers with small or negative pre-2015 excess shift by more than a per-season standard deviation: `k = 1`, `2`, and `8` move by `+0.010` to `+0.012` from a deficit toward zero; `k = 5` moves by `+0.019` from a clear deficit toward zero; `k = 6` moves by `+0.018` from `+0.008` to `+0.026`.
4. `k = 10` and `k = 21` weaken by `0.012`–`0.014` post-2015 — drift larger than the per-season standard deviation at those integers; `k = 4` drifts by a similar magnitude (`-0.008`) into a small deficit.
5. The PAT rule change does not invert the sign of the strongest signals (`k = 3`, `7`, `9`, `12`); it shifts magnitudes within the existing structure and brings several smaller-magnitude integers closer to zero.

## Conclusion

The data shows three things relevant to the Key module's design:

1. Integer-level excess is large at a handful of integers (`+/-3`, `+/-7`, `+/-9`, `+/-12`) and meaningful at a wider second tier (`+/-6`, `+/-10`, `+/-11`, `+/-14`, `+/-15`, `+/-19`, `+/-21`). The model tracks all 40 integers uniformly because deciding which are "key" from a single sample window risks back-leaking the data; the data confirms that signal is concentrated but not exclusively at the canonical key numbers.

2. Per-season CV is low at the strongest signals (`0.26`–`0.44`) and high at the weaker ones (`>= 0.86` at `k = 10`, `14`; near `1.0` at `k = 6`; `> 1.5` for most integers above `k = 20`). The credibility-weighted ratio form pulls high-CV integers toward `1.0` until they accumulate enough effective exposure, which matches the variance profile in the data.

3. Several integers shift measurably across the 2015 boundary: `k = 5` and `k = 6` each move by roughly `+0.018`; `k = 1`, `2`, and `8` close deficits by `+0.010` to `+0.012`; `k = 10` and `k = 21` lose `0.012`–`0.014` of positive excess; the strongest signals (`k = 3`, `7`, `9`, `12`) do not invert. The exponential-decay form lets each tracker follow that drift without a hard reset.

Aggregate magnitudes, per-season excess rates, and era-conditioned rates are reproducible from `output.csv`, `output_stability.csv`, and `output_era.csv` respectively.
