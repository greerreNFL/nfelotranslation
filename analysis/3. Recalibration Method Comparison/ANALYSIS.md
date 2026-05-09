# Recalibration Method Comparison

## Hypothesis

Given the miscalibration identified in `1. Market Moneyline Calibration/` and its stability over time shown in `2. Calibration Stationarity/`, a simple logit-linear (two-parameter Platt) transform achieves calibration quality competitive with more flexible alternatives under leave-one-season-out evaluation. Under this hypothesis, adding parameters or switching to a non-logit-linear family will not improve pooled out-of-sample Expected Calibration Error.

## Method

Seven methods are evaluated as functions that map raw implied probability `p` (equivalently its logit `z = logit(p)`) to a calibrated probability:

- `Raw ML` — identity, `p_cal = expit(z)`.
- `Platt` — `p_cal = expit(a * z + b)`, two parameters.
- `Beta` — `p_cal = expit(a * log(p) + b * log(1 - p) + c)`, three parameters.
- `Poly Logit 2` — `p_cal = expit(a * z + b * z^2 + c)`, three parameters.
- `Platt+ReLU` — `p_cal = expit(a * z + b + c * max(z - k, 0))`, four parameters with extra slope above `k`.
- `Platt+Soft` — `p_cal = expit(a * z + b + c * softplus(z - k))`, four parameters with smooth tail boost.
- `Piecewise` — two linear segments in logit space with continuity at a learned breakpoint, four parameters.

Each method is fit by minimizing negative log-likelihood on the favorite-perspective training sample. Evaluation uses leave-one-season-out cross-validation over seasons 2006 to 2025: for each held-out season the method is refit on the remaining seasons, predictions are made for the held-out season, and the per-game predictions are pooled across folds. Metrics on the pooled out-of-sample predictions:

- Log loss — binary cross-entropy, reported for reference; low discrimination between methods at this sample size.
- Expected Calibration Error (full) — population-weighted bin error using the same edges as `1. Market Moneyline Calibration/`.
- ECE core — restricted to bins below 0.70 implied probability.
- ECE tail — restricted to bins at or above 0.70 implied probability.

Outputs (probabilities as floats in `[0, 1]`, four decimal places):

- `output.csv` — per-method log loss, Brier score, ECE full, ECE core, ECE tail, and parameter count.
- `output_bins.csv` — per-method, per-bin observed rate, mean prediction, and signed error.
- `chart.png` — four panels: ECE by method (core, full, tail), calibration curves over the full range, per-bin residuals, and a tail zoom (0.70 to 1.00).

## Findings

Sample: 5,281 games across seasons 2006 to 2025 (ties excluded). Pooled LOSO results sorted by ECE full:

| Rank | Method | Params | Log loss | ECE full | ECE core | ECE tail |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Platt | 2 | 0.6095 | 0.0107 | 0.0114 | 0.0095 |
| 2 | Platt+Soft | 4 | 0.6096 | 0.0112 | 0.0074 | 0.0144 |
| 3 | Poly Logit 2 | 3 | 0.6095 | 0.0114 | 0.0080 | 0.0137 |
| 4 | Beta | 3 | 0.6095 | 0.0116 | 0.0069 | 0.0167 |
| 5 | Piecewise | 4 | 0.6096 | 0.0118 | 0.0107 | 0.0117 |
| 6 | Raw ML | 0 | 0.6095 | 0.0122 | 0.0135 | 0.0102 |
| 7 | Platt+ReLU | 4 | 0.6100 | 0.0125 | 0.0114 | 0.0120 |

- Log loss spans a 0.0005 range across all seven methods and does not separate them.
- Platt has the lowest ECE full (0.0107) and the lowest ECE tail (0.0095).
- Methods with three or four parameters have lower ECE core than Platt (Beta 0.0069, Platt+Soft 0.0074, Poly Logit 2 0.0080) but higher ECE tail (Beta 0.0167, Poly Logit 2 0.0137, Platt+Soft 0.0144).

### Per-bin residual (observed - predicted), pooled out-of-sample

| Bin | Raw ML | Platt | Beta | Poly Logit 2 | Platt+ReLU | Platt+Soft | Piecewise |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 - 0.55 | -0.0043 | +0.0184 | +0.0082 | +0.0076 | +0.0113 | +0.0079 | +0.0067 |
| 0.55 - 0.60 | -0.0149 | +0.0015 | -0.0001 | -0.0005 | -0.0002 | -0.0006 | -0.0005 |
| 0.60 - 0.65 | -0.0294 | -0.0201 | -0.0161 | -0.0161 | -0.0183 | -0.0165 | -0.0144 |
| 0.65 - 0.70 | -0.0001 | +0.0026 | +0.0087 | +0.0093 | +0.0063 | +0.0090 | +0.0109 |
| 0.70 - 0.75 | +0.0098 | +0.0065 | +0.0115 | +0.0123 | +0.0113 | +0.0125 | +0.0105 |
| 0.75 - 0.80 | +0.0083 | -0.0002 | +0.0006 | +0.0013 | +0.0047 | +0.0020 | -0.0018 |
| 0.80 - 0.85 | -0.0020 | -0.0143 | -0.0192 | -0.0193 | -0.0145 | -0.0185 | -0.0201 |
| 0.85 - 0.90 | +0.0276 | +0.0132 | +0.0026 | +0.0005 | -0.0019 | +0.0001 | +0.0039 |
| 0.90 - 1.00 | +0.0437 | +0.0301 | +0.0158 | +0.0132 | +0.0105 | +0.0111 | +0.0209 |

The 0.60 to 0.65 bin retains a negative residual in every method (range -0.0144 to -0.0201) that does not close further with additional parameters. In the upper tail, the four-parameter methods reduce the 0.85 to 1.00 residuals relative to Platt but concurrently enlarge the 0.80 to 0.85 negative residual; the population weight in the 0.80 to 0.85 bin (n=334 out of 602 above 0.80) dominates the ECE tail calculation.

## Conclusion

Two-parameter Platt scaling matches or beats the more flexible alternatives on pooled out-of-sample ECE at this sample size, with no material log-loss penalty and the lowest parameter count among the fitted methods. The shipping `Recalibrator` therefore uses Platt scaling; the parameters stored in `src/nfelotranslation/Calibration/platt_params.json` are the output of refitting Platt on the full pooled sample once the form has been selected. The residual in the 0.60 to 0.65 bin is a feature of the raw input that is preserved through all tested recalibrations and warrants attention only if a future analysis identifies a covariate that explains it.
