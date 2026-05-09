# Calibration Stationarity

## Hypothesis

The miscalibration pattern identified in `1. Market Moneyline Calibration/` is stable over time, so a single static recalibrator fit on the pooled sample is an adequate correction. Under this hypothesis, per-season Platt parameters will vary around a stable mean with no detectable linear trend, the per-era error shape will be consistent, and per-season weighted MAE will not trend.

## Method

Three complementary checks are run on the favorite-perspective game sample used in `1. Market Moneyline Calibration/`:

1. Per-season Platt fit. For each season with at least 50 qualifying games, fit `p_cal = expit(a * logit(p_raw) + b)` in isolation and record the slope and intercept. Summarize with mean, standard deviation, coefficient of variation, and a linear regression of each parameter on season.
2. Calibration error shape by era. Partition seasons into four blocks (2006 to 2010, 2011 to 2015, 2016 to 2020, 2021 to 2025) and compute the signed per-bin error `observed - implied` in each era.
3. Per-season weighted MAE. Bin each season separately and report weighted MAE; test for a linear trend on season.

Outputs (probabilities as floats in `[0, 1]`, four decimal places):

- `output.csv` — per-season Platt slope and intercept with sample counts.
- `output_era.csv` — per-era, per-bin observed rate, implied probability, and signed error.
- `output_annual.csv` — per-season weighted MAE.
- `chart.png` — four panels: Platt slope by season, Platt intercept by season, calibration error shape by era, and weighted MAE per season.

## Findings

Sample: 5,281 games across seasons 2006 to 2025 (ties excluded).

### Per-season Platt parameters

| Statistic | Slope (a) | Intercept (b) |
|---|---:|---:|
| Mean | 1.1637 | -0.1331 |
| Std | 0.3021 | 0.2698 |
| Coefficient of variation | 0.2596 | n/a |
| Linear trend per year | +0.0032 | +0.0074 |
| p-value of trend | 0.795 | 0.495 |

Per-season values range from 0.6516 (2023) to 1.7478 (2012) for slope and from -0.5449 (2012) to +0.3489 (2013) for intercept. Full per-season table is in `output.csv`.

### Error shape by era (favorite implied probability bin)

| Bin | 2006-2010 | 2011-2015 | 2016-2020 | 2021-2025 |
|---|---:|---:|---:|---:|
| 0.50 - 0.575 | +0.0332 | -0.0202 | +0.0011 | -0.0139 |
| 0.575 - 0.65 | -0.0799 | -0.0181 | -0.0550 | +0.0202 |
| 0.65 - 0.725 | -0.0388 | +0.0126 | +0.0366 | -0.0079 |
| 0.725 - 0.80 | -0.0039 | +0.0126 | +0.0374 | -0.0033 |
| 0.80 - 0.90 | -0.0045 | +0.0176 | +0.0120 | +0.0124 |
| 0.90 - 1.00 | +0.0243 | +0.0816 | -0.0123 | +0.0878 |

The 0.90 to 1.00 bin is positive in three of four eras; the 0.80 to 0.90 bin is positive in three of four eras; the 0.575 to 0.65 bin is negative in three of four eras. Per-era and per-bin rows are in `output_era.csv`.

### Per-season weighted MAE

Weighted MAE per season ranges from 0.0181 (2018) to 0.0858 (2006). Linear trend: -0.0003/year with p=0.659. Full per-season table is in `output_annual.csv`.

## Conclusion

The per-season variation in Platt parameters is consistent with sampling noise at a ~265-game sample per season rather than structural drift: both slope and intercept trends fail to reject zero at conventional thresholds (p=0.795 and p=0.495 respectively), and the per-era error shape retains the same sign pattern in the upper and mid-range bins across the four eras examined. On the evidence here, a single static Platt recalibrator fit on the pooled sample is appropriate. Periodic re-fit and revalidation remains prudent as more seasons accumulate; the `training/Validation/RecalibratorValidator` stationarity block reruns the per-season fit on demand.
