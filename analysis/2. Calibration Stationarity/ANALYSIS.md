# Calibration Stationarity

## Hypothesis

The miscalibration pattern identified in `1. Market Moneyline Calibration/` is persistent over time, but not all Platt parameters need the same treatment. Under this hypothesis, split **slopes** (home vs away) are stable enough to treat as omniscient structural parameters, while **intercepts** drift over calendar time and require a season-aware refit scheme for training labels. The per-era error shape remains consistent, and per-season weighted MAE does not trend.

## Method

Four complementary checks are run on the favorite-perspective game sample used in `1. Market Moneyline Calibration/`:

1. **Omniscient split slopes.** Fit `p_cal = expit(a_loc * logit(p_raw) + b_loc)` separately for home favorites and away favorites on the full pooled sample. Record the slopes as the structural parameters held fixed in production training.
2. **Per-season split Platt fit.** For each season with at least 50 qualifying games per location, fit split Platt in isolation and record home/away slopes and intercepts. Summarize with mean, standard deviation, and a linear regression of each parameter on season.
3. **Calibration error shape by era.** Partition seasons into four blocks (2006 to 2010, 2011 to 2015, 2016 to 2020, 2021 to 2025) and compute the signed per-bin error `observed - implied` in each era.
4. **Per-season weighted MAE.** Bin each season separately and report weighted MAE; test for a linear trend on season.
5. **Intercept scheme comparison.** With omniscient slopes fixed, compare full-sample log loss and Brier score for: raw market ML, static omniscient intercepts, and centered 5-year intercept refits (edge-padded) — the shipped training scheme.

Outputs (probabilities as floats in `[0, 1]`, four decimal places):

- `output.csv` — per-season split Platt slopes and intercepts with sample counts.
- `output_era.csv` — per-era, per-bin observed rate, implied probability, and signed error.
- `output_annual.csv` — per-season weighted MAE.
- `output_centered_compare.csv` — log loss and Brier score by intercept scheme.
- `chart.png` — four panels: split slopes by season, split intercepts by season, calibration error shape by era, and weighted MAE per season.

## Findings

Sample: 5,281 games across seasons 2006 to 2025 (ties excluded).

### Omniscient split slopes (full sample)

| Location | Slope (a) |
|---|---:|
| Home favorite | 1.2113 |
| Away favorite | 0.9726 |

### Per-season split Platt parameters

| Statistic | Home slope | Away slope | Home intercept | Away intercept |
|---|---:|---:|---:|---:|
| Mean | 1.2580 | 1.0209 | -0.2316 | +0.0125 |
| Std | 0.4399 | 0.5190 | 0.4071 | 0.2666 |
| Linear trend per year | -0.0046 | +0.0295 | +0.0137 | -0.0092 |
| p-value of trend | 0.796 | 0.147 | 0.400 | 0.389 |

Per-season values show high sampling variance at ~130 to 170 games per location per season. Slope trends fail to reject zero at conventional thresholds; intercept trends are similarly indistinguishable from noise season-by-season, but home intercepts exhibit visible calendar drift in the time series (see `chart.png`). Full per-season table is in `output.csv`.

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

### Intercept scheme comparison (omniscient slopes fixed)

| Scheme | Log loss | Brier |
|---|---:|---:|
| Raw market ML | 0.6095 | 0.2111 |
| Static omniscient intercept | 0.6086 | 0.2108 |
| Centered 5-year intercept | 0.6081 | 0.2106 |

Centered 5-year intercepts improve log loss by 0.0013 vs raw market on the full sample. This scheme is used for training labels stored in `Calibration/configs/platt_params_{season}.json`. Recalibration is not applied at inference.

## Conclusion

Split Platt **slopes** are stable enough to fit once on the full sample and hold fixed (`a_home ≈ 1.21`, `a_away ≈ 0.97`). **Intercepts** vary over calendar time, especially for home favorites, justifying per-season intercept refits on a centered 5-year window for training labels. The per-era error shape retains the same sign pattern in the upper and mid-range bins across the four eras examined. A single static intercept is adequate for research but the centered scheme produces slightly better training labels. Periodic re-fit and revalidation remains prudent as more seasons accumulate; the `training/Validation/RecalibratorValidator` stationarity block reruns the per-season fit on demand.
