# Changelog


## [0.2.2] - 2026-08-01

### Fixed

- **`Normalizer` spread bisection** — feasible integer and half-integer posted spreads now enforce `P(margin < spread) = P(margin > spread)` exactly while preserving win probability, tie probability, and integer push mass. At `±1`, the push bin adjusts because the between-zero-and-spread region is empty. At `0` and `±0.5`, win and tie probabilities retain precedence because the discrete support makes all constraints incompatible.
- **Equal-line expected value** — exact bisection prevents positive expected value at the model's own feasible posted line from being created by unequal cover and loss mass.

### Changed

- **KeyModel hyperparameters** — `forgetting_rate` `0.04` → `0.0325`, `threshold` `15` → `11.25`, and `initial_prior_size` `52` → `61`; margin-distribution `beta` remains `1.24`.
- **Retrained KeyModel configs** — regenerated seasonal snapshots through the 2026 config using data through 2025.

### Validation snapshot

| Component | Key metric | Value |
|-----------|------------|------:|
| Margin distribution | OOS SAE | 11.2% (487 / 4,363 games) |
| Margin distribution | Close bias | +0.5% |
| Margin distribution | Mid bias | +1.3% |
| Margin distribution | Tail bias | −5.3% |
| KeyModel | OOS RMSE excess | 1.1243 pp |
| KeyModel | OOS MAE excess | 0.7836 pp |
| KeyModel | Model / baseline RMSE | 0.5391 |
| Normalizer | Max feasible-line bisection error | 2.22e−16 |

## [0.2.1] - 2026-07-07

### Fixed

- **`BaseDistribution` spread clamp** — removed construction-time rounding to the 0.5 grid. `Translator` and `MarginDistributionValidator` already pass `Spread.continuous`; clamping forced `loc` to grid while scale still depended on the continuous spread, breaking `(spread, win_prob) → scale` consistency and making `cover_prob(line)` non-monotone in `win_prob`. Win prob could increase while cover prob decreased depending on how the clamp impacted scale.

### Changed

- **`BaseDistribution`** — spread input is kept continuous for PDF center (`loc`) and scale derivation; degenerate fallback unchanged (`OR` guard on spread ≈ 0 or wp ≈ 0.5).
- **`Normalizer`** — region bisection uses `round(base.spread × 2) / 2` (same rule as `Spread.posted`); continuous PDF is unchanged at evaluation time.
- **`MarginDistribution.spread`** — still the grid bisection spread returned from `predict()` (not the internal continuous center).

### Validation snapshot (post-fix, same pipeline `628c245d` configs)

| Component | Key metric | Value |
|-----------|------------|------:|
| Margin distribution | OOS SAE | 11.2% (488 / 4,363 games) |
| Margin distribution | Tail bias | −5.5% |
| Margin distribution | Close bias | +0.2% |

## [0.2.0] - 2026-06-11

### Breaking

- **`Translator` input types** — only `'win_prob'` and `'spread'` are accepted. Removed `'market_win_prob'` and `'market_spread'`.
- **`Translator` properties** — removed `win_prob_market`, `away_win_prob_market`, `home_win_prob_market`, and `market_spread`. Inference operates in model win-probability space only.
- **`SpreadMapper`** — removed dual MODEL/MARKET instances and the `MapType` enum. A single mapper is shipped, fitted on recalibrated market win probabilities vs actual game margins (`intercept = 0`).
- **`Recalibrator` in inference** — no longer composed into `Translator`. Recalibration is a training-label primitive only.

### Changed

- **`Recalibrator`** — split Platt scaling by favorite location (`a_home`, `a_away` omniscient slopes; per-season intercepts in `Calibration/configs/platt_params_{season}.json` using a centered 5-year window).
- **`SpreadMapper` validation** — removed the `market_roundtrip_r2 > 0.98` gate (market mapper removed). Gated checks are now `slope_positive` and `bisection_rate_centered` only. `binned_r2` is tracked against empirical medians of actual margins (not posted spreads).
- **Margin distribution hyperparameters** — `beta` `1.35` → `1.24`; KeyModel hyperparameters `forgetting_rate` `0.087` → `0.04`, `threshold` `25` → `15` (`initial_prior_size` remains `52`).
- **Retrained configs** — full pipeline rerun (`pipeline_id` `628c245d`, data through 2025).

### Added

- `src/nfelotranslation/Calibration/configs/platt_params_{season}.json` — per-season intercept configs for training labels.
- `CHANGELOG.md`.

### Validation snapshot (pipeline `628c245d`)

| Component | Key metric | Value |
|-----------|------------|------:|
| Margin distribution | OOS SAE | 11.3% (494 / 4,363 games) |
| Margin distribution | Tail bias | −5.5% |
| Margin distribution | Close bias | +0.3% |
| SpreadMapper | OOS bisection | 0.502 |
| SpreadMapper | OOS MAE | 10.23 pts |
| SpreadMapper | Binned R² (vs margin medians) | 0.809 |

## [0.1.0]

Initial public release with dual SpreadMapper (MODEL/MARKET), pooled Platt recalibrator composed into `Translator`, and four input types.
