# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
