# Key

Per-integer excess trackers using a credibility-weighted ratio model.

## Why it exists

Though the base distribution is a generalized normal, NFL margins are not normally distributed at the integer level. Certain margins occur substantially more often than a smooth distribution predicts (key numbers: +-3, +-7), while others occur less often (dead zones: +-9, +-12). These deviations are structural to football's scoring system, which makes them 1) generally predictable, but also 2) influenced temporally by rule changes, strategy changes, and performance changes. The Key module models expected per-integer excesses through time, which is then added to the base distribution to create a realistic distribution of margin outcomes.

The prior version of nfelo's margin distribution model (nfeloapp.com/analysis/margin-probabilities-from-nfl-spreads/) baked key number structure into the a single distribution in one shot via mixture models, tiered spikes, etc. This approach is decidedly less principled as it required assumptions about which numbers to model and necisitated complex joint parameter optimization, which itself is prone to overfitting.

By tracking individual integers using a credibility-weighted model, key numbers naturally emerge from data and can be more fluid through time.

## Evidence driving the solution

### The excess is real, and generally stable

Across 5,011 games (2006–2025), the excess rate at each integer is the share of games landing on `±k` minus the share predicted by the gennorm baseline at `beta = 1.35`. The CV column is the across-season `|std / mean|` over 20 seasons:

| k | Excess rate | Ratio | CV (per-season) |
|---:|---:|---:|---:|
| 1  | -0.0059 | 0.88 | 2.11 |
| 2  | -0.0065 | 0.87 | 1.76 |
| 3  | +0.0971 | 2.99 | 0.26 |
| 4  | -0.0006 | 0.99 | 100.23 |
| 5  | -0.0092 | 0.81 | 1.67 |
| 6  | +0.0178 | 1.38 | 1.07 |
| 7  | +0.0417 | 1.93 | 0.44 |
| 8  | -0.0059 | 0.86 | 1.64 |
| 9  | -0.0264 | 0.37 | 0.41 |
| 10 | +0.0135 | 1.34 | 0.89 |
| 11 | -0.0155 | 0.59 | 0.70 |
| 12 | -0.0182 | 0.49 | 0.39 |
| 13 | -0.0090 | 0.73 | 1.15 |
| 14 | +0.0187 | 1.60 | 0.86 |
| 15 | -0.0129 | 0.55 | 0.74 |
| 16 | -0.0038 | 0.86 | 2.48 |
| 17 | +0.0083 | 1.34 | 1.52 |
| 18 | +0.0012 | 1.05 | 9.03 |
| 19 | -0.0103 | 0.51 | 0.64 |
| 20 | +0.0025 | 1.13 | 4.01 |
| 21 | +0.0089 | 1.50 | 1.50 |

Most of the seasonal noise observed occurs in instances where there is little to no excess, which could be read as more a structural product of dividing by a number near zero. Importantly, the numbers generally thought to be key show strong and stable signal (Analysis 8). Where there is true movement, it occurs due to shifts across eras.

### Some integers shift materially across eras

Comparing pre-2015 and 2015+ subsamples (the 2015 boundary is the PAT-distance rule change), the strongest signals are stable but several integers move by more than a per-season standard deviation:

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

In this example, it's quite clear that change to PAT-distance coincided with a large increase to the excess rate of 6 (Analysis 8). For this reason, the key model leverages exponential-decay in its updating to bias towards more recent data and adapt to changes in excess.


### Key numbers are proximity-dependent

Key numbers also experience some proximity dependence. The further the spread is from a number, the more its expected excess shifts towards 0. This is observed by regressing observed excess at different distance bins against the distance from the spread (Analysis 9). Key numbers with positive excess have negative slope, while key numbers with negative excess have positive slope:

| k | Weighted mean | Intercept | Slope per point | Weighted R² |
|---:|---:|---:|---:|---:|
| 1  | -0.0049 | +0.0023 | -0.00135 | 0.151 |
| 2  | -0.0053 | -0.0005 | -0.00089 | 0.046 |
| 3  | +0.0471 | +0.0620 | -0.00270 | 0.336 |
| 4  | -0.0018 | +0.0003 | -0.00036 | 0.022 |
| 5  | -0.0057 | -0.0053 | -0.00007 | 0.001 |
| 6  | +0.0081 | +0.0092 | -0.00015 | 0.003 |
| 7  | +0.0205 | +0.0285 | -0.00105 | 0.131 |
| 8  | -0.0038 | -0.0042 | +0.00005 | 0.001 |
| 9  | -0.0147 | -0.0259 | +0.00121 | 0.496 |
| 10 | +0.0057 | +0.0096 | -0.00038 | 0.028 |
| 11 | -0.0086 | -0.0179 | +0.00084 | 0.255 |
| 12 | -0.0105 | -0.0219 | +0.00095 | 0.382 |
| 13 | -0.0056 | -0.0148 | +0.00070 | 0.221 |
| 14 | +0.0083 | +0.0170 | -0.00062 | 0.114 |
| 15 | -0.0075 | -0.0240 | +0.00110 | 0.414 |
| 16 | -0.0030 | -0.0105 | +0.00046 | 0.168 |
| 17 | +0.0031 | +0.0012 | +0.00011 | 0.006 |
| 18 | +0.0001 | -0.0031 | +0.00018 | 0.014 |
| 19 | -0.0058 | -0.0199 | +0.00074 | 0.412 |
| 20 | +0.0007 | -0.0004 | +0.00006 | 0.002 |
| 21 | +0.0038 | +0.0133 | -0.00045 | 0.126 |

This dynamic is not currently addressed by the model.

### Why track all integers, not just known key numbers

Occurrence rates are generally stable, but can change over time. Proclaiming which numbers are key based on domain knowledge actually leaks data backward (since we collectively only know which numbers are key they've been observed to be so), and is not a practical approach for updating key numbers overtime as they change.

Tracking all integers and allowing instances of excess occurance drive a model that "discovers" key numbers eliminates back leakage, can adapt to changes, and scales.

## The approach

### Credibility-weighted ratio

Each integer k has a tracker that maintains three exponentially decayed state variables: `eff_hits` (observed hits at +-k), `exp_eff_hits` (expected hits from the baseline distribution), and `eff_games` (total games observed). The raw ratio measures how often the number is actually hit relative to what the baseline predicts:

```
raw_ratio = eff_hits / exp_eff_hits
```

This raw ratio is then blended toward 1.0 (no excess) using a credibility weight that ramps linearly from 0 to 1 as the tracker accumulates expected observations:

```
credibility = min(1, exp_eff_hits / threshold)
ratio = 1 + (raw_ratio - 1) * credibility
```

With `threshold=25`, a number needs ~25 expected hits of effective (decay-weighted) exposure before the model reports its full ratio. Low-frequency numbers (like +-30) naturally self-regulate toward ratio=1.0 because they never accumulate enough expected hits to reach full credibility, while high-frequency numbers (like +-3) reach full credibility quickly and report their true ratio.

A ratio > 1.0 means the number occurs more often than the baseline predicts (key number). A ratio < 1.0 means less (dead zone). A ratio of exactly 1.0 means no adjustment.

### Multiplicative application

The ratio is applied multiplicatively to the baseline PMF. For each tracked integer k, the excess at +k and -k is:

```
excess_pos = (ratio - 1) * baseline_pmf[+k]
excess_neg = (ratio - 1) * baseline_pmf[-k]
```

The result is denominated in **excess probability** so it can be added directly to the raw PMF: `raw_pmf[k+75] += excess_pos`. The math is equivalent to scaling each bin by the ratio: `baseline + (ratio-1)*baseline = ratio*baseline`.

This approach naturally handles three things that required separate mechanisms in the prior model:

1. **Side splitting**: Each side (+k and -k) gets excess proportional to its own baseline, so asymmetric spreads produce asymmetric excess — more on the side closer to the spread, less on the far side.

2. **Distance dependence**: When the spread is far from k, the baseline PMF at that bin is small, so `(ratio - 1) * small_baseline` produces small excess. An explicit distance decay parameter could potentially better capture this effect, as the excess ratio itself _also_ decreases, but this adds additional complexitiy and overfitting risk.

3. **No double-counting**: Total excess = `(ratio - 1) * (baseline_pos + baseline_neg)`, which correctly scales with the combined baseline rather than applying the same excess independently to both sides.

### Exponential decay

Older seasons are exponentially decayed (`forgetting_rate=0.087` per season). This means the tracker naturally adapts to structural changes (like the PAT rule change) without requiring a hard reset. The effective sample size stabilizes around `n_games / forgetting_rate ~ 3,200` games, weighting recent data more heavily. The low forgetting rate reflects that season-to-season noise exceeds real drift for most numbers - longer memory slightly improves accuracy.

### Prior initialization

On first update, the tracker is seeded with `initial_prior_size=52` pseudo-games at the baseline rate. This creates a balanced prior (ratio = 1.0, zero excess) with enough weight to prevent the model from overreacting to the first season of data. The prior is small enough that it decays away within a few seasons.

## Pipeline position

Key sits alongside Base in the Distribution pipeline. The MarginDistributionModel passes the discretized baseline PMF to the KeyModel, which computes per-integer excess using `(ratio - 1) * baseline_per_bin`. This excess is added to the raw PMF before normalization.

```
(spread, wp) -> [BaseDistribution] -> continuous gennorm(spread, scale, beta)
                                         |
                                   [Normalizer.discretize] -> raw PMF (baseline)
                                         |
               [KeyModel] (ratio-1)*baseline -> add to raw PMF -> dirty PMF
                                         |
                                   [Normalizer.normalize] -> final PMF
```

## Modules

### `KeyModel`

Collection of 40 `NumberOutcome` trackers. State container + accessor.

- `from_default(params)` - create fresh model with all-zero trackers
- `excess_at(number, baseline_pmf)` - excess for a single number given the baseline PMF
- `get_all_excess(baseline_pmf)` - excess for all 40 numbers (used by MarginDistributionModel)
- `to_file(filepath)` / `from_file(filepath, params)` - JSON persistence

### `NumberOutcome`

Credibility-weighted ratio tracker for a single margin integer.

- `update(hits, n_games, baseline_rate, season)` - ingest one season of data
- `get_ratio()` - credibility-weighted ratio (1.0 = no excess)
- `excess_at(baseline_pos, baseline_neg)` - multiplicative excess at +-k given per-bin baselines
- `get_state_at(season)` - historical lookup for backtesting
- `to_dict()` / `from_dict(data, params)` - serialization

### Types

- `NumberOutcomeRecord` - per-season snapshot (pre-update ratio for backtesting)

## Parameters


| Parameter            | Value | Role                                                      |
| -------------------- | ----- | --------------------------------------------------------- |
| `forgetting_rate`    | 0.087 | Exponential decay rate per season for all state variables |
| `threshold`          | 25    | Expected-hit count at which credibility reaches 1.0       |
| `initial_prior_size` | 52    | Pseudo-games seeded on first update (balanced prior)      |


### How parameters were determined

Learning parameters (`forgetting_rate`, `threshold`, `initial_prior_size`) were optimized on season-aggregate out-of-sample RMSE via Nelder-Mead with 5 random restarts. The objective trains the model across all seasons with candidate params and measures how well each season's pre-update ratio predictions match actual excess. RMSE moved only marginally during optimization (`1.011` to `1.006` pp) — the architecture does the work, not the parameter values.

**When to re-optimize**: These parameters are insensitive — re-optimization is unlikely to yield meaningful improvement unless there is a structural change to the game (rule change, schedule expansion).

## Training

The seasonal fitter, validation report, and integration with the system-level
margin distribution validator live in `training/TRAINING.md` under the `Key`
subsection.
