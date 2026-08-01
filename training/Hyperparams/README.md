# Hyperparams

Hyperparameter tuning for the shipped margin distribution. Optimizes the four tunable knobs against the same objective the production validator reports on, so what we tune and what we ship are the same thing.

## What this module does

The shipped Distribution module has four scalar hyperparameters that control how the margin PMF is composed:

| Param | File | What it controls |
|---|---|---|
| `beta` | `Distribution/margin_hyperparams.json` | Tail weight of the gennorm base distribution. β=2 → Gaussian; β<2 → heavier tails. |
| `forgetting_rate` | `Distribution/Key/key_hyperparams.json` | Per-season exponential decay of accumulated KeyModel state. Higher → adapt faster, retain less history. |
| `threshold` | `Distribution/Key/key_hyperparams.json` | Credibility-shrinkage threshold on the KeyModel ratio. Low-evidence ratios are pulled toward 1.0; the threshold sets how much evidence counts as "enough." |
| `initial_prior_size` | `Distribution/Key/key_hyperparams.json` | Starting pseudo-count weight applied to each NumberOutcome's prior at first update. Higher → more smoothing of sparse early-season counts. |

`HyperparamOptimizer` runs scipy Nelder-Mead over any subset of these against the per-season-averaged SAE objective described below. The fitted Recalibrator and SpreadMappers are held fixed during tuning — this module does not optimize them. Their parameters come from their respective Fitters in `training/Calibration/` and `training/SpreadMap/`.

## The objective: per-season-averaged SAE

For each OOS season `s` (default cutoff: 2010), the optimizer:

1. Reconstructs the KeyModel snapshot trained through season `s-1` (the same snapshot `MarginDistributionValidator` uses).
2. Builds a `MarginDistributionModel` with the candidate `beta` and that snapshot.
3. Predicts the margin histogram for every game in season `s`, summed over `(model_spread, cal_wp)` groups.
4. Computes `SAE_s = Σ_b |predicted_b − actual_b|` across all 151 margin bins.

The objective is `mean(SAE_s)` across all OOS seasons. This is intentionally NOT the aggregate-then-absolute SAE the older optimizer used.

### Why per-season instead of aggregate

Aggregating predictions across seasons before taking the absolute value lets per-season errors cancel:

* If the model over-predicts margin=3 by 10 games in 2020 and under-predicts margin=3 by 10 games in 2021, the aggregate residual at margin=3 is zero. Aggregate SAE awards this a perfect score.
* But the user never experiences "the average across years." They use the snapshot for the season they're predicting. The 2020 user saw a 10-game miss; the 2021 user saw a 10-game miss.

A toy example: if we always over-react to last season's count at one bin, we'd alternate over-prediction and under-prediction year over year. Per-season SAE penalizes both years; aggregate SAE shrugs.

When we first switched objectives, the unconstrained optimum under aggregate SAE wanted to **disable** the credibility mechanism (`forgetting_rate=0.999`, `threshold=0.1`) — let last year's raw counts pass straight through. That's classic noise-chasing behavior, and it scored well on aggregate SAE because the noise canceled. Under per-season-averaged SAE the same optimization moves the parameters in the opposite direction (`forgetting_rate≈0.04`, large prior), recovering the smoothing the credibility mechanism is designed to provide.

## The structural trade-off: center vs tails

`beta` is by far the dominant lever and the source of the only meaningful tradeoff at the package's current state. Smaller `beta` (heavier-tailed gennorm) requires a wider scale to satisfy the win-probability constraint. That wider scale doesn't just put mass in the far tails — it also strips mass from the **intermediate region** (margins ~5-20) where most of the high-occurrence key numbers live.

Concretely, comparing the base PDF at the same `(spread, wp)` constraint:

| margin | β=1.10 (heavy tails) | β=2.00 (Gaussian) |
|---|---|---|
| 3 (center) | 0.0349 | 0.0337 |
| 7 | 0.0298 | 0.0318 ← higher |
| 10 | 0.0249 | 0.0283 ← higher |
| 14 | 0.0187 | 0.0219 ← higher |
| 28 (far tail) | 0.0053 | 0.0036 ← lower |

Three regional checks in `MarginDistributionValidator` constrain where `beta` can land:

* **close (1-6)**: low `beta` over-concentrates mass at the immediate center and the meaty intermediate region (the favorite's typical margin), pulling close-bias negative. Going below `beta ≈ 1.15` pushes close-bias past the −5% threshold.
* **mid (7-16)**: roughly indifferent across the feasible range. Mid bias is the slack region.
* **tail (17+)**: high `beta` thins the tails. Going above `beta ≈ 1.50` pushes tail-bias past the −10% threshold.

So the feasible window is `beta ∈ [1.15, 1.50]`. The center and the tail want different things; the validator's bias checks make us pick a balance.

Inside the feasible window, per-season SAE is relatively flat — the absolute SAE difference between `beta=1.20` and `beta=1.50` is on the order of 1-2 games per season (out of ~272). Most of the available SAE improvement is captured by `beta ≈ 1.25`, with diminishing returns above that.

## Current shipped value: beta = 1.24

Selected to pass the regional bias gates with headroom under the refactored pipeline. At `beta=1.24`, with the other three hyperparams at their shipped values (`forgetting_rate=0.0325`, `threshold=11.25`, `initial_prior_size=61`):

* per-season-averaged SAE: 112.42 games
* aggregate OOS SAE: 11.16%
* close-bias: +0.5% (4.5pp headroom against the ±5% threshold)
* tail-bias: −5.3% (4.7pp headroom against the ±10% threshold)
* mid-bias: +1.3% (no threshold)

The headroom matters as much as the SAE number. Higher `beta` values can improve SAE but push tail-bias toward the −10% gate; `beta=1.24` is the value that clears the validator on the current training pipeline (`pipeline_id` `628c245d`).

The four shipped hyperparameters are evaluated jointly against mean per-season SAE, aggregate OOS SAE, and close/mid/tail bias. The current values are `forgetting_rate=0.0325`, `threshold=11.25`, `initial_prior_size=61`, and `beta=1.24`. They produce mean per-season SAE of `112.42`, aggregate OOS SAE of `11.16%`, and absolute regional biases comfortably inside their validation gates. Requiring acceptable regional bias prevents lower SAE from being selected at the expense of distribution calibration.

## Train-time vs inference-time beta — calibrated, not a bug

The trainer's per-integer baseline computation in `_compute_baselines` uses `scipy_norm.pdf` (a standard Normal, equivalent to `gennorm(beta=2)` up to scale convention) regardless of the inference-time `beta`. This looks like it should be a bug — training computes "expected hits at margin k" against one shape while inference applies the resulting credibility ratio against a different shape — but a matrix sweep of `(beta_train × beta_inference)` shows the current decoupling sits in a comfortably-passing corner of the feasible region.

The mechanism is the redistribution effect described above. Higher `beta_train` (training baseline thinner-tailed and more peaked) produces higher "expected hits" at the intermediate keys (3, 7, 10, 14, 17), which yields lower credibility ratios there, which yields weaker corrections at inference. That weaker amplification is exactly what's needed because inference uses gennorm(1.24) which already has heavier tails than the trainer's Normal baseline. The two effects compose to produce well-calibrated key-number predictions.

If you "fix" the inconsistency naively (use `gennorm(beta=1.24)` baselines at training, matching inference), the corrections can become too strong at intermediate keys and too weak at far tails. The matrix sweep at `.local/hyperparam_optimization/beta_matrix.py` is the empirical proof — re-run it after any beta change.

So the train/inference decoupling is intentional in effect even if it wasn't designed deliberately. Do not unify the betas without re-running the matrix sweep and confirming the regional bias checks still pass.

## What this module does NOT tune

* `tie_prob` (in `margin_hyperparams.json`). Set empirically from the historical rate of integer-margin=0 outcomes; not a fitted hyperparameter.
* The Recalibrator's split Platt slopes and per-season intercepts. Fitted by `RecalibratorFitter` in `training/Calibration/`.
* The SpreadMapper's `slope` / `intercept`. Fitted by `SpreadMapperFitter` in `training/SpreadMap/`.

These three are independent fits with their own loss functions and validators. The hyperparams here only affect how the KeyModel and the gennorm base shape compose into the final margin PMF.

## Modules

### `HyperparamOptimizer`

* `__init__(games=None)` — pre-computes per-season actual counts, per-season spread groups, and per-season training data once (none depend on the hyperparams). Subsequent `objective()` evaluations only rebuild the KeyModel and re-evaluate the per-season SAE.
* `objective(x, optimize_params, fixed_params)` — single SAE evaluation for a candidate parameter vector. Returns mean per-season SAE.
* `optimize(optimize_params, bounds, fixed_params=None, x0=None, method='Nelder-Mead')` — runs scipy.optimize. Returns the OptimizeResult with `.optimized_params` attached.

### Constants

* `_OOS_MIN_SEASON = 2010` — matches `MarginDistributionValidator._MIN_SEASON`. Pre-OOS seasons warm up the KeyModel; only OOS seasons are scored.
* `DEFAULT_PARAMS` — live snapshot of the shipped hyperparams (loaded from `MARGIN_HYPERPARAMS` and `KEY_MODEL_PARAMS`). Single source of truth; updates automatically when shipped values change.

## How to re-run

A bounded sweep over all four hyperparams (typical reference run):

```python
from training.Hyperparams import HyperparamOptimizer

opt = HyperparamOptimizer()
result = opt.optimize(
    optimize_params=['forgetting_rate', 'threshold', 'initial_prior_size', 'beta'],
    bounds={
        'forgetting_rate':    (0.001, 0.999),
        'threshold':          (0.1, 10000.0),
        'initial_prior_size': (0.1, 10000.0),
        'beta':               (1.0, 2.0),
    },
)
print(result.optimized_params)
```

The optimizer does NOT write the result anywhere. Updating shipped hyperparams is a manual step:

```python
from nfelotranslation.Utilities.JsonIo import (
    write_config_envelope, ConfigMetadata, generate_pipeline_id,
)
write_config_envelope(
    'src/nfelotranslation/Distribution/margin_hyperparams.json',
    {'beta': 1.24, 'tie_prob': 0.002},
    ConfigMetadata.new(pipeline_id=generate_pipeline_id()),
)
```

After updating shipped hyperparams, re-run the full training pipeline (`from training.Scripts import train_all; train_all()`) so the per-season fitted configs are regenerated against the new hyperparam values, and verify that `MarginDistributionValidator` reports all three regional checks passing with reasonable headroom.

## References

The empirical findings in this README come from analysis scripts under `.local/hyperparam_optimization/`. Re-run any of these to reproduce or update the numbers cited above:

* `verify.py` — full-bounds optimization over all four hyperparams.
* `unconstrained.py` — same with widened bounds, used to confirm interior optima vs corner solutions.
* `sensitivity.py` — single-parameter sweeps to attribute SAE swings.
* `beta_sweep.py` — `beta`-only sweep with the other three at shipped values; produced the feasible window analysis.
* `beta_matrix.py` — `(beta_train × beta_inference)` matrix sweep; produced the train/inference decoupling analysis.
* `beta_train_high.py` — extends `beta_train` past 2.0 into platykurtic territory.
* `baseline_consistency.py` — compares scipy_norm baselines to gennorm baselines at fixed hyperparams.
* `distribution_at.py`, `chart.py`, `per_season_keys.py` — visualization and diagnostic outputs.
