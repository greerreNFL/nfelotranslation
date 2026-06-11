# nfelotranslation

Translates NFL win probabilities and spreads into discrete margin distributions, with cover probability, push probability, and expected margin derived coherently from that distribution. Designed for prediction-model workflows that need a consistent set of `(win_prob, spread, distribution)` from any single input.

## Install

```bash
pip install nfelotranslation
```

See [CHANGELOG.md](CHANGELOG.md) for version history and breaking changes.

## What it does

The most accurate pair-wise ranking models tend to denominate their predictions in terms of win probabilities. Win probabilities are percise, more extensible to things like simulation, and are the same shape as a classification problem, meaning they can borrow from a wide range of established optimizations, techniques, loss functions, etc from other domains.

While win probabilities are the best choice for the model, they do not represent the most popular market for prediction, which is the spread. Win probabilities must be converted, which introduces opportunity for a new kind of accuracy loss, especially in the context of a sport like the NFL, where spreads are governed by nuanced dynamics like key numbers, or non-stationarity.

nfelotranslation is a package that handles translation between different denominations of prediction with minimal accuracy loss (spread to win prob, win prob to spread, alt lines, etc). Critically, the model is trained to derive expected margin from actual game outcomes rather than mapping to historical spreads, meaning it avoids potential bias embedded into markets and better models sparse tails.

The `Translator` accepts `'win_prob'` or `'spread'` and resolves each to a home-perspective model win probability through the appropriate composition of the underlying primitives. From that win probability it derives every other property: the side-flipped variants, the model spread, the discrete margin PMF over integers `-75..+75`, and the integral quantities (`cover_prob`, `push_prob`, `expected_margin`).

## Quick start

```python
from nfelotranslation import Translator

## convention: positive spread = home favorite ##
t = Translator(3.0, 'spread', season=2025, side='home')

t.win_prob              ## model home WP
t.spread                ## model spread (posted + continuous)
t.cover_prob(3.0)       ## P(margin > 3) from home perspective
t.expected_margin       ## E[margin] from home perspective (which is equivalent to t.spread, but derived via the distribution)
t.pmf                   ## ndarray (151,) over integer margins -75..+75

## reuse loaded models, recompute state ##
t.update(7.0, 'spread')
```

Full API reference in [`Translation/README.md`](src/nfelotranslation/Translation/README.md).

## Repo Layout

```
nfelotranslation/
├── src/nfelotranslation/    Shipping package (pip-installed)
├── training/                Refitting pipeline (repo-only, not shipped)
├── analysis/                Standalone empirical analyses
├── tests/                   pytest suite
└── pyproject.toml           Package metadata and extras
```

The `src/` tree is the only thing the installed wheel contains. The `training/`, `analysis/`, and `tests/` directories live at the repository root and require a clone to use. These modules are not required to run the package, but shared publicly for transparency and reproducibility.

## Modules

### Recalibrator — moneyline win-probability correction

Market moneylines have a tail bias: implied win probabilities for slight favorites overstate their actual win rate, and implied probabilities for heavy favorites understate theirs. To train the models, the dataset must have a notion of expected win probabilities, but since moneyline implied win probabilities exhibit bias, the `Recalibrator` applies an invertible split Platt / logit-linear correction `p_cal = expit(a_loc · logit(p_market) + b_loc)` so downstream fitters can work in calibrated probability space. Slopes are fit once on the full sample; intercepts vary by season config (`platt_params_{season}.json`) using a centered 5-year window. The Recalibrator is used for training labels only and is not composed into the inference `Translator`. `calibrate(p_market)` goes market → calibrated; `uncalibrate(p_cal)` goes the other direction. Detailed reference: [`Calibration/README.md`](src/nfelotranslation/Calibration/README.md).

### SpreadMap — bidirectional WP ↔ spread mapping

The `SpreadMapper` parametrizes the relationship between a win probability and a spread as `spread = slope · logit(wp) + intercept`, an analytically invertible form that handles either direction with one expression. It is fitted on recalibrated win probabilities versus actual game margins with intercept fixed at zero and exposes `win_prob_to_spread()` and `spread_to_win_prob()`. Detailed reference: [`SpreadMap/README.md`](src/nfelotranslation/SpreadMap/README.md).

### Distribution — discrete margin distribution

The `MarginDistributionModel` turns a `(spread, win_prob)` pair into a discrete PMF over every integer margin from `-75` to `+75`. The PMF starts as a continuous generalized normal anchored at the spread, then gets adjusted at every integer to capture the heavy concentration around key margins (`±3`, `±7`) and the dead zones (`±9`, `±12`), then is renormalized so `P(margin > 0) = win_prob`, `P(margin = 0) = tie_prob`, and the spread bisects the PMF. From the result, `cover_prob(line)`, `push_prob(line)`, and `expected_margin()` are direct PMF queries. Detailed reference: [`Distribution/README.md`](src/nfelotranslation/Distribution/README.md).

### Translation — composed top-level API

The `Translator` composes the SpreadMapper and MarginDistributionModel into a single stateful object. Construct it with `'win_prob'` or `'spread'`, a season, and a side; access every derived property as an attribute. State is held on the instance, so subsequent `update()` calls reuse the loaded models without reopening config files. Detailed reference: [`Translation/README.md`](src/nfelotranslation/Translation/README.md).

## Conventions

- **Spread sign**: positive = home favorite. While markets typically display favorites as a negative number, Translator uses the positive convention to better align with the expected margin it is meant to predict. If using market data, callers should determine whether or not a sign flip is required.
- **Win probabilities**: `win_prob > 0.5` means the home team is favored.
- **Distribution perspective**: `pmf`, `cover_prob`, `push_prob`, and `expected_margin` are always reported from the home perspective (positive margin = home win). The sign-flipping properties (`win_prob`, `spread`) flip with the `side` argument; the `home_*` and `away_*` properties are fixed-perspective.
- **Per-season configs**: each module with per-season fits (`SpreadMap`, `Distribution/Key`) places them under `<module>/configs/`. `Model.from_file(season=...)` loads the file matching that season exactly, falls back to the most recent prior season's file with a warning when no exact match exists, and raises `FileNotFoundError` when the requested season predates the earliest fit.

## Training and refitting

Refitting is a once-per-year operation, run after the NFL season completes. The training pipeline lives in the `training/` directory at the repository root and is **not** part of the installed distribution — `pip install nfelotranslation` provides inference only. Training requires `nfelodcm` for historical game data and access to the source tree, both of which the slim wheel intentionally omits.

The pipeline runs four phases in dependency order. Each phase writes its config back into `src/nfelotranslation/` so the editable install picks up the new state immediately. A shared `pipeline_id` is stamped on every config produced by the same run, and downstream phases verify upstream `pipeline_id`s before starting to catch stale-state bugs.

```
1. Recalibrator    → platt_params.json + configs/platt_params_{season}.json
2. SpreadMapper    → spread_map_params.json (root) + configs/spread_map_params_{season+1}.json
3. KeyModel        → key_model.json (root) + configs/key_model_{season+1}.json
4. Distribution validation (system-level integration check)
```

Setup uses a dedicated environment to keep the editable source-pointer install separate from any production install of the same package:

```bash
conda create -n nfelotranslation-dev python=3.12
conda activate nfelotranslation-dev
pip install -e ".[dev]"
```

The full pipeline is one command from the repository root:

```bash
PYTHONPATH=. python -m training.Scripts
```

Each `Fitter` produces a `ValidationReport` with gated checks (structural invariants that must pass) and tracked metrics (numbers monitored across runs for drift). Reports are written to `<module>/validation/`. The system-level `MarginDistributionValidator` then evaluates the full composed distribution against historical outcomes using per-season `KeyModel` configs, so each season is scored against a model trained only on prior seasons.

Hyperparameters of the margin distribution (`beta`, `tie_prob`, and the three `KeyModel` credibility knobs) are tuned separately by the `HyperparamOptimizer` against the same per-season-averaged objective the production validator uses.

Full instructions, per-component validation guidance, and troubleshooting in [`training/TRAINING.md`](training/TRAINING.md). Hyperparameter optimization details in [`training/Hyperparams/README.md`](training/Hyperparams/README.md).

## Development

```bash
pip install -e ".[dev]"
pytest
python -m build
```
