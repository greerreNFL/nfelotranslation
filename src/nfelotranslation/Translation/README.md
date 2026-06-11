# Translation

Top-level user-facing API. Composes the SpreadMapper and MarginDistributionModel into a single stateful object that produces every derived translation quantity from one input.

## Why it exists

The other modules in this package each operate in their own input space: the `SpreadMapper` works on `(win_prob, spread)` pairs, and the `MarginDistributionModel` works on `(spread, win_prob)` pairs. A consumer of the package typically starts from a model win probability or a model spread and needs the full margin distribution in a coherent set.

The Translator is the layer that resolves the input to an internal home-perspective model win probability, runs the relevant primitives in the right order, and exposes every quantity as a property on a single object. Recalibration is a training primitive only — it is not composed into the Translator because applying market-derived corrections at inference would inject information not knowable at prediction time.

## Input types

The constructor takes a numeric value and an `input_type`. The two valid types and their resolution paths to the internal home-perspective model win probability:

| `input_type` | Resolution |
|---|---|
| `'win_prob'` | value is treated as the model WP (home perspective if `side='home'`, away perspective flipped if `side='away'`). |
| `'spread'` | `SpreadMapper.spread_to_win_prob(value)`. |

All other properties on the Translator are computed from this internal model WP.

## Sign convention and `side`

Inputs and outputs use the convention that a **positive spread means the home team is favored**, and a model WP greater than `0.5` means the home team is favored. Sportsbook feeds usually display favorites as negative spreads; callers reading raw sportsbook data must negate before passing in.

The `side` parameter (`'home'` or `'away'`) controls only which perspective the two sign-flipping properties report from: `win_prob` and `spread`. With `side='away'`, those properties return the away-perspective values (away WP, sign-flipped spread).

The fixed-perspective properties — `home_win_prob`, `away_win_prob` — always report their named side regardless of `side`. The discrete distribution and its derived quantities — `pmf`, `tie_prob`, `expected_margin`, `cover_prob(line)`, `push_prob(line)` — are always reported from the home perspective.

## Per-season configs

Because aspects like the key numbers and spread mapping are non-stationary, the Translator takes a `season` argument so each translation uses only configs trained on data available at that season. This matters for backtesting and for capturing non-stationary effects.

- **SpreadMapper** — loaded via `SpreadMapper.from_file(season=season)`. The class handles seasonal resolution.
- **KeyModel** — loaded via `KeyModel.from_file(season=season, params=KEY_MODEL_PARAMS)`. The class handles seasonal resolution.

Both seasonal classes share the same resolution rule, implemented at the model class level (no longer in the Translator):

1. **Exact match.** If `<module>/configs/<prefix>_{season}.json` exists, load it.
2. **Past the latest trained season.** `find_config_path` warns (`falling back to <prefix>_{prior}.json`) and returns the most recent prior season's config. The Translator surfaces that warning to the caller and continues.
3. **Before the earliest trained season.** No prior config exists. `from_file` raises `FileNotFoundError`. This is intentional — using a 2007-trained model on a pre-2007 game is out of the model's domain, and a hard error forces the caller to acknowledge that.

## Pipeline

```
              value, input_type, season, side
                            ↓
     ┌────── load per-season models for the season ──────┐
     │              SpreadMapper, KeyModel               │
     └─────────────────────┬─────────────────────────────┘
                           ↓
              resolve input → home model WP
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
       [SpreadMapper]          [MarginDistributionModel]
        win_prob_to_spread       (Base + Key + Normalizer)
              ↓                         ↓
         model spread            MarginDistribution
                                  (PMF, cover_prob,
                                   push_prob,
                                   expected_margin)
              └────────────┬────────────┘
                           ↓
                  Translator state
        (side flips spread / WP pairs as documented above)
```

Properties are computed eagerly inside `__init__` and `update`. Subsequent property access is plain attribute lookup.

## Usage

```python
from nfelotranslation import Translator

## convention: positive spread = home favorite ##
t = Translator(3.0, 'spread', season=2025, side='home')

t.win_prob              ## model WP from input side
t.spread                ## model Spread (posted + continuous)
t.cover_prob(3.0)       ## P(margin > 3) from home perspective
t.pmf                   ## ndarray (151,)

## reuse loaded models, recompute state ##
t.update(7.0, 'spread')
```

## Modules

### `Translator`

Stateful translator. Loads all season-specific models once on construction and recomputes state on `update`.

- `__init__(value, input_type, season, side='home')`
- `update(value, input_type, side=None)` — recompute state with a new input. Reuses loaded models. `side=None` keeps the current side.

**Core properties** (perspective controlled by `side`):

- `win_prob` — model WP.
- `spread` — model-derived `Spread`.

**Side-fixed properties** (always from the named side):

- `home_win_prob`, `away_win_prob` — model WPs.

**Distribution properties** (always from the home perspective):

- `pmf` — `ndarray` of shape `(151,)`, discrete PMF over integer margins `-75..+75`.
- `tie_prob` — `P(margin = 0)`.
- `expected_margin` — `E[margin]`.
- `cover_prob(line)` — `P(margin > line)`. Integer lines treat `margin == line` as a push.
- `push_prob(line)` — `P(margin == line)`. Zero for non-integer lines.
