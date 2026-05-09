# Translation

Top-level user-facing API. Composes the Recalibrator, SpreadMappers, and MarginDistributionModel into a single stateful object that produces every derived translation quantity from one input.

## Why it exists

The other modules in this package each operate in their own input space: the `Recalibrator` works on win probabilities, the `SpreadMapper` works on `(win_prob, spread)` pairs, and the `MarginDistributionModel` works on `(spread, win_prob)` pairs. A consumer of the package typically starts from one of four inputs — a market moneyline implied win probability, a market posted spread, a true (calibrated) win probability, or a true spread — and needs all of the others, plus the full margin distribution, in a coherent set.

The Translator is the layer that picks the right starting point, runs the relevant primitives in the right order, and exposes every quantity as a property on a single object. Two callers handing it different but equivalent inputs (e.g. a market spread and the corresponding market WP) will read the same values back from every property.

## Input types

The constructor takes a numeric value and an `input_type`. The four valid types and their resolution paths to the internal home-perspective calibrated win probability:

| `input_type` | Resolution |
|---|---|
| `'win_prob'` | value is treated as the calibrated WP. |
| `'market_win_prob'` | `Recalibrator.calibrate(value)`. |
| `'spread'` | `model_mapper.spread_to_win_prob(value)`. |
| `'market_spread'` | `market_mapper.spread_to_win_prob(value)` → `Recalibrator.calibrate(market_wp)`. |

All other properties on the Translator are computed from this internal calibrated WP, so the result is the same regardless of which input space the user started in.

## Sign convention and `side`

Inputs and outputs use the convention that a **positive spread means the home team is favored**, and a calibrated or market WP greater than `0.5` means the home team is favored. Sportsbook feeds usually display favorites as negative spreads; callers reading raw sportsbook data must negate before passing in.

The `side` parameter (`'home'` or `'away'`) controls only which perspective the four sign-flipping properties report from: `win_prob`, `win_prob_market`, `spread`, `market_spread`. With `side='away'`, those four properties return the away-perspective values (away WP, sign-flipped spreads).

The fixed-perspective properties — `home_win_prob`, `away_win_prob`, `home_win_prob_market`, `away_win_prob_market` — always report their named side regardless of `side`. The discrete distribution and its derived quantities — `pmf`, `tie_prob`, `expected_margin`, `cover_prob(line)`, `push_prob(line)` — are always reported from the home perspective.

## Per-season configs

Because aspects like the key numbers and spread mapping are non-stationary, the Translator takes a `season` argument so each translation uses only configs trained on data available at that season. This matters for backtesting and for capturing non-stationary effects.

- **SpreadMappers** — loaded via `SpreadMapper.from_file(MapType.X, season=season)`. The class handles seasonal resolution.
- **KeyModel** — loaded via `KeyModel.from_file(season=season, params=KEY_MODEL_PARAMS)`. The class handles seasonal resolution.
- **Recalibrator** — single static fit loaded via `Recalibrator.from_file()`; no per-season indexing.

Both seasonal classes share the same resolution rule, implemented at the model class level (no longer in the Translator):

1. **Exact match.** If `<module>/configs/<prefix>_{season}.json` exists, load it.
2. **Past the latest trained season.** `find_config_path` warns (`falling back to <prefix>_{prior}.json`) and returns the most recent prior season's config. The Translator surfaces that warning to the caller and continues.
3. **Before the earliest trained season.** No prior config exists. `from_file` raises `FileNotFoundError`. This is intentional — using a 2007-trained model on a pre-2007 game is out of the model's domain, and a hard error forces the caller to acknowledge that.

## Pipeline

```
              value, input_type, season, side
                            ↓
     ┌────── load per-season models for the season ──────┐
     │   Recalibrator (static), SpreadMappers, KeyModel  │
     └─────────────────────┬──────────────────────────────┘
                           ↓
              resolve input → home calibrated WP
                           ↓
    ┌──────────────────────┼──────────────────────┐
    ↓                      ↓                      ↓
[SpreadMappers]      [Recalibrator           [MarginDistributionModel]
  win_prob_to_         .uncalibrate            (Base + Key + Normalizer)
  spread, both          → home market WP)
  instances)
    ↓                      ↓                      ↓
 model spread,         home & away          MarginDistribution
 market spread         market WPs           (PMF, cover_prob,
                                             push_prob,
                                             expected_margin)
    └──────────────────────┼──────────────────────┘
                           ↓
                  Translator state
        (side flips spread / WP pairs as documented above)
```

Properties are computed eagerly inside `__init__` and `update`. Subsequent property access is plain attribute lookup.

## Usage

```python
from nfelotranslation import Translator

## convention: positive spread = home favorite ##
t = Translator(3.0, 'market_spread', season=2025, side='home')

t.win_prob              ## calibrated WP from input side
t.spread                ## model Spread (posted + continuous)
t.market_spread         ## market Spread
t.cover_prob(3.0)       ## P(margin > 3) from home perspective
t.pmf                   ## ndarray (151,)

## reuse loaded models, recompute state ##
t.update(7.0, 'market_spread')
```

## Modules

### `Translator`

Stateful translator. Loads all season-specific models once on construction and recomputes state on `update`.

- `__init__(value, input_type, season, side='home')`
- `update(value, input_type, side=None)` — recompute state with a new input. Reuses loaded models. `side=None` keeps the current side.

**Core properties** (perspective controlled by `side`):

- `win_prob` — calibrated WP.
- `win_prob_market` — market (uncalibrated) WP.
- `spread` — model-derived `Spread`.
- `market_spread` — market-derived `Spread`.

**Side-fixed properties** (always from the named side):

- `home_win_prob`, `away_win_prob` — calibrated WPs.
- `home_win_prob_market`, `away_win_prob_market` — market WPs.

**Distribution properties** (always from the home perspective):

- `pmf` — `ndarray` of shape `(151,)`, discrete PMF over integer margins `-75..+75`.
- `tie_prob` — `P(margin = 0)`.
- `expected_margin` — `E[margin]`.
- `cover_prob(line)` — `P(margin > line)`. Integer lines treat `margin == line` as a push.
- `push_prob(line)` — `P(margin == line)`. Zero for non-integer lines.
