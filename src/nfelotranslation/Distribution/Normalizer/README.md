# Normalizer

Discretizes a continuous BaseDistribution into integer bins and normalizes per-region to restore win probability, tie probability, and spread bisection constraints.

## Why it exists

The BaseDistribution provides a smooth continuous normal, and the KeyModel provides per-integer excess adjustments. But the final margin distribution must be a discrete PMF at integer margins that satisfies three hard constraints:

1. **P(margin > 0) = wp** — win probability is preserved exactly
2. **P(margin = 0) = tie_prob** — tie probability is set explicitly
3. **P(margin < 0) = 1 - wp - tie_prob** — loss probability is the complement

Additionally, the spread must **bisect** the distribution — SpreadMapper determines which spread bisects the distribution of outcomes from a given win probability, based on historical observations. Definitionally, a spread must be interpretable as the centerpoint of the expected distribution of outcomes. Therefore, just as win probability must be preserved, so too must the equal weight on each side of the spread.

Because key numbers do not add distribution mass uniformally, and because clamping to integers introduces asymetric rounding, simply summing the continuous PDF at integers and normalizing so they collectively sum to 1 is not enough to satisfy these *hard* constraints. To perform normalization and satisfy constraints, the distribution must be decomposed into parts that can then be normalized to their weights in the identically decomposed original distribution (see the region scheme below). ie, if the section of [-75,0] (aka p(margin<0)) is 0.34 in the original distribution, then the identical section ([-75,0]) in the adjusted distribution most be normalized to .34 as well.

The Normalizer's job is to perform this decomposition and scaling.

### Why tie probability is separate

The continuous normal is a smooth distribution with zero probability at any single point — P(margin = 0) = 0. But NFL games do end in ties (margin = 0), albeit rarely. The tie probability cannot be derived from the continuous distribution — it must be an explicit parameter that carves out mass at margin 0.

This is why tie_prob lives on the Normalizer (a discrete concept) rather than on BaseDistribution (a continuous object).

## The region scheme: How spread bisection and win probability are preserved

The margin space -75..+75 is divided by two landmarks: **zero** (the win/loss boundary) and **the spread** (the bisection point). These create up to five labelled regions:

| Region | Definition | Role |
|--------|-----------|------|
| **A** | The entire side of zero that does NOT contain the spread | Scaled to its side's total target in one block |
| **T** | Margin 0 | Tie bin — always exactly tie_prob |
| **B** | Margins between zero and the spread (exclusive of both) | Sub-region scaled proportionally to continuous CDF |
| **C** | The spread bin itself | Integer spreads only — scaled proportionally |
| **D** | Margins beyond the spread (same side as the spread) | Sub-region scaled proportionally to continuous CDF |

In the language of our constraints:
- WIN PROB CONSTRAINT: B + C + D = wp
- SPREAD BISECTION CONSTRAINT: A + T + B = D

### Region layout by spread sign

**Positive integer spread** (e.g., s = 7):

```
Region:    A          T     B            C              D
---------------------------------------------------------
Margins:   -75..-1    0     1..s-1       s        s+1..75
Target:    1-wp-tp    tp    ←—— these three sum to wp ——→   (WIN PROB CONSTRAINT)
Target:    ←——      these       ——→      =    ←— this  —→   (SPREAD BISECTION CONSTRAINT)
```

A is the loss side (opposite the spread). B/C/D are the win side, sub-divided around the spread. The C bin (margin = s) is special: its value passes through from the dirty PMF, so the key model's adjustment at the push number is preserved. B and D collectively absorb the remainder of the win-side mass:

```
C        = dirty_pmf[s]                     (passed through unchanged)
B + D    = wp - C
bd_factor = (wp - C) / (sum_dirty_B + sum_dirty_D)
```

The same `bd_factor` is applied to every bin in B and every bin in D, which preserves the dirty PMF's internal shape inside each sub-region and the dirty B/D ratio between them.

**Positive half-integer spread** (e.g., s = 7.5):

```
Region:  A          T     B              D
Margins: -75..-1    0     1..floor(s)    ceil(s)..75
Target:  1-wp-tp    tp    ←— sum to wp —→
```

No C region — no integer bin sits on the spread. B and D use the spread value itself as the boundary, and each sub-region's target is proportional to the continuous CDF slice it covers:

```
target_B = wp × [CDF(s) - CDF(0.5)] / SF(0.5)
target_D = wp × SF(s) / SF(0.5)
```

The denominator `SF(0.5)` is the continuous probability of the entire win side (margin > 0.5), ensuring B + D = wp.

**Negative integer spread** (e.g., s = -7):

```
Region:  D          C     B          T     A
Margins: -75..s-1   s     s+1..-1    0     1..75
Target:  ←— these three sum to 1-wp-tp —→  tp    wp
```

A is the win side. D/C/B are the loss side, sub-divided around the spread. As in the positive integer case, the C bin (margin = s) passes through from the dirty PMF, and B + D jointly absorb the remainder of the loss-side mass:

```
C        = dirty_pmf[s]                     (passed through unchanged)
B + D    = loss - C
bd_factor = (loss - C) / (sum_dirty_B + sum_dirty_D)
```

Where `loss = 1 - wp - tp`. The same `bd_factor` scales every bin in D and every bin in B.

**Negative half-integer spread** (e.g., s = -7.5):

```
Region:  D              B          T     A
Margins: -75..floor(s)  ceil(s)..-1 0     1..75
Target:  ←— sum to 1-wp-tp —→      tp    wp
```

No C region. B and D use the spread value itself as the boundary, with sub-region targets proportional to the continuous CDF slice:

```
target_D = loss × CDF(s) / CDF(-0.5)
target_B = loss × [CDF(-0.5) - CDF(s)] / CDF(-0.5)
```

The denominator `CDF(-0.5)` is the continuous probability of the entire loss side, ensuring D + B = loss.

**Spread = 0** (pick-em):

```
Region:  A          T     D
Margins: -75..-1    0     1..75
Target:  1-wp-tp    tp    wp
```

The spread sits on the tie bin. No B or C — only the two flanking regions.

### Why sub-divide around the spread

Without sub-division, a single scale factor is applied to the entire win (or loss) side. This preserves the total win probability but distorts the shape around the spread. A margin 2 points below the spread and a margin 20 points beyond the spread get the same multiplicative adjustment — which is wrong, because the continuous normal assigns very different density to those regions.

The B/C/D sub-division preserves the continuous CDF's proportions within each side. This means:
- Margins near the spread keep approximately the right density
- Margins far from the spread keep approximately the right density
- The spread remains the approximate bisection point of the discrete PMF

### Symmetry limitation

With nonzero tie_prob, the PMF is not perfectly symmetric: `pmf(k; s, wp)` vs `pmf(-k; -s, 1-wp)` differs by O(tie_prob). This is because the win target is `wp` but the mirror's loss target is `(1-wp) - tie_prob = wp - tie_prob`. The asymmetry is inherent to the three-region scheme and is ~1% in practice — acceptable for a discrete model.

## Pipeline position

The Normalizer sits between Base and the final MarginDistribution output. It is created and used internally by MarginDistributionModel — users don't construct it directly in normal use.

```
[BaseDistribution]
       ↓ continuous density
[Normalizer.discretize()] → raw PMF (151 bins), likely dirty due to rounding
       ↓
[KeyModel excess added] → dirty PMF
       ↓
[Normalizer.normalize()] → final PMF (A/T/B/C/D regions scaled)
       ↓
[MarginDistribution] → result container
```

## Modules

### `Normalizer`

Discrete operations on a BaseDistribution. Init with a base distribution and tie probability.

- `__init__(base, tie_prob=0.002)` — wrap a BaseDistribution with a tie probability
- `discretize()` → ndarray(151) — continuous PDF at integers -75..+75, normalized to sum to 1
- `normalize(dirty_pmf)` → ndarray(151) — per-region scaling to restore A/T/B/C/D targets

Class constants:

- `MARGINS = numpy.arange(-75, 76)` — the 151 integer margin values
