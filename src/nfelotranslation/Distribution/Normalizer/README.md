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
| **B** | Margins between zero and the spread (exclusive of both) | Sub-region scaled so A + T + B equals D |
| **C** | The spread bin itself | Integer spreads only — preserves key-model push mass except at ±1 |
| **D** | Margins beyond the spread (same side as the spread) | Sub-region scaled to half of all non-push mass |

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

A is the loss side (opposite the spread). B/C/D are the win side, sub-divided around the spread. For integer spreads of at least 2, the C bin (margin = s) passes through from the dirty PMF so the key model's adjustment at the push number is preserved. D receives half of all non-push mass, and B receives the remainder of the win-side target:

```
C        = dirty_pmf[s]                     (passed through unchanged)
D*       = (1 - C) / 2
B*       = D* - (A + T)
         = D* - (1 - wp)
         = wp - 0.5 - C/2
```

B and D are scaled independently to these totals. Each region's internal dirty-PMF shape is preserved, but their relative totals change as required to enforce bisection.

At `s = 1`, B is empty. C cannot remain fixed while both constraints hold, so `D* = 1 - wp` and `C* = 2wp - 1`.

**Positive half-integer spread** (e.g., s = 7.5):

```
Region:  A          T     B              D
Margins: -75..-1    0     1..floor(s)    ceil(s)..75
Target:  1-wp-tp    tp    ←— sum to wp —→
```

No C region — no integer bin sits on the spread. For spreads of at least 1.5, exact bisection fixes D at half the distribution, and B receives the remainder of the win-side target:

```
target_B = wp - 0.5
target_D = 0.5
```

At `s = 0.5`, no B or C region exists: D is the entire win side. Exact bisection is therefore incompatible with preserving a mapped `wp != 0.5`; win probability takes precedence.

**Negative integer spread** (e.g., s = -7):

```
Region:  D          C     B          T     A
Margins: -75..s-1   s     s+1..-1    0     1..75
Target:  ←— these three sum to 1-wp-tp —→  tp    wp
```

A is the win side. D/C/B are the loss side, sub-divided around the spread. For integer spreads of at most -2, C passes through from the dirty PMF. D receives half of all non-push mass, and B receives the remainder of the loss-side target:

```
C        = dirty_pmf[s]                     (passed through unchanged)
D*       = (1 - C) / 2
B*       = D* - (A + T)
         = D* - (wp + tp)
         = 0.5 - wp - tp - C/2
```

Where `loss = 1 - wp - tp`. B and D are scaled independently while preserving each region's internal shape.

At `s = -1`, B is empty, so `D* = wp + tp` and `C* = loss - D*`.

**Negative half-integer spread** (e.g., s = -7.5):

```
Region:  D              B          T     A
Margins: -75..floor(s)  ceil(s)..-1 0     1..75
Target:  ←— sum to 1-wp-tp —→      tp    wp
```

No C region. For spreads of at most -1.5, exact bisection fixes D at half the distribution, and B receives the remainder of the loss-side target:

```
target_D = 0.5
target_B = loss - 0.5
```

At `s = -0.5`, no B or C region exists. The loss target takes precedence over exact bisection.

**Spread = 0** (pick-em):

```
Region:  A          T     D
Margins: -75..-1    0     1..75
Target:  1-wp-tp    tp    wp
```

The spread sits on the tie bin. No B or C — only the two flanking regions.

Exact bisection at zero would require `wp = (1 - tie_prob) / 2`. When the mapped win probability differs, win and tie probability take precedence.

### Why sub-divide around the spread

Without sub-division, a single scale factor is applied to the entire win (or loss) side. This preserves the total win probability but distorts the shape around the spread. A margin 2 points below the spread and a margin 20 points beyond the spread get the same multiplicative adjustment — which is wrong, because the continuous normal assigns very different density to those regions.

Independent scaling preserves the dirty PMF's proportions within B and within D while changing their relative totals. This keeps each sub-region's shape and makes the spread an exact bisection point wherever the discrete support permits all constraints to hold.

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
