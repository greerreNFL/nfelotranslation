# 6. Within-Spread Price Signal

## Hypothesis

Within a single posted spread level, the no-vig favorite cover
probability implied by the spread's American prices is uninformative
about the actual game margin. If the price variation carried a margin
signal, games with a higher implied cover probability should produce a
higher median favorite margin and a higher favorite cover rate than
games with a lower implied cover probability at the same spread.

## Method

* Source data: 3,406 games across seasons 2009–2025 with closing
  favorite-side and underdog-side spread prices, a posted spread that
  agrees with the favorite assignment (`fav_spread > 0`), and a recorded
  margin.
* For each game:
  * `fav_cover_prob = american_to_prob(fav_price) /
     (american_to_prob(fav_price) + american_to_prob(dog_price))`.
  * `fav_spread_grid = round(fav_spread * 2) / 2` — the half-point grid
    actually posted.
* For each `fav_spread_grid` with at least 200 games, the games are split
  into terciles by `fav_cover_prob`. Reported per tercile: sample size,
  mean cover probability, median margin, and favorite cover rate
  (excluding pushes).
* Pooled view: `cover_prob` and `fav_margin` are demeaned within their
  posted spread; the slope of demeaned margin on demeaned cover
  probability summarizes the within-spread price → margin relationship.

## Findings

Eligible posted spreads (n ≥ 200): 2.5, 3.0, 3.5, 7.0.

Top tercile minus bottom tercile per spread:

| Spread | n low / high | Mean cp gap | Median margin gap | Cover rate gap |
|--------|--------------|-------------|-------------------|----------------|
| 2.5    | 115 / 89     | +0.0292     | −2.0000           | −0.1132        |
| 3.0    | 195 / 155    | +0.0386     | +0.0000           | −0.0670        |
| 3.5    | 127 / 101    | +0.0272     | +0.0000           | −0.0384        |
| 7.0    | 82 / 64      | +0.0239     | −1.0000           | +0.0032        |

* Mean median-margin gap across the four spreads: −0.7500 points.
* Mean cover-rate gap across the four spreads: −0.0538.
* Number of spreads with a positive median-margin gap: 0 / 4.
* Number of spreads with a positive cover-rate gap: 1 / 4.

Pooled within-spread regression of demeaned margin on demeaned cover
probability:

* Slope: 20.1111 points per unit `cover_prob`.
* Correlation: 0.0170.

## Conclusion

* The expected sign of the within-spread price signal is positive (a
  higher cover probability should accompany a higher favorite margin).
  Three of four eligible spreads produce a non-positive median-margin
  gap, three of four produce a negative cover-rate gap, and the pooled
  correlation is 0.017 — indistinguishable from zero on a sample of
  3,406 games.
* The pooled slope of 20.11 points per unit cover probability is large
  in magnitude but rests on a near-zero correlation; it is driven by a
  small number of sparsely populated demeaned bins and does not survive
  inspection of the per-spread terciles.
* Within a posted spread, the price variation around the standard
  −110 / −110 quote does not predict the conditional margin
  distribution. Treating posted spreads as the sufficient statistic and
  ignoring price-level juice is consistent with the data.
