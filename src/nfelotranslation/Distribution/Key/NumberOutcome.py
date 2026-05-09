'''
NumberOutcome — ratio-based tracker for a single margin integer.

Tracks all integers 1-40 uniformly.  Strong signals (3, 7) accumulate
ratios well above 1; weak signals self-regularize toward ratio=1 via
the credibility threshold.  Handles both positive excess (key numbers,
ratio > 1) and negative excess (dead zones, ratio < 1).

Ratio model:
  raw_ratio = eff_hits / exp_eff_hits
  credibility = min(1, exp_eff_hits / threshold)
  ratio = 1 + (raw_ratio - 1) * credibility

Application:
  excess_pos = (ratio - 1) * baseline_pos
  excess_neg = (ratio - 1) * baseline_neg

The result is denominated in excess probability so it can be added to
the existing PMF loop (raw_pmf[k+75] += ex_pos).  The math is equivalent
to scaling each bin by the ratio: baseline + (ratio-1)*baseline = ratio*baseline.

The form is multiplicative: one parameter per number, with side-splitting
and distance dependence handled entirely by the baseline PMF.
'''

## built-ins ##
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

## local ##
from .Types import NumberOutcomeRecord


@dataclass
class NumberOutcome:
    '''
    Ratio-based tracker for a single margin integer.

    Parameters:
    * number: the margin integer (1-40)
    * eff_hits: exponentially decayed hit count
    * exp_eff_hits: exponentially decayed expected hit count
    * eff_games: exponentially decayed game count
    * trained_through: most recent season ingested (None if no data)
    * history: list of per-season snapshots
    * params: injected config dict, excluded from serialization
    '''
    number: int
    eff_hits: float = 0.0
    exp_eff_hits: float = 0.0
    eff_games: float = 0.0
    trained_through: Optional[int] = None
    history: List[NumberOutcomeRecord] = field(default_factory=list)
    params: Optional[Dict[str, Any]] = field(default=None, repr=False)

    ## ==================== Public Interface ==================== ##

    def update(self, hits: float, n_games: float, baseline_rate: float, season: int) -> None:
        '''
        Ingest one season of data for this number.

        1. Initialize prior on first update (balanced — ratio = 1)
        2. Snapshot current state (pre-update ratio) to history
        3. Decay existing counts
        4. Add new season counts

        Parameters:
        * hits: count of games landing on ±k this season
        * n_games: total games this season
        * baseline_rate: aggregate baseline rate for ±k this season
        * season: season identifier
        '''
        ## initialize prior on first update ##
        if self.trained_through is None:
            prior = self._param('initial_prior_size')
            self.eff_hits = prior * baseline_rate
            self.exp_eff_hits = prior * baseline_rate
            self.eff_games = float(prior)
        ## snapshot pre-update ratio ##
        record = NumberOutcomeRecord(
            season=season,
            ratio=self.get_ratio(),
            eff_hits=self.eff_hits,
            exp_eff_hits=self.exp_eff_hits,
            eff_games=self.eff_games,
        )
        self.history.append(record)
        ## decay existing counts ##
        forget = self._param('forgetting_rate')
        self.eff_hits *= (1.0 - forget)
        self.exp_eff_hits *= (1.0 - forget)
        self.eff_games *= (1.0 - forget)
        ## add new season ##
        self.eff_hits += hits
        self.exp_eff_hits += baseline_rate * n_games
        self.eff_games += n_games
        ## update trained_through ##
        self.trained_through = season

    def get_ratio(self) -> float:
        '''
        Credibility-weighted ratio of hits to expected.

        raw_ratio = eff_hits / exp_eff_hits
        credibility = min(1, exp_eff_hits / threshold)
        ratio = 1 + (raw_ratio - 1) * credibility

        At zero credibility, ratio = 1.0 (no excess).
        At full credibility, ratio = raw_ratio.

        Returns:
        * ratio (1.0 = no excess, >1 = key number, <1 = dead zone)
        '''
        if self.eff_games == 0.0 or self.exp_eff_hits <= 0.0:
            return 1.0
        raw_ratio = self.eff_hits / self.exp_eff_hits
        threshold = self._param('threshold')
        if self.exp_eff_hits >= threshold:
            return raw_ratio
        credibility = self.exp_eff_hits / threshold
        return 1.0 + (raw_ratio - 1.0) * credibility

    def excess_at(self, baseline_pos: float, baseline_neg: float) -> Tuple[float, float]:
        '''
        Multiplicative excess at +k and -k given per-bin baselines.

        excess_pos = (ratio - 1) * baseline_pos
        excess_neg = (ratio - 1) * baseline_neg

        The result is denominated in excess probability — add it to the
        raw PMF bin to get the adjusted value: baseline + excess = ratio * baseline.

        This naturally handles:
        - Side splitting: each side gets excess proportional to its baseline
        - Distance dependence: distant bins have small baselines → small excess
        - No double-counting: total excess = (ratio - 1) * (base_pos + base_neg)

        Parameters:
        * baseline_pos: discretized baseline P(margin = +k)
        * baseline_neg: discretized baseline P(margin = -k)

        Returns:
        * (excess_pos, excess_neg)
        '''
        r_minus_1 = self.get_ratio() - 1.0
        return (r_minus_1 * baseline_pos, r_minus_1 * baseline_neg)

    def get_state_at(self, season: int) -> Optional[NumberOutcomeRecord]:
        '''
        Historical lookup for backtesting.

        Parameters:
        * season: season to look up

        Returns:
        * NumberOutcomeRecord for that season, or None if not found
        '''
        for record in self.history:
            if record.season == season:
                return record
        return None

    ## ==================== Serialization ==================== ##

    def to_dict(self, include_history: bool = True) -> Dict[str, Any]:
        '''
        Serialize to dictionary (params excluded — re-injected on load).

        Parameters:
        * include_history: if False, omit the per-season history list
        '''
        d: Dict[str, Any] = {
            'number': self.number,
            'eff_hits': self.eff_hits,
            'exp_eff_hits': self.exp_eff_hits,
            'eff_games': self.eff_games,
            'trained_through': self.trained_through,
        }
        if include_history:
            d['history'] = [r.to_dict() for r in self.history]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> 'NumberOutcome':
        '''Deserialize from dictionary with params re-injected.'''
        raw_tt = data.get('trained_through')
        return cls(
            number=int(data['number']),
            eff_hits=float(data['eff_hits']),
            exp_eff_hits=float(data['exp_eff_hits']),
            eff_games=float(data['eff_games']),
            trained_through=int(raw_tt) if raw_tt is not None else None,
            history=[NumberOutcomeRecord.from_dict(r) for r in data.get('history', [])],
            params=params,
        )

    ## ==================== Private ==================== ##

    def _param(self, key: str) -> Any:
        '''Strict param lookup — raises if missing.'''
        if self.params is None:
            raise ValueError(f'No params dict provided — cannot look up {key!r}')
        if key not in self.params:
            raise KeyError(f'Missing required param {key!r}')
        return self.params[key]
