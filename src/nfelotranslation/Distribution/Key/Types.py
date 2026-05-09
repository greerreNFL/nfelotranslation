'''
Key number data structures — ratio-based tracker architecture.
'''

## built-ins ##
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class NumberOutcomeRecord:
    '''
    Snapshot of a NumberOutcome's state at a specific season.

    Stored pre-update so that `ratio` is the prediction for that season
    (i.e., using only data prior to this season).

    Parameters:
    * season - the season this record corresponds to
    * ratio - pre-update ratio (1.0 = no excess, >1 = key, <1 = anti-key)
    * eff_hits - exponentially decayed hit count at time of snapshot
    * exp_eff_hits - exponentially decayed expected hit count at time of snapshot
    * eff_games - exponentially decayed game count at time of snapshot
    '''
    season: int
    ratio: float
    eff_hits: float
    exp_eff_hits: float
    eff_games: float

    def to_dict(self) -> Dict[str, Any]:
        '''Serialize to dictionary.'''
        return {
            'season': self.season,
            'ratio': self.ratio,
            'eff_hits': self.eff_hits,
            'exp_eff_hits': self.exp_eff_hits,
            'eff_games': self.eff_games,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NumberOutcomeRecord':
        '''Deserialize from dictionary.'''
        return cls(
            season = int(data['season']),
            ratio = float(data['ratio']),
            eff_hits = float(data['eff_hits']),
            exp_eff_hits = float(data['exp_eff_hits']),
            eff_games = float(data['eff_games']),
        )
