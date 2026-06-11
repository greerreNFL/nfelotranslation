'''
SpreadMap data structures.
'''

from dataclasses import dataclass
from typing import Dict, Any
import json
import pathlib


@dataclass
class LinearMapParams:
    '''
    Parameters for linear-in-logit spread mapping.

    The model maps win probability → spread via:
        z      = logit(win_prob)
        spread = slope * z + intercept

    Semantically distinct from PlattParams (which maps logit-WP → logit-WP);
    this maps logit-WP → spread in points.

    Interpretation:
    * slope — points of spread per unit of logit. With slope ≈ 6 to 7, each
      unit of logit (≈ 10pp of WP at the center) corresponds to about 6 to 7
      points of spread.
    * intercept — the spread at 50% win probability. The shipped mapper is
      fit with intercept fixed at 0 to anchor the mapping at WP=0.50 ↔ spread=0.
    '''
    slope: float
    intercept: float

    def to_dict(self) -> Dict[str, float]:
        '''Serialize to dictionary'''
        return {'slope': self.slope, 'intercept': self.intercept}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinearMapParams':
        '''Deserialize from dictionary'''
        return cls(slope=float(data['slope']), intercept=float(data['intercept']))


@dataclass
class Spread:
    '''
    A spread value with both posted (clamped to 0.5 grid) and continuous forms.

    posted    — rounded to nearest 0.5, suitable for display and comparison
                with market lines.
    continuous — raw formula output, used for lossless round-trip precision
                 when feeding back into spread_to_win_prob().
    '''
    posted: float
    continuous: float

    def __repr__(self) -> str:
        return f'Spread(posted={self.posted}, continuous={self.continuous:.4f})'


@dataclass
class SpreadMapResult:
    '''
    Diagnostic output from evaluating a SpreadMapper on a dataset.

    Contains fitted parameters alongside quality metrics for assessing
    how well the mapping captures the relationship between win probability
    and game margin.
    '''
    params: LinearMapParams
    n_games: int
    mae: float
    bisection_rate: float

    def summary(self) -> str:
        '''Human-readable summary of spread-map quality'''
        lines = [
            f'LinearMapParams: slope={self.params.slope:.4f}  intercept={self.params.intercept:.4f}',
            f'n_games={self.n_games:,}',
            f'MAE:             {self.mae:.4f}',
            f'Bisection rate:  {self.bisection_rate:.4f}',
        ]
        return '\n'.join(lines)
