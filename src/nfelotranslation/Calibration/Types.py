'''
Calibration data structures.
'''

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SplitPlattParams:
    '''
    Split Platt parameters for home-favorite vs away-favorite calibration.

    The model maps market ML win prob → true win prob via:
        z     = logit(p_market)
        z_cal = slope * z + intercept   (per location)
        p_cal = expit(z_cal)

    Slopes are fit once on the full sample. Intercepts vary by season config
    file when using centered-window training labels.
    '''
    slopes: Dict[str, float]
    intercepts: Dict[str, float]
    fit: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'slopes': {k: float(v) for k, v in self.slopes.items()},
            'intercepts': {k: float(v) for k, v in self.intercepts.items()},
        }
        if self.fit is not None:
            fit = dict(self.fit)
            if 'seasons_used' in fit:
                fit['seasons_used'] = [int(s) for s in fit['seasons_used']]
            if 'label_season' in fit:
                fit['label_season'] = int(fit['label_season'])
            out['fit'] = fit
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitPlattParams':
        return cls(
            slopes={k: float(v) for k, v in data['slopes'].items()},
            intercepts={k: float(v) for k, v in data['intercepts'].items()},
            fit=data.get('fit'),
        )


## backward alias for diagnostics that reference a single param record ##
PlattParams = SplitPlattParams


@dataclass
class CalibrationResult:
    '''
    Diagnostic output from fitting or evaluating a Recalibrator.
    '''
    params: SplitPlattParams
    n_games: int
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    seasons: List[int] = field(default_factory=list)

    @property
    def log_loss_improvement(self) -> float:
        return self.log_loss_before - self.log_loss_after

    @property
    def brier_improvement(self) -> float:
        return self.brier_before - self.brier_after

    def summary(self) -> str:
        lines = [
            'SplitPlattParams:',
            f'  slopes:     home={self.params.slopes["home"]:.4f}  '
            f'away={self.params.slopes["away"]:.4f}',
            f'  intercepts: home={self.params.intercepts["home"]:+.4f}  '
            f'away={self.params.intercepts["away"]:+.4f}',
            f'n_games={self.n_games:,}',
            f'Log loss:  {self.log_loss_before:.6f} → {self.log_loss_after:.6f}'
            f'  (Δ={self.log_loss_improvement:+.6f})',
            f'Brier:     {self.brier_before:.6f} → {self.brier_after:.6f}'
            f'  (Δ={self.brier_improvement:+.6f})',
        ]
        if self.seasons:
            lines.append(f'Seasons:   {min(self.seasons)}–{max(self.seasons)}')
        return '\n'.join(lines)
