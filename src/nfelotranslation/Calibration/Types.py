'''
Calibration data structures.
'''

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class PlattParams:
    '''
    Parameters for Platt / logit-linear recalibration.

    The model maps market ML win prob → recalibrated win prob via:
        z     = log(p / (1 - p))           (logit transform)
        z_cal = slope * z + intercept      (linear correction on logit scale)
        p_cal = 1 / (1 + exp(-z_cal))      (back to probability)

    Interpretation:
    * slope > 1 — market hedges toward 50% (long-shot bias); recalibration
      stretches probabilities outward toward the extremes.
    * intercept ≈ 0 — no systematic directional bias.
    '''
    slope: float
    intercept: float

    def to_dict(self) -> Dict[str, float]:
        '''Serialize to dictionary'''
        return {'slope': self.slope, 'intercept': self.intercept}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlattParams':
        '''Deserialize from dictionary'''
        return cls(slope=float(data['slope']), intercept=float(data['intercept']))


@dataclass
class CalibrationResult:
    '''
    Diagnostic output from fitting or evaluating a Recalibrator.

    Contains the fitted parameters alongside before/after quality metrics,
    making it easy to verify that recalibration improves rather than hurts fit.
    Positive improvement values indicate the calibrated model is better.
    '''
    params: PlattParams
    n_games: int
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    seasons: List[int] = field(default_factory=list)

    @property
    def log_loss_improvement(self) -> float:
        '''Reduction in log loss after recalibration (positive = better)'''
        return self.log_loss_before - self.log_loss_after

    @property
    def brier_improvement(self) -> float:
        '''Reduction in Brier score after recalibration (positive = better)'''
        return self.brier_before - self.brier_after

    def summary(self) -> str:
        '''Human-readable summary of calibration quality'''
        lines = [
            f'PlattParams: slope={self.params.slope:.4f}  intercept={self.params.intercept:.4f}',
            f'n_games={self.n_games:,}',
            f'Log loss:  {self.log_loss_before:.6f} → {self.log_loss_after:.6f}'
            f'  (Δ={self.log_loss_improvement:+.6f})',
            f'Brier:     {self.brier_before:.6f} → {self.brier_after:.6f}'
            f'  (Δ={self.brier_improvement:+.6f})',
        ]
        if self.seasons:
            lines.append(f'Seasons:   {min(self.seasons)}–{max(self.seasons)}')
        return '\n'.join(lines)
