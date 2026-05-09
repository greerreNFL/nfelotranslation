'''
Calibration Module

Platt / logit-linear recalibration of market ML win probabilities.

Corrects the systematic long-shot bias in NFL betting markets: the market
overestimates slight favorites and underestimates big favorites.  The bias
is stable across eras, so a single static fit over the full ML era suffices.
'''

from .Types import PlattParams, CalibrationResult
from .Recalibrator import Recalibrator

__all__ = [
    'PlattParams',
    'CalibrationResult',
    'Recalibrator',
]
