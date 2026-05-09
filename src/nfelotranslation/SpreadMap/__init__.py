'''
SpreadMap Module

Linear-in-logit mapping between win probability and spread.

Two mapper instances exist — model (fitted against actual margins) and
market (fitted against market-posted lines).  Both use MAE loss
(median-targeting).  The distinction is the target variable, not the
loss function.  Each is represented by a separate SpreadMapper instance
with its own LinearMapParams.
'''

from .Types import MapType, LinearMapParams, Spread, SpreadMapResult
from .SpreadMapper import SpreadMapper

__all__ = [
    'MapType',
    'LinearMapParams',
    'Spread',
    'SpreadMapResult',
    'SpreadMapper',
]
