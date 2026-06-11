'''
SpreadMap Module

Linear-in-logit mapping between win probability and spread.
'''

from .Types import LinearMapParams, Spread, SpreadMapResult
from .SpreadMapper import SpreadMapper

__all__ = [
    'LinearMapParams',
    'Spread',
    'SpreadMapResult',
    'SpreadMapper',
]
