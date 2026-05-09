'''
Base Distribution Module

Continuous generalized normal foundation for the margin distribution.
Given (spread, win_prob, beta), derives the scale analytically and
provides the continuous density, CDF, and survival function consumed
by downstream modules.  Discretization and region-based normalization
live in the Normalizer module.
'''

from .BaseDistribution import BaseDistribution

__all__ = [
    'BaseDistribution',
]
