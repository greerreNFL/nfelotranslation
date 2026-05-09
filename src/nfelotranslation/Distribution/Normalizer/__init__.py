'''
Normalizer Module

Discretizes a continuous BaseDistribution into integer bins and normalizes
per-region to restore win probability, tie probability, and spread bisection
constraints.

The normalization operates on five labelled regions (A/T/B/C/D) defined by
two landmarks in the margin space: zero (win/loss boundary) and the spread
(bisection point).  See Normalizer.py and README.md for the full region
layout.
'''

from .Normalizer import Normalizer

__all__ = [
    'Normalizer',
]
