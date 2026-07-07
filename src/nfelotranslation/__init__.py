'''
nfelotranslation — NFL Win Probability / Spread Translation

Translates model-derived win probabilities into empirically-grounded margin
distributions, from which spread, cover probability, push probability, and
expected value are derived coherently.

Pipeline:
    win_prob  →  [SpreadMapper]  →  spread
    (spread, win_prob)  →  [MarginDistributionModel]  →  margin distribution

Recalibration is training-only (see Calibration module).
    margin distribution  →  { spread, cover_prob, push_prob, EV }

Because spread is derived as the median of the margin distribution rather
than being an intermediate translation step, win probability and spread
are guaranteed self-consistent throughout the pipeline.

Model refitting and training pipelines live in the repository ``training/``
package (not installed from PyPI by default).
'''

__version__ = '0.2.1'

## calibration ##
from .Calibration import PlattParams, CalibrationResult, Recalibrator

## distribution ##
from .Distribution import (
    BaseDistribution,
    NumberOutcomeRecord, NumberOutcome, KeyModel,
    Normalizer,
    MarginDistribution, MarginDistributionModel,
)

## persistence helpers (config paths, envelopes) ##
from .Utilities.JsonIo import find_config_path

## spread map ##
from .SpreadMap import LinearMapParams, Spread, SpreadMapResult, SpreadMapper

## translation ##
from .Translation import Translator

__all__ = [
    ## calibration ##
    'PlattParams',
    'CalibrationResult',
    'Recalibrator',
    ## distribution — base ##
    'BaseDistribution',
    ## distribution — key ##
    'NumberOutcomeRecord',
    'NumberOutcome',
    'KeyModel',
    ## distribution — normalizer ##
    'Normalizer',
    ## distribution — composer ##
    'MarginDistribution',
    'MarginDistributionModel',
    ## persistence ##
    'find_config_path',
    ## spread map ##
    'LinearMapParams',
    'Spread',
    'SpreadMapResult',
    'SpreadMapper',
    ## translation ##
    'Translator',
]
