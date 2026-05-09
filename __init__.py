'''
nfelotranslation — development bootstrap.

Redirects submodule resolution to src/nfelotranslation/ so the package
works directly from site-packages without pip install -e.

Also adds the package root to sys.path so ``import training`` resolves
to the sibling training/ directory.
'''

## built-ins ##
import os as _os
import sys as _sys

## ==================== Path Setup ==================== ##

_pkg_root = _os.path.dirname(_os.path.abspath(__file__))

## redirect nfelotranslation.* submodule lookups to src/nfelotranslation/ ##
__path__ = [_os.path.join(_pkg_root, 'src', 'nfelotranslation')]

## make training/ importable as a top-level package ##
if _pkg_root not in _sys.path:
    _sys.path.insert(0, _pkg_root)

## ==================== Re-export Public API ==================== ##

__version__ = '0.1.0'

## calibration ##
from nfelotranslation.Calibration import PlattParams, CalibrationResult, Recalibrator

## distribution ##
from nfelotranslation.Distribution import (
    BaseDistribution,
    NumberOutcomeRecord, NumberOutcome, KeyModel,
    Normalizer,
    MarginDistribution, MarginDistributionModel,
)

## persistence helpers ##
from nfelotranslation.Utilities.JsonIo import find_config_path

## spread map ##
from nfelotranslation.SpreadMap import MapType, LinearMapParams, Spread, SpreadMapResult, SpreadMapper

## translation ##
from nfelotranslation.Translation import Translator

__all__ = [
    'PlattParams', 'CalibrationResult', 'Recalibrator',
    'BaseDistribution',
    'NumberOutcomeRecord', 'NumberOutcome', 'KeyModel',
    'Normalizer',
    'MarginDistribution', 'MarginDistributionModel',
    'find_config_path',
    'MapType', 'LinearMapParams', 'Spread', 'SpreadMapResult', 'SpreadMapper',
    'Translator',
]
