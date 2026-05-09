'''
Distribution Module

Two-stage NFL margin distribution pipeline:

    Stage 1 — Base:  continuous generalized normal parameterized by
                      (spread, wp, beta).  Derives scale analytically.
                      Provides region mass targets (loss, win) that
                      downstream normalization must preserve.

    Stage 2 — Key:   per-integer credibility-weighted ratio trackers for
                      all integers 1-40.  Self-regularizing: strong signals
                      accumulate ratios above 1, noise stays near 1.
                      Multiplicative application via the baseline PMF —
                      no distance decay parameter needed.

    Composer — MarginDistributionModel:  composes Base + Key, discretizes
                      into integer bins, normalizes per-region to lock
                      constraints.

This module takes (spread, wp) as inputs and produces distributions /
derived probabilities (cover, push, PMF).  It does NOT handle wp↔spread
translation — that belongs to SpreadMapper.

Shipped hyperparameters are loaded at import from ``margin_hyperparams.json``
and exposed as ``MARGIN_HYPERPARAMS``.  Consumers (production defaults,
tests, downstream callers) read the same dict — no path handling outside
this module.
'''

## built-ins ##
import pathlib
from typing import Any, Dict

## local ##
from ..Utilities.JsonIo import read_config_envelope

## load shipped hyperparams once at import — single source of truth ##
_HYPER_PATH = pathlib.Path(__file__).resolve().parent / 'margin_hyperparams.json'
MARGIN_HYPERPARAMS: Dict[str, Any]
MARGIN_HYPERPARAMS, _ = read_config_envelope(str(_HYPER_PATH))

from .Base import BaseDistribution
from .Key import NumberOutcomeRecord, NumberOutcome, KeyModel
from .Normalizer import Normalizer
from .Types import MarginDistribution
from .MarginDistributionModel import MarginDistributionModel

__all__ = [
    ## shipped hyperparams ##
    'MARGIN_HYPERPARAMS',
    ## base ##
    'BaseDistribution',
    ## key ##
    'NumberOutcomeRecord',
    'NumberOutcome',
    'KeyModel',
    ## normalizer ##
    'Normalizer',
    ## composer ##
    'MarginDistribution',
    'MarginDistributionModel',
]
