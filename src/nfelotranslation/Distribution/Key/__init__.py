'''
Key Number Module

Tracks per-integer excess above baseline using credibility-weighted
trackers.  All integers 1-40 are tracked uniformly — strong signals
(3, 7) accumulate real excess while noise numbers self-regularize
toward 0 via the credibility threshold.

Handles both positive excess (key numbers like ±3, ±7) and negative
excess (dead zones like ±9, ±12).

Default NumberOutcome hyperparameters are loaded from ``key_hyperparams.json``
(same envelope format as other shipped configs).
'''

## built-ins ##
import pathlib
from typing import Any, Dict

## local ##
from ...Utilities.JsonIo import read_config_envelope
from .Types import NumberOutcomeRecord
from .NumberOutcome import NumberOutcome
from .KeyModel import KeyModel

_HYPER_PATH = pathlib.Path(__file__).resolve().parent / 'key_hyperparams.json'
KEY_MODEL_PARAMS: Dict[str, Any]
KEY_MODEL_PARAMS, _ = read_config_envelope(str(_HYPER_PATH))


__all__ = [
    'NumberOutcomeRecord',
    'NumberOutcome',
    'KeyModel',
    'KEY_MODEL_PARAMS',
]
