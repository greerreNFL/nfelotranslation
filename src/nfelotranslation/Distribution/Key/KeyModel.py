'''
KeyModel — collection of 40 NumberOutcome trackers.

State container + accessor.  Holds one NumberOutcome per integer 1-40.
Does NOT compose PMFs or normalize — that is the MarginDistributionModel's
responsibility.  The Key module's job: "given my learned ratios and a
baseline PMF, what is the excess at each number?"

On-disk format — key_model.json is a config envelope:

    {
        "metadata": {"pipeline_id": "...", "generated_at": "..."},
        "params": {
            "1":  {...NumberOutcome...},
            "2":  {...NumberOutcome...},
            ...
            "40": {...NumberOutcome...}
        }
    }
'''

## built-ins ##
import pathlib
from typing import Any, Dict, Optional, Tuple

## external ##
import numpy

## local ##
from .NumberOutcome import NumberOutcome
from ...Utilities.JsonIo import (
    ConfigMetadata,
    find_config_path,
    read_config_envelope,
    write_config_envelope,
)


## default range of tracked integers ##
_MIN_NUMBER = 1
_MAX_NUMBER = 40

## default path for the root (latest) KeyModel config ##
_STATE_PATH: pathlib.Path = pathlib.Path(__file__).parent / 'key_model.json'

## per-season fitted configs.  find_config_path resolves season -> file
## with a warning fallback to the most recent prior season; from_file
## raises FileNotFoundError when the requested season predates the
## earliest config (out-of-domain ask). ##
_CONFIG_DIR: pathlib.Path = pathlib.Path(__file__).parent / 'configs'
_CONFIG_PREFIX: str = 'key_model'


class KeyModel:
    '''
    Collection of NumberOutcome trackers for integers 1-40.

    Usage:
        params = {'forgetting_rate': 0.087, 'threshold': 25,
                  'initial_prior_size': 52}
        model = KeyModel.from_initial(params)
        model.outcomes[3].update(hits=120, n_games=2500, baseline_rate=0.04, season=2023)
        excess_pos, excess_neg = model.excess_at(3, baseline_pmf)
    '''

    def __init__(
        self,
        outcomes: Dict[int, NumberOutcome],
        metadata: Optional[ConfigMetadata] = None,
    ):
        self.outcomes = outcomes
        self.metadata = metadata or ConfigMetadata()

    ## ==================== Public Interface ==================== ##

    def excess_at(self, number: int, baseline_pmf: numpy.ndarray) -> Tuple[float, float]:
        '''
        Excess for a number given the full baseline PMF.

        Extracts per-bin baselines and delegates to NumberOutcome.excess_at().

        Parameters:
        * number:       the margin integer (1-40)
        * baseline_pmf: 151-element discretized PMF

        Returns:
        * (excess_at_plus_k, excess_at_minus_k)

        Raises:
        * ValueError if number is not tracked
        '''
        if number not in self.outcomes:
            raise ValueError(f'{number} is not a tracked number (must be {_MIN_NUMBER}-{_MAX_NUMBER})')
        baseline_pos = float(baseline_pmf[number + 75])
        baseline_neg = float(baseline_pmf[-number + 75])
        return self.outcomes[number].excess_at(baseline_pos, baseline_neg)

    def get_all_excess(self, baseline_pmf: numpy.ndarray) -> Dict[int, Tuple[float, float]]:
        '''
        Excess for all 40 numbers given the full baseline PMF.

        Parameters:
        * baseline_pmf: 151-element discretized PMF

        Returns:
        * dict mapping number -> (excess_at_plus_k, excess_at_minus_k)
        '''
        result = {}
        for k, outcome in self.outcomes.items():
            baseline_pos = float(baseline_pmf[k + 75])
            baseline_neg = float(baseline_pmf[-k + 75])
            result[k] = outcome.excess_at(baseline_pos, baseline_neg)
        return result

    ## ==================== Config I/O ==================== ##

    def to_file(
        self,
        filepath: Optional[str] = None,
        include_history: bool = True,
    ) -> None:
        '''
        Save all NumberOutcome states to a JSON envelope.

        Parameters:
        * filepath: path to output file (defaults to package state path)
        * include_history: if False, omit per-season history from each outcome
        '''
        path = str(filepath) if filepath is not None else str(_STATE_PATH)
        payload = {
            str(k): outcome.to_dict(include_history=include_history)
            for k, outcome in self.outcomes.items()
        }
        write_config_envelope(path, payload, self.metadata)

    @classmethod
    def from_file(
        cls,
        filepath: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        *,
        season: Optional[int] = None,
    ) -> 'KeyModel':
        '''
        Load from JSON file with params re-injected into each NumberOutcome.

        Supports both the envelope format and the legacy flat format
        (raw {"1": {...}, ...} dict at top level).

        Resolution order:
        1. ``filepath`` if given.
        2. ``season`` if given — resolves via ``find_config_path``; falls back
           to the most recent prior season's file with a warning, or raises
           ``FileNotFoundError`` if no config exists for or before ``season``.
        3. Otherwise, the package root snapshot ``key_model.json``.

        Parameters:
        * filepath: explicit override (highest precedence)
        * params:   config dict to inject into each NumberOutcome
        * season:   target season, used only when ``filepath`` is None

        Returns:
        * KeyModel with restored state

        Raises:
        * FileNotFoundError: if ``season`` is given and no per-season config
          exists for that season or any earlier season.
        '''
        if filepath is not None:
            path = str(filepath)
        elif season is not None:
            resolved = find_config_path(_CONFIG_PREFIX, season, str(_CONFIG_DIR))
            if resolved is None:
                raise FileNotFoundError(
                    f'No {_CONFIG_PREFIX} config available for season {season} '
                    f'or any earlier season under {_CONFIG_DIR}'
                )
            path = resolved
        else:
            path = str(_STATE_PATH)
        payload, metadata = read_config_envelope(path)
        outcomes = {
            int(k): NumberOutcome.from_dict(v, params=params)
            for k, v in payload.items()
        }
        return cls(outcomes, metadata=metadata)

    ## ==================== Factory ==================== ##

    @classmethod
    def from_initial(cls, params: Optional[Dict[str, Any]] = None) -> 'KeyModel':
        '''
        Create a fresh, untrained model with 40 zero-state NumberOutcomes.

        Used by training (initial state before the first season's update)
        and by the OOS-warmup branch of MarginDistributionValidator (where
        the no-key-signal fallback is the deliberate, correct choice).
        Inference paths should NOT call this — they should use ``from_file``
        with a ``filepath`` or ``season`` argument.

        Parameters:
        * params: config dict injected into each NumberOutcome

        Returns:
        * KeyModel with all outcomes at zero state
        '''
        outcomes = {
            k: NumberOutcome(number=k, params=params)
            for k in range(_MIN_NUMBER, _MAX_NUMBER + 1)
        }
        return cls(outcomes)
