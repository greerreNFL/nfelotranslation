'''
SpreadMapper — linear-in-logit mapping between win probability and spread.

Sign convention:
    positive spread = favorite is expected to win by that many points.
    This is the unified package-wide convention: matches the sign of margin
    (positive = home/favorite won), matches the DataLoader's spread_line,
    and matches the Translator's input/output conventions for `spread`.

Maps win probabilities to spreads using a parametric linear-in-logit model:

    spread = slope * logit(win_prob) + intercept

Fitted on recalibrated training labels (ml_wp_cal) vs actual margins with
intercept fixed at 0 at fit time.

On-disk format — spread_map_params.json is a config envelope:

    {
        "metadata": {"pipeline_id": "...", "generated_at": "..."},
        "params": {"slope": ..., "intercept": ...}
    }
'''

## built-ins ##
import pathlib
from typing import Optional, Union

## external ##
import numpy

## local ##
from .Types import LinearMapParams, Spread
from ..Utilities.JsonIo import (
    ConfigMetadata,
    find_config_path,
    read_config_envelope,
    write_config_envelope,
)
from ..Utilities.MathUtils import logit, expit, clip_prob


## default path for persisted spread-map parameters ##
_STATE_PATH: pathlib.Path = pathlib.Path(__file__).parent / 'spread_map_params.json'

## per-season fitted configs.  find_config_path resolves season -> file
## with a warning fallback to the most recent prior season; from_file
## raises FileNotFoundError when the requested season predates the
## earliest config (out-of-domain ask). ##
_CONFIG_DIR: pathlib.Path = pathlib.Path(__file__).parent / 'configs'
_CONFIG_PREFIX: str = 'spread_map_params'


class SpreadMapper:
    '''
    Applies linear-in-logit spread mapping.

    Translates between win probability and spread.  The mapping:

        spread = slope * logit(wp) + intercept
        wp     = expit((spread - intercept) / slope)

    is analytically invertible, monotonic, and generalizes better than
    lookup tables at sparse tails (analysis 09).

    Usage:
        mapper = SpreadMapper.from_file(season=2025)
        spread = mapper.win_prob_to_spread(0.60)
        wp     = mapper.spread_to_win_prob(spread.continuous)
    '''

    def __init__(
        self,
        params: LinearMapParams,
        metadata: Optional[ConfigMetadata] = None,
    ):
        '''
        Initialize with pre-fitted parameters.

        Parameters:
        * params: LinearMapParams (slope, intercept)
        * metadata: ConfigMetadata from the file that produced these params
          (defaults to an empty record)
        '''
        self.params = params
        self.metadata = metadata or ConfigMetadata()

    ## ==================== Public Interface ==================== ##

    def win_prob_to_spread(self, win_prob: Union[float, numpy.ndarray]) -> Spread:
        '''
        Map win probability to spread.

        Accepts a single float or a numpy array; returns Spread fields
        matching the input shape (scalar in → scalar out, array in → array out).

        Parameters:
        * win_prob: win probability in (0, 1) ## positive = favorite

        Returns:
        * Spread ## positive = favorite expected margin
        '''
        wp = clip_prob(numpy.asarray(win_prob, dtype=float))
        z = logit(wp)
        continuous = self.params.slope * z + self.params.intercept
        posted = _clamp_half(continuous)
        return Spread(posted=posted, continuous=continuous)

    def spread_to_win_prob(self, spread: Union[float, numpy.ndarray]) -> Union[float, numpy.ndarray]:
        '''
        Map spread back to win probability (inverse of win_prob_to_spread).

        Accepts a single float or a numpy array; returns the same shape.

        Parameters:
        * spread: continuous spread value ## positive = favorite expected margin

        Returns:
        * win probability in (0, 1)
        '''
        s = numpy.asarray(spread, dtype=float)
        return expit((s - self.params.intercept) / self.params.slope)

    def to_file(self, filepath: Optional[str] = None) -> None:
        '''
        Persist parameters to JSON.

        Parameters:
        * filepath: override path (defaults to package state path)
        '''
        path = str(filepath) if filepath is not None else str(_STATE_PATH)
        write_config_envelope(path, self.params.to_dict(), self.metadata)

    ## ==================== Factory Methods ==================== ##

    @classmethod
    def from_file(
        cls,
        filepath: Optional[str] = None,
        *,
        season: Optional[int] = None,
    ) -> 'SpreadMapper':
        '''
        Load a SpreadMapper from JSON.

        Resolution order:
        1. ``filepath`` if given.
        2. ``season`` if given — resolves via ``find_config_path``; falls back
           to the most recent prior season's file with a warning, or raises
           ``FileNotFoundError`` if no config exists for or before ``season``.
        3. Otherwise, the package root snapshot ``spread_map_params.json``.

        Parameters:
        * filepath: explicit override (highest precedence)
        * season:   target season, used only when ``filepath`` is None

        Returns:
        * SpreadMapper initialized with the stored parameters and metadata

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
        params = _parse_params(payload)
        return cls(params, metadata=metadata)

    @classmethod
    def from_params(cls, slope: float, intercept: float) -> 'SpreadMapper':
        '''
        Construct from known parameter values.

        Parameters:
        * slope: logit-linear slope coefficient
        * intercept: logit-linear intercept
        '''
        return cls(LinearMapParams(slope=slope, intercept=intercept))


## ==================== Private ==================== ##


def _parse_params(payload: dict) -> LinearMapParams:
    '''Accept current single-param envelope or legacy model/market envelope.'''
    if 'slope' in payload:
        return LinearMapParams.from_dict(payload)
    if 'model' in payload:
        return LinearMapParams.from_dict(payload['model'])
    raise KeyError('SpreadMapper config must contain slope/intercept or legacy model key')


def _clamp_half(x: numpy.ndarray) -> numpy.ndarray:
    '''Round to nearest 0.5 (e.g. 2.7 → 2.5, 2.8 → 3.0)'''
    return numpy.round(numpy.asarray(x, dtype=float) * 2.0) / 2.0
