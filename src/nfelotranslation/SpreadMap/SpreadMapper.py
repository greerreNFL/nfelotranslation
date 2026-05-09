'''
SpreadMapper — linear-in-logit mapping between win probability and spread.

Sign convention:
    positive spread = favorite is expected to win by that many points.
    This is the unified package-wide convention: matches the sign of margin
    (positive = home/favorite won), matches the DataLoader's spread_line,
    and matches the Translator's input/output conventions for both
    `spread` and `market_spread`.

Maps win probabilities to spreads using a parametric linear-in-logit model:

    spread = slope * logit(win_prob) + intercept

Two instances are needed in practice:
    * MODEL  mapper — fitted against actual game margins
    * MARKET mapper — fitted against market-posted lines

Both use MAE loss (median-targeting).  The distinction is the target variable,
not the loss function.

On-disk format — spread_map_params.json is a config envelope:

    {
        "metadata": {"pipeline_id": "...", "generated_at": "..."},
        "params": {
            "model":  {"slope": ..., "intercept": ...},
            "market": {"slope": ..., "intercept": ...}
        }
    }

Both mappers live in the same file; callers typically save them as a pair
via save_mapper_pair() and load a single mapper via SpreadMapper.from_file().
'''

## built-ins ##
import pathlib
from typing import Optional, Tuple, Union

## external ##
import numpy

## local ##
from .Types import MapType, LinearMapParams, Spread
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

    Translates between win probability and spread in a single coordinate
    system (model or market).  The mapping:

        spread = slope * logit(wp) + intercept
        wp     = expit((spread - intercept) / slope)

    is analytically invertible, monotonic, and generalizes better than
    lookup tables at sparse tails (analysis 09).

    Usage:
        mapper = SpreadMapper.from_file(MapType.MODEL)
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

    def to_file(self, map_type: MapType, filepath: Optional[str] = None) -> None:
        '''
        Persist parameters to JSON, merging into the shared envelope under
        the map_type key.  Preserves whatever other map_type is already in
        the file along with its existing metadata (bulk writes should use
        save_mapper_pair() instead to stamp fresh metadata).

        Parameters:
        * map_type: MapType.MODEL or MapType.MARKET
        * filepath: override path (defaults to package state path)
        '''
        path = str(filepath) if filepath is not None else str(_STATE_PATH)
        payload, metadata = _read_or_empty(path)
        payload[map_type.value] = self.params.to_dict()
        write_config_envelope(path, payload, metadata)

    ## ==================== Factory Methods ==================== ##

    @classmethod
    def from_file(
        cls,
        map_type: MapType,
        filepath: Optional[str] = None,
        *,
        season: Optional[int] = None,
    ) -> 'SpreadMapper':
        '''
        Load a SpreadMapper from JSON, selecting the map_type sub-dict.

        Resolution order:
        1. ``filepath`` if given.
        2. ``season`` if given — resolves via ``find_config_path``; falls back
           to the most recent prior season's file with a warning, or raises
           ``FileNotFoundError`` if no config exists for or before ``season``.
        3. Otherwise, the package root snapshot ``spread_map_params.json``.

        Parameters:
        * map_type: MapType.MODEL or MapType.MARKET
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
        return cls(LinearMapParams.from_dict(payload[map_type.value]), metadata=metadata)

    @classmethod
    def from_params(cls, slope: float, intercept: float) -> 'SpreadMapper':
        '''
        Construct from known parameter values.

        Parameters:
        * slope: logit-linear slope coefficient
        * intercept: logit-linear intercept
        '''
        return cls(LinearMapParams(slope=slope, intercept=intercept))


## ==================== Pair IO ==================== ##


def save_mapper_pair(
    model_mapper: SpreadMapper,
    market_mapper: SpreadMapper,
    metadata: ConfigMetadata,
    filepath: Optional[str] = None,
) -> str:
    '''
    Write both mappers to a single envelope file atomically.

    Parameters:
    * model_mapper: fitted model-coordinate SpreadMapper
    * market_mapper: fitted market-coordinate SpreadMapper
    * metadata: ConfigMetadata to stamp on the envelope
    * filepath: override path (defaults to package state path)

    Returns:
    * path to the written file
    '''
    path = str(filepath) if filepath is not None else str(_STATE_PATH)
    payload = {
        MapType.MODEL.value: model_mapper.params.to_dict(),
        MapType.MARKET.value: market_mapper.params.to_dict(),
    }
    write_config_envelope(path, payload, metadata)
    ## mirror the written metadata onto the mapper instances ##
    model_mapper.metadata = metadata
    market_mapper.metadata = metadata
    return path


def load_mapper_pair(
    filepath: Optional[str] = None,
) -> Tuple[SpreadMapper, SpreadMapper, ConfigMetadata]:
    '''
    Read both mappers and shared metadata from a single envelope file.

    Parameters:
    * filepath: override path (defaults to package state path)

    Returns:
    * (model_mapper, market_mapper, metadata) tuple
    '''
    path = str(filepath) if filepath is not None else str(_STATE_PATH)
    payload, metadata = read_config_envelope(path)
    model = SpreadMapper(LinearMapParams.from_dict(payload[MapType.MODEL.value]), metadata=metadata)
    market = SpreadMapper(LinearMapParams.from_dict(payload[MapType.MARKET.value]), metadata=metadata)
    return model, market, metadata


## ==================== Private ==================== ##


def _read_or_empty(path: str) -> Tuple[dict, ConfigMetadata]:
    '''Read an existing envelope, or return empty payload + metadata if missing.'''
    if not pathlib.Path(path).exists():
        return {}, ConfigMetadata()
    return read_config_envelope(path)


def _clamp_half(x: numpy.ndarray) -> numpy.ndarray:
    '''Round to nearest 0.5 (e.g. 2.7 → 2.5, 2.8 → 3.0)'''
    return numpy.round(numpy.asarray(x, dtype=float) * 2.0) / 2.0
