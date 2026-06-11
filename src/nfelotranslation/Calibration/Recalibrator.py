'''
Recalibrator — split Platt recalibration of market ML win probabilities.

Used to derive true win probabilities for training labels and retrospective
analysis. Not composed into the inference Translator.
'''

## built-ins ##
import pathlib
from typing import Optional, Union

## external ##
import numpy

## local ##
from .Types import SplitPlattParams
from ..Utilities.JsonIo import (
    ConfigMetadata,
    find_config_path,
    read_config_envelope,
    write_config_envelope,
)
from ..Utilities.MathUtils import logit, expit, clip_prob


_STATE_PATH: pathlib.Path = pathlib.Path(__file__).parent / 'platt_params.json'
_CONFIG_DIR: pathlib.Path = pathlib.Path(__file__).parent / 'configs'
_CONFIG_PREFIX: str = 'platt_params'


class Recalibrator:
    '''
    Split Platt recalibration: home favorites and away favorites use
    separate intercepts; slopes are shared structural parameters.

        z     = logit(p_market)
        p_cal = expit(a_loc * z + b_loc)

    Parameters:
    * params: SplitPlattParams (slopes + intercepts for home/away)
    * metadata: ConfigMetadata from the training run
    '''

    def __init__(
        self,
        params: SplitPlattParams,
        metadata: Optional[ConfigMetadata] = None,
    ):
        self.params = params
        self.metadata = metadata or ConfigMetadata()

    def calibrate(
        self,
        win_prob: numpy.ndarray,
        is_home_fav: Optional[Union[numpy.ndarray, bool]] = None,
    ) -> numpy.ndarray:
        '''
        Apply recalibration to market ML win probabilities.

        Parameters:
        * win_prob: probabilities in (0, 1). When ``is_home_fav`` is omitted,
          each value is treated as home-perspective WP and location is inferred
          as home favorite when wp >= 0.5.
        * is_home_fav: optional bool or array — True for home favorites.
          When provided, ``win_prob`` must be from the favorite's perspective.

        Returns:
        * recalibrated win probabilities in (0, 1), same perspective as input
        '''
        wp = clip_prob(numpy.asarray(win_prob, dtype=float))
        z = logit(wp)
        if is_home_fav is None:
            home_fav = wp >= 0.5
        else:
            home_fav = numpy.asarray(is_home_fav, dtype=bool)
        a = numpy.where(home_fav, self.params.slopes['home'], self.params.slopes['away'])
        b = numpy.where(
            home_fav,
            self.params.intercepts['home'],
            self.params.intercepts['away'],
        )
        return expit(a * z + b)

    def uncalibrate(
        self,
        cal_wp: numpy.ndarray,
        is_home_fav: Optional[Union[numpy.ndarray, bool]] = None,
    ) -> numpy.ndarray:
        '''
        Invert split Platt recalibration.

        Parameters:
        * cal_wp: recalibrated win probabilities in (0, 1)
        * is_home_fav: same convention as ``calibrate``

        Returns:
        * market-implied win probabilities in (0, 1)
        '''
        p = clip_prob(numpy.asarray(cal_wp, dtype=float))
        z_cal = logit(p)
        if is_home_fav is None:
            home_fav = p >= 0.5
        else:
            home_fav = numpy.asarray(is_home_fav, dtype=bool)
        a = numpy.where(home_fav, self.params.slopes['home'], self.params.slopes['away'])
        b = numpy.where(
            home_fav,
            self.params.intercepts['home'],
            self.params.intercepts['away'],
        )
        return expit((z_cal - b) / a)

    def to_file(self, filepath: Optional[str] = None) -> None:
        path = str(filepath) if filepath is not None else str(_STATE_PATH)
        write_config_envelope(path, self.params.to_dict(), self.metadata)

    @classmethod
    def from_params(
        cls,
        slope_home: float,
        slope_away: float,
        intercept_home: float,
        intercept_away: float,
        fit: Optional[dict] = None,
    ) -> 'Recalibrator':
        return cls(
            SplitPlattParams(
                slopes={'home': slope_home, 'away': slope_away},
                intercepts={'home': intercept_home, 'away': intercept_away},
                fit=fit,
            )
        )

    @classmethod
    def from_file(
        cls,
        filepath: Optional[str] = None,
        *,
        season: Optional[int] = None,
    ) -> 'Recalibrator':
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
        return cls(SplitPlattParams.from_dict(payload), metadata=metadata)
