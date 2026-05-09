'''
Recalibrator — Platt / logit-linear recalibration of market ML win probabilities.
'''

## built-ins ##
import pathlib
from typing import Optional

## external ##
import numpy

## local ##
from .Types import PlattParams
from ..Utilities.JsonIo import (
    ConfigMetadata,
    read_config_envelope,
    write_config_envelope,
)
from ..Utilities.MathUtils import logit, expit


## default path for persisted Platt parameters ##
_STATE_PATH: pathlib.Path = pathlib.Path(__file__).parent / 'platt_params.json'


class Recalibrator:
    '''
    Applies Platt / logit-linear recalibration to market ML win probabilities.

    Corrects the systematic miscalibration observed in NFL betting markets:
    the market overestimates slight favorites (~55–65%) and underestimates
    big favorites (>80%).

    The recalibration model:
    * z = logit(p_market)
    * z_cal = slope * z + intercept
    * p_cal = expit(z_cal)

    Usage:
        rec = Recalibrator.from_file()
        calibrated = rec.calibrate(new_win_probs)
        rec.to_file()
    '''

    def __init__(
        self,
        params: PlattParams,
        metadata: Optional[ConfigMetadata] = None,
    ):
        '''
        Initialize with parameters.

        Parameters:
        * params: PlattParams (slope, intercept)
        * metadata: ConfigMetadata describing the training run that
          produced these params (defaults to an empty record)
        '''
        self.params = params
        self.metadata = metadata or ConfigMetadata()

    ## ==================== Public Interface ==================== ##

    def calibrate(self, win_prob: numpy.ndarray) -> numpy.ndarray:
        '''
        Apply recalibration to an array of market ML win probabilities.

        Parameters:
        * win_prob: market-implied win probabilities in [0, 1]; any side
          (home, away, favorite) — caller is responsible for consistency

        Returns:
        * recalibrated win probabilities in (0, 1)
        '''
        z = logit(numpy.asarray(win_prob, dtype=float))
        return expit(self.params.slope * z + self.params.intercept)

    def uncalibrate(self, cal_wp: numpy.ndarray) -> numpy.ndarray:
        '''
        Invert the Platt recalibration to recover market win probabilities.

        Parameters:
        * cal_wp: calibrated win probabilities in (0, 1)

        Returns:
        * market-implied win probabilities in (0, 1)
        '''
        z_cal = logit(numpy.asarray(cal_wp, dtype=float))
        return expit((z_cal - self.params.intercept) / self.params.slope)

    def to_file(self, filepath: Optional[str] = None) -> None:
        '''
        Persist fitted parameters to JSON (defaults to package state path)
        '''
        path = str(filepath) if filepath is not None else str(_STATE_PATH)
        write_config_envelope(path, self.params.to_dict(), self.metadata)

    ## ==================== Factory Methods ==================== ##

    @classmethod
    def from_params(cls, slope: float, intercept: float) -> 'Recalibrator':
        '''
        Construct from known parameter values (e.g., loaded from a prior fit).

        Parameters:
        * slope: logit-linear slope coefficient
        * intercept: logit-linear intercept
        '''
        return cls(PlattParams(slope=slope, intercept=intercept))

    @classmethod
    def from_file(cls, filepath: Optional[str] = None) -> 'Recalibrator':
        '''
        Load a previously fitted Recalibrator from JSON (defaults to package state path)

        Parameters:
        * filepath: override path (defaults to package state path)

        Returns:
        * Recalibrator initialized with the stored parameters and metadata
        '''
        path = str(filepath) if filepath is not None else str(_STATE_PATH)
        payload, metadata = read_config_envelope(path)
        return cls(PlattParams.from_dict(payload), metadata=metadata)
