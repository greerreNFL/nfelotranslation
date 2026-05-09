'''
RecalibratorFitter — fits and evaluates Platt / logit-linear recalibration.

Extends Fitter with Platt scaling.  Provides both instance methods
(fit/diagnostics using self.games) and static array-level methods
(fit_from_arrays/diagnostics_from_arrays) for direct usage.
'''

## built-ins ##
import pathlib
from typing import List, Optional

## external ##
import numpy
import pandas
from scipy.optimize import minimize

import nfelotranslation

## local ##
from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Calibration.Types import CalibrationResult, PlattParams
from nfelotranslation.Utilities.JsonIo import ConfigMetadata
from nfelotranslation.Utilities.ValidationTypes import ValidationReport
from nfelotranslation.Utilities.MathUtils import brier_score, clip_prob, expit, log_loss, logit
from training.Base.Fitter import Fitter

## ==================== Constants ==================== ##

_MODULE_DIR = str(pathlib.Path(nfelotranslation.__file__).resolve().parent / 'Calibration')


class RecalibratorFitter(Fitter):
    '''
    Fits a Recalibrator from data and evaluates calibration quality.

    Instance usage (auto-loads data):
        fitter = RecalibratorFitter()
        rec = fitter.fit()
        result = fitter.diagnostics()

    Static usage (bring your own arrays):
        rec = RecalibratorFitter.fit_from_arrays(win_probs, outcomes)
        result = RecalibratorFitter.diagnostics_from_arrays(rec, win_probs, outcomes)
    '''

    def __init__(
        self,
        games: pandas.DataFrame = None,
        pipeline_id: Optional[str] = None,
    ):
        super().__init__(games=games, pipeline_id=pipeline_id)

    ## ==================== Fitter Interface ==================== ##

    @property
    def model_name(self) -> str:
        return 'recalibrator'

    @property
    def output_dir(self) -> str:
        return _MODULE_DIR

    def prepare_data(self):
        '''
        Extract (win_probs, outcomes) from self.games in favorite perspective.

        Folds all games to favorite perspective (win_prob >= 50%) so the
        Platt fit isolates the market's compression-toward-50% bias without
        muddling it with home/away dynamics.

        Returns:
        * tuple of (win_probs, outcomes) numpy arrays
        '''
        games = self.games
        ## filter to games with ML win probs and decisive results ##
        mask = games['ml_wp_close'].notna() & games['result'].notna()
        df = games[mask].copy()
        ## exclude ties for binary calibration ##
        df = df[df['result'] != 0]
        ## fold to favorite perspective ##
        home_wp = df['ml_wp_close'].values.astype(float)
        home_won = (df['result'] > 0).values.astype(float)
        is_home_fav = home_wp >= 0.5
        win_probs = numpy.where(is_home_fav, home_wp, 1.0 - home_wp)
        outcomes = numpy.where(is_home_fav, home_won, 1.0 - home_won)
        return win_probs, outcomes

    def fit(self) -> Recalibrator:
        '''
        Fit a Recalibrator from self.games via maximum likelihood.

        Persists the fitted parameters to the canonical config path
        (stamped with self.pipeline_id) as a side effect, so downstream
        fitters in the same training run can pick them up.

        Returns:
        * fitted Recalibrator
        '''
        win_probs, outcomes = self.prepare_data()
        self._fitted_model = self.fit_from_arrays(win_probs, outcomes)
        self.save_config()
        return self._fitted_model

    def diagnostics(self, recalibrator: Optional[Recalibrator] = None) -> CalibrationResult:
        '''
        Evaluate calibration quality on self.games.

        Parameters:
        * recalibrator: optional fitted Recalibrator (uses self.fitted_model if omitted)

        Returns:
        * CalibrationResult with before/after log loss and Brier score
        '''
        recalibrator = recalibrator or self._ensure_fitted()
        win_probs, outcomes = self.prepare_data()
        seasons = sorted(self.games['season'].unique().tolist())
        return self.diagnostics_from_arrays(recalibrator, win_probs, outcomes, seasons)

    ## ==================== Validation ==================== ##

    def validate(self) -> ValidationReport:
        '''
        Run LOSO cross-validation and stationarity checks.

        Delegates to RecalibratorValidator for the actual validation logic.

        Returns:
        * ValidationReport with checks and metrics
        '''
        self._ensure_fitted()
        from training.Validation import RecalibratorValidator
        validator = RecalibratorValidator(
            games=self.games,
            fitted_model=self.fitted_model,
        )
        return validator.validate()

    ## ==================== Config Persistence ==================== ##

    def save_config(self) -> str:
        '''
        Persist the fitted Recalibrator to its canonical config path,
        stamped with a ConfigMetadata envelope carrying self.pipeline_id.

        Returns:
        * path to the written file
        '''
        model = self._ensure_fitted()
        model.metadata = ConfigMetadata.new(pipeline_id=self._pipeline_id)
        model.to_file()
        return str(pathlib.Path(_MODULE_DIR) / 'platt_params.json')

    ## ==================== Static Methods ==================== ##

    @staticmethod
    def fit_from_arrays(
        win_probs: numpy.ndarray,
        outcomes: numpy.ndarray,
    ) -> Recalibrator:
        '''
        Fit a Recalibrator from arrays via maximum likelihood.

        Parameters:
        * win_probs: market ML win probabilities (any consistent side)
        * outcomes: binary outcomes for the same side (1 = win, 0 = loss or tie excluded)

        Returns:
        * fitted Recalibrator
        '''
        params = _fit(
            numpy.asarray(win_probs, dtype=float),
            numpy.asarray(outcomes, dtype=float),
        )
        return Recalibrator(params)

    @staticmethod
    def diagnostics_from_arrays(
        recalibrator: Recalibrator,
        win_probs: numpy.ndarray,
        outcomes: numpy.ndarray,
        seasons: Optional[List[int]] = None,
    ) -> CalibrationResult:
        '''
        Evaluate calibration quality on arrays.

        Parameters:
        * recalibrator: fitted Recalibrator instance
        * win_probs: market ML win probabilities
        * outcomes: binary game outcomes matching the win_prob perspective (1 = win)
        * seasons: optional list of seasons for record-keeping

        Returns:
        * CalibrationResult with before/after log loss and Brier score
        '''
        wp  = numpy.asarray(win_probs, dtype=float)
        out = numpy.asarray(outcomes, dtype=float)
        cal = recalibrator.calibrate(wp)
        return CalibrationResult(
            params=recalibrator.params,
            n_games=len(wp),
            log_loss_before=log_loss(out, wp),
            log_loss_after=log_loss(out, cal),
            brier_before=brier_score(out, wp),
            brier_after=brier_score(out, cal),
            seasons=sorted(seasons) if seasons is not None else [],
        )


## ==================== Private ==================== ##

def _fit(win_probs: numpy.ndarray, outcomes: numpy.ndarray) -> PlattParams:
    '''
    Fit logit-linear recalibration parameters via MLE.

    Minimizes binary cross-entropy over the (win_prob, outcome) pairs.
    Uses Nelder-Mead with tight tolerances; raises on non-convergence
    rather than silently returning a bad fit.

    Parameters:
    * win_probs: market ML win probabilities
    * outcomes: binary outcomes

    Returns:
    * fitted PlattParams
    '''
    z = logit(win_probs)
    y = outcomes
    ## objective function ##
    def nll(params: numpy.ndarray) -> float:
        ## negative log-likelihood (binary cross-entropy) ##
        p = clip_prob(expit(params[0] * z + params[1]))
        return float(-numpy.sum(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))
    ## optimization ##
    result = minimize(
        nll,
        x0=numpy.array([1.0, 0.0]),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10_000},
    )
    if not result.success:
        raise RuntimeError(
            f'Platt scaling optimization did not converge: {result.message}'
        )
    return PlattParams(slope=float(result.x[0]), intercept=float(result.x[1]))
