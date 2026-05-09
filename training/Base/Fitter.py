'''
Fitter — base ABC for all model fitters.

Shared contract:
  * Data loading: all fitters receive a games DataFrame (auto-loaded via
    DataLoader if not injected).
  * IO: all fitters persist their fitted state (save_config) and their
    validation output (save_validation) to disk in a consistent layout.
  * Validation: all fitters produce a ValidationReport (via validate()).

Each concrete fitter owns its own fit/prepare/diagnostics logic.  Those
methods are not part of the abstract contract because their signatures
and return types differ across fitters (Recalibrator returns a model,
Seasonal fitters return a per-season result, etc.).  The ABC mandates
only what is truly polymorphic:

    model_name, output_dir, validate(), save_config(pipeline_id)

plus the shared pipeline_id that threads through a training run.
'''

## built-ins ##
import os
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Tuple

## external ##
import pandas

## local ##
from nfelotranslation.Utilities.ValidationTypes import ValidationReport
from ..Data.DataLoader import DataLoader


class Fitter(ABC):
    '''
    Base class for all model fitters.

    Lifecycle (owned by the orchestrator):
        pipeline_id = generate_pipeline_id()
        fitter = SomeFitter(pipeline_id=pipeline_id)
        fitter.fit()                         ## runs fit + save_config
        report = fitter.validate()           ## ValidationReport
        fitter.save_validation(report)       ## persisted alongside config

    Subclasses must implement:
    * model_name (property) — string identifier
    * output_dir (property) — directory for configs and validation reports
    * validate() — produce a ValidationReport
    * save_config() — persist fitted state to disk, stamped with pipeline_id
    '''

    def __init__(
        self,
        games: pandas.DataFrame = None,
        pipeline_id: Optional[str] = None,
    ):
        ## accept injected games or load via singleton ##
        if games is not None:
            self._games = games
        else:
            self._games = DataLoader.get().games
        self._fitted_model: Any = None
        self._pipeline_id = pipeline_id

    ## ==================== Concrete ==================== ##

    @property
    def games(self) -> pandas.DataFrame:
        '''Formatted games DataFrame.'''
        return self._games

    @property
    def fitted_model(self) -> Any:
        '''The fitted model object, or None if fit() has not been called.'''
        return self._fitted_model

    @property
    def is_fitted(self) -> bool:
        '''Whether fit() has been called successfully.'''
        return self._fitted_model is not None

    @property
    def pipeline_id(self) -> Optional[str]:
        '''Training-run identifier (shared across fitters in one pipeline).'''
        return self._pipeline_id

    def _ensure_fitted(self) -> Any:
        '''Return the fitted model, calling fit() first if needed.'''
        if self._fitted_model is None:
            self.fit()
        return self._fitted_model

    def _loso_folds(self) -> Iterator[Tuple[int, pandas.DataFrame, pandas.DataFrame]]:
        '''
        Yield leave-one-season-out folds from self.games.

        Returns:
        * iterator of (season, train_games, test_games) tuples
        '''
        seasons = sorted(self.games['season'].unique())
        for season in seasons:
            train = self.games[self.games['season'] != season]
            test = self.games[self.games['season'] == season]
            yield season, train, test

    def save_validation(self, report: ValidationReport, output_dir: str = None) -> str:
        '''
        Save a validation report alongside configs.

        Writes to {output_dir}/validation/{model_name}.json.

        Parameters:
        * report: the ValidationReport to save
        * output_dir: base directory (defaults to self.output_dir)

        Returns:
        * path to the saved file
        '''
        output_dir = output_dir or self.output_dir
        val_dir = os.path.join(output_dir, 'validation')
        filepath = os.path.join(val_dir, f'{report.model_name}.json')
        report.to_file(filepath)
        return filepath

    ## ==================== Abstract ==================== ##

    @property
    @abstractmethod
    def model_name(self) -> str:
        '''Identifier for this model (used in file names).'''

    @property
    @abstractmethod
    def output_dir(self) -> str:
        '''Directory where configs and validation reports are stored.'''

    @abstractmethod
    def validate(self) -> ValidationReport:
        '''
        Run cross-validation and stationarity checks.

        Returns a ValidationReport with gated checks (structural invariants)
        and tracked metrics (detailed numbers for drift detection).
        '''

    @abstractmethod
    def save_config(self) -> str:
        '''
        Persist the fitted model's state to its canonical config path,
        stamped with self.pipeline_id.

        Should call _ensure_fitted() internally and return the written path.
        '''
