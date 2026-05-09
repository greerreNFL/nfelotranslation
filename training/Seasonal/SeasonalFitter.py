'''
SeasonalFitter — ABC for seasonal training pipelines.

Template method pattern: fit() handles the season loop, subclasses
implement data prep, training, scoring, and saving.

Key design: score_season() is called BEFORE train_season() so that
diagnostics measure genuine out-of-sample quality.
'''

## built-ins ##
from abc import abstractmethod
from typing import Dict, List, Optional

## external ##
import pandas

## local ##
from .Types import SeasonalDiagnostics, SeasonalResult
from ..Base.Fitter import Fitter


class SeasonalFitter(Fitter):
    '''
    Abstract base class for seasonal model training.

    Extends Fitter with a season-loop template.  Subclasses must implement:
    * model_name (property) — inherited from Fitter
    * output_dir (property)
    * prepare_data() → filtered DataFrame (uses self.games)
    * initialize_model() → set up fresh model state
    * train_season(season_data, season) → ingest one season
    * save_season(season) → persist state after training
    * score_season(season_data, season) → out-of-sample diagnostics
    '''

    def __init__(
        self,
        games: pandas.DataFrame = None,
        output_path: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ):
        super().__init__(games=games, pipeline_id=pipeline_id)

    ## ==================== Abstract Interface ==================== ##

    @abstractmethod
    def initialize_model(self) -> None:
        '''Set up fresh model state before the season loop.'''

    @abstractmethod
    def train_season(self, season_data: pandas.DataFrame, season: int) -> None:
        '''
        Ingest one season of data into the model.

        Parameters:
        * season_data: games for this season only
        * season: season identifier
        '''

    @abstractmethod
    def save_season(self, season: int) -> None:
        '''
        Persist model state after training this season.

        Parameters:
        * season: the season just trained
        '''

    @abstractmethod
    def score_season(self, season_data: pandas.DataFrame, season: int) -> SeasonalDiagnostics:
        '''
        Evaluate out-of-sample quality BEFORE training on this season.

        Parameters:
        * season_data: games for this season (held out from model)
        * season: season identifier

        Returns:
        * SeasonalDiagnostics with fitter-specific metrics
        '''

    ## ==================== Template Method ==================== ##

    def fit(self, seasons: Optional[List[int]] = None) -> SeasonalResult:
        '''
        Execute the full seasonal training loop.

        For each season (in chronological order):
            1. score_season() — measure out-of-sample prediction quality
            2. train_season() — ingest that season's data
            3. save_season() — persist state

        Parameters:
        * seasons: optional list of seasons to train on (default: all available)

        Returns:
        * SeasonalResult with per-season diagnostics and aggregates
        '''
        data = self.prepare_data()
        available = sorted(data['season'].unique())
        train_seasons = [s for s in available if s in seasons] if seasons else available
        self.initialize_model()
        per_season = []
        for season in train_seasons:
            season_data = data[data['season'] == season]
            ## score BEFORE train (out-of-sample) ##
            diagnostics = self.score_season(season_data, season)
            per_season.append(diagnostics)
            self.train_season(season_data, season)
            self.save_season(season)
        aggregate = self._compute_aggregate(per_season)
        result = SeasonalResult(
            model_name=self.model_name,
            seasons_trained=list(train_seasons),
            per_season=per_season,
            aggregate_metrics=aggregate,
            output_dir=self.output_dir,
        )
        self._fitted_model = result
        return result

    def diagnostics(self) -> SeasonalResult:
        '''
        Return the SeasonalResult from the last fit, fitting first if needed.

        Returns:
        * SeasonalResult with per-season diagnostics and aggregates
        '''
        return self._ensure_fitted()

    ## ==================== Aggregation ==================== ##

    def _compute_aggregate(self, per_season: List[SeasonalDiagnostics]) -> Dict[str, float]:
        '''
        Compute aggregate metrics as the mean of each metric across seasons.

        Subclasses can override for custom aggregation logic.

        Parameters:
        * per_season: list of per-season diagnostics

        Returns:
        * dict of metric_name → mean_value
        '''
        if not per_season:
            return {}
        ## collect all metric names from first season ##
        metric_names = list(per_season[0].metrics.keys())
        aggregate = {}
        for name in metric_names:
            values = [d.metrics[name] for d in per_season if name in d.metrics]
            aggregate[name] = sum(values) / len(values) if values else 0.0
        return aggregate
