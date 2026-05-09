'''
DataLoader — singleton wrapper around nfelodcm for training pipelines.

Loads games once, formats derived fields, shares across multiple fitters.
'''

## built-ins ##
from typing import ClassVar, Optional

## external ##
import numpy
import pandas
import nfelodcm

## local ##
from nfelotranslation.Utilities.MathUtils import american_to_prob, hold_adj_probs


class DataLoader:
    '''
    Singleton data loader for seasonal training.

    Usage:
        loader = DataLoader.get()
        games = loader.games
        DataLoader.reset()   ## clear for fresh reload

    The games DataFrame includes derived fields:
    * margin: round(result) as int
    * ml_wp_close: hold-adjusted closing moneyline win probability (raw market)
    * ml_wp_cal: recalibrated win probability via Platt scaling
    * spread_line: in positive = favorite convention (matches the package-wide
      internal convention; nflfastR source already uses this, so no flip applied)
    '''

    _instance: ClassVar[Optional['DataLoader']] = None

    def __init__(self):
        self._games = self._load_and_format()

    ## ==================== Singleton ==================== ##

    @classmethod
    def get(cls) -> 'DataLoader':
        '''Return the singleton instance, creating it on first call.'''
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        '''Clear the singleton instance (forces reload on next get()).'''
        cls._instance = None

    ## ==================== Public ==================== ##

    @property
    def games(self) -> pandas.DataFrame:
        '''Formatted games DataFrame.'''
        return self._games

    ## ==================== Private ==================== ##

    @staticmethod
    def _load_and_format() -> pandas.DataFrame:
        '''Load from nfelodcm and derive training-relevant fields.'''
        ## lazy import — keeps nfelodcm load out of import time ##
        from nfelotranslation.Calibration.Recalibrator import Recalibrator
        db = nfelodcm.load(['games'])
        games = db['games'][[
            'season', 'week', 'game_type',
            'home_team', 'away_team',
            'home_score', 'away_score',
            'spread_line', 'result',
            'home_moneyline', 'away_moneyline',
        ]].copy()
        ## filter to completed games with spreads ##
        games = games[games['result'].notna() & games['spread_line'].notna()].copy()
        ## spread_line kept in nflfastR's positive = favorite convention
        ## (matches package-wide internal convention — no flip needed) ##
        ## margin as integer ##
        games['margin'] = games['result'].round().astype(int)
        ## hold-adjusted closing ML win probability ##
        ml_valid = games['home_moneyline'].notna() & games['away_moneyline'].notna()
        h_wp, _ = hold_adj_probs(
            games.loc[ml_valid, 'home_moneyline'].values,
            games.loc[ml_valid, 'away_moneyline'].values,
        )
        games.loc[ml_valid, 'ml_wp_close'] = h_wp
        ## recalibrated win probability ##
        rec = Recalibrator.from_file()
        games.loc[ml_valid, 'ml_wp_cal'] = rec.calibrate(
            games.loc[ml_valid, 'ml_wp_close'].values
        )
        return games
