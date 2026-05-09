'''
Translator — the user-facing API for nfelotranslation.

Composes Recalibrator, SpreadMapper, and MarginDistributionModel into
a single stateful object.  All properties are computed eagerly on init
and update; property access is fast dict-like lookup.

Usage:
    from nfelotranslation import Translator

    ## convention: positive spread = home favorite (applies to both
    ## input and output for spread and market_spread; callers with
    ## raw sportsbook data — where favorites appear as negatives —
    ## must negate before passing in) ##
    t = Translator(3.0, 'market_spread', season=2025, side='home')
    t.win_prob              # calibrated WP from input side
    t.spread                # model Spread (posted + continuous)
    t.cover_prob(3.0)       # P(margin > 3)
    t.pmf                   # ndarray (151,)

    t.update(7.0, 'market_spread')   # recompute, reuse loaded models
'''

## built-ins ##
from typing import Optional

## external ##
import numpy

## local ##
from ..Calibration.Recalibrator import Recalibrator
from ..SpreadMap.SpreadMapper import SpreadMapper, MapType
from ..SpreadMap.Types import Spread
from ..Distribution.Key import KeyModel, KEY_MODEL_PARAMS
from ..Distribution.MarginDistributionModel import MarginDistributionModel
from ..Distribution.Types import MarginDistribution


## ==================== Constants ==================== ##

_VALID_INPUT_TYPES = {'win_prob', 'market_win_prob', 'spread', 'market_spread'}
_VALID_SIDES = {'home', 'away'}


class Translator:
    '''
    Stateful translator: input → calibrated WP → margin distribution → properties.

    Accepts a numeric value with a declared input type, season, and side.
    On construction, loads all models for the given season and computes
    the full margin distribution.  Properties are stored eagerly.

    Parameters:
    * value:      numeric input (spread, win probability, etc.)
    * input_type: one of 'win_prob', 'market_win_prob', 'spread', 'market_spread'
    * season:     NFL season year (selects per-season model configs)
    * side:       'home' or 'away' — determines sign conventions
    '''

    def __init__(
        self,
        value: float,
        input_type: str,
        season: int,
        side: str = 'home',
    ):
        ## validate ##
        _validate_input_type(input_type)
        _validate_side(side)
        ## load models for this season (cached on self).  Per-season
        ## resolution: SpreadMapper and KeyModel both warn-and-fall-back
        ## to the latest available config when ``season`` is past the
        ## last training cycle, and raise FileNotFoundError when ``season``
        ## predates the earliest config (out-of-domain ask). ##
        self._recalibrator = Recalibrator.from_file()
        self._model_mapper  = SpreadMapper.from_file(MapType.MODEL,  season=season)
        self._market_mapper = SpreadMapper.from_file(MapType.MARKET, season=season)
        key_model = KeyModel.from_file(season=season, params=KEY_MODEL_PARAMS)
        self._margin_model = MarginDistributionModel(key_model)
        self._season = season
        ## build state ##
        self._build(value, input_type, side)

    ## ==================== Update ==================== ##

    def update(
        self,
        value: float,
        input_type: str,
        side: Optional[str] = None,
    ) -> None:
        '''
        Recompute all properties with a new input.  Reuses loaded models.

        Parameters:
        * value:      numeric input
        * input_type: one of 'win_prob', 'market_win_prob', 'spread', 'market_spread'
        * side:       'home' or 'away'; None keeps current side
        '''
        _validate_input_type(input_type)
        if side is not None:
            _validate_side(side)
        else:
            side = self._side
        self._build(value, input_type, side)

    ## ==================== Core Properties ==================== ##

    @property
    def win_prob(self) -> float:
        '''Calibrated win probability from the input side's perspective.'''
        return self._win_prob

    @property
    def win_prob_market(self) -> float:
        '''Market (uncalibrated) win probability from the input side's perspective.'''
        return self._win_prob_market

    @property
    def spread(self) -> Spread:
        '''Model-derived Spread from the input side's perspective.'''
        return self._spread

    @property
    def market_spread(self) -> Spread:
        '''Market-derived Spread from the input side's perspective.'''
        return self._market_spread

    ## ==================== Side-Aware Properties ==================== ##

    @property
    def home_win_prob(self) -> float:
        '''Calibrated home win probability (always from home perspective).'''
        return self._home_cal_wp

    @property
    def away_win_prob(self) -> float:
        '''Calibrated away win probability (always from away perspective).'''
        return self._away_cal_wp

    @property
    def home_win_prob_market(self) -> float:
        '''Market home win probability (always from home perspective).'''
        return self._home_market_wp

    @property
    def away_win_prob_market(self) -> float:
        '''Market away win probability (always from away perspective).'''
        return self._away_market_wp

    ## ==================== Distribution Properties ==================== ##

    @property
    def pmf(self) -> numpy.ndarray:
        '''Discrete PMF over integer margins -75..+75, shape (151,).'''
        return self._distribution.pmf

    @property
    def tie_prob(self) -> float:
        '''P(margin = 0) from the discrete PMF.'''
        return self._distribution.tie_prob

    @property
    def expected_margin(self) -> float:
        '''E[margin] from the discrete PMF, from home perspective.'''
        return self._distribution.expected_margin()

    def cover_prob(self, line: float) -> float:
        '''
        P(margin > line) from home perspective.

        Parameters:
        * line: betting line to evaluate (e.g. 3.0 for -3)

        Returns:
        * probability in [0, 1]
        '''
        return self._distribution.cover_prob(line)

    def push_prob(self, line: float) -> float:
        '''
        P(margin == line) from the discrete PMF.

        Parameters:
        * line: betting line to evaluate

        Returns:
        * probability in [0, 1]
        '''
        return self._distribution.push_prob(line)

    ## ==================== Internal ==================== ##

    def _build(self, value: float, input_type: str, side: str) -> None:
        '''Resolve input to calibrated home WP, build distribution, store state.'''
        self._side = side
        ## step 1: resolve to home-perspective calibrated WP ##
        home_cal_wp = self._resolve_to_home_cal_wp(value, input_type)
        ## step 2: derive all WP variants ##
        self._home_cal_wp = float(home_cal_wp)
        home_market_wp = float(self._recalibrator.uncalibrate(numpy.array([home_cal_wp]))[0])
        self._home_market_wp = home_market_wp
        self._away_cal_wp = 1.0 - self._home_cal_wp - self._margin_model.tie_prob
        self._away_market_wp = 1.0 - home_market_wp - self._margin_model.tie_prob
        ## step 3: compute spreads (home perspective) ##
        home_model_spread = self._model_mapper.win_prob_to_spread(self._home_cal_wp)
        home_market_spread = self._market_mapper.win_prob_to_spread(home_market_wp)
        ## step 4: build distribution (always home perspective) ##
        self._distribution = self._margin_model.predict(
            float(home_model_spread.continuous),
            self._home_cal_wp,
        )
        ## step 5: store side-oriented properties ##
        if side == 'home':
            self._win_prob = self._home_cal_wp
            self._win_prob_market = self._home_market_wp
            self._spread = home_model_spread
            self._market_spread = home_market_spread
        else:
            self._win_prob = self._away_cal_wp
            self._win_prob_market = self._away_market_wp
            self._spread = Spread(
                posted=-home_model_spread.posted,
                continuous=-home_model_spread.continuous,
            )
            self._market_spread = Spread(
                posted=-home_market_spread.posted,
                continuous=-home_market_spread.continuous,
            )

    def _resolve_to_home_cal_wp(self, value: float, input_type: str) -> float:
        '''
        Map any input type to a home-perspective calibrated win probability.

        Input convention: positive spread = home favorite, wp > 0.5 = home
        favoured.  Side does not affect resolution.

        Resolution paths:
            'market_spread'    → market_mapper.spread_to_win_prob(value) → calibrate → cal_wp
            'market_win_prob'  → calibrate(value) → cal_wp
            'win_prob'         → value IS cal_wp
            'spread'           → model_mapper.spread_to_win_prob(value) → cal_wp
        '''
        if input_type == 'market_spread':
            market_wp = float(self._market_mapper.spread_to_win_prob(value))
            cal_wp = float(self._recalibrator.calibrate(numpy.array([market_wp]))[0])
        elif input_type == 'market_win_prob':
            cal_wp = float(self._recalibrator.calibrate(numpy.array([value]))[0])
        elif input_type == 'win_prob':
            cal_wp = value
        elif input_type == 'spread':
            cal_wp = float(self._model_mapper.spread_to_win_prob(value))
        return cal_wp


## ==================== Module-Level Helpers ==================== ##

def _validate_input_type(input_type: str) -> None:
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {sorted(_VALID_INPUT_TYPES)}, "
            f"got '{input_type}'"
        )

def _validate_side(side: str) -> None:
    if side not in _VALID_SIDES:
        raise ValueError(
            f"side must be one of {sorted(_VALID_SIDES)}, "
            f"got '{side}'"
        )

