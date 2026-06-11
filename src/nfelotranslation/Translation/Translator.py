'''
Translator — the user-facing API for nfelotranslation.

Composes SpreadMapper and MarginDistributionModel into a single stateful
object.  All properties are computed eagerly on init and update; property
access is fast dict-like lookup.

Usage:
    from nfelotranslation import Translator

    ## convention: positive spread = home favorite (applies to both
    ## input and output for spread; callers with raw sportsbook data —
    ## where favorites appear as negatives — must negate before passing in) ##
    t = Translator(3.0, 'spread', season=2025, side='home')
    t.win_prob              # model WP from input side
    t.spread                # Spread (posted + continuous)
    t.cover_prob(3.0)       # P(margin > 3)
    t.pmf                   # ndarray (151,)

    t.update(7.0, 'spread')   # recompute, reuse loaded models
'''

## built-ins ##
from typing import Optional

## external ##
import numpy

## local ##
from ..SpreadMap.SpreadMapper import SpreadMapper
from ..SpreadMap.Types import Spread
from ..Distribution.Key import KeyModel, KEY_MODEL_PARAMS
from ..Distribution.MarginDistributionModel import MarginDistributionModel
from ..Distribution.Types import MarginDistribution


## ==================== Constants ==================== ##

_VALID_INPUT_TYPES = {'win_prob', 'spread'}
_VALID_SIDES = {'home', 'away'}


class Translator:
    '''
    Stateful translator: input → model WP → margin distribution → properties.

    Accepts a numeric value with a declared input type, season, and side.
    On construction, loads all models for the given season and computes
    the full margin distribution.  Properties are stored eagerly.

    Parameters:
    * value:      numeric input (spread or win probability)
    * input_type: one of 'win_prob' or 'spread'
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
        ## load models for this season (cached on self) ##
        self._mapper = SpreadMapper.from_file(season=season)
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
        * input_type: one of 'win_prob' or 'spread'
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
        '''Model win probability from the input side's perspective.'''
        return self._win_prob

    @property
    def spread(self) -> Spread:
        '''Model-derived Spread from the input side's perspective.'''
        return self._spread

    ## ==================== Side-Aware Properties ==================== ##

    @property
    def home_win_prob(self) -> float:
        '''Home win probability (always from home perspective).'''
        return self._home_wp

    @property
    def away_win_prob(self) -> float:
        '''Away win probability (always from away perspective).'''
        return self._away_wp

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
        '''Resolve input to home-perspective model WP, build distribution, store state.'''
        self._side = side
        ## step 1: resolve to home-perspective model WP ##
        home_wp = self._resolve_to_home_wp(value, input_type)
        ## step 2: derive all WP variants ##
        self._home_wp = float(home_wp)
        self._away_wp = 1.0 - self._home_wp - self._margin_model.tie_prob
        ## step 3: compute spread (home perspective) ##
        home_spread = self._mapper.win_prob_to_spread(self._home_wp)
        ## step 4: build distribution (always home perspective) ##
        self._distribution = self._margin_model.predict(
            float(home_spread.continuous),
            self._home_wp,
        )
        ## step 5: store side-oriented properties ##
        if side == 'home':
            self._win_prob = self._home_wp
            self._spread = home_spread
        else:
            self._win_prob = self._away_wp
            self._spread = Spread(
                posted=-home_spread.posted,
                continuous=-home_spread.continuous,
            )

    def _resolve_to_home_wp(self, value: float, input_type: str) -> float:
        '''
        Map any input type to a home-perspective model win probability.

        Input convention: positive spread = home favorite, wp > 0.5 = home
        favoured.  Side does not affect resolution.

        Resolution paths:
            'win_prob' → value IS model WP (home perspective)
            'spread'   → mapper.spread_to_win_prob(value) → model WP
        '''
        if input_type == 'win_prob':
            return value
        if input_type == 'spread':
            return float(self._mapper.spread_to_win_prob(value))
        raise ValueError(f'unsupported input_type: {input_type}')


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
