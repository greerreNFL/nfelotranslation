## Unit tests for the Translator user-facing API.
##
## Properties tested:
##   1. Init and basic properties: all expected attributes exist and are reasonable
##   2. Input type consistency: all four input types produce valid distributions
##   3. Side handling: home/away perspectives are consistent
##   4. Update reuse: update() recomputes without reloading models
##   5. WP round-trip: calibrate → uncalibrate is consistent
##   6. PMF validity: sum=1, non-negative for all input types
##   7. Pick-em edge case: spread=0 does not produce NaN
##   8. Validation: bad input_type and bad side raise ValueError
##   9. Spread sign convention: negative market spread = home favourite
##  10. Seasonal config: different seasons load different configs
##
import pytest
import numpy as np
from nfelotranslation import Translator, Recalibrator
from nfelotranslation.Distribution import MARGIN_HYPERPARAMS


## ============================================================
## TOLERANCES
## ============================================================

PMF_SUM_TOL = 1e-10
WP_TOL      = 1e-12     ## calibrate → uncalibrate is pure analytic inverse (expit∘logit∘logit∘expit)

## tie_prob from the shipped config — same dict production reads ##
TIE_PROB = float(MARGIN_HYPERPARAMS['tie_prob'])


## ============================================================
## FIXTURES
## ============================================================

@pytest.fixture(scope='module')
def home_translator():
    '''Home-perspective translator for a 3-point home favorite.'''
    return Translator(3.0, 'market_spread', season=2025, side='home')

@pytest.fixture(scope='module')
def away_translator():
    '''Away-perspective translator for the same 3-point home favorite game.'''
    return Translator(3.0, 'market_spread', season=2025, side='away')


## ============================================================
## PROPERTY 1: init and basic properties
## ============================================================

class TestBasicProperties:
    '''Translator should have all expected properties after init.'''

    def test_win_prob_in_range(self, home_translator):
        assert 0.0 < home_translator.win_prob < 1.0

    def test_win_prob_market_in_range(self, home_translator):
        assert 0.0 < home_translator.win_prob_market < 1.0

    def test_spread_is_spread(self, home_translator):
        from nfelotranslation.SpreadMap.Types import Spread
        assert isinstance(home_translator.spread, Spread)

    def test_market_spread_is_spread(self, home_translator):
        from nfelotranslation.SpreadMap.Types import Spread
        assert isinstance(home_translator.market_spread, Spread)

    def test_pmf_shape(self, home_translator):
        assert home_translator.pmf.shape == (151,)

    def test_tie_prob(self, home_translator):
        assert home_translator.tie_prob == TIE_PROB

    def test_expected_margin_near_spread(self, home_translator):
        ## bounded by rounding (|continuous - posted| <= 0.25) plus key-model
        ## mean shift (<~0.3 for modest spreads); 0.5 is empirical max + headroom ##
        assert abs(home_translator.expected_margin - home_translator.spread.posted) < 0.5


## ============================================================
## PROPERTY 2: all input types produce valid distributions
## ============================================================

class TestInputTypes:
    '''
    All four input types should produce a valid PMF AND preserve the
    calibration invariant the Translator guarantees:
        win_prob == recalibrate(win_prob_market)   (equivalently,
        win_prob_market == uncalibrate(win_prob))
    This catches calibration being skipped, double-applied, or applied
    in the wrong direction for any input type.
    '''

    def _assert_calibration_invariant(self, t):
        '''calibrate(win_prob_market) must equal win_prob exactly.'''
        rec = Recalibrator.from_file()
        expected = float(rec.calibrate(np.array([t.win_prob_market]))[0])
        assert abs(t.win_prob - expected) < 1e-10, (
            f'calibration invariant broken: win_prob={t.win_prob:.12f}, '
            f'calibrate(win_prob_market)={expected:.12f}'
        )

    def test_market_spread(self):
        t = Translator(3.0, 'market_spread', season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert t.win_prob > 0.5   ## positive market_spread = home favorite ##
        self._assert_calibration_invariant(t)

    def test_market_win_prob(self):
        t = Translator(0.60, 'market_win_prob', season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert abs(t.win_prob_market - 0.60) < WP_TOL   ## input preserved (to float roundoff) ##
        self._assert_calibration_invariant(t)

    def test_win_prob(self):
        t = Translator(0.62, 'win_prob', season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert t.win_prob == 0.62   ## input preserved verbatim ##
        self._assert_calibration_invariant(t)

    def test_spread(self):
        t = Translator(3.0, 'spread', season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert t.win_prob > 0.5   ## positive model spread = home favorite ##
        self._assert_calibration_invariant(t)


## ============================================================
## PROPERTY 3: side handling
## ============================================================

class TestSideHandling:
    '''Home and away perspectives should be consistent.'''

    def test_home_away_wp_sum(self, home_translator):
        total = home_translator.home_win_prob + home_translator.away_win_prob + TIE_PROB
        assert abs(total - 1.0) < 1e-10

    def test_home_away_market_wp_sum(self, home_translator):
        total = home_translator.home_win_prob_market + home_translator.away_win_prob_market + TIE_PROB
        assert abs(total - 1.0) < 1e-10

    def test_side_symmetry(self, home_translator, away_translator):
        '''Same input, different sides: home_wp of one = away_wp of other.'''
        assert abs(home_translator.home_win_prob - away_translator.home_win_prob) < 1e-10
        assert abs(home_translator.away_win_prob - away_translator.away_win_prob) < 1e-10

    def test_away_win_prob_is_perspective(self, away_translator):
        '''Away-side translator.win_prob should equal away WP.'''
        assert abs(away_translator.win_prob - away_translator.away_win_prob) < 1e-10

    def test_spread_sign_flips_with_side(self, home_translator, away_translator):
        '''Away spread should be negated relative to home spread.'''
        assert abs(home_translator.spread.continuous + away_translator.spread.continuous) < 1e-10
        assert abs(home_translator.spread.posted + away_translator.spread.posted) < 1e-10


## ============================================================
## PROPERTY 4: update reuse
## ============================================================

class TestUpdate:
    '''update() should recompute properties without reloading models.'''

    def test_update_changes_wp(self):
        t = Translator(3.0, 'market_spread', season=2025)
        wp_before = t.win_prob
        t.update(7.0, 'market_spread')
        assert t.win_prob != wp_before
        assert t.win_prob > wp_before  ## bigger home-favorite spread = higher home WP ##

    def test_update_changes_side(self):
        t = Translator(3.0, 'market_spread', season=2025, side='home')
        home_wp = t.win_prob
        t.update(3.0, 'market_spread', side='away')
        assert abs(t.win_prob - (1.0 - home_wp - TIE_PROB)) < 1e-10

    def test_update_preserves_side_when_none(self):
        t = Translator(3.0, 'market_spread', season=2025, side='away')
        t.update(7.0, 'market_spread')
        ## should still be away ##
        assert abs(t.win_prob - t.away_win_prob) < 1e-10

    def test_pmf_valid_after_update(self):
        t = Translator(3.0, 'market_spread', season=2025)
        t.update(0.65, 'win_prob')
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert (t.pmf >= 0).all()


## ============================================================
## PROPERTY 5: Full-pipeline round-trip invariants
## ============================================================
## The core app-level guarantee: whatever spread/wp a caller puts in,
## the Translator must round-trip back to the same value through the
## composed primitives (mapper + recalibrator). These tests compose
## multiple invariants in a way the component-level mapper round-trip
## tests don't — a bug in how the Translator routes inputs through the
## mappers (e.g., mixing model/market, skipping calibration) would
## break these even if the underlying components are fine.

class TestWPRoundTrip:
    '''win_prob ↔ win_prob_market should round-trip through calibration.'''

    MARKET_WPS = [0.55, 0.60, 0.65, 0.70, 0.80]

    @pytest.mark.parametrize('mwp', MARKET_WPS)
    def test_market_wp_round_trip(self, mwp):
        t = Translator(mwp, 'market_win_prob', season=2025)
        ## win_prob_market should be close to mwp (calibrate then uncalibrate) ##
        assert abs(t.win_prob_market - mwp) < WP_TOL


class TestMarketSpreadRoundTrip:
    '''
    market_spread input → t.market_spread round-trips through the full
    pipeline (market_mapper.spread_to_wp → calibrate → uncalibrate →
    market_mapper.wp_to_spread). Input and output both use positive =
    home favorite, so round-trip recovers input exactly.
    '''

    MARKET_SPREADS = [-7.0, -3.0, -1.5, 0.0, 1.5, 3.0, 7.0, 14.0]

    @pytest.mark.parametrize('ms', MARKET_SPREADS)
    def test_market_spread_round_trip_home(self, ms):
        t = Translator(ms, 'market_spread', season=2025, side='home')
        assert abs(t.market_spread.continuous - ms) < WP_TOL, (
            f'market_spread round-trip: in={ms}, out={t.market_spread.continuous}'
        )


class TestSpreadRoundTrip:
    '''
    spread input → t.spread round-trips through the model mapper
    (spread_to_win_prob → win_prob_to_spread). No calibration in this
    path — spread is already in model (calibrated) coordinates.
    '''

    SPREADS = [-14.0, -7.0, -3.0, -1.5, 0.0, 1.5, 3.0, 7.0, 14.0]

    @pytest.mark.parametrize('s', SPREADS)
    def test_spread_round_trip_home(self, s):
        t = Translator(s, 'spread', season=2025, side='home')
        ## same sign — model convention is positive = home favorite on both ends ##
        assert abs(t.spread.continuous - s) < WP_TOL, (
            f'spread round-trip: in={s}, out={t.spread.continuous}'
        )


## ============================================================
## PROPERTY 6: PMF validity
## ============================================================

class TestPMFValidity:
    '''PMF should be valid for a range of inputs.'''

    CASES = [
        (-14.0, 'market_spread'),
        (-7.0, 'market_spread'),
        (-3.0, 'market_spread'),
        (0.0, 'market_spread'),
        (3.0, 'market_spread'),
        (7.0, 'market_spread'),
        (0.55, 'win_prob'),
        (0.70, 'win_prob'),
        (0.90, 'win_prob'),
    ]

    @pytest.mark.parametrize('value,itype', CASES)
    def test_sums_to_one(self, value, itype):
        t = Translator(value, itype, season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL

    @pytest.mark.parametrize('value,itype', CASES)
    def test_non_negative(self, value, itype):
        t = Translator(value, itype, season=2025)
        assert (t.pmf >= 0).all()


## ============================================================
## PROPERTY 7: pick-em edge case
## ============================================================

class TestPickEm:
    '''Spread=0 must not produce NaN.'''

    def test_market_spread_zero(self):
        t = Translator(0.0, 'market_spread', season=2025)
        assert not np.any(np.isnan(t.pmf))
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL

    def test_win_prob_50(self):
        t = Translator(0.50, 'win_prob', season=2025)
        assert not np.any(np.isnan(t.pmf))
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL


## ============================================================
## PROPERTY 8: validation
## ============================================================

class TestValidation:
    '''Bad inputs should raise ValueError.'''

    def test_bad_input_type(self):
        with pytest.raises(ValueError, match='input_type'):
            Translator(-3.0, 'bad_type', season=2025)

    def test_bad_side(self):
        with pytest.raises(ValueError, match='side'):
            Translator(-3.0, 'market_spread', season=2025, side='favorite')

    def test_bad_update_input_type(self):
        t = Translator(-3.0, 'market_spread', season=2025)
        with pytest.raises(ValueError, match='input_type'):
            t.update(-3.0, 'invalid')

    def test_bad_update_side(self):
        t = Translator(-3.0, 'market_spread', season=2025)
        with pytest.raises(ValueError, match='side'):
            t.update(-3.0, 'market_spread', side='neutral')


## ============================================================
## PROPERTY 9: spread sign convention
## ============================================================

class TestSpreadSignConvention:
    '''
    Positive spread = home favorite, throughout input and output.
    Applies uniformly to market_spread, spread, and all sub-components.
    '''

    def test_positive_market_spread_is_home_favourite(self):
        t = Translator(7.0, 'market_spread', season=2025, side='home')
        assert t.win_prob > 0.5
        assert t.spread.continuous > 0

    def test_negative_market_spread_is_away_favourite(self):
        t = Translator(-7.0, 'market_spread', season=2025, side='home')
        assert t.win_prob < 0.5
        assert t.spread.continuous < 0


## ============================================================
## PROPERTY 10: seasonal config
## ============================================================

class TestSeasonalConfig:
    '''Different seasons load distinct configs and produce results that
    reflect those specific configs — not merely "different numbers."'''

    def test_shipped_seasonal_configs_differ(self):
        '''Two seasonal spread_map configs must have distinct params; if they
        ever become identical, the seasonal infrastructure is silently
        collapsing.'''
        from nfelotranslation.SpreadMap.SpreadMapper import load_mapper_pair
        from nfelotranslation.Utilities.JsonIo import find_config_path
        import pathlib, nfelotranslation.SpreadMap as _sm
        config_dir = str(pathlib.Path(_sm.__file__).resolve().parent / 'configs')
        path_2020 = find_config_path('spread_map_params', 2020, config_dir)
        path_2025 = find_config_path('spread_map_params', 2025, config_dir)
        assert path_2020 is not None and path_2025 is not None
        assert path_2020 != path_2025, (
            'expected distinct 2020 and 2025 seasonal config paths'
        )
        m2020, _, _ = load_mapper_pair(filepath=path_2020)
        m2025, _, _ = load_mapper_pair(filepath=path_2025)
        params_differ = (
            m2020.params.slope != m2025.params.slope
            or m2020.params.intercept != m2025.params.intercept
        )
        assert params_differ, (
            f'2020 and 2025 model mappers have identical params — seasonal '
            f'configs appear degenerate: slope={m2020.params.slope}, '
            f'intercept={m2020.params.intercept}'
        )

    def test_translator_reflects_seasonal_config(self):
        '''Translator output for a given input must match win_prob_to_spread
        computed with each season's *own* model mapper params — proving the
        Translator routed through the correct seasonal config, not a wrong
        one that happens to give a different number.'''
        from nfelotranslation.SpreadMap.SpreadMapper import load_mapper_pair
        from nfelotranslation.Utilities.JsonIo import find_config_path
        import pathlib, nfelotranslation.SpreadMap as _sm
        config_dir = str(pathlib.Path(_sm.__file__).resolve().parent / 'configs')
        for season in (2020, 2025):
            t = Translator(7.0, 'market_spread', season=season)
            path = find_config_path('spread_map_params', season, config_dir)
            mm, _, _ = load_mapper_pair(filepath=path)
            ## model spread for this Translator should match the model mapper
            ## at this WP — proves the right seasonal mapper was loaded ##
            expected_spread = mm.win_prob_to_spread(t.win_prob).continuous
            assert abs(t.spread.continuous - expected_spread) < 1e-10, (
                f'season={season}: Translator.spread.continuous={t.spread.continuous:.8f} '
                f'does not match season-{season} model mapper={expected_spread:.8f}'
            )


## ============================================================
## PROPERTY 11: out-of-domain season handling
## ============================================================

class TestOutOfDomainSeason:
    '''Translator's behavior for seasons outside the trained range:
       - past the latest trained season: warn and use the latest config
       - before the earliest trained season: raise FileNotFoundError
    '''

    def test_past_latest_season_warns_and_loads(self, recwarn):
        '''A season past the latest trained config falls back to the
        latest fit with a warning, NOT to a no-key-signal model.  Catches
        the historical silent-degrade bug: today this season has neither
        a per-season SpreadMap nor a per-season KeyModel, but the
        Translator should still produce sensible output, not garbage.'''
        import warnings
        future = 2099
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t = Translator(3.0, 'market_spread', season=future, side='home')
        ## a fall-back warning should have been emitted ##
        assert any('falling back' in str(w.message).lower() for w in caught), (
            f'expected a fall-back warning when constructing a Translator for '
            f'season={future}; got: {[str(w.message) for w in caught]}'
        )
        ## and the result should still be a sensible Translator: home favored
        ## by 3 points → wp > 0.5, valid PMF, key-model corrections active
        ## (not a no-signal model — this is the bug-fix assertion) ##
        assert t.win_prob > 0.5
        assert abs(t.pmf.sum() - 1.0) < 1e-10
        ## key-model is active: ratio at ±3 should differ meaningfully from 1.0
        ## (i.e. the credibility tracker has been trained, not zero-state) ##
        ratio_3 = t._margin_model.key_model.outcomes[3].get_ratio()
        assert ratio_3 != 1.0, (
            'past-latest fallback returned a no-signal KeyModel (ratio at 3 = 1.0); '
            'the silent-degrade bug has resurfaced'
        )

    def test_before_earliest_season_raises(self):
        '''A season before the earliest trained config raises
        FileNotFoundError.  Prevents using a 2007-trained model on a
        pre-2007 game (out of model domain).'''
        import pytest as _pytest
        ancient = 1990
        with _pytest.raises(FileNotFoundError):
            Translator(3.0, 'market_spread', season=ancient, side='home')
