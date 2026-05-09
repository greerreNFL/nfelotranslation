## Unit tests for the SpreadMapper linear-in-logit spread mapping.
##
## Properties tested:
##   1a. WP round-trip:     spread_to_win_prob(win_prob_to_spread(wp).continuous) ≈ wp
##   1b. Spread round-trip: win_prob_to_spread(spread_to_win_prob(s)).continuous ≈ s
##       Both directions are critical app-level invariants: we must be
##       able to map a spread to a win prob and back to the same spread
##       (and vice versa), for both MODEL and MARKET mappers.
##   2. Known values: 50% → intercept, 60% → formula from loaded params
##   3. Two-instance isolation: MODEL and MARKET give different results
##   4. Clamping: posted on 0.5 grid, continuous off-grid
##   5. Vectorization: numpy array inputs work correctly
##   6. Serialization: to_file / from_file round-trip preserves params
##   7. Monotonicity: higher WP → higher spread
##
import pytest
import numpy as np
import tempfile
import os
from scipy.special import logit
from nfelotranslation.SpreadMap import (
    SpreadMapper, MapType, LinearMapParams, Spread,
)


## ============================================================
## TOLERANCES
## ============================================================

ROUNDTRIP_TOL = 1e-12      ## pure analytic inverse — expit∘logit∘logit∘expit ##
KNOWN_VALUE_TOL = 0.5      ## points — known-value sanity check
PARAM_TOL = 1e-10          ## serialization round-trip precision


## ============================================================
## FIXTURES
## ============================================================

@pytest.fixture(scope='module')
def model_mapper():
    return SpreadMapper.from_file(MapType.MODEL)

@pytest.fixture(scope='module')
def market_mapper():
    return SpreadMapper.from_file(MapType.MARKET)


## ============================================================
## PROPERTY 1a: win_prob → spread → win_prob round-trip
## ============================================================

class TestWpRoundTrip:
    '''
    spread_to_win_prob(win_prob_to_spread(wp).continuous) ≈ wp.
    The app-level guarantee for both MODEL and MARKET mappers: any wp
    passed through the mapper and back must recover exactly.
    '''

    WIN_PROBS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]

    @pytest.mark.parametrize('wp', WIN_PROBS)
    def test_model_round_trip(self, model_mapper, wp):
        spread = model_mapper.win_prob_to_spread(wp)
        wp_back = model_mapper.spread_to_win_prob(spread.continuous)
        assert abs(float(wp_back) - wp) < ROUNDTRIP_TOL, (
            f'Model round-trip: in={wp:.4f}  spread={spread.continuous:.4f}  '
            f'out={float(wp_back):.8f}  err={float(wp_back) - wp:+.2e}'
        )

    @pytest.mark.parametrize('wp', WIN_PROBS)
    def test_market_round_trip(self, market_mapper, wp):
        spread = market_mapper.win_prob_to_spread(wp)
        wp_back = market_mapper.spread_to_win_prob(spread.continuous)
        assert abs(float(wp_back) - wp) < ROUNDTRIP_TOL, (
            f'Market round-trip: in={wp:.4f}  spread={spread.continuous:.4f}  '
            f'out={float(wp_back):.8f}  err={float(wp_back) - wp:+.2e}'
        )


## ============================================================
## PROPERTY 1b: spread → win_prob → spread round-trip
## ============================================================

class TestSpreadRoundTrip:
    '''
    win_prob_to_spread(spread_to_win_prob(s)).continuous ≈ s.
    The complementary direction: a spread passed into the mapper and
    back must recover exactly. Critical for the app-level guarantee
    that we can translate a spread → wp and back to the same spread
    in both MODEL and MARKET coordinates.
    '''

    SPREADS = [-14.0, -10.0, -7.0, -6.5, -3.0, -0.5, 0.0, 0.5, 3.0, 6.5, 7.0, 10.0, 14.0]

    @pytest.mark.parametrize('s', SPREADS)
    def test_model_spread_round_trip(self, model_mapper, s):
        p = model_mapper.spread_to_win_prob(s)
        s_out = model_mapper.win_prob_to_spread(float(p))
        assert abs(s_out.continuous - s) < ROUNDTRIP_TOL, (
            f'Model spread round-trip: in={s:.2f}  p={float(p):.6f}  '
            f'out={s_out.continuous:.4f}  err={s_out.continuous - s:+.2e}'
        )

    @pytest.mark.parametrize('s', SPREADS)
    def test_market_spread_round_trip(self, market_mapper, s):
        p = market_mapper.spread_to_win_prob(s)
        s_out = market_mapper.win_prob_to_spread(float(p))
        assert abs(s_out.continuous - s) < ROUNDTRIP_TOL, (
            f'Market spread round-trip: in={s:.2f}  p={float(p):.6f}  '
            f'out={s_out.continuous:.4f}  err={s_out.continuous - s:+.2e}'
        )


## ============================================================
## PROPERTY 2: known values
## ============================================================

class TestKnownValues:
    '''
    Sanity checks against the linear-in-logit formula using each mapper's
    actual loaded params (no hardcoded slope/intercept values):
    * 50% win prob → spread = intercept (since logit(0.5)=0)
    * 60% win prob → spread = slope * logit(0.60) + intercept
    '''

    def test_50pct_equals_intercept_model(self, model_mapper):
        spread = model_mapper.win_prob_to_spread(0.50)
        assert abs(spread.continuous - model_mapper.params.intercept) < 1e-6, (
            f'50% spread should equal intercept: got {spread.continuous:.4f}, '
            f'expected {model_mapper.params.intercept:.4f}'
        )

    def test_50pct_equals_intercept_market(self, market_mapper):
        spread = market_mapper.win_prob_to_spread(0.50)
        assert abs(spread.continuous - market_mapper.params.intercept) < 1e-6, (
            f'50% spread should equal intercept: got {spread.continuous:.4f}, '
            f'expected {market_mapper.params.intercept:.4f}'
        )

    def test_60pct_model(self, model_mapper):
        spread = model_mapper.win_prob_to_spread(0.60)
        expected = model_mapper.params.slope * logit(0.60) + model_mapper.params.intercept
        assert abs(spread.continuous - expected) < 1e-6, (
            f'60% model spread should match linear-in-logit formula: '
            f'got {spread.continuous:.6f}, expected {expected:.6f}'
        )

    def test_60pct_market(self, market_mapper):
        spread = market_mapper.win_prob_to_spread(0.60)
        expected = market_mapper.params.slope * logit(0.60) + market_mapper.params.intercept
        assert abs(spread.continuous - expected) < 1e-6, (
            f'60% market spread should match linear-in-logit formula: '
            f'got {spread.continuous:.6f}, expected {expected:.6f}'
        )


## ============================================================
## PROPERTY 3: two-instance isolation
## ============================================================

class TestTwoInstanceIsolation:
    '''
    MODEL and MARKET mappers have different parameters because they
    are fitted to different WP scales (ml_wp_cal vs ml_wp_close) and
    different target variables (actual margins vs market spreads).
    '''

    def test_params_differ(self, model_mapper, market_mapper):
        '''Model and market mappers must have different slope and intercept.'''
        assert model_mapper.params.slope != market_mapper.params.slope
        assert model_mapper.params.intercept != market_mapper.params.intercept

    def test_market_slope_steeper(self, model_mapper, market_mapper):
        '''Market slope > model slope because ml_wp_close is compressed
        toward 50% relative to ml_wp_cal, requiring a steeper slope
        to map to the same spread range.'''
        assert market_mapper.params.slope > model_mapper.params.slope


## ============================================================
## PROPERTY 4: clamping
## ============================================================

class TestClamping:
    '''
    posted values are on 0.5 grid; continuous values are off-grid.
    '''

    def test_posted_on_half_grid(self, model_mapper):
        ## test several win probs and verify posted is always a multiple of 0.5 ##
        for wp in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            spread = model_mapper.win_prob_to_spread(wp)
            remainder = spread.posted % 0.5
            assert abs(remainder) < 1e-10 or abs(remainder - 0.5) < 1e-10, (
                f'Posted spread {spread.posted} is not on 0.5 grid for wp={wp}'
            )

    def test_continuous_off_grid(self, model_mapper):
        ## most continuous values should NOT be exact multiples of 0.5 ##
        off_grid_count = 0
        for wp in [0.55, 0.60, 0.65, 0.70, 0.75]:
            spread = model_mapper.win_prob_to_spread(wp)
            remainder = float(spread.continuous) % 0.5
            if abs(remainder) > 1e-6 and abs(remainder - 0.5) > 1e-6:
                off_grid_count += 1
        assert off_grid_count >= 3, (
            'Expected most continuous values to be off the 0.5 grid'
        )


## ============================================================
## PROPERTY 5: vectorization
## ============================================================

class TestVectorization:
    '''numpy array inputs produce correct array outputs'''

    def test_array_win_prob_to_spread(self, model_mapper):
        wps = np.array([0.50, 0.60, 0.70, 0.80])
        spread = model_mapper.win_prob_to_spread(wps)
        assert isinstance(spread.continuous, np.ndarray), 'continuous should be ndarray'
        assert isinstance(spread.posted, np.ndarray), 'posted should be ndarray'
        assert len(spread.continuous) == 4
        assert len(spread.posted) == 4

    def test_array_spread_to_win_prob(self, model_mapper):
        spreads = np.array([0.0, 3.0, 7.0, 10.0])
        wps = model_mapper.spread_to_win_prob(spreads)
        assert isinstance(wps, np.ndarray)
        assert len(wps) == 4
        ## monotonicity: higher spread → higher win prob ##
        for i in range(len(wps) - 1):
            assert wps[i] < wps[i + 1]

    def test_array_round_trip(self, model_mapper):
        wps = np.array([0.30, 0.50, 0.70, 0.90])
        spread = model_mapper.win_prob_to_spread(wps)
        wps_back = model_mapper.spread_to_win_prob(spread.continuous)
        assert np.allclose(wps_back, wps, atol=ROUNDTRIP_TOL)


## ============================================================
## PROPERTY 6: serialization
## ============================================================

class TestSerialization:
    '''to_file / from_file round-trip preserves params'''

    def test_model_round_trip(self, model_mapper):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_params.json')
            model_mapper.to_file(MapType.MODEL, filepath=filepath)
            loaded = SpreadMapper.from_file(MapType.MODEL, filepath=filepath)
            assert abs(loaded.params.slope - model_mapper.params.slope) < PARAM_TOL
            assert abs(loaded.params.intercept - model_mapper.params.intercept) < PARAM_TOL

    def test_both_types_in_one_file(self, model_mapper, market_mapper):
        '''Writing both MODEL and MARKET to the same file merges correctly'''
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_params.json')
            model_mapper.to_file(MapType.MODEL, filepath=filepath)
            market_mapper.to_file(MapType.MARKET, filepath=filepath)
            ## load both back ##
            loaded_model = SpreadMapper.from_file(MapType.MODEL, filepath=filepath)
            loaded_market = SpreadMapper.from_file(MapType.MARKET, filepath=filepath)
            assert abs(loaded_model.params.slope - model_mapper.params.slope) < PARAM_TOL
            assert abs(loaded_market.params.slope - market_mapper.params.slope) < PARAM_TOL

    def test_from_params_factory(self):
        mapper = SpreadMapper.from_params(slope=6.0, intercept=0.5)
        assert mapper.params.slope == 6.0
        assert mapper.params.intercept == 0.5


## ============================================================
## PROPERTY 7: monotonicity
## ============================================================

class TestMonotonicity:
    '''Higher win probability → higher spread'''

    def test_monotonic_model(self, model_mapper):
        wps = np.linspace(0.10, 0.90, 50)
        spreads = model_mapper.win_prob_to_spread(wps).continuous
        for i in range(len(spreads) - 1):
            assert spreads[i] < spreads[i + 1], (
                f'Monotonicity violated at wp={wps[i]:.4f}: '
                f'{spreads[i]:.4f} >= {spreads[i + 1]:.4f}'
            )

    def test_monotonic_market(self, market_mapper):
        wps = np.linspace(0.10, 0.90, 50)
        spreads = market_mapper.win_prob_to_spread(wps).continuous
        for i in range(len(spreads) - 1):
            assert spreads[i] < spreads[i + 1], (
                f'Monotonicity violated at wp={wps[i]:.4f}: '
                f'{spreads[i]:.4f} >= {spreads[i + 1]:.4f}'
            )
