## Unit tests for the SpreadMapper linear-in-logit spread mapping.
import pytest
import numpy as np
import tempfile
import os
from scipy.special import logit
from nfelotranslation.SpreadMap import SpreadMapper, LinearMapParams, Spread


ROUNDTRIP_TOL = 1e-12
PARAM_TOL = 1e-10


@pytest.fixture(scope='module')
def mapper():
    return SpreadMapper.from_file()


class TestWpRoundTrip:
    WIN_PROBS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]

    @pytest.mark.parametrize('wp', WIN_PROBS)
    def test_round_trip(self, mapper, wp):
        spread = mapper.win_prob_to_spread(wp)
        wp_back = mapper.spread_to_win_prob(spread.continuous)
        assert abs(float(wp_back) - wp) < ROUNDTRIP_TOL


class TestSpreadRoundTrip:
    SPREADS = [-14.0, -10.0, -7.0, -6.5, -3.0, -0.5, 0.0, 0.5, 3.0, 6.5, 7.0, 10.0, 14.0]

    @pytest.mark.parametrize('s', SPREADS)
    def test_spread_round_trip(self, mapper, s):
        p = mapper.spread_to_win_prob(s)
        s_out = mapper.win_prob_to_spread(float(p))
        assert abs(s_out.continuous - s) < ROUNDTRIP_TOL


class TestKnownValues:
    def test_50pct_equals_intercept(self, mapper):
        spread = mapper.win_prob_to_spread(0.50)
        assert abs(spread.continuous - mapper.params.intercept) < 1e-6

    def test_60pct(self, mapper):
        spread = mapper.win_prob_to_spread(0.60)
        expected = mapper.params.slope * logit(0.60) + mapper.params.intercept
        assert abs(spread.continuous - expected) < 1e-6


class TestClamping:
    def test_posted_on_half_grid(self, mapper):
        for wp in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            spread = mapper.win_prob_to_spread(wp)
            remainder = spread.posted % 0.5
            assert abs(remainder) < 1e-10 or abs(remainder - 0.5) < 1e-10


class TestVectorization:
    def test_array_win_prob_to_spread(self, mapper):
        wps = np.array([0.50, 0.60, 0.70, 0.80])
        spread = mapper.win_prob_to_spread(wps)
        assert isinstance(spread.continuous, np.ndarray)
        assert len(spread.continuous) == 4

    def test_array_round_trip(self, mapper):
        wps = np.array([0.30, 0.50, 0.70, 0.90])
        spread = mapper.win_prob_to_spread(wps)
        wps_back = mapper.spread_to_win_prob(spread.continuous)
        assert np.allclose(wps_back, wps, atol=ROUNDTRIP_TOL)


class TestSerialization:
    def test_round_trip(self, mapper):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_params.json')
            mapper.to_file(filepath=filepath)
            loaded = SpreadMapper.from_file(filepath=filepath)
            assert abs(loaded.params.slope - mapper.params.slope) < PARAM_TOL
            assert abs(loaded.params.intercept - mapper.params.intercept) < PARAM_TOL

    def test_from_params_factory(self):
        mapper = SpreadMapper.from_params(slope=6.0, intercept=0.5)
        assert mapper.params.slope == 6.0
        assert mapper.params.intercept == 0.5


class TestMonotonicity:
    def test_monotonic(self, mapper):
        wps = np.linspace(0.10, 0.90, 50)
        spreads = mapper.win_prob_to_spread(wps).continuous
        for i in range(len(spreads) - 1):
            assert spreads[i] < spreads[i + 1]
