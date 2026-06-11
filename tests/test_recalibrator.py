## Unit tests for the Recalibrator (split Platt recalibration).
import pytest
import numpy as np
from scipy.special import logit, expit

from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Calibration.Types import SplitPlattParams
from nfelotranslation.Utilities.JsonIo import ConfigMetadata


PLATT_TOL = 1e-12


class TestCalibrateMath:
    SLOPES = {'home': 1.2, 'away': 0.97}
    INTERCEPTS = {'home': -0.15, 'away': 0.05}
    WIN_PROBS = [0.10, 0.30, 0.50, 0.55, 0.75, 0.90]

    @pytest.fixture(scope='class')
    def rec(self):
        return Recalibrator.from_params(
            slope_home=self.SLOPES['home'],
            slope_away=self.SLOPES['away'],
            intercept_home=self.INTERCEPTS['home'],
            intercept_away=self.INTERCEPTS['away'],
        )

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_home_fav_matches_analytic(self, rec, p):
        expected = float(expit(self.SLOPES['home'] * logit(p) + self.INTERCEPTS['home']))
        actual = float(rec.calibrate(np.array([p]), is_home_fav=True)[0])
        assert abs(actual - expected) < PLATT_TOL

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_away_fav_matches_analytic(self, rec, p):
        expected = float(expit(self.SLOPES['away'] * logit(p) + self.INTERCEPTS['away']))
        actual = float(rec.calibrate(np.array([p]), is_home_fav=False)[0])
        assert abs(actual - expected) < PLATT_TOL

    def test_vectorized(self, rec):
        ps = np.array([0.30, 0.50, 0.70])
        out = rec.calibrate(ps, is_home_fav=True)
        assert out.shape == (3,)


class TestUncalibrateInverse:
    WIN_PROBS = [0.10, 0.20, 0.50, 0.55, 0.65, 0.80, 0.90]

    @pytest.fixture(scope='class')
    def rec(self):
        return Recalibrator.from_params(
            slope_home=1.211, slope_away=0.973,
            intercept_home=-0.15, intercept_away=0.05,
        )

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_calibrate_then_uncalibrate_home(self, rec, p):
        back = float(rec.uncalibrate(rec.calibrate(np.array([p]), is_home_fav=True), is_home_fav=True)[0])
        assert abs(back - p) < PLATT_TOL

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_calibrate_then_uncalibrate_away(self, rec, p):
        back = float(rec.uncalibrate(rec.calibrate(np.array([p]), is_home_fav=False), is_home_fav=False)[0])
        assert abs(back - p) < PLATT_TOL


class TestShippedParams:
    def test_loads(self):
        rec = Recalibrator.from_file()
        assert isinstance(rec, Recalibrator)
        assert np.isfinite(rec.params.slopes['home'])
        assert np.isfinite(rec.params.intercepts['home'])

    def test_home_slope_reflects_market_bias(self):
        rec = Recalibrator.from_file()
        assert rec.params.slopes['home'] > 1.0

    def test_seasonal_from_file(self):
        rec = Recalibrator.from_file(season=2020)
        assert rec.params.fit is not None
        assert 2020 in rec.params.fit['seasons_used']

    def test_metadata_has_generated_at(self):
        rec = Recalibrator.from_file()
        assert rec.metadata.generated_at is not None


class TestFromParams:
    def test_stores_params(self):
        rec = Recalibrator.from_params(1.5, 0.9, 0.1, -0.05)
        assert rec.params.slopes['home'] == 1.5
        assert rec.params.intercepts['away'] == -0.05


class TestSerialization:
    def test_params_round_trip(self, tmp_path):
        original = Recalibrator(
            SplitPlattParams(
                slopes={'home': 1.234, 'away': 0.987},
                intercepts={'home': -0.056, 'away': 0.012},
            ),
            metadata=ConfigMetadata(
                pipeline_id='abc12345',
                generated_at='2026-04-23T12:00:00+00:00',
            ),
        )
        path = str(tmp_path / 'platt_params.json')
        original.to_file(path)
        loaded = Recalibrator.from_file(path)
        assert loaded.params.slopes == original.params.slopes
        assert loaded.params.intercepts == original.params.intercepts

    def test_envelope_format_on_disk(self, tmp_path):
        import json
        rec = Recalibrator.from_params(1.0, 1.0, 0.0, 0.0)
        path = str(tmp_path / 'platt.json')
        rec.to_file(path)
        with open(path) as f:
            raw = json.load(f)
        assert 'metadata' in raw
        assert 'params' in raw
        assert 'slopes' in raw['params']
        assert 'intercepts' in raw['params']
