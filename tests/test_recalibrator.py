## Unit tests for the Recalibrator (Platt / logit-linear recalibration).
##
## The Recalibrator is a shipped class with zero direct coverage before this
## file — all calibration-related tests historically tested the spread mapper's
## self-consistency, not the calibrator itself.
##
## Properties tested:
##   1. calibrate math: calibrate(p) = expit(slope * logit(p) + intercept)
##   2. uncalibrate inverse: uncalibrate(calibrate(p)) = p (and reverse)
##   3. Vectorization: array inputs produce array outputs
##   4. Shipped platt_params.json loads via from_file() with sane values
##   5. from_params factory
##   6. to_file / from_file round-trip preserves params + metadata
##
import pytest
import numpy as np
from scipy.special import logit, expit

from nfelotranslation.Calibration.Recalibrator import Recalibrator
from nfelotranslation.Calibration.Types import PlattParams
from nfelotranslation.Utilities.JsonIo import ConfigMetadata


## ============================================================
## TOLERANCES
## ============================================================

## Platt math is a pure analytic composition of expit/logit;
## precision is limited only by float roundoff.
PLATT_TOL = 1e-12


## ============================================================
## PROPERTY 1: calibrate math
## ============================================================

class TestCalibrateMath:
    '''calibrate(p) = expit(slope * logit(p) + intercept), pointwise.'''

    SLOPE, INTERCEPT = 1.1, 0.05
    WIN_PROBS = [0.10, 0.30, 0.50, 0.55, 0.75, 0.90]

    @pytest.fixture(scope='class')
    def rec(self):
        return Recalibrator.from_params(slope=self.SLOPE, intercept=self.INTERCEPT)

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_matches_analytic(self, rec, p):
        expected = float(expit(self.SLOPE * logit(p) + self.INTERCEPT))
        actual = float(rec.calibrate(np.array([p]))[0])
        assert abs(actual - expected) < PLATT_TOL, (
            f'calibrate({p})={actual:.12f}, expected={expected:.12f}'
        )

    def test_vectorized(self, rec):
        ps = np.array([0.30, 0.50, 0.70])
        out = rec.calibrate(ps)
        assert out.shape == (3,)
        for i, p in enumerate(ps):
            expected = float(expit(self.SLOPE * logit(p) + self.INTERCEPT))
            assert abs(out[i] - expected) < PLATT_TOL

    def test_identity_slope_1_intercept_0(self):
        '''slope=1, intercept=0 → calibrate is identity.'''
        rec = Recalibrator.from_params(slope=1.0, intercept=0.0)
        ps = np.array([0.25, 0.50, 0.75])
        out = rec.calibrate(ps)
        assert np.allclose(out, ps, atol=PLATT_TOL)


## ============================================================
## PROPERTY 2: uncalibrate is the inverse of calibrate
## ============================================================

class TestUncalibrateInverse:
    '''uncalibrate(calibrate(p)) = p and calibrate(uncalibrate(p)) = p.'''

    WIN_PROBS = [0.10, 0.20, 0.50, 0.55, 0.65, 0.80, 0.90]

    @pytest.fixture(scope='class')
    def rec(self):
        ## use actual shipped params (nonzero intercept tests the hardest case) ##
        return Recalibrator.from_params(slope=1.124, intercept=-0.103)

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_calibrate_then_uncalibrate(self, rec, p):
        back = float(rec.uncalibrate(rec.calibrate(np.array([p])))[0])
        assert abs(back - p) < PLATT_TOL, (
            f'cal→uncal round-trip: in={p}, out={back}, err={back - p:.2e}'
        )

    @pytest.mark.parametrize('p', WIN_PROBS)
    def test_uncalibrate_then_calibrate(self, rec, p):
        back = float(rec.calibrate(rec.uncalibrate(np.array([p])))[0])
        assert abs(back - p) < PLATT_TOL, (
            f'uncal→cal round-trip: in={p}, out={back}, err={back - p:.2e}'
        )


## ============================================================
## PROPERTY 3: vectorization
## ============================================================

class TestVectorization:
    '''Array inputs produce array outputs (same shape).'''

    def test_calibrate_array(self):
        rec = Recalibrator.from_params(slope=1.1, intercept=0.0)
        ps = np.array([0.30, 0.50, 0.70, 0.85])
        out = rec.calibrate(ps)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4,)

    def test_uncalibrate_array(self):
        rec = Recalibrator.from_params(slope=1.1, intercept=0.0)
        ps = np.array([0.30, 0.50, 0.70, 0.85])
        out = rec.uncalibrate(ps)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4,)


## ============================================================
## PROPERTY 4: shipped platt_params.json
## ============================================================

class TestShippedParams:
    '''Default from_file() loads the shipped platt_params.json.'''

    def test_loads(self):
        rec = Recalibrator.from_file()
        assert isinstance(rec, Recalibrator)
        assert np.isfinite(rec.params.slope)
        assert np.isfinite(rec.params.intercept)

    def test_slope_reflects_market_bias(self):
        '''Market compresses probabilities toward 50%, so Platt slope > 1.
        Historical fits are ~1.12; a slope <= 1 would indicate either a
        retrain against bad data or a direction flip.'''
        rec = Recalibrator.from_file()
        assert rec.params.slope > 1.0, (
            f'shipped slope={rec.params.slope:.4f} is not > 1.0 — market bias '
            f'should compress toward 50%, yielding slope > 1'
        )

    def test_intercept_bounded(self):
        '''Intercept should be small — |b| < 1 in logit space.
        Large values suggest an unfixed upstream bias leaking into Platt.'''
        rec = Recalibrator.from_file()
        assert abs(rec.params.intercept) < 1.0, (
            f'shipped intercept={rec.params.intercept:.4f} is suspiciously large'
        )

    def test_metadata_has_generated_at(self):
        '''Shipped params must carry a generated_at timestamp for release tracking.'''
        rec = Recalibrator.from_file()
        assert rec.metadata.generated_at is not None


## ============================================================
## PROPERTY 5: from_params factory
## ============================================================

class TestFromParams:

    def test_stores_params(self):
        rec = Recalibrator.from_params(slope=1.5, intercept=0.1)
        assert rec.params.slope == 1.5
        assert rec.params.intercept == 0.1

    def test_empty_metadata(self):
        '''from_params produces a Recalibrator with empty metadata (no training run).'''
        rec = Recalibrator.from_params(slope=1.0, intercept=0.0)
        assert rec.metadata.pipeline_id is None
        assert rec.metadata.generated_at is None


## ============================================================
## PROPERTY 6: serialization round-trip
## ============================================================

class TestSerialization:
    '''to_file / from_file preserves params and metadata.'''

    def test_params_round_trip(self, tmp_path):
        original = Recalibrator(
            PlattParams(slope=1.234, intercept=-0.056),
            metadata=ConfigMetadata(
                pipeline_id='abc12345',
                generated_at='2026-04-23T12:00:00+00:00',
            ),
        )
        path = str(tmp_path / 'platt_params.json')
        original.to_file(path)
        loaded = Recalibrator.from_file(path)
        assert loaded.params.slope == original.params.slope
        assert loaded.params.intercept == original.params.intercept

    def test_metadata_round_trip(self, tmp_path):
        meta = ConfigMetadata(pipeline_id='test0001', generated_at='2026-04-23T00:00:00+00:00')
        original = Recalibrator(PlattParams(slope=1.1, intercept=0.0), metadata=meta)
        path = str(tmp_path / 'platt.json')
        original.to_file(path)
        loaded = Recalibrator.from_file(path)
        assert loaded.metadata.pipeline_id == 'test0001'
        assert loaded.metadata.generated_at == '2026-04-23T00:00:00+00:00'

    def test_envelope_format_on_disk(self, tmp_path):
        '''Written file uses the {metadata, params} envelope format.'''
        import json
        rec = Recalibrator.from_params(slope=1.0, intercept=0.0)
        path = str(tmp_path / 'platt.json')
        rec.to_file(path)
        with open(path) as f:
            raw = json.load(f)
        assert 'metadata' in raw
        assert 'params' in raw
        assert 'slope' in raw['params']
        assert 'intercept' in raw['params']
