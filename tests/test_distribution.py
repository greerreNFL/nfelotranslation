## Unit tests for the BaseDistribution stateful continuous distribution.
##
## Uses generalized normal (gennorm) with shape param beta loaded from
## margin_hyperparams.json — the same config the production model reads —
## so tests always validate against the currently shipped hyperparameter.
##
## Properties tested:
##   1. WP constraint: sf(0) = wp (survival function at 0 recovers win prob)
##   2. Known values: scale at spread=7, wp=0.75 → 7 / gennorm.ppf(0.75, beta)
##   3. Symmetry: BaseDistribution(s, wp).scale = BaseDistribution(-s, 1-wp).scale
##   4. Degenerate case: BaseDistribution(0, 0.5).scale returns fallback
##   5. Monotonicity: higher wp → smaller scale (tighter distribution)
##   6. Region masses: loss_mass + win_mass = 1.0; win_mass = wp
##   7. CDF/SF consistency: cdf(x) + sf(x) = 1.0
##   8. PDF integration: pdf integrates to ~1.0 over wide range
##   9. Continuous spread: non-grid input preserved as loc
##
import pytest
import numpy as np
from scipy.stats import gennorm as scipy_gennorm
from nfelotranslation.Distribution import MARGIN_HYPERPARAMS
from nfelotranslation.Distribution.Base import BaseDistribution


## ============================================================
## TOLERANCES
## ============================================================

WP_TOL    = 1e-10     ## continuous WP constraint
SCALE_TOL = 1e-10     ## known-value scale check
MASS_TOL  = 1e-10     ## region mass consistency
CDF_TOL   = 1e-10     ## CDF + SF = 1
INTEG_TOL = 0.001     ## PDF integration over finite range

## beta from the shipped config — same dict production reads ##
BETA = float(MARGIN_HYPERPARAMS['beta'])


## ============================================================
## PROPERTY 1: P(margin > 0) = wp
## ============================================================

class TestWPConstraint:
    '''sf(0) should exactly recover wp.'''

    CASES = [
        (3.0, 0.60),
        (7.0, 0.75),
        (-3.0, 0.40),
        (10.0, 0.80),
        (1.0, 0.53),
        (14.0, 0.90),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_sf_at_zero(self, spread, wp):
        base = BaseDistribution(spread, wp)
        recovered = float(base.sf(0.0))
        assert abs(recovered - wp) < WP_TOL, (
            f'WP constraint violated: spread={spread}, wp={wp}, '
            f'recovered={recovered:.12f}  err={recovered - wp:+.2e}'
        )


## ============================================================
## PROPERTY 2: known values
## ============================================================

class TestKnownValues:
    '''scale at known inputs matches analytical formula.'''

    def test_scale_7_75(self):
        expected = 7.0 / scipy_gennorm.ppf(0.75, BETA)
        base = BaseDistribution(7.0, 0.75)
        assert abs(base.scale - expected) < SCALE_TOL, (
            f'scale(7, 0.75): expected={expected:.10f}  actual={base.scale:.10f}'
        )

    def test_scale_3_60(self):
        expected = 3.0 / scipy_gennorm.ppf(0.60, BETA)
        base = BaseDistribution(3.0, 0.60)
        assert abs(base.scale - expected) < SCALE_TOL, (
            f'scale(3, 0.60): expected={expected:.10f}  actual={base.scale:.10f}'
        )

## ============================================================
## PROPERTY 3: symmetry
## ============================================================

class TestSymmetry:
    '''scale(s, wp) = scale(-s, 1-wp); pdf(x, s, wp) mirrors pdf(-x, -s, 1-wp).'''

    CASES = [
        (3.0, 0.60),
        (7.0, 0.75),
        (10.0, 0.80),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_scale_mirror(self, spread, wp):
        base_pos = BaseDistribution(spread, wp)
        base_neg = BaseDistribution(-spread, 1.0 - wp)
        assert abs(base_pos.scale - base_neg.scale) < SCALE_TOL, (
            f'scale symmetry violated: scale({spread},{wp})={base_pos.scale:.10f}  '
            f'scale({-spread},{1-wp})={base_neg.scale:.10f}'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_pdf_mirror(self, spread, wp):
        x = np.linspace(-40, 40, 161)
        pdf_pos = BaseDistribution(spread, wp).pdf(x)
        pdf_neg = BaseDistribution(-spread, 1.0 - wp).pdf(-x)
        assert np.allclose(pdf_pos, pdf_neg, atol=1e-12), (
            f'PDF symmetry violated at spread={spread}, wp={wp}: '
            f'max diff={np.max(np.abs(pdf_pos - pdf_neg)):.2e}'
        )


## ============================================================
## PROPERTY 4: degenerate case
## ============================================================

class TestDegenerateCase:
    '''BaseDistribution(0, 0.5) returns fallback without crashing.'''

    def test_pick_em(self):
        base = BaseDistribution(0.0, 0.5)
        assert base.scale == BaseDistribution._FALLBACK_SCALE, (
            f'Pick-em scale should be fallback {BaseDistribution._FALLBACK_SCALE}, got {base.scale}'
        )

    def test_pick_em_pdf(self):
        '''pdf at 0 with pick-em should be the mode density.'''
        base = BaseDistribution(0.0, 0.5)
        val = float(base.pdf(0.0))
        expected = float(scipy_gennorm.pdf(0.0, BETA, loc=0.0, scale=BaseDistribution._FALLBACK_SCALE))
        assert abs(val - expected) < 1e-12

    def test_pick_em_sf(self):
        '''sf(0) should be ~0.5 for pick-em.'''
        base = BaseDistribution(0.0, 0.5)
        val = float(base.sf(0.0))
        assert abs(val - 0.5) < WP_TOL


## ============================================================
## PROPERTY 5: monotonicity — higher wp → smaller scale
## ============================================================

class TestMonotonicity:
    '''At a fixed spread, higher wp means tighter distribution (smaller scale).'''

    def test_scale_decreases_with_wp(self):
        spread = 7.0
        wps = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        scales = [BaseDistribution(spread, wp).scale for wp in wps]
        for i in range(len(scales) - 1):
            assert scales[i] > scales[i + 1], (
                f'Monotonicity violated: scale(wp={wps[i]})={scales[i]:.4f} '
                f'<= scale(wp={wps[i+1]})={scales[i+1]:.4f}'
            )


## ============================================================
## PROPERTY 6: region masses
## ============================================================

class TestRegionMasses:
    '''loss_mass + win_mass = 1.0; win_mass = wp.'''

    CASES = [
        (3.0, 0.60),
        (7.0, 0.75),
        (-3.0, 0.40),
        (0.0, 0.50),
        (10.0, 0.80),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_masses_sum_to_one(self, spread, wp):
        base = BaseDistribution(spread, wp)
        total = base.loss_mass + base.win_mass
        assert abs(total - 1.0) < MASS_TOL, (
            f'Region masses sum to {total:.12f}, expected 1.0'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_win_mass_equals_wp(self, spread, wp):
        base = BaseDistribution(spread, wp)
        assert abs(base.win_mass - wp) < MASS_TOL, (
            f'win_mass={base.win_mass:.12f} != wp={wp}'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_loss_mass_equals_complement(self, spread, wp):
        base = BaseDistribution(spread, wp)
        assert abs(base.loss_mass - (1.0 - wp)) < MASS_TOL, (
            f'loss_mass={base.loss_mass:.12f} != 1-wp={1.0-wp}'
        )


## ============================================================
## PROPERTY 7: CDF + SF = 1.0
## ============================================================

class TestCDFSFConsistency:
    '''cdf(x) + sf(x) = 1.0 at arbitrary points.'''

    def test_sum_to_one(self):
        x = np.array([-10.0, -3.0, 0.0, 3.0, 7.0, 15.0])
        base = BaseDistribution(7.0, 0.75)
        cdf_vals = base.cdf(x)
        sf_vals  = base.sf(x)
        sums = cdf_vals + sf_vals
        assert np.allclose(sums, 1.0, atol=CDF_TOL), (
            f'CDF + SF != 1.0: max deviation={np.max(np.abs(sums - 1.0)):.2e}'
        )


## ============================================================
## PROPERTY 8: PDF integrates to ~1.0
## ============================================================

class TestPDFIntegration:
    '''PDF over a wide range should integrate close to 1.0.'''

    CASES = [
        (3.0, 0.60),
        (7.0, 0.75),
        (0.0, 0.50),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_pdf_integrates(self, spread, wp):
        ## trapezoidal integration over [-100, 100] ##
        x = np.linspace(-100, 100, 10001)
        base = BaseDistribution(spread, wp)
        pdf_vals = base.pdf(x)
        integral = float(np.trapezoid(pdf_vals, x))
        assert abs(integral - 1.0) < INTEG_TOL, (
            f'PDF integral={integral:.6f}, expected ~1.0'
        )


## ============================================================
## PROPERTY 9: continuous spread preserved
## ============================================================

class TestContinuousSpread:
    '''Non-grid spreads are kept continuous (not snapped to 0.5 grid).'''

    CASES = [
        (7.0, 7.0),
        (7.5, 7.5),
        (7.2, 7.2),
        (7.3, 7.3),
        (7.8, 7.8),
        (-3.3, -3.3),
        (-3.1, -3.1),
        (0.0, 0.0),
    ]

    @pytest.mark.parametrize('input_spread,expected', CASES)
    def test_preserved(self, input_spread, expected):
        base = BaseDistribution(input_spread, 0.65)
        assert base.spread == expected, (
            f'Spread {input_spread} should stay {expected}, got {base.spread}'
        )
