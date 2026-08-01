## Unit tests for MarginDistributionModel, MarginDistribution, and Normalizer.
##
## Properties tested:
##   1. PMF sums to 1.0
##   2. P(margin > 0) from PMF ≈ win_prob
##   3. PMF is all non-negative
##   4. Key number bins have more mass than base-only for strong numbers
##   5. cover_prob / push_prob produce sensible values
##   6. win_prob_from_pmf matches input
##   7. expected_margin is near spread
##   8. Works for integer, half-integer, and pick-em spreads
##   9. Symmetry: predict(s, wp) mirrors predict(-s, 1-wp)
##  10. Normalizer: discretize returns valid PMF
##  11. Normalizer: normalize preserves region targets
##  12. Spread bisection and near-zero exception policy
##  13. Region probs: tie_prob_from_pmf and loss_prob_from_pmf
##
import pytest
import numpy as np
from nfelotranslation.Distribution import (
    BaseDistribution, KeyModel, Normalizer,
    MarginDistribution, MarginDistributionModel,
    MARGIN_HYPERPARAMS,
)
from nfelotranslation.Distribution.Key import KEY_MODEL_PARAMS


## ============================================================
## TOLERANCES
## ============================================================

PMF_SUM_TOL = 1e-10     ## PMF sums to 1
WP_TOL      = 1e-10     ## WP constraint is exact — normalization enforces P(margin>0) = wp
MARGIN_TOL  = 1.0       ## expected margin near spread
NORM_TOL    = 1e-10     ## normalized PMF sum = 1

## shipped tie_prob — same dict production reads ##
TIE_PROB = float(MARGIN_HYPERPARAMS['tie_prob'])


## ============================================================
## FIXTURES
## ============================================================

@pytest.fixture(scope='module')
def key_model():
    '''Fresh KeyModel with shipped hyperparams (all zero excess).'''
    return KeyModel.from_initial(KEY_MODEL_PARAMS)


@pytest.fixture(scope='module')
def model(key_model):
    return MarginDistributionModel(key_model)


@pytest.fixture(scope='module')
def trained_key_model():
    '''KeyModel with some trained numbers for excess testing.'''
    km = KeyModel.from_initial(KEY_MODEL_PARAMS)
    ## simulate strong key numbers: 3 and 7 with high excess ##
    km.outcomes[3].update(hits=200, n_games=2500, baseline_rate=0.04, season=2023)
    km.outcomes[7].update(hits=250, n_games=2500, baseline_rate=0.04, season=2023)
    return km


@pytest.fixture(scope='module')
def trained_model(trained_key_model):
    return MarginDistributionModel(trained_key_model)


## ============================================================
## PROPERTY 1: PMF sums to 1.0
## ============================================================

class TestPMFSum:
    '''PMF from predict() should sum to 1.0.'''

    CASES = [
        (7.0, 0.75),      ## integer positive
        (3.5, 0.62),      ## half-int positive
        (0.0, 0.50),      ## pick-em
        (-7.0, 0.25),     ## integer negative
        (-3.5, 0.38),     ## half-int negative
        (14.0, 0.90),     ## large favourite
        (1.0, 0.53),      ## small favourite
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_sum(self, model, spread, wp):
        result = model.predict(spread, wp)
        assert abs(result.pmf.sum() - 1.0) < PMF_SUM_TOL, (
            f'PMF sum={result.pmf.sum():.12f}, expected 1.0'
        )


## ============================================================
## PROPERTY 2: P(margin > 0) ≈ wp
## ============================================================

class TestWPRecovery:
    '''win_prob_from_pmf should be close to input wp.'''

    CASES = [
        (7.0, 0.75),
        (3.0, 0.60),
        (0.0, 0.50),
        (-3.0, 0.40),
        (10.0, 0.80),
        (3.5, 0.62),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_wp_recovery(self, model, spread, wp):
        result = model.predict(spread, wp)
        recovered = result.win_prob_from_pmf()
        assert abs(recovered - wp) < WP_TOL, (
            f'WP recovery: input={wp}, recovered={recovered:.6f}, '
            f'err={recovered - wp:+.4f}'
        )


## ============================================================
## PROPERTY 3: PMF non-negative
## ============================================================

class TestNonNegative:
    '''PMF should have no negative values.'''

    CASES = [
        (7.0, 0.75),
        (0.0, 0.50),
        (-7.0, 0.25),
        (3.5, 0.62),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_non_negative(self, model, spread, wp):
        result = model.predict(spread, wp)
        assert (result.pmf >= 0).all(), (
            f'PMF has negative values: min={result.pmf.min():.2e}'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_trained_non_negative(self, trained_model, spread, wp):
        result = trained_model.predict(spread, wp)
        assert (result.pmf >= 0).all(), (
            f'Trained PMF has negative values: min={result.pmf.min():.2e}'
        )


## ============================================================
## PROPERTY 4: key number excess
## ============================================================

class TestKeyNumberExcess:
    '''Trained key numbers should have more mass than base-only.'''

    def test_key_3_has_excess(self, model, trained_model):
        '''margin=3 should be higher with trained key model.'''
        base_result = model.predict(7.0, 0.75)
        trained_result = trained_model.predict(7.0, 0.75)
        ## margin 3 → index 78 ##
        assert trained_result.pmf[78] > base_result.pmf[78], (
            f'Key number 3 should have more mass when trained: '
            f'base={base_result.pmf[78]:.6f}, trained={trained_result.pmf[78]:.6f}'
        )

    def test_key_7_has_excess(self, model, trained_model):
        '''margin=7 should be higher with trained key model at a non-7 spread.'''
        ## use spread=3 so margin=7 falls in region D, not on the C boundary ##
        base_result = model.predict(3.0, 0.60)
        trained_result = trained_model.predict(3.0, 0.60)
        ## margin 7 → index 82 ##
        assert trained_result.pmf[82] > base_result.pmf[82], (
            f'Key number 7 should have more mass when trained: '
            f'base={base_result.pmf[82]:.6f}, trained={trained_result.pmf[82]:.6f}'
        )


## ============================================================
## PROPERTY 5: cover_prob / push_prob
## ============================================================

class TestCoverPush:
    '''cover_prob and push_prob produce sensible values.'''

    def test_cover_integer_line(self, model):
        result = model.predict(7.0, 0.75)
        ## cover at 6.5 should be > cover at 7.5 ##
        cover_6_5 = result.cover_prob(6.5)
        cover_7_5 = result.cover_prob(7.5)
        assert cover_6_5 > cover_7_5, (
            f'cover(6.5)={cover_6_5:.6f} should be > cover(7.5)={cover_7_5:.6f}'
        )

    def test_push_integer_line(self, model):
        result = model.predict(7.0, 0.75)
        ## push at 7 should be nonzero ##
        push_7 = result.push_prob(7)
        assert push_7 > 0, f'push(7)={push_7:.6f} should be > 0'

    def test_push_half_line(self, model):
        result = model.predict(7.0, 0.75)
        ## push at 7.5 should be 0 ##
        push_7_5 = result.push_prob(7.5)
        assert push_7_5 == 0.0, f'push(7.5)={push_7_5:.6f} should be 0'

    def test_cover_bounds(self, model):
        result = model.predict(7.0, 0.75)
        ## cover at -100 should be ~1.0, cover at 100 should be ~0.0 ##
        assert result.cover_prob(-100) == 1.0
        assert result.cover_prob(100) == 0.0

    def test_cover_plus_push_plus_loss(self, model):
        '''cover(k) + push(k) + P(margin < k) should sum to ~1.0 for integer k.'''
        result = model.predict(7.0, 0.75)
        k = 7
        cover = result.cover_prob(k)
        push = result.push_prob(k)
        ## P(margin < k) = sum of bins at margins -75..k-1 ##
        loss_side = float(result.pmf[:k + 75].sum())
        total = cover + push + loss_side
        assert abs(total - 1.0) < 1e-10, (
            f'cover + push + loss = {total:.12f}, expected 1.0'
        )


## ============================================================
## PROPERTY 6: win_prob_from_pmf
## ============================================================

class TestWinProbFromPMF:
    '''win_prob_from_pmf returns sum of positive margin bins.'''

    def test_consistency_with_cover(self, model):
        result = model.predict(7.0, 0.75)
        ## win_prob_from_pmf = cover_prob(0) since both = P(margin > 0) ##
        wp = result.win_prob_from_pmf()
        cover_0 = result.cover_prob(0)
        assert abs(wp - cover_0) < 1e-12, (
            f'win_prob_from_pmf={wp:.12f} != cover_prob(0)={cover_0:.12f}'
        )


## ============================================================
## PROPERTY 7: expected margin near spread
## ============================================================

class TestExpectedMargin:
    '''Expected margin should be close to spread (mean of normal).'''

    CASES = [
        (7.0, 0.75),
        (3.0, 0.60),
        (0.0, 0.50),
        (-3.0, 0.40),
        (3.5, 0.62),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_expected_margin(self, model, spread, wp):
        result = model.predict(spread, wp)
        em = result.expected_margin()
        assert abs(em - result.spread) < MARGIN_TOL, (
            f'E[margin]={em:.4f}, spread={result.spread}, diff={em - result.spread:+.4f}'
        )


## ============================================================
## PROPERTY 8: result container fields
## ============================================================

class TestResultContainer:
    '''MarginDistribution has correct shape and fields.'''

    def test_shape(self, model):
        result = model.predict(7.0, 0.75)
        assert result.pmf.shape == (151,)

    def test_spread_clamped(self, model):
        result = model.predict(7.3, 0.72)
        assert result.spread == 7.5

    def test_win_prob_passed_through(self, model):
        result = model.predict(7.0, 0.75)
        assert result.win_prob == 0.75

    def test_tie_prob_passed_through(self, model):
        result = model.predict(7.0, 0.75)
        assert result.tie_prob == TIE_PROB

    def test_isinstance(self, model):
        result = model.predict(7.0, 0.75)
        assert isinstance(result, MarginDistribution)


## ============================================================
## PROPERTY 9: symmetry
## ============================================================

class TestSymmetry:
    '''predict(s, wp) should mirror predict(-s, 1-wp).'''

    CASES = [
        (7.0, 0.75),
        (3.0, 0.60),
        (3.5, 0.62),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_pmf_mirror(self, model, spread, wp):
        result_pos = model.predict(spread, wp)
        result_neg = model.predict(-spread, 1.0 - wp)
        ## pmf at margin k for (s, wp) should ≈ pmf at margin -k for (-s, 1-wp) ##
        pmf_flipped = result_neg.pmf[::-1]
        ## Nonzero tie probability prevents an exact mirror because the paired
        ## inputs use 1-wp while loss mass is 1-wp-tp.  The residual remains
        ## bounded by O(tie_prob). ##
        assert np.allclose(result_pos.pmf, pmf_flipped, atol=TIE_PROB * 0.55), (
            f'Symmetry violated at spread={spread}: '
            f'max diff={np.max(np.abs(result_pos.pmf - pmf_flipped)):.2e}'
        )


## ============================================================
## PROPERTY 10: Normalizer discretize
## ============================================================

class TestNormalizerDiscretize:
    '''Normalizer.discretize() returns a valid PMF.'''

    CASES = [
        (7.0, 0.75),
        (0.0, 0.50),
        (-3.0, 0.40),
        (3.5, 0.62),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_shape_and_sum(self, spread, wp):
        base = BaseDistribution(spread, wp)
        normalizer = Normalizer(base)
        pmf = normalizer.discretize()
        assert pmf.shape == (151,), f'Expected shape (151,), got {pmf.shape}'
        assert abs(pmf.sum() - 1.0) < NORM_TOL, (
            f'PMF sum={pmf.sum():.12f}, expected 1.0'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_non_negative(self, spread, wp):
        base = BaseDistribution(spread, wp)
        normalizer = Normalizer(base)
        pmf = normalizer.discretize()
        assert (pmf >= 0).all(), 'PMF contains negative values'


## ============================================================
## PROPERTY 11: Normalizer normalize preserves region targets
## ============================================================

class TestNormalizerNormalize:
    '''Normalizer.normalize() scales regions to match targets.'''

    CASES = [
        (7.0, 0.75),     ## integer positive
        (3.5, 0.62),     ## half-int positive
        (0.0, 0.50),     ## pick-em
        (-7.0, 0.25),    ## integer negative
        (-3.5, 0.38),    ## half-int negative
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_sums_to_one(self, spread, wp):
        base = BaseDistribution(spread, wp)
        normalizer = Normalizer(base)
        dirty = normalizer.discretize()
        ## add some noise to simulate key adjustments ##
        dirty[78] += 0.005   ## margin 3
        dirty[72] += 0.003   ## margin -3
        dirty = np.maximum(dirty, 0.0)
        pmf = normalizer.normalize(dirty)
        assert abs(pmf.sum() - 1.0) < 1e-8, (
            f'Normalized PMF sum={pmf.sum():.12f}, expected 1.0'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_non_negative(self, spread, wp):
        base = BaseDistribution(spread, wp)
        normalizer = Normalizer(base)
        pmf = normalizer.normalize(normalizer.discretize())
        assert (pmf >= 0).all(), 'Normalized PMF contains negative values'

    def test_identity_on_clean_discretize(self):
        '''normalize(discretize()) should not change the base PMF much.'''
        base = BaseDistribution(7.0, 0.75)
        normalizer = Normalizer(base)
        clean = normalizer.discretize()
        normed = normalizer.normalize(clean)
        ## differs by O(tie_prob) ≈ 1e-2 because normalization ##
        ## enforces P(margin=0)=tie_prob exactly while discretize() doesn't ##
        assert np.allclose(clean, normed, atol=3e-2), (
            f'Normalize changed clean PMF: max diff={np.max(np.abs(clean - normed)):.2e}'
        )


# ============================================================
# PROPERTY 12: spread bisection
# ============================================================

class TestSpreadBisection:
    '''Feasible posted spreads should exactly bisect the discrete PMF.'''

    CASES = [
        (step / 2, 0.5 + 0.03 * (step / 2))
        for step in range(-20, 21)
        if abs(step) >= 2
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_exact_bisection(self, model, spread, wp):
        result = model.predict(spread, wp)
        margins = np.arange(-75, 76)
        below = float(result.pmf[margins < spread].sum())
        above = float(result.pmf[margins > spread].sum())
        assert abs(above - below) < 1e-10, (
            f'Bisection failed at spread={spread}: above={above:.12f}, below={below:.12f}'
        )

    @pytest.mark.parametrize('spread,wp', [(7.0, 0.75), (-7.0, 0.25)])
    def test_integer_push_preserved(self, spread, wp):
        base = BaseDistribution(spread, wp)
        normalizer = Normalizer(base)
        dirty = normalizer.discretize()
        idx = int(spread) + 75
        dirty[idx] += 0.01
        push = dirty[idx]
        pmf = normalizer.normalize(dirty)
        assert abs(pmf[idx] - push) < 1e-12

    @pytest.mark.parametrize('spread,wp', [(1.0, 0.54), (-1.0, 0.46)])
    def test_one_point_adjusts_push(self, model, spread, wp):
        result = model.predict(spread, wp)
        margins = np.arange(-75, 76)
        below = float(result.pmf[margins < spread].sum())
        above = float(result.pmf[margins > spread].sum())
        assert abs(above - below) < 1e-10
        if spread > 0:
            expected_push = 2.0 * wp - 1.0
        else:
            expected_push = 1.0 - 2.0 * wp - 2.0 * result.tie_prob
        assert abs(result.push_prob(spread) - expected_push) < 1e-10

    @pytest.mark.parametrize('spread,wp', [
        (0.0, 0.50),
        (0.5, 0.52),
        (-0.5, 0.48),
    ])
    def test_near_zero_preserves_side_targets(self, model, spread, wp):
        result = model.predict(spread, wp)
        assert abs(result.win_prob_from_pmf() - wp) < WP_TOL
        assert abs(result.tie_prob_from_pmf() - result.tie_prob) < WP_TOL
        assert abs(result.loss_prob_from_pmf() - (1.0 - wp - result.tie_prob)) < WP_TOL
        margins = np.arange(-75, 76)
        below = float(result.pmf[margins < spread].sum())
        above = float(result.pmf[margins > spread].sum())
        if spread > 0:
            expected_error = 2.0 * wp - 1.0
        elif spread < 0:
            expected_error = 2.0 * (wp + result.tie_prob) - 1.0
        else:
            expected_error = 2.0 * wp + result.tie_prob - 1.0
        assert abs((above - below) - expected_error) < 1e-10

    def test_infeasible_target_raises(self):
        base = BaseDistribution(2.0, 0.51)
        normalizer = Normalizer(base)
        dirty = normalizer.discretize()
        dirty[77] = 0.20
        with pytest.raises(ValueError, match='B target is negative at spread=2.0'):
            normalizer.normalize(dirty)


## ============================================================
## PROPERTY 13: region probs from PMF
## ============================================================

class TestRegionProbs:
    '''tie_prob_from_pmf and loss_prob_from_pmf return correct values.'''

    CASES = [
        (7.0, 0.75),
        (0.0, 0.50),
        (-7.0, 0.25),
        (3.5, 0.62),
    ]

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_tie_prob_recovery(self, model, spread, wp):
        result = model.predict(spread, wp)
        recovered = result.tie_prob_from_pmf()
        assert abs(recovered - result.tie_prob) < 1e-10, (
            f'tie_prob_from_pmf={recovered:.12f} != tie_prob={result.tie_prob}'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_loss_prob_recovery(self, model, spread, wp):
        result = model.predict(spread, wp)
        recovered = result.loss_prob_from_pmf()
        expected = 1.0 - wp - result.tie_prob
        assert abs(recovered - expected) < 1e-10, (
            f'loss_prob_from_pmf={recovered:.12f} != expected={expected}'
        )

    @pytest.mark.parametrize('spread,wp', CASES)
    def test_regions_sum_to_one(self, model, spread, wp):
        result = model.predict(spread, wp)
        total = result.win_prob_from_pmf() + result.tie_prob_from_pmf() + result.loss_prob_from_pmf()
        assert abs(total - 1.0) < 1e-10, (
            f'Region probs sum to {total:.12f}, expected 1.0'
        )
