## End-to-end smoke tests for the shipped package surface.
##
## These tests exist to catch (a) broken top-level imports, (b) malformed
## or missing shipped JSON configs, (c) breaking changes in the classes a
## consumer imports after `pip install nfelotranslation`.
##
## Every test here exercises ONLY public, shipped API — no reaching into
## training/, no pathlib manipulation. If any of these fails, something
## the end user will hit is broken.
##
import numpy as np
import pytest

## top-level namespace — all of this must import cleanly ##
from nfelotranslation import (
    Recalibrator,
    SpreadMapper,
    MarginDistributionModel,
    Translator,
    PlattParams,
    CalibrationResult,
    BaseDistribution,
    NumberOutcome,
    NumberOutcomeRecord,
    KeyModel,
    Normalizer,
    MarginDistribution,
    MapType,
    LinearMapParams,
    Spread,
    SpreadMapResult,
)
## shipped module-level constants ##
from nfelotranslation.Distribution import MARGIN_HYPERPARAMS
from nfelotranslation.Distribution.Key import KEY_MODEL_PARAMS


## ============================================================
## TOP-LEVEL IMPORT SMOKE
## ============================================================

class TestTopLevelImports:
    '''All public symbols advertised by the package must be importable.'''

    def test_classes_are_classes(self):
        for cls in [Recalibrator, SpreadMapper, MarginDistributionModel,
                    Translator, BaseDistribution, KeyModel,
                    NumberOutcome, Normalizer]:
            assert isinstance(cls, type), f'{cls!r} is not a class'

    def test_dataclasses_importable(self):
        for cls in [PlattParams, CalibrationResult, LinearMapParams,
                    Spread, SpreadMapResult, MarginDistribution,
                    NumberOutcomeRecord]:
            assert isinstance(cls, type)

    def test_maptype_enum_values(self):
        assert MapType.MODEL.value == 'model'
        assert MapType.MARKET.value == 'market'


## ============================================================
## SHIPPED HYPERPARAMS
## ============================================================

class TestShippedHyperparams:
    '''Module-level hyperparam constants are loaded with sane values.'''

    def test_margin_hyperparams_schema(self):
        assert 'beta' in MARGIN_HYPERPARAMS
        assert 'tie_prob' in MARGIN_HYPERPARAMS
        assert np.isfinite(MARGIN_HYPERPARAMS['beta'])
        assert np.isfinite(MARGIN_HYPERPARAMS['tie_prob'])

    def test_margin_beta_in_gennorm_range(self):
        '''beta=2 is Gaussian, beta=1 is Laplace; shipped value is between
        the two (heavier tails than Gaussian).'''
        beta = float(MARGIN_HYPERPARAMS['beta'])
        assert 1.0 <= beta <= 2.0, f'beta={beta} outside expected [1, 2] range'

    def test_margin_tie_prob_bounded(self):
        '''tie_prob is a probability, should be small (ties are rare).'''
        tp = float(MARGIN_HYPERPARAMS['tie_prob'])
        assert 0.0 < tp < 0.05, f'tie_prob={tp} outside expected (0, 0.05) range'

    def test_key_params_schema(self):
        for key in ['forgetting_rate', 'threshold', 'initial_prior_size']:
            assert key in KEY_MODEL_PARAMS, f'missing key_hyperparams key: {key}'

    def test_key_params_positive(self):
        assert 0 < float(KEY_MODEL_PARAMS['forgetting_rate']) < 1
        assert float(KEY_MODEL_PARAMS['threshold']) > 0
        assert float(KEY_MODEL_PARAMS['initial_prior_size']) > 0


## ============================================================
## SHIPPED CONFIG FILES LOAD CLEANLY
## ============================================================

class TestShippedConfigsLoad:
    '''Each shipped JSON config file loads into its class without error.'''

    def test_recalibrator_from_file(self):
        rec = Recalibrator.from_file()
        assert np.isfinite(rec.params.slope)
        assert np.isfinite(rec.params.intercept)

    def test_spread_mapper_model_from_file(self):
        m = SpreadMapper.from_file(MapType.MODEL)
        assert np.isfinite(m.params.slope)
        assert np.isfinite(m.params.intercept)

    def test_spread_mapper_market_from_file(self):
        m = SpreadMapper.from_file(MapType.MARKET)
        assert np.isfinite(m.params.slope)
        assert np.isfinite(m.params.intercept)

    def test_margin_model_from_shipped_hyperparams(self):
        '''MarginDistributionModel composes correctly from shipped hyperparams.'''
        km = KeyModel.from_initial(KEY_MODEL_PARAMS)
        m = MarginDistributionModel(km)
        assert float(m.beta) == float(MARGIN_HYPERPARAMS['beta'])
        assert float(m.tie_prob) == float(MARGIN_HYPERPARAMS['tie_prob'])


## ============================================================
## END-TO-END: TRANSLATOR FROM SHIPPED CONFIGS
## ============================================================

class TestEndToEnd:
    '''A Translator built from shipped configs produces a valid output.'''

    def test_market_spread_flow(self):
        '''Full market_spread → cal_wp → PMF pipeline.'''
        t = Translator(3.0, 'market_spread', season=2025, side='home')
        assert 0.5 < t.win_prob < 1.0
        assert np.isclose(t.pmf.sum(), 1.0, atol=1e-10)
        assert (t.pmf >= 0).all()

    def test_spread_flow(self):
        t = Translator(7.0, 'spread', season=2025, side='home')
        assert 0.5 < t.win_prob < 1.0
        assert np.isclose(t.pmf.sum(), 1.0, atol=1e-10)

    def test_win_prob_flow(self):
        t = Translator(0.62, 'win_prob', season=2025, side='home')
        assert t.win_prob == 0.62
        assert np.isclose(t.pmf.sum(), 1.0, atol=1e-10)

    def test_market_win_prob_flow(self):
        t = Translator(0.60, 'market_win_prob', season=2025, side='home')
        assert abs(t.win_prob_market - 0.60) < 1e-12
        assert np.isclose(t.pmf.sum(), 1.0, atol=1e-10)
