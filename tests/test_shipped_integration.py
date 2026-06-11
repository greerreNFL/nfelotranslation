## End-to-end smoke tests for the shipped package surface.
import numpy as np
import pytest

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
    LinearMapParams,
    Spread,
    SpreadMapResult,
)
from nfelotranslation.Distribution import MARGIN_HYPERPARAMS
from nfelotranslation.Distribution.Key import KEY_MODEL_PARAMS


class TestTopLevelImports:
    def test_classes_are_classes(self):
        for cls in [Recalibrator, SpreadMapper, MarginDistributionModel,
                    Translator, BaseDistribution, KeyModel,
                    NumberOutcome, Normalizer]:
            assert isinstance(cls, type)

    def test_dataclasses_importable(self):
        for cls in [PlattParams, CalibrationResult, LinearMapParams,
                    Spread, SpreadMapResult, MarginDistribution,
                    NumberOutcomeRecord]:
            assert isinstance(cls, type)


class TestShippedHyperparams:
    def test_margin_hyperparams_schema(self):
        assert 'beta' in MARGIN_HYPERPARAMS
        assert 'tie_prob' in MARGIN_HYPERPARAMS

    def test_key_params_schema(self):
        for key in ['forgetting_rate', 'threshold', 'initial_prior_size']:
            assert key in KEY_MODEL_PARAMS


class TestShippedConfigsLoad:
    def test_recalibrator_from_file(self):
        rec = Recalibrator.from_file()
        assert np.isfinite(rec.params.slopes['home'])
        assert np.isfinite(rec.params.intercepts['home'])

    def test_spread_mapper_from_file(self):
        m = SpreadMapper.from_file()
        assert np.isfinite(m.params.slope)
        assert np.isfinite(m.params.intercept)

    def test_margin_model_from_shipped_hyperparams(self):
        km = KeyModel.from_initial(KEY_MODEL_PARAMS)
        m = MarginDistributionModel(km)
        assert float(m.beta) == float(MARGIN_HYPERPARAMS['beta'])


class TestEndToEnd:
    def test_spread_flow(self):
        t = Translator(3.0, 'spread', season=2025, side='home')
        assert 0.5 < t.win_prob < 1.0
        assert np.isclose(t.pmf.sum(), 1.0, atol=1e-10)
        assert (t.pmf >= 0).all()

    def test_win_prob_flow(self):
        t = Translator(0.62, 'win_prob', season=2025, side='home')
        assert t.win_prob == 0.62
        assert np.isclose(t.pmf.sum(), 1.0, atol=1e-10)
