'''
Tests for Distribution/Key module — ratio-based NumberOutcome architecture.

Uses KEY_MODEL_PARAMS from the shipped package — the same dict production
consumes — so tests always validate against the currently shipped config.
'''

import numpy as np
import pytest

from nfelotranslation.Distribution.Key import (
    NumberOutcomeRecord, NumberOutcome, KeyModel, KEY_MODEL_PARAMS,
)


## ==================== Config ==================== ##

DEFAULT_PARAMS = KEY_MODEL_PARAMS
## shorthands used in expected-value arithmetic below ##
_FR = float(DEFAULT_PARAMS['forgetting_rate'])
_DECAY = 1.0 - _FR           ## per-season decay factor
_PRIOR_N = int(DEFAULT_PARAMS['initial_prior_size'])
_THR = int(DEFAULT_PARAMS['threshold'])


## ==================== NumberOutcomeRecord ==================== ##

class TestNumberOutcomeRecord:

    def test_serialization_roundtrip(self):
        rec = NumberOutcomeRecord(season=2023, ratio=1.25, eff_hits=50.0, exp_eff_hits=48.0, eff_games=1200.0)
        d = rec.to_dict()
        rec2 = NumberOutcomeRecord.from_dict(d)
        assert rec2.season == 2023
        assert rec2.ratio == pytest.approx(1.25)
        assert rec2.eff_hits == pytest.approx(50.0)
        assert rec2.exp_eff_hits == pytest.approx(48.0)
        assert rec2.eff_games == pytest.approx(1200.0)


## ==================== NumberOutcome ==================== ##

class TestNumberOutcome:

    def test_initial_state(self):
        '''Fresh NumberOutcome has ratio = 1.0 (no excess).'''
        no = NumberOutcome(number=3, params=DEFAULT_PARAMS)
        assert no.get_ratio() == 1.0

    def test_prior_initialization(self):
        '''First update initializes prior from baseline_rate and initial_prior_size.'''
        no = NumberOutcome(number=3, params=DEFAULT_PARAMS)
        no.update(hits=120, n_games=2500, baseline_rate=0.04, season=2023)
        ## prior: eff_hits = _PRIOR_N * 0.04, eff_games = _PRIOR_N ##
        ## then decay by _DECAY, then add this season's counts ##
        assert no.eff_hits == pytest.approx(_PRIOR_N * 0.04 * _DECAY + 120, abs=0.01)
        assert no.exp_eff_hits == pytest.approx(_PRIOR_N * 0.04 * _DECAY + 0.04 * 2500, abs=0.01)
        assert no.eff_games == pytest.approx(_PRIOR_N * _DECAY + 2500, abs=0.01)

    def test_update_mechanics(self):
        '''After update on pre-populated state, counts reflect decay + new.'''
        ## start with state as if prior was already initialized ##
        no = NumberOutcome(
            number=3,
            eff_hits=100.0, exp_eff_hits=95.0, eff_games=2000.0,
            trained_through=2022,
            params=DEFAULT_PARAMS,
        )
        no.update(hits=120, n_games=2500, baseline_rate=0.045, season=2023)
        ## existing state decays by _DECAY, then new season's counts added ##
        assert no.eff_hits == pytest.approx(100 * _DECAY + 120, abs=0.1)
        assert no.exp_eff_hits == pytest.approx(95 * _DECAY + 0.045 * 2500, abs=0.1)
        assert no.eff_games == pytest.approx(2000 * _DECAY + 2500, abs=0.1)

    def test_ratio_math(self):
        '''Hand-computed credibility ratio matches get_ratio().'''
        ## state strong enough that exp_eff_hits >= _THR → full credibility ##
        strong_hits, strong_exp = 200.0, 180.0
        assert strong_exp >= _THR, 'test assumes strong_exp clears the credibility threshold'
        no = NumberOutcome(
            number=3,
            eff_hits=strong_hits, exp_eff_hits=strong_exp, eff_games=4000.0,
            trained_through=2023,
            params=DEFAULT_PARAMS,
        )
        expected = strong_hits / strong_exp
        assert no.get_ratio() == pytest.approx(expected, abs=1e-10)

    def test_credibility_below_threshold(self):
        '''Below threshold, ratio is pulled toward 1.0 by credibility weight.'''
        ## state deliberately below threshold to exercise credibility blending ##
        weak_hits, weak_exp = 3.0, 2.0
        assert weak_exp < _THR, 'test assumes weak_exp is below the credibility threshold'
        no = NumberOutcome(
            number=30,
            eff_hits=weak_hits, exp_eff_hits=weak_exp, eff_games=100.0,
            trained_through=2023,
            params=DEFAULT_PARAMS,
        )
        credibility = weak_exp / _THR
        expected = 1.0 + (weak_hits / weak_exp - 1.0) * credibility
        assert no.get_ratio() == pytest.approx(expected, abs=1e-10)

    def test_strong_signal_converges(self):
        '''Large hits → ratio pulls away from 1.0.'''
        no = NumberOutcome(number=3, params=DEFAULT_PARAMS)
        ## feed many seasons with excess hits ##
        for season in range(2015, 2025):
            no.update(hits=130, n_games=2500, baseline_rate=0.04, season=season)
        ## 130/2500 = 0.052, baseline = 0.04, so ratio should be > 1 ##
        assert no.get_ratio() > 1.0

    def test_noise_stays_near_one(self):
        '''Baseline-rate hits → ratio stays near 1.0.'''
        no = NumberOutcome(number=25, params=DEFAULT_PARAMS)
        for season in range(2015, 2025):
            no.update(hits=100, n_games=2500, baseline_rate=0.04, season=season)
        ## 100/2500 = 0.04 = baseline, ratio ≈ 1.0 ##
        assert abs(no.get_ratio() - 1.0) < 0.05

    def test_multiplicative_excess(self):
        '''excess_at() scales proportionally to baseline — larger baseline → larger excess.'''
        no = NumberOutcome(
            number=3,
            eff_hits=200.0, exp_eff_hits=180.0, eff_games=4000.0,
            trained_through=2023,
            params=DEFAULT_PARAMS,
        )
        ## large baselines (spread near key number) ##
        ex_large_pos, ex_large_neg = no.excess_at(0.05, 0.03)
        ## small baselines (spread far from key number) ##
        ex_small_pos, ex_small_neg = no.excess_at(0.005, 0.003)
        ## larger baseline → larger excess ##
        assert abs(ex_large_pos) > abs(ex_small_pos)
        assert abs(ex_large_neg) > abs(ex_small_neg)
        ## ratio is constant regardless of baseline ##
        ratio = no.get_ratio()
        assert ex_large_pos == pytest.approx((ratio - 1.0) * 0.05, abs=1e-10)
        assert ex_small_pos == pytest.approx((ratio - 1.0) * 0.005, abs=1e-10)

    def test_dead_zone(self):
        '''Dead zone number produces ratio < 1.0.'''
        no = NumberOutcome(number=9, params=DEFAULT_PARAMS)
        ## feed below-baseline hits ##
        for season in range(2015, 2025):
            no.update(hits=70, n_games=2500, baseline_rate=0.04, season=season)
        ## 70/2500 = 0.028 < baseline 0.04, so ratio < 1 ##
        assert no.get_ratio() < 1.0
        ## excess is negative ##
        ex_pos, ex_neg = no.excess_at(0.02, 0.02)
        assert ex_pos < 0.0
        assert ex_neg < 0.0

    def test_history_recording(self):
        '''Each update() appends to history.'''
        no = NumberOutcome(number=7, params=DEFAULT_PARAMS)
        no.update(hits=50, n_games=1000, baseline_rate=0.04, season=2022)
        no.update(hits=55, n_games=1000, baseline_rate=0.04, season=2023)
        assert len(no.history) == 2
        assert no.history[0].season == 2022
        assert no.history[1].season == 2023

    def test_get_state_at(self):
        '''Historical lookup returns correct season.'''
        no = NumberOutcome(number=7, params=DEFAULT_PARAMS)
        no.update(hits=50, n_games=1000, baseline_rate=0.04, season=2022)
        no.update(hits=55, n_games=1000, baseline_rate=0.04, season=2023)
        rec = no.get_state_at(2022)
        assert rec is not None
        assert rec.season == 2022
        assert no.get_state_at(2020) is None

    def test_serialization_roundtrip(self):
        '''to_dict → from_dict preserves state.'''
        no = NumberOutcome(number=3, params=DEFAULT_PARAMS)
        no.update(hits=120, n_games=2500, baseline_rate=0.04, season=2023)
        d = no.to_dict()
        no2 = NumberOutcome.from_dict(d, params=DEFAULT_PARAMS)
        assert no2.number == 3
        assert no2.eff_hits == pytest.approx(no.eff_hits)
        assert no2.exp_eff_hits == pytest.approx(no.exp_eff_hits)
        assert no2.eff_games == pytest.approx(no.eff_games)
        assert len(no2.history) == 1
        assert no2.get_ratio() == pytest.approx(no.get_ratio(), abs=1e-10)

    def test_strict_params_missing_raises(self):
        '''Missing param raises KeyError, not silent default.'''
        no = NumberOutcome(
            number=3, eff_hits=10.0, exp_eff_hits=8.0, eff_games=200.0,
            trained_through=2023, params={'forgetting_rate': 0.087},
        )
        with pytest.raises(KeyError):
            no.get_ratio()  ## needs 'threshold' ##

    def test_no_params_raises(self):
        '''No params dict at all raises ValueError.'''
        no = NumberOutcome(number=3, eff_hits=10.0, exp_eff_hits=8.0, eff_games=200.0, trained_through=2023)
        with pytest.raises(ValueError):
            no.get_ratio()


## ==================== KeyModel ==================== ##

def _make_baseline_pmf(spread=7.0):
    '''Helper: build a simple 151-element baseline PMF centered at spread.'''
    from scipy.stats import norm
    margins = np.arange(-75, 76, dtype=float)
    raw = norm.pdf(margins, loc=spread, scale=13.0)
    return raw / raw.sum()


class TestKeyModel:

    def test_from_initial(self):
        '''Factory creates 40 zero-state NumberOutcomes.'''
        model = KeyModel.from_initial(DEFAULT_PARAMS)
        assert len(model.outcomes) == 40
        assert all(k in model.outcomes for k in range(1, 41))

    def test_empty_model_zero_excess(self):
        '''Fresh model returns ~0 excess for all numbers.'''
        model = KeyModel.from_initial(DEFAULT_PARAMS)
        baseline_pmf = _make_baseline_pmf()
        for k in range(1, 41):
            ex_pos, ex_neg = model.excess_at(k, baseline_pmf)
            assert ex_pos == 0.0
            assert ex_neg == 0.0

    def test_excess_at_delegates(self):
        '''excess_at() delegates to the correct NumberOutcome.'''
        model = KeyModel.from_initial(DEFAULT_PARAMS)
        ## feed data into number 3 only ##
        model.outcomes[3].update(hits=130, n_games=2500, baseline_rate=0.04, season=2023)
        baseline_pmf = _make_baseline_pmf(spread=3.0)
        ex3_pos, ex3_neg = model.excess_at(3, baseline_pmf)
        ex7_pos, ex7_neg = model.excess_at(7, baseline_pmf)
        ## number 3 should have excess, number 7 should not ##
        assert ex3_pos != 0.0
        assert ex7_pos == 0.0

    def test_get_all_excess(self):
        '''Returns dict with entries for all 40 numbers.'''
        model = KeyModel.from_initial(DEFAULT_PARAMS)
        baseline_pmf = _make_baseline_pmf()
        all_excess = model.get_all_excess(baseline_pmf)
        assert len(all_excess) == 40
        assert all(isinstance(v, tuple) and len(v) == 2 for v in all_excess.values())

    def test_excess_at_unknown_raises(self):
        '''Raises ValueError for number not in outcomes.'''
        model = KeyModel.from_initial(DEFAULT_PARAMS)
        baseline_pmf = _make_baseline_pmf()
        with pytest.raises(ValueError):
            model.excess_at(99, baseline_pmf)

    def test_file_roundtrip(self, tmp_path):
        '''to_file → from_file preserves all NumberOutcome state.'''
        model = KeyModel.from_initial(DEFAULT_PARAMS)
        model.outcomes[3].update(hits=130, n_games=2500, baseline_rate=0.04, season=2023)
        model.outcomes[7].update(hits=110, n_games=2500, baseline_rate=0.04, season=2023)
        path = str(tmp_path / 'test_key_state.json')
        model.to_file(path)
        loaded = KeyModel.from_file(path, params=DEFAULT_PARAMS)
        ## verify state preserved ##
        assert len(loaded.outcomes) == 40
        baseline_pmf = _make_baseline_pmf()
        for k in [3, 7]:
            orig_ex = model.excess_at(k, baseline_pmf)
            load_ex = loaded.excess_at(k, baseline_pmf)
            assert load_ex[0] == pytest.approx(orig_ex[0], abs=1e-10)
            assert load_ex[1] == pytest.approx(orig_ex[1], abs=1e-10)
