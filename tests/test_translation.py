## Unit tests for the Translator user-facing API.
##
## Properties tested:
##   1. Init and basic properties
##   2. Input type consistency: win_prob and spread produce valid distributions
##   3. Side handling: home/away perspectives are consistent
##   4. Update reuse: update() recomputes without reloading models
##   5. Spread round-trip through the full pipeline
##   6. PMF validity
##   7. Pick-em edge case
##   8. Validation: bad input_type and bad side raise ValueError
##   9. Spread sign convention
##  10. Seasonal config
##
import pytest
import numpy as np
from nfelotranslation import Translator
from nfelotranslation.Distribution import MARGIN_HYPERPARAMS


PMF_SUM_TOL = 1e-10
WP_TOL      = 1e-12
TIE_PROB = float(MARGIN_HYPERPARAMS['tie_prob'])


@pytest.fixture(scope='module')
def home_translator():
    return Translator(3.0, 'spread', season=2025, side='home')

@pytest.fixture(scope='module')
def away_translator():
    return Translator(3.0, 'spread', season=2025, side='away')


class TestBasicProperties:
    def test_win_prob_in_range(self, home_translator):
        assert 0.0 < home_translator.win_prob < 1.0

    def test_spread_is_spread(self, home_translator):
        from nfelotranslation.SpreadMap.Types import Spread
        assert isinstance(home_translator.spread, Spread)

    def test_pmf_shape(self, home_translator):
        assert home_translator.pmf.shape == (151,)

    def test_tie_prob(self, home_translator):
        assert home_translator.tie_prob == TIE_PROB

    def test_expected_margin_near_spread(self, home_translator):
        assert abs(home_translator.expected_margin - home_translator.spread.posted) < 0.5


class TestInputTypes:
    def test_spread(self):
        t = Translator(3.0, 'spread', season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert t.win_prob > 0.5

    def test_win_prob(self):
        t = Translator(0.62, 'win_prob', season=2025)
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert t.win_prob == 0.62


class TestSideHandling:
    def test_home_away_wp_sum(self, home_translator):
        total = home_translator.home_win_prob + home_translator.away_win_prob + TIE_PROB
        assert abs(total - 1.0) < 1e-10

    def test_side_symmetry(self, home_translator, away_translator):
        assert abs(home_translator.home_win_prob - away_translator.home_win_prob) < 1e-10
        assert abs(home_translator.away_win_prob - away_translator.away_win_prob) < 1e-10

    def test_away_win_prob_is_perspective(self, away_translator):
        assert abs(away_translator.win_prob - away_translator.away_win_prob) < 1e-10

    def test_spread_sign_flips_with_side(self, home_translator, away_translator):
        assert abs(home_translator.spread.continuous + away_translator.spread.continuous) < 1e-10
        assert abs(home_translator.spread.posted + away_translator.spread.posted) < 1e-10


class TestUpdate:
    def test_update_changes_wp(self):
        t = Translator(3.0, 'spread', season=2025)
        wp_before = t.win_prob
        t.update(7.0, 'spread')
        assert t.win_prob != wp_before
        assert t.win_prob > wp_before

    def test_update_changes_side(self):
        t = Translator(3.0, 'spread', season=2025, side='home')
        home_wp = t.win_prob
        t.update(3.0, 'spread', side='away')
        assert abs(t.win_prob - (1.0 - home_wp - TIE_PROB)) < 1e-10

    def test_update_preserves_side_when_none(self):
        t = Translator(3.0, 'spread', season=2025, side='away')
        t.update(7.0, 'spread')
        assert abs(t.win_prob - t.away_win_prob) < 1e-10

    def test_pmf_valid_after_update(self):
        t = Translator(3.0, 'spread', season=2025)
        t.update(0.65, 'win_prob')
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL
        assert (t.pmf >= 0).all()


class TestSpreadRoundTrip:
    SPREADS = [-14.0, -7.0, -3.0, -1.5, 0.0, 1.5, 3.0, 7.0, 14.0]

    @pytest.mark.parametrize('s', SPREADS)
    def test_spread_round_trip_home(self, s):
        t = Translator(s, 'spread', season=2025, side='home')
        assert abs(t.spread.continuous - s) < WP_TOL, (
            f'spread round-trip: in={s}, out={t.spread.continuous}'
        )


class TestPMFValidity:
    CASES = [
        (-14.0, 'spread'),
        (-7.0, 'spread'),
        (-3.0, 'spread'),
        (0.0, 'spread'),
        (3.0, 'spread'),
        (7.0, 'spread'),
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


class TestPickEm:
    def test_spread_zero(self):
        t = Translator(0.0, 'spread', season=2025)
        assert not np.any(np.isnan(t.pmf))
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL

    def test_win_prob_50(self):
        t = Translator(0.50, 'win_prob', season=2025)
        assert not np.any(np.isnan(t.pmf))
        assert abs(t.pmf.sum() - 1.0) < PMF_SUM_TOL


class TestValidation:
    def test_bad_input_type(self):
        with pytest.raises(ValueError, match='input_type'):
            Translator(-3.0, 'bad_type', season=2025)

    def test_bad_side(self):
        with pytest.raises(ValueError, match='side'):
            Translator(-3.0, 'spread', season=2025, side='favorite')

    def test_bad_update_input_type(self):
        t = Translator(-3.0, 'spread', season=2025)
        with pytest.raises(ValueError, match='input_type'):
            t.update(-3.0, 'invalid')

    def test_bad_update_side(self):
        t = Translator(-3.0, 'spread', season=2025)
        with pytest.raises(ValueError, match='side'):
            t.update(-3.0, 'spread', side='neutral')


class TestSpreadSignConvention:
    def test_positive_spread_is_home_favourite(self):
        t = Translator(7.0, 'spread', season=2025, side='home')
        assert t.win_prob > 0.5
        assert t.spread.continuous > 0

    def test_negative_spread_is_away_favourite(self):
        t = Translator(-7.0, 'spread', season=2025, side='home')
        assert t.win_prob < 0.5
        assert t.spread.continuous < 0


class TestSeasonalConfig:
    def test_shipped_seasonal_configs_differ(self):
        from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper
        from nfelotranslation.Utilities.JsonIo import find_config_path
        import pathlib, nfelotranslation.SpreadMap as _sm
        config_dir = str(pathlib.Path(_sm.__file__).resolve().parent / 'configs')
        path_2020 = find_config_path('spread_map_params', 2020, config_dir)
        path_2025 = find_config_path('spread_map_params', 2025, config_dir)
        assert path_2020 is not None and path_2025 is not None
        assert path_2020 != path_2025
        m2020 = SpreadMapper.from_file(filepath=path_2020)
        m2025 = SpreadMapper.from_file(filepath=path_2025)
        params_differ = (
            m2020.params.slope != m2025.params.slope
            or m2020.params.intercept != m2025.params.intercept
        )
        assert params_differ

    def test_translator_reflects_seasonal_config(self):
        from nfelotranslation.SpreadMap.SpreadMapper import SpreadMapper
        from nfelotranslation.Utilities.JsonIo import find_config_path
        import pathlib, nfelotranslation.SpreadMap as _sm
        config_dir = str(pathlib.Path(_sm.__file__).resolve().parent / 'configs')
        for season in (2020, 2025):
            t = Translator(7.0, 'spread', season=season)
            path = find_config_path('spread_map_params', season, config_dir)
            mapper = SpreadMapper.from_file(filepath=path)
            expected_spread = mapper.win_prob_to_spread(t.win_prob).continuous
            assert abs(t.spread.continuous - expected_spread) < 1e-10


class TestOutOfDomainSeason:
    def test_past_latest_season_warns_and_loads(self):
        import warnings
        future = 2099
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t = Translator(3.0, 'spread', season=future, side='home')
        assert any('falling back' in str(w.message).lower() for w in caught)
        assert t.win_prob > 0.5
        assert abs(t.pmf.sum() - 1.0) < 1e-10
        ratio_3 = t._margin_model.key_model.outcomes[3].get_ratio()
        assert ratio_3 != 1.0

    def test_before_earliest_season_raises(self):
        with pytest.raises(FileNotFoundError):
            Translator(3.0, 'spread', season=1990, side='home')
