'''
Shared data loader for the analyses in this directory.

Pulls games and market data via nfelodcm, computes derived fields used
across the analyses, and caches the merged frame to disk for fast reuse.
'''

## external ##
import numpy
import pandas
import nfelodcm

## local ##
from .utils import DATA_DIR, hold_adj_probs, spread_to_prob_538

## package (shipping recalibration params) ##
from nfelotranslation.Calibration.Recalibrator import Recalibrator


_CACHE_PATH = DATA_DIR / 'merged.csv'


def load_data(force_refresh: bool = False) -> pandas.DataFrame:
    '''
    Load the merged games + market data frame used by the analyses.

    Uses a CSV cache at ``analysis/_shared/data/merged.csv`` unless
    ``force_refresh`` is set.

    Parameters:
    * force_refresh: if True, re-pull from nfelodcm and overwrite the cache

    Returns:
    * merged pandas DataFrame with derived fields
    '''
    if _CACHE_PATH.exists() and not force_refresh:
        return pandas.read_csv(_CACHE_PATH)
    db = nfelodcm.load(['games', 'market_data'])
    games = _prepare_games(db['games'])
    mkt = _prepare_market(db['market_data'])
    join_keys = ['season', 'week', 'home_team', 'away_team']
    df = pandas.merge(
        games,
        mkt[join_keys + [
            'spread_open', 'spread_close_mkt', 'spread_move',
            'ml_wp_open', 'ml_wp_close_mkt', 'ml_formula_divergence_open',
            'home_spread_close_price', 'away_spread_close_price',
        ]],
        on=join_keys,
        how='left',
    )
    df = _derive_favorite_perspective(df)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CACHE_PATH, index=False)
    return df


## ==================== Private ==================== ##

def _prepare_games(games: pandas.DataFrame) -> pandas.DataFrame:
    '''
    Derive game-level fields used across analyses. ``spread_line`` uses
    the nflfastR convention of positive = home favored; it is flipped
    to the internal convention of negative = home favored.
    '''
    cols = [
        'season', 'week', 'game_type',
        'home_team', 'away_team',
        'home_score', 'away_score',
        'spread_line', 'result',
        'home_moneyline', 'away_moneyline',
        'home_spread_odds', 'away_spread_odds',
        'div_game', 'total_line',
    ]
    out = games[cols].copy()
    out = out[out['result'].notna() & out['spread_line'].notna()].copy()
    out['spread_line'] = -out['spread_line']
    out['home_win'] = (out['result'] > 0).astype(int)
    out['is_tie'] = (out['result'] == 0).astype(int)
    out['is_playoff'] = out['game_type'].isin(
        ['WC', 'DIV', 'CON', 'SB', 'POST']
    ).astype(int)
    out['is_home_fav'] = (out['spread_line'] < 0).astype(int)
    out['spread_bucket'] = numpy.round(out['spread_line'] * 2.0) / 2.0
    out['spread_wp_538'] = spread_to_prob_538(out['spread_line'])
    ml_valid = out['home_moneyline'].notna() & out['away_moneyline'].notna()
    h_wp, _ = hold_adj_probs(
        out.loc[ml_valid, 'home_moneyline'],
        out.loc[ml_valid, 'away_moneyline'],
    )
    out.loc[ml_valid, 'ml_wp_close'] = h_wp
    out['ml_formula_divergence'] = out['ml_wp_close'] - out['spread_wp_538']
    return out


def _prepare_market(market: pandas.DataFrame) -> pandas.DataFrame:
    '''Derive open/close ML probabilities and spread movement from market data.'''
    cols = [
        'season', 'week', 'home_team', 'away_team',
        'home_spread_open', 'home_spread_last',
        'home_spread_last_price', 'away_spread_last_price',
        'home_ml_open', 'away_ml_open',
        'home_ml_last', 'away_ml_last',
    ]
    mkt = market[cols].copy().rename(columns={
        'home_spread_open': 'spread_open',
        'home_spread_last': 'spread_close_mkt',
        'home_spread_last_price': 'home_spread_close_price',
        'away_spread_last_price': 'away_spread_close_price',
    })
    ml_open_valid = mkt['home_ml_open'].notna() & mkt['away_ml_open'].notna()
    h_wp_open, _ = hold_adj_probs(
        mkt.loc[ml_open_valid, 'home_ml_open'],
        mkt.loc[ml_open_valid, 'away_ml_open'],
    )
    mkt.loc[ml_open_valid, 'ml_wp_open'] = h_wp_open
    ml_close_valid = mkt['home_ml_last'].notna() & mkt['away_ml_last'].notna()
    h_wp_close, _ = hold_adj_probs(
        mkt.loc[ml_close_valid, 'home_ml_last'],
        mkt.loc[ml_close_valid, 'away_ml_last'],
    )
    mkt.loc[ml_close_valid, 'ml_wp_close_mkt'] = h_wp_close
    mkt['spread_move'] = mkt['spread_close_mkt'] - mkt['spread_open']
    mkt['spread_wp_538_open'] = spread_to_prob_538(mkt['spread_open'])
    mkt['ml_formula_divergence_open'] = mkt['ml_wp_open'] - mkt['spread_wp_538_open']
    return mkt


def _derive_favorite_perspective(df: pandas.DataFrame) -> pandas.DataFrame:
    '''
    Add favorite-perspective fields and recalibrated WP to the merged frame.

    Recalibration uses the shipped ``Recalibrator`` state. Because the Platt
    model was fit on favorite-side probabilities, the recalibration is
    applied to the favorite-side WP and then folded back into home-side.

    New columns:
    * ``ml_wp_cal``        — recalibrated home win probability in (0, 1)
    * ``home_is_fav_ml``   — 1 if home is the ML favorite (raw market WP)
    * ``fav_ml_wp``        — raw market WP from the favorite's side
    * ``fav_wp_cal``       — recalibrated WP from the favorite's side
    * ``fav_margin``       — actual margin from the favorite's side
    * ``fav_spread``       — posted market spread from the favorite's side
                             (positive = favorite expected to win by that much)
    * ``fav_spread_close_price`` / ``dog_spread_close_price``
    '''
    out = df.copy()
    fav_ml_mask = out['ml_wp_close'].notna()
    out['home_is_fav_ml'] = numpy.nan
    out.loc[fav_ml_mask, 'home_is_fav_ml'] = (
        out.loc[fav_ml_mask, 'ml_wp_close'] >= 0.5
    ).astype(int)
    ## fold raw WP to favorite perspective, recalibrate, fold back ##
    rec = Recalibrator.from_file()
    home_wp = out.loc[fav_ml_mask, 'ml_wp_close'].values
    home_is_fav = out.loc[fav_ml_mask, 'home_is_fav_ml'].values.astype(bool)
    fav_wp_raw = numpy.where(home_is_fav, home_wp, 1.0 - home_wp)
    fav_wp_cal = rec.calibrate(fav_wp_raw)
    out.loc[fav_ml_mask, 'fav_ml_wp'] = fav_wp_raw
    out.loc[fav_ml_mask, 'fav_wp_cal'] = fav_wp_cal
    out.loc[fav_ml_mask, 'ml_wp_cal'] = numpy.where(
        home_is_fav, fav_wp_cal, 1.0 - fav_wp_cal
    )
    ## favorite-perspective outcome and spread ##
    home_fav_bool = out['home_is_fav_ml'] == 1
    out['fav_margin'] = numpy.where(home_fav_bool, out['result'], -out['result'])
    ## internal spread_line is negative when home is favored; invert so that
    ## fav_spread is positive when the ML favorite is expected to win ##
    out['fav_spread'] = numpy.where(
        home_fav_bool, -out['spread_line'], out['spread_line']
    )
    out['fav_spread_close_price'] = numpy.where(
        home_fav_bool,
        out['home_spread_close_price'],
        out['away_spread_close_price'],
    )
    out['dog_spread_close_price'] = numpy.where(
        home_fav_bool,
        out['away_spread_close_price'],
        out['home_spread_close_price'],
    )
    return out
