'''
Shared plotting style, statistical helpers, and odds conversions used by
the analyses in this directory.
'''

## built-ins ##
import pathlib

## external ##
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


## paths ##
BASE: pathlib.Path = pathlib.Path(__file__).resolve().parent
DATA_DIR: pathlib.Path = BASE / 'data'

## color palette ##
C_EMPIRICAL: str = '#2E86AB'
C_MARKET: str = '#E84855'
C_FORMULA: str = '#3BB273'
C_NEUTRAL: str = '#888888'
C_BAND_FILL: str = '#E84855'


## ==================== Probability Helpers ==================== ##

def american_to_prob(odds: numpy.ndarray) -> numpy.ndarray:
    '''
    Convert American odds to raw implied probability.

    Parameters:
    * odds: American odds (negative = favorite, positive = underdog)

    Returns:
    * raw implied probabilities (not hold-adjusted)
    '''
    odds = numpy.asarray(odds, dtype=float)
    out = numpy.full(odds.shape, numpy.nan)
    neg = odds < 0
    pos = odds > 0
    out[neg] = (-odds[neg]) / (100.0 - odds[neg])
    out[pos] = 100.0 / (100.0 + odds[pos])
    return out


def hold_adj_probs(home_odds: numpy.ndarray, away_odds: numpy.ndarray) -> tuple:
    '''
    Compute hold-adjusted win probability pair from American odds.

    Parameters:
    * home_odds: American odds for home team
    * away_odds: American odds for away team

    Returns:
    * (home_prob, away_prob) hold-adjusted
    '''
    hp = american_to_prob(home_odds)
    ap = american_to_prob(away_odds)
    total = hp + ap
    return hp / total, ap / total


def spread_to_prob_538(spread: numpy.ndarray) -> numpy.ndarray:
    '''
    538-style spread to win probability mapping.

    Parameters:
    * spread: point spread (negative = home favorite)

    Returns:
    * home win probability in (0, 1)
    '''
    elo_dif = -numpy.asarray(spread, dtype=float) * 25.0
    return 1.0 / (10.0 ** (-elo_dif / 400.0) + 1.0)


def wilson_ci(n_wins: numpy.ndarray, n: numpy.ndarray, z: float = 1.96) -> tuple:
    '''
    Wilson confidence interval for a binomial proportion.

    Parameters:
    * n_wins: count of successes
    * n: sample size
    * z: z-score for the desired coverage (default 1.96 = 95%)

    Returns:
    * (lower, upper) bounds as arrays aligned with the inputs
    '''
    n_wins = numpy.asarray(n_wins, dtype=float)
    n = numpy.asarray(n, dtype=float)
    p = n_wins / n
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2.0 * n)) / denom
    margin = z * numpy.sqrt(p * (1.0 - p) / n + z ** 2 / (4.0 * n ** 2)) / denom
    return center - margin, center + margin


## ==================== Plot Style ==================== ##

def setup_style() -> None:
    '''Apply the shared matplotlib style used across all analyses.'''
    plt.rcParams.update({
        'figure.facecolor':   'white',
        'axes.facecolor':     '#F8F9FA',
        'axes.grid':          True,
        'grid.color':         '#DDDDDD',
        'grid.linewidth':     0.7,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'font.family':        'sans-serif',
        'font.size':          10,
        'axes.labelsize':     11,
        'axes.titlesize':     12,
        'figure.titlesize':   13,
        'figure.titleweight': 'bold',
        'legend.fontsize':    9,
        'legend.framealpha':  0.8,
    })
