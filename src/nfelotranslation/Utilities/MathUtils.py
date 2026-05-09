'''
Shared mathematical utilities for the nfelotranslation package.
'''

import numpy
from scipy.special import logit as _scipy_logit, expit as _scipy_expit


## probability clipping bounds — keeps log transforms finite ##
_PROB_EPS: float = 1e-9


def clip_prob(p: numpy.ndarray, eps: float = _PROB_EPS) -> numpy.ndarray:
    '''Clip probabilities to (eps, 1-eps) to avoid log(0) in transforms'''
    return numpy.clip(numpy.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p: numpy.ndarray) -> numpy.ndarray:
    '''Log-odds transform: log(p / (1-p)), with clipping for numerical safety'''
    return _scipy_logit(clip_prob(p))


def expit(z: numpy.ndarray) -> numpy.ndarray:
    '''Sigmoid / inverse logit: 1 / (1 + exp(-z))'''
    return _scipy_expit(numpy.asarray(z, dtype=float))


def log_loss(outcomes: numpy.ndarray, probs: numpy.ndarray) -> float:
    '''
    Binary cross-entropy (log loss).

    Parameters:
    * outcomes: binary array (0 / 1)
    * probs: predicted probabilities in (0, 1)

    Returns:
    * mean log loss (lower = better calibration)
    '''
    y = numpy.asarray(outcomes, dtype=float)
    p = clip_prob(numpy.asarray(probs, dtype=float))
    return float(-numpy.mean(y * numpy.log(p) + (1.0 - y) * numpy.log(1.0 - p)))


def brier_score(outcomes: numpy.ndarray, probs: numpy.ndarray) -> float:
    '''
    Brier score: mean squared error of probability predictions.

    Parameters:
    * outcomes: binary array (0 / 1)
    * probs: predicted probabilities in (0, 1)

    Returns:
    * mean Brier score (lower = better)
    '''
    y = numpy.asarray(outcomes, dtype=float)
    p = numpy.asarray(probs, dtype=float)
    return float(numpy.mean((y - p) ** 2))


def american_to_prob(odds: numpy.ndarray) -> numpy.ndarray:
    '''
    Convert American odds to raw implied probability.

    Parameters:
    * odds: American odds (negative = favourite, positive = underdog)

    Returns:
    * raw implied probabilities (not hold-adjusted)
    '''
    odds = numpy.asarray(odds, dtype=float)
    ## numpy.where evaluates both branches for all elements, so guard ##
    ## denominators to avoid divide-by-zero on the unselected side    ##
    is_neg = odds < 0
    neg_denom = numpy.where(is_neg, 100.0 - odds, 1.0)
    pos_denom = numpy.where(is_neg, 1.0, 100.0 + odds)
    return numpy.where(is_neg, (-odds) / neg_denom, 100.0 / pos_denom)


def hold_adj_probs(
    home_odds: numpy.ndarray,
    away_odds: numpy.ndarray,
) -> tuple:
    '''
    Compute hold-adjusted win probability pair from American odds.

    Removes the vig by normalizing raw implied probabilities to sum to 1.

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
