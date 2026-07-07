'''
MarginDistributionModel — composes BaseDistribution + KeyModel into a
final discrete PMF with key number adjustments and per-region normalization.

This is the top-level composer for the Distribution module.  Given a
(spread, win_prob) pair, it:
    1. Creates the continuous base distribution
    2. Discretizes to integer bins
    3. Adds key number excess from the KeyModel
    4. Clamps negatives
    5. Normalizes per-region to restore WP, tie, and spread bisection constraints

Hyperparameters (gennorm ``beta``, discrete ``tie_prob``) are read from
``margin_hyperparams.json`` via ``Utilities.JsonIo.read_config_envelope``,
same envelope convention as other shipped configs.
'''

## external ##
import numpy

## local ##
from . import MARGIN_HYPERPARAMS
from .Base import BaseDistribution
from .Key import KeyModel
from .Normalizer import Normalizer
from .Types import MarginDistribution

_BETA = float(MARGIN_HYPERPARAMS['beta'])
_TIE_PROB = float(MARGIN_HYPERPARAMS['tie_prob'])


class MarginDistributionModel:
    '''
    Composes BaseDistribution + KeyModel into a MarginDistribution.

    Usage:
        key_model = KeyModel.from_initial(params)
        model = MarginDistributionModel(key_model)
        result = model.predict(7.0, 0.75)
        print(result.pmf.sum())       # 1.0
        print(result.cover_prob(6.5))  # P(margin > 6.5)

    Parameters:
    * key_model: KeyModel with per-integer excess trackers
    * tie_prob:  P(margin = 0) discrete tie probability (default from margin_hyperparams.json)
    * beta:      generalized-normal shape for the base (default from margin_hyperparams.json)
    '''

    def __init__(
        self,
        key_model: KeyModel,
        tie_prob: float = _TIE_PROB,
        beta: float = _BETA,
    ):
        self.key_model = key_model
        self.tie_prob = tie_prob
        self.beta = beta

    ## ==================== Public Interface ==================== ##

    def predict(self, spread: float, win_prob: float) -> MarginDistribution:
        '''
        Build a full discrete margin distribution.

        Parameters:
        * spread:   point spread (positive = favourite)
        * win_prob: P(margin > 0) moneyline win probability

        Returns:
        * MarginDistribution with normalized PMF
        '''
        ## 1. create base distribution ##
        base = BaseDistribution(spread, win_prob, beta=self.beta)
        ## 2. create normalizer ##
        normalizer = Normalizer(base, self.tie_prob)
        ## 3. discretize ##
        raw_pmf = normalizer.discretize()
        ## 4. add key number excess (ratio model: pass baseline PMF) ##
        all_excess = self.key_model.get_all_excess(raw_pmf)
        for k, (ex_pos, ex_neg) in all_excess.items():
            raw_pmf[k + 75] += ex_pos
            raw_pmf[-k + 75] += ex_neg
        ## 5. clamp negatives ##
        raw_pmf = numpy.maximum(raw_pmf, 0.0)
        ## 6. normalize ##
        pmf = normalizer.normalize(raw_pmf)
        ## 7. return ##
        ## convert continuous spread to grid spread for distribution ##
        grid_spread = round(base.spread * 2) / 2
        return MarginDistribution(
            spread=grid_spread,
            win_prob=win_prob,
            tie_prob=self.tie_prob,
            pmf=pmf,
        )
