'''
BaseDistribution — stateful continuous distribution for the margin distribution.

Generalized normal (``scipy.stats.gennorm``) with shape parameter ``beta``.
Given (spread, win_prob, beta), derives scale analytically and provides
the continuous density, CDF, and survival function.

``beta`` is the generalized-normal shape: ``beta=2`` recovers the standard
Gaussian, ``beta<2`` produces heavier tails.  The shipped default is loaded
from ``margin_hyperparams.json`` via ``MarginDistributionModel``.

'''

## built-ins ##
from dataclasses import dataclass, field
from typing import ClassVar

## external ##
import numpy
from scipy.stats import gennorm as scipy_gennorm

## local ##
from .. import MARGIN_HYPERPARAMS
_DEFAULT_BETA = float(MARGIN_HYPERPARAMS['beta'])


@dataclass
class BaseDistribution:
    '''
    Continuous generalized normal parameterized by (spread, win_prob, beta).

    beta=2 is the standard Gaussian; beta<2 gives heavier tails.  The default
    is loaded at import from the shipped ``margin_hyperparams.json`` so this
    class and ``MarginDistributionModel`` agree on what "default beta" means.

    The constraints uniquely determine the distribution:
        loc   = spread
        scale = spread / gennorm.ppf(win_prob, beta)

    When spread ≈ 0 or wp ≈ 0.5, the formula degenerates
    (numerator or denominator → 0) and a fallback scale is used instead.
    The Normalizer enforces the wp constraint via region scaling regardless.

    Parameters:
    * spread: point spread (positive = favorite)
    * win_prob: P(margin > 0) moneyline win probability
    * beta: shape parameter (2.0 = Gaussian, 1.0 = Laplace, <2 = heavier tails).
            Defaults to the shipped value in margin_hyperparams.json.

    Derived:
    * scale: scale parameter of the generalized normal
    * loss_mass: P(margin < 0) from continuous CDF
    * win_mass: P(margin > 0) from continuous SF
    '''

    spread: float
    win_prob: float
    beta: float = _DEFAULT_BETA
    ## derived in __post_init__ ##
    scale: float = field(init=False)
    loss_mass: float = field(init=False)
    win_mass: float = field(init=False)

    ## empirical fallback scale for pick'em games (degenerate case) ##
    _FALLBACK_SCALE: ClassVar[float] = 13.2

    ## threshold for treating spread / wp as degenerate ##
    _DEGEN_EPS: ClassVar[float] = 1e-6

    ## ==================== Init ==================== ##

    def __post_init__(self):
        ## derive scale ##
        ## handle degenerate case ##
        if abs(self.spread) < self._DEGEN_EPS or abs(self.win_prob - 0.5) < self._DEGEN_EPS:
            self.scale = self._FALLBACK_SCALE
        else:
            self.scale = self.spread / scipy_gennorm.ppf(self.win_prob, self.beta)
        ## region masses from continuous CDF ##
        self.loss_mass = float(scipy_gennorm.cdf(0.0, self.beta, loc=self.spread, scale=self.scale))
        self.win_mass = float(scipy_gennorm.sf(0.0, self.beta, loc=self.spread, scale=self.scale))

    ## ==================== Continuous Evaluators ==================== ##

    def pdf(self, x) -> numpy.ndarray:
        '''
        Evaluate the continuous density at arbitrary points.

        Parameters:
        * x: points at which to evaluate (scalar or array)

        Returns:
        * density values (same shape as x)
        '''
        return scipy_gennorm.pdf(numpy.asarray(x, dtype=float), self.beta, loc=self.spread, scale=self.scale)

    def cdf(self, x) -> numpy.ndarray:
        '''
        Evaluate the continuous CDF at arbitrary points.

        Parameters:
        * x: points at which to evaluate (scalar or array)

        Returns:
        * cumulative probabilities (same shape as x)
        '''
        return scipy_gennorm.cdf(numpy.asarray(x, dtype=float), self.beta, loc=self.spread, scale=self.scale)

    def sf(self, x) -> numpy.ndarray:
        '''
        Evaluate the continuous survival function at arbitrary points.

        Parameters:
        * x: points at which to evaluate (scalar or array)

        Returns:
        * survival probabilities P(X > x) (same shape as x)
        '''
        return scipy_gennorm.sf(numpy.asarray(x, dtype=float), self.beta, loc=self.spread, scale=self.scale)
