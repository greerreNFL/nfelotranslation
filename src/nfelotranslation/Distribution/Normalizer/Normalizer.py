'''
Normalizer — discretizes a continuous BaseDistribution into integer bins
and normalizes per-region to restore win probability, tie probability,
and spread bisection constraints.

The margin space -75..+75 is divided by two landmarks: zero (the win/loss
boundary) and the spread (the bisection point).  With tie carved out at
margin 0, the resulting regions are labelled A through D:

    A — the entire side of zero opposite the spread (one block, no sub-division)
    T — the tie bin (margin 0, always a single bin)
    B — margins between zero and the spread (exclusive of both)
    C — the spread bin itself (integer spreads only)
    D — margins beyond the spread on the same side as the spread

For positive integer spread (e.g., s=7):

    Region:  A          T     B          C     D
    Margins: -75..-1    0     1..s-1     s     s+1..75
    Target:  1-wp-tp    tp    ←—— these three sum to wp ——→

For negative integer spread (e.g., s=-7):

    Region:  D          C     B          T     A
    Margins: -75..s-1   s     s+1..-1    0     1..75
    Target:  ←—— these three sum to 1-wp-tp ——→  tp    wp

For half-integer spread, C does not exist (no bin sits on the spread).
For spread = 0, only A, T, D exist.

For integer spreads, region C (the spread bin) is a single bin whose
dirty PMF value passes through untouched — this preserves the key
model's excess at push numbers.  B and D absorb the remainder of
their side's total target (wp or 1 - wp - tp), scaled proportionally
using continuous CDF ratios at region boundaries.

For half-integer spreads, no region C exists; B and D targets are
proportional to the continuous CDF evaluated at region boundaries,
then scaled so they sum to their side's total target.
'''

## built-ins ##
from typing import ClassVar

## external ##
import numpy

## local ##
from ..Base import BaseDistribution


class Normalizer:
    '''
    Discretizes and normalizes a BaseDistribution into a valid PMF.

    Owns all discrete operations: integer binning, region scaling,
    and spread bisection sub-region logic.  BaseDistribution stays
    purely continuous.

    Parameters:
    * base:     continuous BaseDistribution to discretize
    * tie_prob: P(margin = 0) discrete tie probability
    '''

    ## integer margins from -75 to +75 ##
    MARGINS: ClassVar[numpy.ndarray] = numpy.arange(-75, 76)

    def __init__(self, base: BaseDistribution, tie_prob: float = 0.002):
        self.base = base
        self.tie_prob = tie_prob

    ## ==================== Discretization ==================== ##

    def discretize(self) -> numpy.ndarray:
        '''
        Continuous PDF evaluated at integer margins -75..+75, normalized to sum to 1.

        Returns:
        * ndarray of shape (151,) — discrete PMF at integer margins
        '''
        raw = self.base.pdf(self.MARGINS)
        return raw / raw.sum()

    ## ==================== Normalization ==================== ##

    def normalize(self, dirty_pmf: numpy.ndarray) -> numpy.ndarray:
        '''
        Per-region scaling to restore win probability, tie probability,
        and spread bisection constraints.

        Regions A/T/B/C/D are defined by position relative to zero and
        the spread.  See module docstring for the full region layout.

        Parameters:
        * dirty_pmf: ndarray of shape (151,) — raw PMF to normalize

        Returns:
        * ndarray of shape (151,) — normalized PMF summing to 1
        '''
        pmf = dirty_pmf.copy()
        s = self.base.spread
        wp = self.base.win_prob
        tp = self.tie_prob
        is_int = (s == round(s))
        ## scale a slice of pmf to match a target mass ##
        def _scale(sl, target):
            actual = pmf[sl].sum()
            if actual > 0:
                pmf[sl] *= target / actual
        ## margin k → index k + 75 ##
        def _i(k):
            return k + 75
        ## T — tie region: margin 0 (index 75) — always a single bin ##
        _scale(slice(_i(0), _i(0) + 1), tp)
        ## dispatch on spread sign for A/B/C/D regions ##
        if s > 0:
            self._normalize_positive(pmf, s, int(s), is_int, wp, tp, _scale, _i)
        elif s < 0:
            self._normalize_negative(pmf, s, int(s), is_int, wp, tp, _scale, _i)
        else:
            self._normalize_zero(pmf, wp, tp, _scale, _i)
        return pmf

    ## ==================== Normalization Helpers ==================== ##

    def _normalize_positive(self, pmf, s, s_int, is_int, wp, tp, _scale, _i):
        '''
        Normalize for spread > 0.  Spread is on the win side.

        A = margins -75..-1 (loss), T already handled,
        B/C/D are win sub-regions around the spread.

        For integer spreads, C (the spread bin) preserves the dirty PMF
        value — its height comes from the key model, not the continuous
        CDF.  B and D absorb the remainder proportionally.
        '''
        ## A — loss: margins -75..-1 ##
        _scale(slice(_i(-75), _i(0)), 1.0 - wp - tp)
        ## B/C/D — win sub-regions ##
        if is_int:
            ## C — spread bin: preserve key-model-derived push height ##
            push = pmf[_i(s_int)]
            ## B + D absorb the remainder of wp ##
            bd_target = wp - push
            bd_actual = pmf[_i(1):_i(s_int)].sum() + pmf[_i(s_int) + 1:_i(75) + 1].sum()
            if bd_actual > 0:
                bd_factor = bd_target / bd_actual
                ## B — between tie and spread: margins 1..s-1 ##
                pmf[_i(1):_i(s_int)] *= bd_factor
                ## D — beyond spread: margins s+1..75 ##
                pmf[_i(s_int) + 1:_i(75) + 1] *= bd_factor
        else:
            sf_half = float(self.base.sf(0.5))
            fi = int(numpy.floor(s))
            ci = int(numpy.ceil(s))
            ## B — below spread: margins 1..floor(s) ##
            target_b = wp * float(self.base.cdf(s) - self.base.cdf(0.5)) / sf_half
            _scale(slice(_i(1), _i(fi) + 1), target_b)
            ## D — above spread: margins ceil(s)..75 ##
            target_d = wp * float(self.base.sf(s)) / sf_half
            _scale(slice(_i(ci), _i(75) + 1), target_d)

    def _normalize_negative(self, pmf, s, s_int, is_int, wp, tp, _scale, _i):
        '''
        Normalize for spread < 0.  Spread is on the loss side.

        A = margins 1..75 (win), T already handled,
        D/C/B are loss sub-regions around the spread.

        For integer spreads, C (the spread bin) preserves the dirty PMF
        value — its height comes from the key model, not the continuous
        CDF.  B and D absorb the remainder proportionally.
        '''
        loss_target = 1.0 - wp - tp
        ## A — win: margins 1..75 ##
        _scale(slice(_i(1), _i(75) + 1), wp)
        ## D/C/B — loss sub-regions ##
        if is_int:
            ## C — spread bin: preserve key-model-derived push height ##
            push = pmf[_i(s_int)]
            ## B + D absorb the remainder of loss_target ##
            bd_target = loss_target - push
            bd_actual = pmf[_i(-75):_i(s_int)].sum() + pmf[_i(s_int) + 1:_i(0)].sum()
            if bd_actual > 0:
                bd_factor = bd_target / bd_actual
                ## D — beyond spread: margins -75..s-1 ##
                pmf[_i(-75):_i(s_int)] *= bd_factor
                ## B — between spread and tie: margins s+1..-1 ##
                pmf[_i(s_int) + 1:_i(0)] *= bd_factor
        else:
            cdf_half = float(self.base.cdf(-0.5))
            fi = int(numpy.floor(s))
            ci = int(numpy.ceil(s))
            ## D — beyond spread: margins -75..floor(s) ##
            target_d = loss_target * float(self.base.cdf(s)) / cdf_half
            _scale(slice(_i(-75), _i(fi) + 1), target_d)
            ## B — between spread and tie: margins ceil(s)..-1 ##
            target_b = loss_target * float(self.base.cdf(-0.5) - self.base.cdf(s)) / cdf_half
            _scale(slice(_i(ci), _i(0)), target_b)

    def _normalize_zero(self, pmf, wp, tp, _scale, _i):
        '''
        Normalize for spread = 0.  Only A, T, D exist.

        T already handled.  Spread sits on the tie bin, so there is
        no B or C — only the two flanking regions.
        '''
        ## A — loss: margins -75..-1 ##
        _scale(slice(_i(-75), _i(0)), 1.0 - wp - tp)
        ## D — win: margins 1..75 ##
        _scale(slice(_i(1), _i(75) + 1), wp)
