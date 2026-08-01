'''
Normalizer — discretizes a continuous BaseDistribution into integer bins
and normalizes per-region to restore win probability, tie probability,
and spread bisection constraints.

The margin space -75..+75 is divided by zero (the win/loss boundary) and
the spread (the bisection point):

    A — the entire side of zero opposite the spread
    T — margin 0
    B — margins between zero and the spread, exclusive
    C — margin equal to the spread; integer spreads only
    D — margins beyond the spread on the same side as the spread

For positive integer spread s:

    Region:  A          T     B          C     D
    Margins: -75..-1    0     1..s-1     s     s+1..75

For negative integer spread s:

    Region:  D          C     B          T     A
    Margins: -75..s-1   s     s+1..-1    0     1..75

Region target derivation
------------------------
Let S be the total probability on the side of zero containing the spread,
and O = 1 - S be the mass on the opposite side, including the tie:

    positive spread: S = wp,            O = A + T = 1 - wp
    negative spread: S = 1 - wp - tp,   O = A + T = wp + tp

The side-mass and exact-bisection constraints are:

    B + C + D = S
    O + B = D

Substitute B = D - O into the side-mass constraint:

    (D - O) + C + D = S
    2D = S + O - C = 1 - C

Therefore the unique feasible targets are:

    D* = (1 - C) / 2
    B* = D* - O

For integer |s| >= 2, C is the dirty PMF's key-model-derived push mass.
For half-integer |s| >= 1.5, C = 0, so D* = 0.5.

At |s| = 1, B has no bins.  Setting B = 0 gives D* = O and
C* = S - O = 2S - 1; C must adjust instead of preserving dirty push mass.

At |s| = 0.5, neither B nor C has bins, so D = S.  Exact bisection would
also require D = O, which is impossible unless S = O.  At s = 0, exact
bisection likewise requires wp = (1 - tp) / 2.  For these exceptions,
win probability and tie probability take precedence over bisection.

Scaling changes only each region's total.  The dirty PMF's internal shape
within A, B, and D remains proportional after normalization.
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
    ## floating-point residue smaller than this is treated as zero ##
    TARGET_TOLERANCE: ClassVar[float] = 1e-12

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
        ## clamp spread so bisection aligns to a 0.5 grid ##
        s = round(self.base.spread * 2) / 2
        wp = self.base.win_prob
        tp = self.tie_prob
        ## T — tie region: margin 0 (index 75) — always a single bin ##
        tie_index = self.margin_index(0)
        self.scale_region(pmf, slice(tie_index, tie_index + 1), tp, 'T', s)
        ## dispatch on spread sign for A/B/C/D regions ##
        if s > 0:
            self.normalize_positive(pmf, s)
        elif s < 0:
            self.normalize_negative(pmf, s)
        else:
            self.normalize_zero(pmf, s)
        return pmf

    ## ==================== Normalization Helpers ==================== ##

    @staticmethod
    def margin_index(margin: int) -> int:
        '''Convert an integer margin to its PMF array index.'''
        return margin + 75

    def scale_region(
        self,
        pmf: numpy.ndarray,
        region_slice: slice,
        target: float,
        region: str,
        spread: float,
    ) -> None:
        '''Scale one region to its target mass, rejecting infeasible targets.'''
        actual = float(pmf[region_slice].sum())
        wp = self.base.win_prob
        tp = self.tie_prob
        if target < -self.TARGET_TOLERANCE:
            raise ValueError(
                f'Normalizer {region} target is negative at spread={spread}: '
                f'target={target}, actual={actual}, wp={wp}, tp={tp}'
            )
        target = max(target, 0.0)
        if actual > 0:
            pmf[region_slice] *= target / actual
        elif target > self.TARGET_TOLERANCE:
            raise ValueError(
                f'Normalizer cannot assign {region} target to an empty region '
                f'at spread={spread}: target={target}, wp={wp}, tp={tp}'
            )

    @staticmethod
    def derive_region_targets(
        spread: float,
        side_target: float,
        push: float,
    ) -> tuple[float, float, float]:
        '''
        Derive B/C/D totals from side mass and exact bisection constraints.

        Parameters:
        * spread:      posted spread on the 0.5 grid
        * side_target: total mass on the side of zero containing the spread
        * push:        dirty C mass for integer spreads; zero otherwise

        Returns:
        * tuple (target_b, target_c, target_d)
        '''
        opposite_mass = 1.0 - side_target
        absolute_spread = abs(spread)

        if absolute_spread == 0.5:
            ## B and C are empty; preserve the side target ##
            return 0.0, 0.0, side_target

        if absolute_spread == 1.0:
            ## B is empty; C must adjust so D equals the opposite mass ##
            target_d = opposite_mass
            target_c = side_target - opposite_mass
            return 0.0, target_c, target_d

        target_c = push if spread == round(spread) else 0.0
        target_d = (1.0 - target_c) / 2.0
        target_b = target_d - opposite_mass
        return target_b, target_c, target_d

    def normalize_positive(self, pmf: numpy.ndarray, spread: float) -> None:
        '''
        Normalize for spread > 0.  Spread is on the win side.

        A = margins -75..-1 (loss), T already handled,
        B/C/D are win sub-regions around the spread.
        '''
        wp = self.base.win_prob
        tp = self.tie_prob
        spread_int = int(spread)
        is_int = (spread == round(spread))

        ## A — loss: margins -75..-1 ##
        self.scale_region(
            pmf,
            slice(self.margin_index(-75), self.margin_index(0)),
            1.0 - wp - tp,
            'A',
            spread,
        )

        if is_int:
            b_slice = slice(self.margin_index(1), self.margin_index(spread_int))
            c_index = self.margin_index(spread_int)
            c_slice = slice(c_index, c_index + 1)
            d_slice = slice(c_index + 1, self.margin_index(75) + 1)
            push = float(pmf[c_index])
        else:
            b_slice = slice(
                self.margin_index(1),
                self.margin_index(int(numpy.floor(spread))) + 1,
            )
            c_slice = None
            d_slice = slice(
                self.margin_index(int(numpy.ceil(spread))),
                self.margin_index(75) + 1,
            )
            push = 0.0

        target_b, target_c, target_d = self.derive_region_targets(
            spread, wp, push
        )
        self.scale_region(pmf, b_slice, target_b, 'B', spread)
        if c_slice is not None:
            self.scale_region(pmf, c_slice, target_c, 'C', spread)
        self.scale_region(pmf, d_slice, target_d, 'D', spread)

    def normalize_negative(self, pmf: numpy.ndarray, spread: float) -> None:
        '''
        Normalize for spread < 0.  Spread is on the loss side.

        A = margins 1..75 (win), T already handled,
        D/C/B are loss sub-regions around the spread.
        '''
        wp = self.base.win_prob
        tp = self.tie_prob
        spread_int = int(spread)
        is_int = (spread == round(spread))
        loss_target = 1.0 - wp - tp

        ## A — win: margins 1..75 ##
        self.scale_region(
            pmf,
            slice(self.margin_index(1), self.margin_index(75) + 1),
            wp,
            'A',
            spread,
        )

        if is_int:
            c_index = self.margin_index(spread_int)
            d_slice = slice(self.margin_index(-75), c_index)
            c_slice = slice(c_index, c_index + 1)
            b_slice = slice(c_index + 1, self.margin_index(0))
            push = float(pmf[c_index])
        else:
            d_slice = slice(
                self.margin_index(-75),
                self.margin_index(int(numpy.floor(spread))) + 1,
            )
            c_slice = None
            b_slice = slice(
                self.margin_index(int(numpy.ceil(spread))),
                self.margin_index(0),
            )
            push = 0.0

        target_b, target_c, target_d = self.derive_region_targets(
            spread, loss_target, push
        )
        self.scale_region(pmf, d_slice, target_d, 'D', spread)
        if c_slice is not None:
            self.scale_region(pmf, c_slice, target_c, 'C', spread)
        self.scale_region(pmf, b_slice, target_b, 'B', spread)

    def normalize_zero(self, pmf: numpy.ndarray, spread: float) -> None:
        '''
        Normalize for spread = 0.  Only A, T, D exist.

        T already handled.  Spread sits on the tie bin, so there is
        no B or C — only the two flanking regions.
        '''
        wp = self.base.win_prob
        tp = self.tie_prob
        ## A — loss: margins -75..-1 ##
        self.scale_region(
            pmf,
            slice(self.margin_index(-75), self.margin_index(0)),
            1.0 - wp - tp,
            'A',
            spread,
        )
        ## D — win: margins 1..75 ##
        self.scale_region(
            pmf,
            slice(self.margin_index(1), self.margin_index(75) + 1),
            wp,
            'D',
            spread,
        )
