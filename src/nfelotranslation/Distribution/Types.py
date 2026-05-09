'''
Distribution result data structures.
'''

## built-ins ##
from dataclasses import dataclass

## external ##
import numpy


@dataclass
class MarginDistribution:
    '''
    Result container for a fully constructed margin distribution.

    Holds the discrete PMF at integer margins -75..+75 along with
    the spread, win probability, and tie probability that produced it.

    Parameters:
    * spread:   point spread (positive = favourite), on the 0.5 grid
    * win_prob: P(margin > 0) moneyline win probability
    * tie_prob: P(margin = 0) discrete tie probability
    * pmf:     ndarray of shape (151,), discrete PMF at integer margins
    '''
    spread: float
    win_prob: float
    tie_prob: float
    pmf: numpy.ndarray

    ## ==================== Derived Quantities ==================== ##

    def cover_prob(self, line: float) -> float:
        '''
        Probability of covering a given line.

        P(margin > line).  For integer lines, margin == line is a push
        (not a cover).  For half-integer lines, no push is possible.

        Parameters:
        * line: the betting line to evaluate

        Returns:
        * P(margin > line)
        '''
        ## first integer margin that counts as a cover ##
        threshold = int(line) + 1 if line == int(line) else int(numpy.ceil(line))
        idx = threshold + 75
        if idx > 150:
            return 0.0
        if idx < 0:
            return 1.0
        return float(self.pmf[idx:].sum())

    def push_prob(self, line: float) -> float:
        '''
        Probability of pushing (landing exactly on) a given line.

        Only nonzero for integer lines since margins are integers.

        Parameters:
        * line: the betting line to evaluate

        Returns:
        * P(margin == line), or 0.0 for non-integer lines
        '''
        if line != int(line):
            return 0.0
        idx = int(line) + 75
        if idx < 0 or idx > 150:
            return 0.0
        return float(self.pmf[idx])

    def win_prob_from_pmf(self) -> float:
        '''
        Compute P(margin > 0) directly from the discrete PMF.

        Returns:
        * sum of PMF for margins 1..75
        '''
        ## margins 1..75 → indices 76..150 ##
        return float(self.pmf[76:].sum())

    def loss_prob_from_pmf(self) -> float:
        '''
        Compute P(margin < 0) directly from the discrete PMF.

        Returns:
        * sum of PMF for margins -75..-1
        '''
        ## margins -75..-1 → indices 0..74 ##
        return float(self.pmf[:75].sum())

    def tie_prob_from_pmf(self) -> float:
        '''
        Compute P(margin = 0) directly from the discrete PMF.

        Returns:
        * PMF at margin 0
        '''
        ## margin 0 → index 75 ##
        return float(self.pmf[75])

    def expected_margin(self) -> float:
        '''
        Expected value of the margin from the discrete PMF.

        Returns:
        * E[margin] = Σ k * pmf[k] for k in -75..75
        '''
        margins = numpy.arange(-75, 76)
        return float((margins * self.pmf).sum())
