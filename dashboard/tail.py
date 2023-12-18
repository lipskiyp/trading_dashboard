"""
Trading dashborad.
"""

from abc import ABC
from pandas.core.frame import DataFrame


class TailDashboard(ABC):
    """
    Abstract base class for trading dashboard with tail-risk metrics.
    """

    def skew(
        self,
        offset: int = 1,
    ) -> DataFrame:
        """
        Returns skew of returns.
        """
        rs = self.ln_rs(offset)
        sigma = self.sigma(offset, annualise=False)
        N = len(rs)

        return (
            (rs - rs.mean()) ** 3
        ).sum() / sigma * N / (N - 1) / (N - 2)


    def kurtosis(
        self,
        offset: int = 1,
        excess: bool = True,
    ) -> DataFrame:
        """
        Returns kurtosis of returns.
        """
        rs = self.ln_rs(offset)
        sigma = self.sigma(offset, annualise=False)
        N = len(rs)

        kurt = (
            (rs - rs.mean()) ** 4
        ).sum() / sigma * N * (N - 1) / (N - 2) / (N - 3)

        if not excess:
            kurt -= 3 * (N - 1) ** 2 / (N - 2) / (N - 3)
        return kurt