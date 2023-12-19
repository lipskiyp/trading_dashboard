"""
Core trading dashborad.
"""

from abc import ABC
import numpy as np
from pandas.core.frame import DataFrame


class CoreDashboard(ABC):
    """
    Abstract base class for trading dashboard with core metrics.
    """

    def avg_ds(
        self,
        offset: int = 1,
    ) -> DataFrame:
        """
        Returns average price change.
        """
        return (
            self.ts.shift(offset) - self.ts
        ).mean()


    def avg_r(
        self,
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns average returns.
        """
        avg_rs = self.ln_rs(offset).mean()

        if annualise:
            avg_rs *= self.TRADING_DAYS_YEAR/offset
        return avg_rs


    def cagr(
        self,
        offset: int = 1,
    ) -> DataFrame:
        """
        Returns compounded annual growth rate (CAGR).
        """
        cumprod = (1 + self.ln_rs(offset)).cumprod()
        power = (len(cumprod.index)) / self.TRADING_DAYS_YEAR
        return cumprod.iloc[-1] ** power - 1


    def var(
        self,
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns var of returns.
        """
        rs = self.ln_rs(offset)
        N = len(rs)

        sigma = (
            (rs - rs.mean()) ** 2
        ).sum() / (N - 1)

        if annualise:
            sigma *= self.TRADING_DAYS_YEAR / offset
        return sigma


    def sigma(
        self,
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns standard deviation of retunrs.
        """
        return np.sqrt(
            self.var(offset, annualise)
        )


    def downside_sigma(
        self,
        threshold: float, # downside return threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns downside (loss) standard deviation of returns.
        """
        rs = self.ln_rs(offset).map(
            lambda x: np.min([x - threshold, 0]) ** 2
        )
        N = len(rs)

        downside_sigma = np.sqrt( rs.sum() / N )

        if annualise:
            downside_sigma *= np.sqrt(self.TRADING_DAYS_YEAR / offset)
        return downside_sigma


    def upside_sigma(
        self,
        threshold: float, # upside return threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns downside standard deviation of returns.
        """
        rs = self.ln_rs(offset).map(
            lambda x: np.max([x - threshold, 0]) ** 2
        )
        N = len(rs)

        upside_sigma = np.sqrt(rs.sum() / N)

        if annualise:
            upside_sigma *= np.sqrt(self.TRADING_DAYS_YEAR / offset)
        return upside_sigma


    def covar(
        self,
        benchmark: str,
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns covariance between time-series and benchmark.
        """
        rs = self.ln_rs(offset)
        N = len(rs)

        covar = (
            (
                (rs - rs.mean()) * (rs[[benchmark]] - rs[[benchmark]].mean()).values
            ).sum() / (N - 1)
        )

        if annualise:
            covar *= self.TRADING_DAYS_YEAR / offset
        return covar


    def corr(
        self,
        benchmark: str,
        offset: int = 1,
    ) -> DataFrame:
        """
        Returns correlation between time-series and benchmark.
        """
        covar = self.covar(benchmark, offset)
        sigma = self.sigma(offset)

        return covar / (sigma * sigma[[benchmark]].values)
