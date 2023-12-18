"""
Trading dashborad.
"""

import numpy as np
from pandas.core.frame import DataFrame

from base import BaseDashboard


class TradingDashboard(BaseDashboard):
    """
    Class to caculate metrics for financial timeseries.
    """
    TRADING_DAYS_YEAR = 252


    def __init__(self, ts: DataFrame):
        self.ts = ts


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
        avg_rs = self._ln_r(offset).mean()

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
        _cumprod = (1 + self._ln_r(offset)).cumprod()
        _power = (len(_cumprod.index)) / self.TRADING_DAYS_YEAR
        return _cumprod.iloc[-1] ** _power - 1


    def sigma(
        self,
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns standard deviation of returns.
        """
        rs = self._ln_r(offset)
        N = len(rs)

        sigma = np.sqrt(
            ( (rs - rs.mean()) ** 2).sum() / (N - 1)
        )

        if annualise:
            sigma *= np.sqrt(self.TRADING_DAYS_YEAR / offset)
        return sigma


    def downside_sigma(
        self,
        threshold: float, # downside return threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> DataFrame:
        """
        Returns downside (loss) standard deviation of returns.
        """
        rs = self._ln_r(offset).map(
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
        rs = self._ln_r(offset).map(
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
        rs = self._ln_r(offset)
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


    def skew(
        self,
        offset: int = 1,
    ) -> DataFrame:
        """
        Returns skew of returns.
        """
        rs = self._ln_r(offset)
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
        rs = self._ln_r(offset)
        sigma = self.sigma(offset, annualise=False)
        N = len(rs)

        kurt = (
            (rs - rs.mean()) ** 4
        ).sum() / sigma * N * (N - 1) / (N - 2) / (N - 3)

        if not excess:
            kurt -= 3 * (N - 1) ** 2 / (N - 2) / (N - 3)
        return kurt

