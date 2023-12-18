"""
Trading dashborad.
"""

import numpy as np
from pandas.core.frame import DataFrame
from typing import List

from base import BaseDashboard


class TradingDashboard(BaseDashboard):
    """
    Class to caculate metrics for financial timeseries.
    """

    DAYS_YEAR = 252


    def __init__(self, ts: DataFrame):
        self.ts = ts


    def avg_ds(
        self,
        offset: int = 1,
    ) -> List[float]:
        """
        Return average price change.
        """
        return (
            self.ts.shift(offset) - self.ts
        ).mean()


    def avg_r(
        self,
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns average returns.
        """
        avg_rs = self._ln_r(offset).mean()

        if annualise:
            avg_rs *= self.DAYS_YEAR/offset
        return avg_rs


    def cagr(
        self,
        offset: int = 1,
    ) -> List[float]:
        """
        Returns compounded annual growth rate (CAGR).
        """
        _cumprod = (1 + self._ln_r(offset)).cumprod()
        _power = (len(_cumprod.index)) / self.DAYS_YEAR
        return _cumprod.iloc[-1] ** _power - 1


    def sigma_r(
        self,
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns standard deviation of returns.
        """
        _rs = self._ln_r(offset)
        sigma = np.sqrt(
            ( (_rs - _rs.mean()) ** 2).sum() / (len(_rs) - 1)
        )

        if annualise:
            sigma *= np.sqrt(252 / offset)
        return sigma


    def downside_sigma_r(
        self,
        threshold: float, # downside retunr threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns downside (loss) standard deviation of returns.
        """
        _rs = self._ln_r(offset).map(
            lambda x: np.min([x - threshold, 0]) ** 2
        )
        downside_sigma = np.sqrt( _rs.sum() / len(_rs) )

        if annualise:
            downside_sigma *= np.sqrt(252 / offset)
        return downside_sigma


    def upside_sigma_r(
        self,
        threshold: float, # upside retunr threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns downside standard deviation of returns.
        """
        _rs = self._ln_r(offset).map(
            lambda x: np.max([x - threshold, 0]) ** 2
        )
        upside_sigma = np.sqrt(_rs.sum()/len(_rs))

        if annualise:
            upside_sigma *= np.sqrt(252 / offset)
        return upside_sigma


    def covar_r(
        self,
        benchmark: str,
        offset: int = 1,
        annualise: bool = True,
    ):
        """
        Returns covariance between time-series and benchmark.
        """
        _rs = self._ln_r(offset)

        covar = (
            (
                (_rs - _rs.mean()) * (_rs[[benchmark]] - _rs[[benchmark]].mean()).values
            ).sum() / (len(_rs) - 1)
        )

        if annualise:
            covar *= 252 / offset
        return covar


    def corr_r(
        self,
        benchmark: str,
        offset: int = 1,
        annualise: bool = True,
    ):
        """
        Returns correlation between time-series and benchmark.
        """
        covar = self.covar_r(benchmark, offset)
        sigma = self.sigma_r(offset)

        return covar / (sigma * sigma[[benchmark]].values)



