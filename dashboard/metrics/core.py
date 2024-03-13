"""
Core metrics service.
"""

import numpy as np
from pandas.core.frame import DataFrame

from ..timeseries import TimeSeries


class CoreMetrics:
    """
    Core metrics service.
    """
    
    @staticmethod
    def var(
        ts: TimeSeries, offset: int = 1, annualise: bool = True,
    ) -> DataFrame:
        """
        Returns var of returns.
        """
        rs = ts.ln_rs(offset)
        N = len(rs)

        sigma = (
            (rs - rs.mean()) ** 2
        ).sum() / (N - 1)

        if annualise:
            sigma *= ts.trading_days_year / offset
        return sigma


    @staticmethod
    def sigma(
        ts: TimeSeries, offset: int = 1, annualise: bool = True,
    ) -> DataFrame:
        """
        Returns standard deviation of retunrs.
        """
        return np.sqrt(
            CoreMetrics.var(ts, offset, annualise)
        )


    @staticmethod
    def cagr(
        ts: TimeSeries, offset: int = 1,
    ) -> DataFrame:
        """
        Returns compounded annual growth rate (CAGR).
        """
        cumprod = (1 + ts.ln_rs(offset)).cumprod()
        power = (len(cumprod.index)) / ts.trading_days_year
        return cumprod.iloc[-1] ** power - 1


    @staticmethod
    def downside_sigma(
        ts: TimeSeries, threshold: float, offset: int = 1, annualise: bool = True,
    ) -> DataFrame:
        """
        Returns downside (loss) standard deviation of returns.
        """
        rs = ts.ln_rs(offset).map(
            lambda x: np.min([x - threshold, 0]) ** 2
        )
        N = len(rs)

        downside_sigma = np.sqrt( rs.sum() / N )

        if annualise:
            downside_sigma *= np.sqrt(ts.trading_days_year / offset)
        return downside_sigma


    @staticmethod
    def upside_sigma(
        ts: TimeSeries, threshold: float, offset: int = 1, annualise: bool = True,
    ) -> DataFrame:
        """
        Returns downside standard deviation of returns.
        """
        rs = ts.ln_rs(offset).map(
            lambda x: np.max([x - threshold, 0]) ** 2
        )
        N = len(rs)

        upside_sigma = np.sqrt(rs.sum() / N)

        if annualise:
            upside_sigma *= np.sqrt(ts.trading_days_year / offset)
        return upside_sigma


    @staticmethod
    def covar(
        ts: TimeSeries, benchmark: str, offset: int = 1, annualise: bool = True,
    ) -> DataFrame:
        """
        Returns covariance between time-series and benchmark.
        """
        rs = ts.ln_rs(offset)
        N = len(rs)

        covar = (
            (
                (rs - rs.mean()) * (rs[[benchmark]] - rs[[benchmark]].mean()).values
            ).sum() / (N - 1)
        )

        if annualise:
            covar *= ts.trading_days_year / offset
        return covar


    @staticmethod
    def corr(
        ts: TimeSeries, benchmark: str, offset: int = 1,
    ) -> DataFrame:
        """
        Returns correlation between time-series and benchmark.
        """
        covar = CoreMetrics.covar(ts, benchmark, offset)
        sigma = CoreMetrics.sigma(ts, offset)

        return covar / (sigma * sigma[[benchmark]].values)
