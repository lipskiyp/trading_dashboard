"""
Tail metrics service.
"""

from pandas.core.frame import DataFrame

from .core import CoreMetrics
from ..timeseries import TimeSeries


class TailMetrics:
    """
    Tail-risk metrics service.
    """

    @staticmethod
    def skew(
        ts: TimeSeries, offset: int = 1,
    ) -> DataFrame:
        """
        Returns skew of returns.
        """
        rs = ts.ln_rs(offset)
        sigma = CoreMetrics.sigma(ts, offset, annualise=False)
        N = len(rs)

        return (
            (rs - rs.mean()) ** 3
        ).sum() / sigma * N / (N - 1) / (N - 2)


    @staticmethod
    def coskew(
        ts: TimeSeries, benchmark: str, offset: int = 1
    ):
        """
        Returns coskew of returns with benchmark.
        """
        rs = ts.ln_rs(offset)
        sigma = CoreMetrics.sigma(ts, offset, annualise=False)
        N = len(rs)

        return (
            (rs - rs.mean()) * ((rs[[benchmark]] - rs[[benchmark]].mean()) ** 2).values
        ).sum() / sigma * N / (N - 1) / (N - 2)


    @staticmethod
    def kurtosis(
        ts: TimeSeries, offset: int = 1, excess: bool = True,
    ) -> DataFrame:
        """
        Returns kurtosis of returns.
        """
        rs = ts.ln_rs(offset)
        sigma = CoreMetrics.sigma(ts, offset, annualise=False)
        N = len(rs)

        kurt = (
            (rs - rs.mean()) ** 4
        ).sum() / sigma * N * (N - 1) / (N - 2) / (N - 3)

        if not excess:
            kurt -= 3 * (N - 1) ** 2 / (N - 2) / (N - 3)
        return kurt


    @staticmethod
    def cokurtosis(
        ts: TimeSeries, benchmark: str, offset: int = 1, excess: bool = True,
    ) -> DataFrame:
        """
        Returns cokurtosis of returns with benchmark.
        """
        rs = ts.ln_rs(offset)
        sigma = CoreMetrics.sigma(ts, offset, annualise=False)
        N = len(rs)

        cokurt = (
            (rs - rs.mean()) * ((rs[[benchmark]] - rs[[benchmark]].mean()) ** 3).values
        ).sum() / sigma * N * (N - 1) / (N - 2) / (N - 3)

        if not excess:
            cokurt -= 3 * (N - 1) ** 2 / (N - 2) / (N - 3)
        return cokurt


    @staticmethod
    def drawdown(
        ts: TimeSeries,
    ) -> DataFrame:
        """
        Returns current loss from last high water mark.
        """
        return ts.timeseries.iloc[-1] / ts.timeseries.max() - 1


    @staticmethod
    def drawdown_dur(
        ts: TimeSeries,
    ) -> DataFrame:
        """
        Returns current drawdown duration.
        """
        return ts.timeseries.index.max() - ts.timeseries.idxmax()


    @staticmethod
    def maxdrawdown(
        ts: TimeSeries,
    ) -> DataFrame:
        """
        Returns maximum loss from last high water mark.
        """
        return (
            ts.timeseries / ts.timeseries.cummax() - 1
        ).min()


    @staticmethod
    def maxdrawdown_dur(
        ts: TimeSeries,
    ) -> DataFrame:
        """
        Returns maximum drawdown duration.
        """
        def maxdd_dur(x):
            dd = x / x.cummax() - 1
            return dd.idxmin() - x[:dd.idxmin()].idxmax()

        return ts.timeseries.apply(
            maxdd_dur
        )


    @staticmethod
    def pain_index(
        ts: TimeSeries,
    ) -> DataFrame:
        """
        Retursn average drawdown from last high water mark.
        """
        return (
            1 - ts.timeseries / ts.timeseries.cummax()
        ).mean()
