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

    def __init__(self, ts: TimeSeries):
        self.ts = ts
        self.core_metrics = CoreMetrics(ts)


    def skew(
        self, offset: int = 1,
    ) -> DataFrame:
        """
        Returns skew of returns.
        """
        rs = self.ts.ln_rs(offset)
        sigma = self.core_metrics.sigma(offset, annualise=False)
        N = len(rs)

        return (
            (rs - rs.mean()) ** 3
        ).sum() / sigma * N / (N - 1) / (N - 2)


    def coskew(
        self, benchmark: str, offset: int = 1
    ):
        """
        Returns coskew of returns with benchmark.
        """
        rs = self.ts.ln_rs(offset)
        sigma = self.core_metrics.sigma(offset, annualise=False)
        N = len(rs)

        return (
            (rs - rs.mean()) * ((rs[[benchmark]] - rs[[benchmark]].mean()) ** 2).values
        ).sum() / sigma * N / (N - 1) / (N - 2)


    def kurtosis(
        self, offset: int = 1, excess: bool = True,
    ) -> DataFrame:
        """
        Returns kurtosis of returns.
        """
        rs = self.ts.ln_rs(offset)
        sigma = self.core_metrics.sigma(offset, annualise=False)
        N = len(rs)

        kurt = (
            (rs - rs.mean()) ** 4
        ).sum() / sigma * N * (N - 1) / (N - 2) / (N - 3)

        if not excess:
            kurt -= 3 * (N - 1) ** 2 / (N - 2) / (N - 3)
        return kurt


    def cokurtosis(
        self, benchmark: str, offset: int = 1, excess: bool = True,
    ) -> DataFrame:
        """
        Returns cokurtosis of returns with benchmark.
        """
        rs = self.ts.ln_rs(offset)
        sigma = self.core_metrics.sigma(offset, annualise=False)
        N = len(rs)

        cokurt = (
            (rs - rs.mean()) * ((rs[[benchmark]] - rs[[benchmark]].mean()) ** 3).values
        ).sum() / sigma * N * (N - 1) / (N - 2) / (N - 3)

        if not excess:
            cokurt -= 3 * (N - 1) ** 2 / (N - 2) / (N - 3)
        return cokurt


    def drawdown(
        self,
    ) -> DataFrame:
        """
        Returns current loss from last high water mark.
        """
        return self.ts.timeseries.iloc[-1] / self.ts.timeseries.max() - 1


    def drawdown_dur(
        self,
    ) -> DataFrame:
        """
        Returns current drawdown duration.
        """
        return self.ts.timeseries.index.max() - self.ts.timeseries.idxmax()


    def maxdrawdown(
        self,
    ) -> DataFrame:
        """
        Returns maximum loss from last high water mark.
        """
        return (
            self.ts.timeseries / self.ts.timeseries.cummax() - 1
        ).min()


    def maxdrawdown_dur(
        self,
    ) -> DataFrame:
        """
        Returns maximum drawdown duration.
        """
        def maxdd_dur(x):
            dd = x / x.cummax() - 1
            return dd.idxmin() - x[:dd.idxmin()].idxmax()

        return self.ts.timeseries.apply(
            maxdd_dur
        )


    def pain_index(
        self,
    ) -> DataFrame:
        """
        Retursn average drawdown from last high water mark.
        """
        return (
            1 - self.ts.timeseries / self.ts.timeseries.cummax()
        ).mean()
