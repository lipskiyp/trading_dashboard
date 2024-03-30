"""
Trading dashborad client.
"""

import pandas as pd
from pandas.core.frame import DataFrame
from typing import Any

from .timeseries import TimeSeries
from .metrics import CoreMetrics, TailMetrics


class TradingDashboard:
    """
    Trading dashboard client.
    """

    def __init__(self, ts: DataFrame):
        self.ts = TimeSeries(ts)
        self.core_metrics = CoreMetrics(self.ts)
        self.tail_metrics = TailMetrics(self.ts)


    @property
    def metrics(self) -> DataFrame:
        """
        Returns metrics for the dashboard.
        """
        _metrics = {
            "Avg. Price Change": self.ts.avg_ds(),
            "Avg. Annual Return": self.ts.avg_r(),
            "CAGR": self.core_metrics.cagr(),
            "Var": self.core_metrics.var(),
            "Vol": self.core_metrics.sigma(),
            "DownsideVol": self.core_metrics.downside_sigma(threshold=0.01),
            "UpsideVol": self.core_metrics.upside_sigma(threshold=0.01),
            "CoVar": self.core_metrics.covar(benchmark="benchmark"),
            "Corr": self.core_metrics.corr(benchmark="benchmark"),
            "Skew": self.tail_metrics.skew(),
            "CoSkew": self.tail_metrics.coskew(benchmark="benchmark"),
            "Kurt": self.tail_metrics.kurtosis(),
            "CoKurt": self.tail_metrics.cokurtosis(benchmark="benchmark"),
            "DD": self.tail_metrics.drawdown(),
            "DDur": self.tail_metrics.drawdown_dur(),
            "MaxDD": self.tail_metrics.maxdrawdown(),
            "MaxDDur": self.tail_metrics.maxdrawdown_dur(),
            "PainIdx": self.tail_metrics.pain_index(),
        }

        metrics = pd.concat(_metrics.values(), axis=1)
        metrics.columns = _metrics.keys()
        return metrics.T
