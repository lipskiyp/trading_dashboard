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

    @property
    def metrics(self) -> DataFrame:
        """
        Returns metrics for the dashboard.
        """
        _metrics = {
            "Avg. Price Change": self.ts.avg_ds(),
            "Avg. Annual Return": self.ts.avg_r(),
            "CAGR": CoreMetrics.cagr(self.ts),
            "Var": CoreMetrics.var(self.ts),
            "Vol": CoreMetrics.sigma(self.ts),
            "DownsideVol": CoreMetrics.downside_sigma(self.ts, threshold=0.01),
            "UpsideVol": CoreMetrics.upside_sigma(self.ts, threshold=0.01),
            "CoVar": CoreMetrics.covar(self.ts, benchmark="benchmark"),
            "Corr": CoreMetrics.corr(self.ts, benchmark="benchmark"),
            "Skew": TailMetrics.skew(self.ts),
            "CoSkew": TailMetrics.coskew(self.ts, benchmark="benchmark"),
            "Kurt": TailMetrics.kurtosis(self.ts),
            "CoKurt": TailMetrics.cokurtosis(self.ts, benchmark="benchmark"),
            "DD": TailMetrics.drawdown(self.ts),
            "DDur": TailMetrics.drawdown_dur(self.ts),
            "MaxDD": TailMetrics.maxdrawdown(self.ts),
            "MaxDDur": TailMetrics.maxdrawdown_dur(self.ts),
            "PainIdx": TailMetrics.pain_index(self.ts),
        }

        metrics = pd.concat(_metrics.values(), axis=1)
        metrics.columns = _metrics.keys()
        return metrics.T
