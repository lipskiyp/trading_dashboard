"""
Execution file.
"""

import pandas as pd

from dashboard import TradingDashboard
from timeseries import TimeSeriesFactory


if __name__ == "__main__":
    # Get time-series
    benchmark = TimeSeriesFactory.get_random_daily_ts(calendar="NYSE")
    ts_1 = TimeSeriesFactory.get_random_daily_ts(calendar="NYSE")
    ts_2 = TimeSeriesFactory.get_random_daily_ts(calendar="NYSE")

    # Concat time-series
    ts = pd.concat([benchmark, ts_1, ts_2], axis=1, join="outer")
    ts.columns = ["benchmark", "ts_1", "ts_2"]

    # Initialise dashboard
    dashboard = TradingDashboard(ts=ts)

    metrics = {
        "Avg. Price Change": dashboard.avg_ds(),
        "Avg. Annual Return": dashboard.avg_r(),
        "CAGR": dashboard.cagr(),
        "Var": dashboard.var(),
        "Vol": dashboard.sigma(),
        "DownsideVol": dashboard.downside_sigma(threshold=0.01),
        "UpsideVol": dashboard.upside_sigma(threshold=0.01),
        "CoVar": dashboard.covar(benchmark="benchmark"),
        "Corr": dashboard.corr(benchmark="benchmark"),
        "Skew": dashboard.skew(),
        "CoSkew": dashboard.coskew(benchmark="benchmark"),
        "Kurt": dashboard.kurtosis(),
        "CoKurt": dashboard.cokurtosis(benchmark="benchmark"),
        "DD": dashboard.drawdown(),
        "DDur": dashboard.drawdown_dur(),
        "MaxDD": dashboard.maxdrawdown(),
        "MaxDDur": dashboard.maxdrawdown_dur(),
        "PainIdx": dashboard.pain_index(),
    }

    res = pd.concat(metrics.values(), axis=1)
    res.columns = metrics.keys()

    print(res.T)
