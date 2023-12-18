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

    res = pd.concat(
        [
            dashboard.avg_ds(),
            dashboard.avg_r(),
            dashboard.cagr(),
            dashboard.sigma(),
            dashboard.downside_sigma(threshold=0.01),
            dashboard.upside_sigma(threshold=0.01),
            dashboard.covar(benchmark="benchmark"),
            dashboard.corr(benchmark="benchmark"),
            dashboard.skew(),
            dashboard.kurtosis(),
        ],
        axis=1
    )

    res.columns = [
        "Avg. Price Change",
        "Avg. Annual Return",
        "CAGR",
        "Vol",
        "DownsideVol",
        "UpsideVol",
        "CoVar",
        "Corr",
        "Skew",
        "Kurt"
    ]

    print(res)
