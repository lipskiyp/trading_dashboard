"""
Execution file.
"""


from dashboard import TradingDashboard
from timeseries import TimeSeriesFactory


if __name__ == "__main__":
    ts = TimeSeriesFactory.get_random_daily_ts(calendar="NYSE")

    print({
        "Avg. Annual Return": TradingDashboard.get_avg_r(ts)[0],
        "CAGR": TradingDashboard.get_cagr(ts)[0],
        "Vol": TradingDashboard.get_sigma_r(ts)[0],
        "DownsideVol": TradingDashboard.get_downside_sigma_r(ts, threshold=0.01)[0],
        "UpsideVol": TradingDashboard.get_upside_sigma_r(ts, threshold=0.01)[0],
    })
