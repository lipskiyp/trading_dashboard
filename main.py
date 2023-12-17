"""
Execution file.
"""


from dashboard import TradingDashboard
from timeseries import TimeSeriesFactory


if __name__ == "__main__":
    ts = TimeSeriesFactory.get_random_daily_ts(calendar="NYSE")
    dashboard = TradingDashboard(ts=ts)

    print({
        "Avg. Annual Return": dashboard.avg_r()[0],
        "CAGR": dashboard.cagr()[0],
        "Vol": dashboard.sigma_r()[0],
        "DownsideVol": dashboard.downside_sigma_r(threshold=0.01)[0],
        "UpsideVol": dashboard.upside_sigma_r(threshold=0.01)[0],
    })
