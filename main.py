"""
Trading dashborad.
"""

import numpy as np
from pandas.core.frame import DataFrame
from timeseries import TimeSeriesFactory


class TradingDashboard:
    """
    Class to caculate metrics for financial timeseries.
    """
    DAYS_YEAR = 252

    @classmethod
    def get_avg_daily_r(
        cls,
        ts: DataFrame,
        annualise: bool = True,
    ):
        """
        Returns average daily return.
        """
        return cls.get_avg_r(
            ts, offset=1, annualise=annualise
        )

    @classmethod
    def get_avg_weekly_r(
        cls,
        ts: DataFrame,
        annualise: bool = True,
    ):
        """
        Returns average weekly return.
        """
        return cls.get_avg_r(
            ts, offset=5, annualise=annualise
        )

    @classmethod
    def get_avg_r(
        cls,
        ts: DataFrame,
        offset: int = 1,
        annualise: bool = True,
    ) -> float:
        """
        Returns average returns.
        """
        avg_rs = np.log(
            ts.shift(offset) / ts
        ).mean().values

        if annualise:
            avg_rs *= cls.DAYS_YEAR/offset
        return avg_rs





ts = TimeSeriesFactory.get_random_daily_ts(calendar="NYSE")
r = TradingDashboard.get_avg_weekly_r(ts)
print(r)

