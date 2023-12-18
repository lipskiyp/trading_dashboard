"""
Timeseries factory.
"""

from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.frame import DataFrame
import pandas_market_calendars as mcal
from typing import Optional


class TimeSeriesFactory:
    """
    Generates random financial timeseries.
    """
    DAYS_YEAR = 252

    @classmethod
    def get_random_daily_ts(
        cls,
        initial_price: int = 100,
        drift: float = 0.05,
        sigma: float = 0.1,
        start_date: date = (datetime.now() - timedelta(days=365)).date(),  # 1 year ago
        end_date: date = datetime.now().date(),  # today
        calendar: Optional[str] = None,
    ) -> DataFrame:
        """
        Returns random financial timeseries for calendar dates in the specified range.
        """
        cal = cls.get_dates_from_cal(
            start_date, end_date, calendar
        )

        daily_rs = np.random.normal(
            drift/cls.DAYS_YEAR, sigma/np.sqrt(cls.DAYS_YEAR), len(cal)
        )

        return pd.DataFrame(
            data = initial_price * np.exp(np.cumsum(daily_rs)),
            index = cal
        )

    @classmethod
    def get_dates_from_cal(
        cls,
        start_date: date,
        end_date: date,
        calendar: Optional[str] = None
    ) -> DatetimeIndex:
        """
        Returns Pandas DatetimeIndex with calendar dates in the specified range.
        """
        if calendar:
            try:
                cal = mcal.get_calendar(calendar)
                return cal.schedule(
                    start_date, end_date
                ).index
            except RuntimeError:
                raise RuntimeError(f"Invalid calendar: {calendar}, using default.")

        return mcal.date_range(
            start_date, end_date, "D"
        )
