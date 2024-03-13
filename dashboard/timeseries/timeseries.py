"""
Time-series object.
"""

import numpy as np
from pandas.core.frame import DataFrame


TRADING_DAYS_YEAR = 252


class TimeSeries:
    """
    Time-series object.
    """

    def __init__(
        self, ts: DataFrame
    ) -> None:
        self.ts = ts
        self.trading_days_year = TRADING_DAYS_YEAR


    @property
    def timeseries(self) -> DataFrame:
        """
        Returns time-series data frame.
        """
        return self.ts
    

    def ln_rs(
        self, offset: int
    ) -> DataFrame:
        """
        Returns log returns for the time-series.
        """
        return np.log(
            self.ts.shift(offset) / self.ts
        )
    

    def avg_ds(
        self, offset: int = 1,
    ) -> DataFrame:
        """
        Returns average price change for the time-series.
        """
        return (
            self.ts.shift(offset) - self.ts
        ).mean()


    def avg_r(
        self, offset: int = 1, annualise: bool = True,
    ) -> DataFrame:
        """
        Returns average returns for the time-series.
        """
        avg_rs = self.ln_rs(offset).mean()

        if annualise:
            avg_rs *= self.trading_days_year/offset
        return avg_rs
    