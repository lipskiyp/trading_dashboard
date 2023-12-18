"""
Base class for trading dashboard.
"""

from abc import ABC
import numpy as np
from pandas.core.frame import DataFrame


class BaseDashboard(ABC):
    """
    Abstract base class for trading dashboard.
    """

    def ln_rs(
        self,
        offset: int
    ) -> DataFrame:
        """
        Returns log returns.
        """
        return np.log(
            self.ts.shift(offset) / self.ts
        )

    def ts(self):
        """
        Returns time-series.
        """
        return self.ts
