"""
Base dashboard.
"""

import numpy as np
from pandas.core.frame import DataFrame

class BaseDashboard:
    """
    Base class
    """
    @classmethod
    def _get_ln_r(
        cls,
        ts: DataFrame,
        offset: int
    ) -> DataFrame:
        """
        Return log returns.
        """
        return np.log(
            ts.shift(offset) / ts
        )
