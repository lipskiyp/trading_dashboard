"""
Base dashboard.
"""

import numpy as np
from pandas.core.frame import DataFrame

class BaseDashboard:
    """
    Base class
    """
    def _ln_r(
        self,
        offset: int
    ) -> DataFrame:
        """
        Return log returns.
        """
        return np.log(
            self.ts.shift(offset) / self.ts
        )
