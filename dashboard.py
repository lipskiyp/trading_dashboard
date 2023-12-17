"""
Trading dashborad.
"""

import numpy as np
from pandas.core.frame import DataFrame
from typing import List

from base import BaseDashboard


class TradingDashboard(BaseDashboard):
    """
    Class to caculate metrics for financial timeseries.
    """
    DAYS_YEAR = 252

    @classmethod
    def get_avg_r(cls,
        ts: DataFrame,
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns average returns.
        """
        avg_rs = cls._get_ln_r(ts, offset).mean().values

        if annualise:
            avg_rs *= cls.DAYS_YEAR/offset
        return avg_rs

    @classmethod
    def get_cagr(
        cls,
        ts: DataFrame,
        offset: int = 1,
    ) -> List[float]:
        """
        Returns compounded annual growth rate (CAGR).
        """
        _cumprod = (1 + cls._get_ln_r(ts, offset)).cumprod()
        _power = (len(_cumprod.index)) / cls.DAYS_YEAR
        return np.power(_cumprod.values[-1], _power) - 1

    @classmethod
    def get_sigma_r(
        cls,
        ts: DataFrame,
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns standard deviation of returns.
        """
        _rs = cls._get_ln_r(ts, offset) ** 2
        sigma = np.sqrt(
            _rs.sum() / (len(_rs) - 1)
        ).values

        if annualise:
            sigma *= np.sqrt(252 / offset)
        return sigma

    @classmethod
    def get_downside_sigma_r(
        cls,
        ts: DataFrame,
        threshold: float, # downside retunr threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns downside (loss) standard deviation of returns.
        """
        _rs = cls._get_ln_r(ts, offset).map(
            lambda x: np.min([x - threshold, 0]) ** 2
        )
        downside_sigma = np.sqrt(_rs.sum()/len(_rs)).values

        if annualise:
            downside_sigma *= np.sqrt(252 / offset)
        return downside_sigma

    @classmethod
    def get_upside_sigma_r(
        cls,
        ts: DataFrame,
        threshold: float, # upside retunr threshold
        offset: int = 1,
        annualise: bool = True,
    ) -> List[float]:
        """
        Returns downside standard deviation of returns.
        """
        _rs = cls._get_ln_r(ts, offset).map(
            lambda x: np.max([x - threshold, 0]) ** 2
        )
        upside_sigma = np.sqrt(_rs.sum()/len(_rs)).values

        if annualise:
            upside_sigma *= np.sqrt(252 / offset)
        return upside_sigma
