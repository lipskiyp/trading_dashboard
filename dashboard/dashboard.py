"""
Trading dashborad.
"""

from pandas.core.frame import DataFrame

from .base import BaseDashboard
from .core import CoreDashboard
from .tail import TailDashboard


class TradingDashboard(
    BaseDashboard, CoreDashboard, TailDashboard
):
    """
    Trading dashboard.
    """

    TRADING_DAYS_YEAR = 252

    def __init__(self, ts: DataFrame):
        self.ts = ts
