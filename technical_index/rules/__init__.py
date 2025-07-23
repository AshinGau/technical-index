"""
规则模块
包含所有技术分析规则实现
"""

from .base_rule import BaseRule, RuleConfig, RuleType, SignalResult, SignalType
from .custom_rule import CustomRule
from .macd_golden_cross_rule import MACDGoldenCrossRule
from .moving_average_rule import MovingAverageRule
from .new_high_low_rule import NewHighLowRule
from .price_breakout_rule import PriceBreakoutRule
from .price_volatility_rule import PriceVolatilityRule
from .rsi_signal_rule import RSISignalRule
from .trend_analysis_rule import TrendAnalysisRule

__all__ = [
    "BaseRule",
    "RuleConfig",
    "SignalResult",
    "SignalType",
    "RuleType",
    "PriceVolatilityRule",
    "PriceBreakoutRule",
    "NewHighLowRule",
    "MACDGoldenCrossRule",
    "MovingAverageRule",
    "RSISignalRule",
    "TrendAnalysisRule",
    "CustomRule",
]
