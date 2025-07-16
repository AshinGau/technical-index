"""
Technical Index Package

A Python package for technical index calculations and price monitoring.
"""

__version__ = "0.1.0"
__author__ = "AshinGau"
__email__ = "helloxiyue@gmail.com"

# 导出监控模块
# 导出技术指标模块
from . import binance, config, index, monitor, plot
from .config import (ConfigManager, GlobalConfig, MonitorConfig,
                     RuleDefinition, SymbolConfig, create_default_config,
                     load_rules_from_config)
# 导出主要类
from .monitor import (BaseRule, CustomRule, MACDGoldenCrossRule,
                      NewHighLowRule, PriceBreakoutRule, PriceMonitor,
                      PriceVolatilityRule, RSISignalRule, RuleEngine, RuleType,
                      SignalResult, SignalType, TrendAnalysisRule,
                      create_breakout_rule, create_custom_rule,
                      create_macd_rule, create_price_volatility_rule,
                      create_rsi_rule, create_trend_rule)

__all__ = [
    # 技术指标相关
    "index",
    "binance",
    "plot",
    # 监控相关
    "monitor",
    "config",
    # 核心类
    "RuleEngine",
    "PriceMonitor",
    "SignalResult",
    "SignalType",
    "RuleType",
    "BaseRule",
    "PriceVolatilityRule",
    "PriceBreakoutRule",
    "NewHighLowRule",
    "MACDGoldenCrossRule",
    "RSISignalRule",
    "TrendAnalysisRule",
    "CustomRule",
    # 便捷函数
    "create_price_volatility_rule",
    "create_breakout_rule",
    "create_macd_rule",
    "create_rsi_rule",
    "create_trend_rule",
    "create_custom_rule",
    # 配置相关
    "ConfigManager",
    "GlobalConfig",
    "MonitorConfig",
    "SymbolConfig",
    "RuleDefinition",
    "create_default_config",
    "load_rules_from_config",
]
