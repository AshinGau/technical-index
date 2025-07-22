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
from .config import (
    ConfigManager,
    GlobalConfig,
    RuleDefinition,
    SymbolConfig,
    create_default_config,
    load_rules_from_config,
)

# 导出常量
from .constants import DEFAULT_CONFIG_FILE, RULE_DESCRIPTIONS, RuleNames

# rules
from .rules import (
    BaseRule,
    RuleConfig,
    RuleType,
    SignalResult,
    SignalType,
    CustomRule,
    PriceVolatilityRule,
    PriceBreakoutRule,
    NewHighLowRule,
    MACDGoldenCrossRule,
    RSISignalRule,
    TrendAnalysisRule,
)

# 导出主要类
from .monitor import (
    PriceMonitor,
    RuleEngine,
    RuleFactory,
)

__all__ = [
    # 技术指标相关
    "index",
    "binance",
    "plot",
    # 监控相关
    "monitor",
    "config",
    # 常量
    "DEFAULT_CONFIG_FILE",
    "RuleNames",
    "RULE_DESCRIPTIONS",
    # 核心类
    "RuleEngine",
    "PriceMonitor",
    "SignalResult",
    "SignalType",
    "RuleType",
    "BaseRule",
    "RuleConfig",
    "PriceVolatilityRule",
    "PriceBreakoutRule",
    "NewHighLowRule",
    "MACDGoldenCrossRule",
    "RSISignalRule",
    "TrendAnalysisRule",
    "CustomRule",
    "RuleFactory",
    # 配置相关
    "ConfigManager",
    "GlobalConfig",
    "SymbolConfig",
    "RuleDefinition",
    "create_default_config",
    "load_rules_from_config",
]
