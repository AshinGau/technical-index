#!/usr/bin/env python3
"""
常量定义模块
定义系统中使用的各种常量
"""


# 配置文件相关常量
DEFAULT_CONFIG_FILE = "config/monitor_config.json"
DEFAULT_SIGNAL_FILE = "log/signals.json"


# 规则名称常量（英文）
class RuleNames:
    """规则名称常量类"""

    PRICE_VOLATILITY = "price_volatility"
    PRICE_BREAKOUT = "price_breakout"
    MACD_GOLDEN_CROSS = "macd_golden_cross"
    RSI_SIGNAL = "rsi_signal"
    TREND_ANALYSIS = "trend_analysis"
    NEW_HIGH_LOW = "new_high_low"


# 规则描述映射
RULE_DESCRIPTIONS = {
    RuleNames.PRICE_VOLATILITY: "价格波动监控",
    RuleNames.PRICE_BREAKOUT: "价格突破监控",
    RuleNames.MACD_GOLDEN_CROSS: "MACD金叉死叉",
    RuleNames.RSI_SIGNAL: "RSI超买超卖",
    RuleNames.TREND_ANALYSIS: "趋势分析",
    RuleNames.NEW_HIGH_LOW: "新高新低监控",
}

# 默认监控配置
DEFAULT_MONITOR_INTERVAL = "1h"
DEFAULT_CHECK_INTERVAL_MINUTES = 15
DEFAULT_LOG_LEVEL = "INFO"

# 默认交易对
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# 文件路径常量
def get_config_file_path(config_file: str = None) -> str:
    """获取配置文件路径"""
    if config_file:
        return config_file
    return DEFAULT_CONFIG_FILE


def get_signal_file_path(signal_file: str = None) -> str:
    """获取信号文件路径"""
    if signal_file:
        return signal_file
    return DEFAULT_SIGNAL_FILE
