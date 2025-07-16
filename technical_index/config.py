#!/usr/bin/env python3
"""
配置管理模块
支持从JSON文件加载和保存规则配置
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .monitor import RuleConfig, RuleType

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """监控配置数据类"""

    symbols: List[str] = field(default_factory=list)
    interval: str = "1h"
    check_interval_minutes: int = 15  # 检查间隔（分钟）
    enabled: bool = True
    log_level: str = "INFO"
    save_signals: bool = True
    signal_file: str = "signals.json"


@dataclass
class RuleDefinition:
    """规则定义数据类"""

    name: str
    rule_type: str  # RuleType的字符串值
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolConfig:
    """交易对配置数据类"""

    symbol: str
    interval: str
    rules: List[RuleDefinition] = field(default_factory=list)
    callback_module: Optional[str] = None  # 自定义回调模块
    callback_function: Optional[str] = None  # 自定义回调函数


@dataclass
class GlobalConfig:
    """全局配置数据类"""

    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    symbols: List[SymbolConfig] = field(default_factory=list)
    global_rules: List[RuleDefinition] = field(default_factory=list)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: str = "config/monitor_config.json"):
        self.config_file = config_file
        self.config = GlobalConfig()

    def load_config(self) -> GlobalConfig:
        """从文件加载配置"""
        if not os.path.exists(self.config_file):
            logger.info(f"配置文件 {self.config_file} 不存在，创建默认配置")
            self.config = create_default_config()
            self.save_config()
            return self.config

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 重建配置对象
            self.config = self._dict_to_config(data)
            logger.info(f"成功加载配置文件: {self.config_file}")
            return self.config

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            # 如果加载失败，使用默认配置
            self.config = create_default_config()
            return self.config

    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            data = self._config_to_dict(self.config)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def _config_to_dict(self, config: GlobalConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        return {
            "monitor": asdict(config.monitor),
            "symbols": [asdict(symbol) for symbol in config.symbols],
            "global_rules": [asdict(rule) for rule in config.global_rules],
        }

    def _dict_to_config(self, data: Dict[str, Any]) -> GlobalConfig:
        """将字典转换为配置对象"""
        monitor_config = MonitorConfig(**data.get("monitor", {}))

        symbols = []
        for symbol_data in data.get("symbols", []):
            rules = [RuleDefinition(**rule_data) for rule_data in symbol_data.get("rules", [])]
            symbol_config = SymbolConfig(
                symbol=symbol_data["symbol"],
                interval=symbol_data["interval"],
                rules=rules,
                callback_module=symbol_data.get("callback_module"),
                callback_function=symbol_data.get("callback_function"),
            )
            symbols.append(symbol_config)

        global_rules = [RuleDefinition(**rule_data) for rule_data in data.get("global_rules", [])]

        return GlobalConfig(monitor=monitor_config, symbols=symbols, global_rules=global_rules)

    def add_symbol_config(self, symbol: str, interval: str = "1h") -> None:
        """添加交易对配置"""
        # 检查是否已存在
        for existing_symbol in self.config.symbols:
            if existing_symbol.symbol == symbol:
                logger.warning(f"交易对 {symbol} 已存在")
                return

        symbol_config = SymbolConfig(symbol=symbol, interval=interval)
        self.config.symbols.append(symbol_config)
        logger.info(f"已添加交易对配置: {symbol}")

    def remove_symbol_config(self, symbol: str) -> bool:
        """移除交易对配置"""
        for i, symbol_config in enumerate(self.config.symbols):
            if symbol_config.symbol == symbol:
                del self.config.symbols[i]
                logger.info(f"已移除交易对配置: {symbol}")
                return True
        return False

    def add_rule_to_symbol(self, symbol: str, rule: RuleDefinition) -> bool:
        """为交易对添加规则"""
        for symbol_config in self.config.symbols:
            if symbol_config.symbol == symbol:
                symbol_config.rules.append(rule)
                logger.info(f"已为 {symbol} 添加规则: {rule.name}")
                return True
        return False

    def remove_rule_from_symbol(self, symbol: str, rule_name: str) -> bool:
        """从交易对移除规则"""
        for symbol_config in self.config.symbols:
            if symbol_config.symbol == symbol:
                for i, rule in enumerate(symbol_config.rules):
                    if rule.name == rule_name:
                        del symbol_config.rules[i]
                        logger.info(f"已从 {symbol} 移除规则: {rule_name}")
                        return True
        return False

    def add_global_rule(self, rule: RuleDefinition) -> None:
        """添加全局规则"""
        self.config.global_rules.append(rule)
        logger.info(f"已添加全局规则: {rule.name}")

    def remove_global_rule(self, rule_name: str) -> bool:
        """移除全局规则"""
        for i, rule in enumerate(self.config.global_rules):
            if rule.name == rule_name:
                del self.config.global_rules[i]
                logger.info(f"已移除全局规则: {rule_name}")
                return True
        return False

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """获取交易对配置"""
        for symbol_config in self.config.symbols:
            if symbol_config.symbol == symbol:
                return symbol_config
        return None

    def update_monitor_config(self, **kwargs) -> None:
        """更新监控配置"""
        for key, value in kwargs.items():
            if hasattr(self.config.monitor, key):
                setattr(self.config.monitor, key, value)
        logger.info("已更新监控配置")


def create_default_config() -> GlobalConfig:
    """创建默认配置"""
    monitor_config = MonitorConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        interval="1h",
        check_interval_minutes=15,
        enabled=True,
        log_level="INFO",
        save_signals=True,
        signal_file="log/signals.json",
    )

    # 默认规则定义
    default_rules = [
        RuleDefinition(
            name="价格波动监控",
            rule_type=RuleType.PRICE_BASED.value,
            enabled=True,
            parameters={"volatility_threshold": 0.03},
        ),
        RuleDefinition(
            name="价格突破监控", rule_type=RuleType.PRICE_BASED.value, enabled=True, parameters={}
        ),
        RuleDefinition(
            name="MACD金叉死叉",
            rule_type=RuleType.TECHNICAL_INDICATOR.value,
            enabled=True,
            parameters={},
        ),
        RuleDefinition(
            name="RSI超买超卖",
            rule_type=RuleType.TECHNICAL_INDICATOR.value,
            enabled=True,
            parameters={},
        ),
        RuleDefinition(
            name="趋势分析",
            rule_type=RuleType.TECHNICAL_INDICATOR.value,
            enabled=True,
            parameters={},
        ),
    ]

    # 为每个交易对创建配置
    symbols = []
    for symbol in monitor_config.symbols:
        symbol_config = SymbolConfig(
            symbol=symbol, interval=monitor_config.interval, rules=default_rules.copy()
        )
        symbols.append(symbol_config)

    return GlobalConfig(monitor=monitor_config, symbols=symbols, global_rules=[])


def load_rules_from_config(config: GlobalConfig) -> List[tuple]:
    """从配置加载规则"""
    from .monitor import (create_breakout_rule, create_custom_rule,
                          create_macd_rule, create_price_volatility_rule,
                          create_rsi_rule, create_trend_rule)

    rules = []

    # 处理全局规则
    for rule_def in config.global_rules:
        if rule_def.rule_type == RuleType.PRICE_BASED.value:
            if rule_def.name == "价格波动监控":
                rule = create_price_volatility_rule(
                    symbol="GLOBAL",
                    interval=config.monitor.interval,
                    volatility_threshold=rule_def.parameters.get("volatility_threshold", 0.05),
                )
                rules.append((rule, True))  # True表示全局规则
        # 可以添加更多全局规则类型...

    # 处理交易对特定规则
    for symbol_config in config.symbols:
        for rule_def in symbol_config.rules:
            if rule_def.rule_type == RuleType.PRICE_BASED.value:
                if rule_def.name == "价格波动监控":
                    rule = create_price_volatility_rule(
                        symbol=symbol_config.symbol,
                        interval=symbol_config.interval,
                        volatility_threshold=rule_def.parameters.get("volatility_threshold", 0.05),
                    )
                    rules.append((rule, False))
                elif rule_def.name == "价格突破监控":
                    rule = create_breakout_rule(
                        symbol=symbol_config.symbol, interval=symbol_config.interval
                    )
                    rules.append((rule, False))
            elif rule_def.rule_type == RuleType.TECHNICAL_INDICATOR.value:
                if rule_def.name == "MACD金叉死叉":
                    rule = create_macd_rule(
                        symbol=symbol_config.symbol, interval=symbol_config.interval
                    )
                    rules.append((rule, False))
                elif rule_def.name == "RSI超买超卖":
                    rule = create_rsi_rule(
                        symbol=symbol_config.symbol, interval=symbol_config.interval
                    )
                    rules.append((rule, False))
                elif rule_def.name == "趋势分析":
                    rule = create_trend_rule(
                        symbol=symbol_config.symbol, interval=symbol_config.interval
                    )
                    rules.append((rule, False))

    return rules
