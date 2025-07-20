#!/usr/bin/env python3
"""
配置管理模块
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .constants import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_MONITOR_INTERVAL,
    DEFAULT_SYMBOLS,
    RULE_DESCRIPTIONS,
    RuleNames,
)
from .monitor import RuleFactory, RuleType

logger = logging.getLogger(__name__)


@dataclass
class RuleDefinition:
    """规则定义数据类"""

    name: str
    rule_type: str  # RuleType的字符串值
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""  # 规则描述

    def merge_with(self, other: "RuleDefinition") -> "RuleDefinition":
        # 合并参数：先复制全局参数，再用本地参数覆盖
        merged_parameters = self.parameters.copy()
        merged_parameters.update(other.parameters)

        # 创建合并后的规则定义
        return RuleDefinition(
            name=other.name,
            rule_type=other.rule_type,
            enabled=other.enabled,
            parameters=merged_parameters,
            description=other.description or self.description,
        )


@dataclass
class SymbolConfig:
    """交易对配置数据类"""

    symbol: str
    interval: str = "1h"  # 交易对的间隔
    use_global_rules: bool = True  # 是否使用全局规则
    rules: List[RuleDefinition] = field(default_factory=list)  # 本地规则（覆盖全局规则）
    callback_module: Optional[str] = None  # 自定义回调模块
    callback_function: Optional[str] = None  # 自定义回调函数


@dataclass
class GlobalConfig:
    """全局配置数据类"""

    global_rules: List[RuleDefinition] = field(default_factory=list)  # 全局规则
    symbols: List[SymbolConfig] = field(default_factory=list)  # 交易对配置


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> GlobalConfig:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.config = self._dict_to_config(data)
            else:
                self.config = create_default_config()
                self.save_config()
                logger.info(f"已创建默认配置文件: {self.config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = create_default_config()

        return self.config

    def save_config(self) -> None:
        """保存配置文件"""
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
            "global_rules": [asdict(rule) for rule in config.global_rules],
            "symbols": [asdict(symbol) for symbol in config.symbols],
        }

    def _dict_to_config(self, data: Dict[str, Any]) -> GlobalConfig:
        """将字典转换为配置对象"""
        global_rules = [RuleDefinition(**rule_data) for rule_data in data.get("global_rules", [])]

        symbols = []
        for symbol_data in data.get("symbols", []):
            rules = [RuleDefinition(**rule_data) for rule_data in symbol_data.get("rules", [])]
            symbol_config = SymbolConfig(
                symbol=symbol_data["symbol"],
                interval=symbol_data.get("interval", "1h"),
                use_global_rules=symbol_data.get("use_global_rules", True),
                rules=rules,
                callback_module=symbol_data.get("callback_module"),
                callback_function=symbol_data.get("callback_function"),
            )
            symbols.append(symbol_config)

        return GlobalConfig(global_rules=global_rules, symbols=symbols)

    def get_symbol_config(self, symbol: str, interval: str) -> Optional[SymbolConfig]:
        """获取特定交易对和间隔的配置"""
        for symbol_config in self.config.symbols:
            if symbol_config.symbol == symbol and symbol_config.interval == interval:
                return symbol_config
        return None

    def add_symbol_config(self, symbol: str, interval: str, use_global_rules: bool = True) -> None:
        """添加交易对配置"""
        # 检查是否已存在
        if self.get_symbol_config(symbol, interval):
            logger.warning(f"交易对 {symbol}:{interval} 已存在")
            return

        symbol_config = SymbolConfig(
            symbol=symbol, interval=interval, use_global_rules=use_global_rules
        )
        self.config.symbols.append(symbol_config)
        logger.info(f"已添加交易对配置: {symbol}:{interval}")

    def remove_symbol_config(self, symbol: str, interval: str) -> bool:
        """移除交易对配置"""
        for i, symbol_config in enumerate(self.config.symbols):
            if symbol_config.symbol == symbol and symbol_config.interval == interval:
                del self.config.symbols[i]
                logger.info(f"已移除交易对配置: {symbol}:{interval}")
                return True
        return False

    def add_rule_to_symbol(self, symbol: str, interval: str, rule: RuleDefinition) -> bool:
        """为交易对添加规则"""
        symbol_config = self.get_symbol_config(symbol, interval)
        if symbol_config:
            symbol_config.rules.append(rule)
            logger.info(f"已为 {symbol}:{interval} 添加规则: {rule.name}")
            return True
        return False

    def remove_rule_from_symbol(self, symbol: str, interval: str, rule_name: str) -> bool:
        """从交易对移除规则"""
        symbol_config = self.get_symbol_config(symbol, interval)
        if symbol_config:
            for i, rule in enumerate(symbol_config.rules):
                if rule.name == rule_name:
                    del symbol_config.rules[i]
                    logger.info(f"已从 {symbol}:{interval} 移除规则: {rule_name}")
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

    def update_symbol_config(self, symbol: str, interval: str, **kwargs) -> bool:
        """更新交易对配置"""
        symbol_config = self.get_symbol_config(symbol, interval)
        if symbol_config:
            for key, value in kwargs.items():
                if hasattr(symbol_config, key):
                    setattr(symbol_config, key, value)
            logger.info(f"已更新交易对配置: {symbol}:{interval}")
            return True
        return False


def create_default_config() -> GlobalConfig:
    """创建默认配置"""
    # 默认全局规则
    global_rules = [
        RuleDefinition(
            name=RuleNames.MACD_GOLDEN_CROSS,
            rule_type=RuleType.TECHNICAL_INDICATOR.value,
            enabled=True,
            description=RULE_DESCRIPTIONS[RuleNames.MACD_GOLDEN_CROSS],
            parameters={},
        ),
        RuleDefinition(
            name=RuleNames.RSI_SIGNAL,
            rule_type=RuleType.TECHNICAL_INDICATOR.value,
            enabled=True,
            description=RULE_DESCRIPTIONS[RuleNames.RSI_SIGNAL],
            parameters={},
        ),
    ]

    # 为每个默认交易对创建配置
    symbols = []
    for symbol in DEFAULT_SYMBOLS:
        symbol_config = SymbolConfig(
            symbol=symbol,
            interval=DEFAULT_MONITOR_INTERVAL,
            use_global_rules=True,
            rules=[],  # 使用全局规则，本地规则为空
        )
        symbols.append(symbol_config)

    return GlobalConfig(global_rules=global_rules, symbols=symbols)


def load_rules_from_config(config: GlobalConfig) -> List:
    """从配置加载规则"""

    rules = []

    # 为每个交易对加载规则
    for symbol_config in config.symbols:
        # 创建规则字典，用于合并全局规则和本地规则
        rule_dict = {}

        # 如果使用全局规则，先添加全局规则
        if symbol_config.use_global_rules:
            for rule_def in config.global_rules:
                if rule_def.enabled:
                    rule_dict[rule_def.name] = rule_def

        # 添加本地规则（继承全局规则参数）
        for rule_def in symbol_config.rules:
            if rule_def.enabled:
                if rule_def.name in rule_dict:
                    rule_dict[rule_def.name] = rule_dict[rule_def.name].merge_with(rule_def)
                else:
                    rule_dict[rule_def.name] = rule_def

        # 创建规则实例
        for rule_def in rule_dict.values():
            rule = _create_rule_instance(rule_def, symbol_config.symbol, symbol_config.interval)
            if rule:
                rules.append(rule)

    return rules


def _create_rule_instance(rule_def: RuleDefinition, symbol: str, interval: str) -> Optional[Any]:
    """创建规则实例"""

    try:
        if rule_def.rule_type == RuleType.PRICE_BASED.value:
            if rule_def.name == RuleNames.PRICE_VOLATILITY:
                return RuleFactory.create_price_volatility_rule(
                    symbol=symbol,
                    interval=interval,
                    volatility_threshold=rule_def.parameters.get("volatility_threshold", 0.05),
                    amplitude_multiplier=rule_def.parameters.get("amplitude_multiplier", 2.0),
                    change_multiplier=rule_def.parameters.get("change_multiplier", 2.0),
                )
            elif rule_def.name == RuleNames.PRICE_BREAKOUT:
                return RuleFactory.create_breakout_rule(symbol=symbol, interval=interval)
            elif rule_def.name == RuleNames.NEW_HIGH_LOW:
                return RuleFactory.create_new_high_low_rule(symbol=symbol, interval=interval)
        elif rule_def.rule_type == RuleType.TECHNICAL_INDICATOR.value:
            if rule_def.name == RuleNames.MACD_GOLDEN_CROSS:
                return RuleFactory.create_macd_rule(symbol=symbol, interval=interval)
            elif rule_def.name == RuleNames.RSI_SIGNAL:
                return RuleFactory.create_rsi_rule(symbol=symbol, interval=interval)
            elif rule_def.name == RuleNames.TREND_ANALYSIS:
                return RuleFactory.create_trend_rule(symbol=symbol, interval=interval)
        elif rule_def.rule_type == RuleType.CUSTOM.value:
            # 自定义规则需要特殊处理
            evaluator = rule_def.parameters.get("evaluator")
            if evaluator:
                return RuleFactory.create_custom_rule(
                    symbol=symbol,
                    interval=interval,
                    name=rule_def.name,
                    evaluator=evaluator,
                )
    except Exception as e:
        logger.error(f"创建规则实例失败: {rule_def.name} - {e}")
        return None

    logger.warning(f"未知的规则类型或名称: {rule_def.rule_type}.{rule_def.name}")
    return None
