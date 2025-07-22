#!/usr/bin/env python3
"""
监控系统测试脚本
"""

import asyncio
import datetime
import json
import logging
import os
import tempfile

import pytest

from technical_index.config import ConfigManager
from technical_index.constants import RuleNames
from technical_index.rules import SignalResult, SignalType
from technical_index.monitor import PriceMonitor, RuleEngine, RuleFactory

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rule_creation():
    """测试规则创建"""
    print("🧪 测试规则创建...")

    # 测试价格波动规则
    volatility_rule = RuleFactory.create_price_volatility_rule(
        "BTCUSDT", "1h", volatility_threshold=0.03
    )
    assert volatility_rule.name == RuleNames.PRICE_VOLATILITY
    assert volatility_rule.symbol == "BTCUSDT"
    assert volatility_rule.interval == "1h"
    assert volatility_rule.volatility_threshold == 0.03

    # 测试突破规则
    breakout_rule = RuleFactory.create_breakout_rule("ETHUSDT", "1h")
    assert breakout_rule.name == RuleNames.PRICE_BREAKOUT
    assert breakout_rule.symbol == "ETHUSDT"

    # 测试MACD规则
    macd_rule = RuleFactory.create_macd_rule("BNBUSDT", "1h")
    assert macd_rule.name == RuleNames.MACD_GOLDEN_CROSS
    assert macd_rule.symbol == "BNBUSDT"

    print("✅ 规则创建测试通过")


def test_rule_engine():
    """测试规则引擎"""
    print("🧪 测试规则引擎...")

    rule_engine = RuleEngine()

    # 添加规则
    rule1 = RuleFactory.create_price_volatility_rule("BTCUSDT", "1h")
    rule2 = RuleFactory.create_macd_rule("ETHUSDT", "1h")
    rule3 = RuleFactory.create_rsi_rule("BTCUSDT", "1h")  # 同一个symbol_interval的另一个规则

    rule_engine.add_rule(rule1)
    rule_engine.add_rule(rule2)
    rule_engine.add_rule(rule3)

    # 测试获取规则
    btc_rules = rule_engine.get_rules_for_symbol_interval("BTCUSDT", "1h")
    assert len(btc_rules) == 2  # 两个规则：价格波动和RSI
    assert btc_rules[0].symbol == "BTCUSDT"
    assert btc_rules[1].symbol == "BTCUSDT"

    eth_rules = rule_engine.get_rules_for_symbol_interval("ETHUSDT", "1h")
    assert len(eth_rules) == 1  # 一个规则：MACD
    assert eth_rules[0].symbol == "ETHUSDT"

    # 测试获取所有symbol_interval组合
    all_combinations = rule_engine.get_all_symbol_intervals()
    assert len(all_combinations) == 2  # BTCUSDT_1h 和 ETHUSDT_1h
    assert ("BTCUSDT", "1h") in all_combinations
    assert ("ETHUSDT", "1h") in all_combinations

    # 测试移除规则
    removed = rule_engine.remove_rule("BTCUSDT", "1h", RuleNames.PRICE_VOLATILITY)
    assert removed

    btc_rules_after = rule_engine.get_rules_for_symbol_interval("BTCUSDT", "1h")
    assert len(btc_rules_after) == 1  # 还剩一个RSI规则

    # 测试移除所有规则
    removed_all = rule_engine.remove_all_rules_for_symbol_interval("BTCUSDT", "1h")
    assert removed_all

    btc_rules_final = rule_engine.get_rules_for_symbol_interval("BTCUSDT", "1h")
    assert len(btc_rules_final) == 0

    print("✅ 规则引擎测试通过")


def test_config_manager():
    """测试配置管理器"""
    print("🧪 测试配置管理器...")

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        config_file = f.name

    try:
        config_manager = ConfigManager(config_file)

        # 测试默认配置创建
        config = config_manager.load_config()
        assert len(config.symbols) == 2  # BTCUSDT, ETHUSDT

        # 测试添加交易对
        config_manager.add_symbol_config("ADAUSDT", "1h")
        config_manager.save_config()

        # 重新加载配置
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()
        assert len(config.symbols) == 3

        # 测试移除交易对
        removed = config_manager.remove_symbol_config("ADAUSDT", "1h")
        assert removed
        config_manager.save_config()

        # 验证移除结果
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()
        assert len(config.symbols) == 2

        print("✅ 配置管理器测试通过")

    finally:
        # 清理临时文件
        if os.path.exists(config_file):
            os.unlink(config_file)


def test_signal_callback():
    """测试信号回调"""
    print("🧪 测试信号回调...")

    signals_received = []

    def test_callback(signal: SignalResult):
        signals_received.append(signal)
        print(f"收到信号: {signal.symbol} - {signal.rule_name}")

    # 创建规则引擎和监控器
    rule_engine = RuleEngine()
    monitor = PriceMonitor(rule_engine)

    # 添加规则
    rule = RuleFactory.create_price_volatility_rule(
        "BTCUSDT", "1h", volatility_threshold=0.01
    )  # 1%阈值
    rule_engine.add_rule(rule)

    # 添加交易对和回调
    monitor.add_symbol_interval("BTCUSDT", "1h", test_callback)

    # 模拟信号触发（这里只是测试回调机制）
    test_signal = SignalResult(
        symbol="BTCUSDT",
        rule_name="测试规则",
        signal_type=SignalType.BULLISH,
        timestamp=datetime.datetime.now(),
        current_price=50000.0,
        interval="1h",
        confidence=0.8,
    )

    test_callback(test_signal)

    assert len(signals_received) == 1
    assert signals_received[0].symbol == "BTCUSDT"

    print("✅ 信号回调测试通过")


@pytest.mark.asyncio
async def test_monitor_integration():
    """测试监控器集成"""
    print("🧪 测试监控器集成...")

    rule_engine = RuleEngine()
    monitor = PriceMonitor(rule_engine)

    # 添加规则
    rule = RuleFactory.create_price_volatility_rule("BTCUSDT", "1h", volatility_threshold=0.01)
    rule_engine.add_rule(rule)

    # 添加交易对
    monitor.add_symbol_interval("BTCUSDT", "1h")

    # 启动监控（只运行很短时间）
    monitor_task = asyncio.create_task(monitor.start_monitoring())

    # 等待一小段时间
    await asyncio.sleep(2)

    # 停止监控
    monitor.stop_monitoring()

    try:
        await asyncio.wait_for(monitor_task, timeout=5)
    except asyncio.TimeoutError:
        monitor_task.cancel()

    print("✅ 监控器集成测试通过")


def test_signal_serialization():
    """测试信号序列化"""
    print("🧪 测试信号序列化...")

    signal = SignalResult(
        symbol="BTCUSDT",
        rule_name="测试规则",
        signal_type=SignalType.BULLISH,
        timestamp=datetime.datetime.now(),
        current_price=50000.0,
        interval="1h",
        confidence=0.8,
        duration=5,
        target_price=52500.0,
        stop_loss=49000.0,
        take_profit=55000.0,
        resistance_level=51000.0,
        support_level=48000.0,
        additional_signals=["关注成交量", "观察突破"],
        metadata={"test": True},
    )

    # 测试转换为字典
    signal_dict = {
        "symbol": signal.symbol,
        "rule_name": signal.rule_name,
        "signal_type": signal.signal_type.value,
        "current_price": signal.current_price,
        "confidence": signal.confidence,
        "duration": signal.duration,
        "target_price": signal.target_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "resistance_level": signal.resistance_level,
        "support_level": signal.support_level,
        "additional_signals": signal.additional_signals,
        "metadata": signal.metadata,
    }

    # 测试JSON序列化
    json_str = json.dumps(signal_dict, ensure_ascii=False)
    assert "BTCUSDT" in json_str
    assert "bullish" in json_str

    # 测试反序列化
    loaded_dict = json.loads(json_str)
    assert loaded_dict["symbol"] == "BTCUSDT"
    assert loaded_dict["signal_type"] == "bullish"

    print("✅ 信号序列化测试通过")


def test_config_loading():
    """测试配置加载和全局规则合并"""
    print("🧪 测试配置加载和全局规则合并...")

    from technical_index.config import (
        GlobalConfig,
        RuleDefinition,
        SymbolConfig,
        load_rules_from_config,
    )
    from technical_index.constants import RuleNames

    # 创建测试配置
    global_rules = [
        RuleDefinition(
            name=RuleNames.PRICE_VOLATILITY,
            rule_type="price_based",
            enabled=True,
            parameters={"volatility_threshold": 0.04},
            description="价格波动监控",
        )
    ]

    symbols = [
        SymbolConfig(
            symbol="ETHUSDT",
            interval="1d",
            use_global_rules=True,
            rules=[
                RuleDefinition(
                    name=RuleNames.MACD_GOLDEN_CROSS,
                    rule_type="technical",
                    enabled=True,
                    parameters={},
                    description="MACD金叉死叉",
                )
            ],
        ),
        SymbolConfig(
            symbol="ETHUSDT",
            interval="1h",
            use_global_rules=False,
            rules=[
                RuleDefinition(
                    name=RuleNames.PRICE_VOLATILITY,
                    rule_type="price_based",
                    enabled=True,
                    parameters={"volatility_threshold": 0.03},  # 覆盖全局规则的参数
                    description="价格波动监控",
                ),
                RuleDefinition(
                    name=RuleNames.RSI_SIGNAL,
                    rule_type="technical",
                    enabled=True,
                    parameters={},
                    description="RSI超买超卖",
                ),
            ],
        ),
    ]

    config = GlobalConfig(global_rules=global_rules, symbols=symbols)

    # 加载规则
    rules = load_rules_from_config(config)

    # 验证规则加载
    assert len(rules) == 4  # ETHUSDT_1d: 2个规则(global+local), ETHUSDT_1h: 2个规则(local only)

    # 创建规则引擎并添加规则
    rule_engine = RuleEngine()
    for rule in rules:
        rule_engine.add_rule(rule)

    # 测试 ETHUSDT_1d (use_global_rules=True)
    ethusdt_1d_rules = rule_engine.get_rules_for_symbol_interval("ETHUSDT", "1d")
    assert len(ethusdt_1d_rules) == 2  # 全局规则 + 本地规则
    rule_names = [rule.name for rule in ethusdt_1d_rules]
    assert RuleNames.PRICE_VOLATILITY in rule_names
    assert RuleNames.MACD_GOLDEN_CROSS in rule_names

    # 测试 ETHUSDT_1h (use_global_rules=False)
    ethusdt_1h_rules = rule_engine.get_rules_for_symbol_interval("ETHUSDT", "1h")
    assert len(ethusdt_1h_rules) == 2  # 只有本地规则
    rule_names = [rule.name for rule in ethusdt_1h_rules]
    assert RuleNames.PRICE_VOLATILITY in rule_names
    assert RuleNames.RSI_SIGNAL in rule_names

    # 验证参数覆盖
    volatility_rule_1h = next(
        rule for rule in ethusdt_1h_rules if rule.name == RuleNames.PRICE_VOLATILITY
    )
    assert volatility_rule_1h.volatility_threshold == 0.03  # 使用本地参数

    print("✅ 配置加载和全局规则合并测试通过")


async def main():
    """主测试函数"""
    print("🚀 开始监控系统测试...")

    try:
        # 运行所有测试
        test_rule_creation()
        test_rule_engine()
        test_config_manager()
        test_signal_callback()
        await test_monitor_integration()
        test_signal_serialization()
        test_config_loading()

        print("\n🎉 所有测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
