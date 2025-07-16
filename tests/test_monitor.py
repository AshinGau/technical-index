#!/usr/bin/env python3
"""
监控系统测试脚本
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import tempfile

import pytest

from technical_index.config import ConfigManager, create_default_config
from technical_index.monitor import (PriceMonitor, RuleEngine, SignalResult,
                                     SignalType, create_breakout_rule,
                                     create_macd_rule,
                                     create_price_volatility_rule,
                                     create_rsi_rule, create_trend_rule)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rule_creation():
    """测试规则创建"""
    print("🧪 测试规则创建...")

    # 测试价格波动规则
    volatility_rule = create_price_volatility_rule("BTCUSDT", "1h", 0.03)
    assert volatility_rule.name == "价格波动监控"
    assert volatility_rule.symbol == "BTCUSDT"
    assert volatility_rule.interval == "1h"
    assert volatility_rule.volatility_threshold == 0.03

    # 测试突破规则
    breakout_rule = create_breakout_rule("ETHUSDT", "1h")
    assert breakout_rule.name == "价格突破监控"
    assert breakout_rule.symbol == "ETHUSDT"

    # 测试MACD规则
    macd_rule = create_macd_rule("BNBUSDT", "1h")
    assert macd_rule.name == "MACD金叉死叉"
    assert macd_rule.symbol == "BNBUSDT"

    print("✅ 规则创建测试通过")


def test_rule_engine():
    """测试规则引擎"""
    print("🧪 测试规则引擎...")

    rule_engine = RuleEngine()

    # 添加规则
    rule1 = create_price_volatility_rule("BTCUSDT", "1h")
    rule2 = create_macd_rule("ETHUSDT", "1h")

    rule_engine.add_rule(rule1)
    rule_engine.add_rule(rule2)

    # 测试获取规则
    btc_rules = rule_engine.get_rules_for_symbol("BTCUSDT")
    assert len(btc_rules) == 1
    assert btc_rules[0].symbol == "BTCUSDT"

    eth_rules = rule_engine.get_rules_for_symbol("ETHUSDT")
    assert len(eth_rules) == 1
    assert eth_rules[0].symbol == "ETHUSDT"

    # 测试移除规则
    removed = rule_engine.remove_rule("BTCUSDT", "价格波动监控")
    assert removed

    btc_rules_after = rule_engine.get_rules_for_symbol("BTCUSDT")
    assert len(btc_rules_after) == 0

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
        assert config.monitor.interval == "1h"
        assert len(config.symbols) == 2  # BTCUSDT, ETHUSDT

        # 测试添加交易对
        config_manager.add_symbol_config("ADAUSDT", "15m")
        config_manager.save_config()

        # 重新加载配置
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()
        assert len(config.symbols) == 3

        # 测试移除交易对
        removed = config_manager.remove_symbol_config("ADAUSDT")
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
    rule = create_price_volatility_rule("BTCUSDT", "1h", 0.01)  # 1%阈值
    rule_engine.add_rule(rule)

    # 添加交易对和回调
    monitor.add_symbol("BTCUSDT", test_callback)

    # 模拟信号触发（这里只是测试回调机制）
    test_signal = SignalResult(
        symbol="BTCUSDT",
        rule_name="测试规则",
        signal_type=SignalType.BULLISH,
        timestamp=datetime.datetime.now(),
        current_price=50000.0,
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
    rule = create_price_volatility_rule("BTCUSDT", "1h", 0.01)
    rule_engine.add_rule(rule)

    # 添加交易对
    monitor.add_symbol("BTCUSDT")

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

        print("\n🎉 所有测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
