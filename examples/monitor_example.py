#!/usr/bin/env python3
"""
币价监控规则系统使用示例
演示如何设置规则、启动监控和自定义回调函数
"""

import asyncio
import logging

from technical_index.binance import get_futures_market_data
from technical_index.constants import RuleNames
from technical_index.index import build_quantitative_analysis
from technical_index.monitor import PriceMonitor, RuleEngine, RuleFactory, SignalResult, SignalType

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def custom_signal_callback(signal: SignalResult) -> None:
    """自定义信号回调函数"""
    print("\n" + "=" * 60)
    print(f"🚨 信号触发: {signal.symbol}")
    print(f"📊 规则: {signal.rule_name}")
    print(f"🎯 信号类型: {signal.signal_type.value}")
    print(f"💰 当前价格: {signal.current_price:.4f}")
    print(f"📈 置信度: {signal.confidence:.2f}")

    if signal.duration:
        print(f"⏱️  预期持续: {signal.duration} 个周期")

    if signal.target_price:
        print(f"🎯 目标价格: {signal.format_price_change(signal.target_price)}")

    if signal.stop_loss:
        print(f"🛑 止损价格: {signal.format_price_change(signal.stop_loss)}")

    if signal.take_profit:
        print(f"✅ 止盈价格: {signal.format_price_change(signal.take_profit)}")

    if signal.resistance_level:
        print(f"🔺 阻力位: {signal.format_price_change(signal.resistance_level)}")

    if signal.support_level:
        print(f"🔻 支撑位: {signal.format_price_change(signal.support_level)}")

    if signal.additional_signals:
        print(f"📋 额外信号: {', '.join(signal.additional_signals)}")

    # 根据信号类型给出建议
    if signal.signal_type == SignalType.BULLISH:
        print("💚 建议: 考虑做多，注意风险控制")
    elif signal.signal_type == SignalType.BEARISH:
        print("🔴 建议: 考虑做空，注意风险控制")
    elif signal.signal_type == SignalType.ALERT:
        print("⚠️  建议: 密切关注价格变化")
    elif signal.signal_type == SignalType.NEUTRAL:
        print("⚪ 建议: 观望为主，等待明确信号")

    print("=" * 60 + "\n")


def custom_volume_rule(df, config):
    """自定义成交量规则"""
    if len(df) < 20:
        return None

    current_volume = df["Volume"].iloc[-1]
    avg_volume = df["Volume"].rolling(window=20).mean().iloc[-1]
    current_price = df["Close"].iloc[-1]

    # 成交量放大且价格上涨
    if current_volume > avg_volume * 2 and current_price > df["Close"].iloc[-2]:
        from technical_index.monitor import SignalResult, SignalType

        return SignalResult(
            symbol=config.symbol,
            rule_name=config.name,
            signal_type=SignalType.BULLISH,
            timestamp=df.index[-1],
            current_price=current_price,
            interval=config.interval,
            confidence=0.7,
            duration=3,
            target_price=current_price * 1.03,
            stop_loss=current_price * 0.98,
            take_profit=current_price * 1.08,
            additional_signals=["成交量确认", "关注持续性"],
            metadata={"volume_ratio": current_volume / avg_volume},
        )

    # 成交量放大且价格下跌
    elif current_volume > avg_volume * 2 and current_price < df["Close"].iloc[-2]:
        from technical_index.monitor import SignalResult, SignalType

        return SignalResult(
            symbol=config.symbol,
            rule_name=config.name,
            signal_type=SignalType.BEARISH,
            timestamp=df.index[-1],
            current_price=current_price,
            interval=config.interval,
            confidence=0.7,
            duration=3,
            target_price=current_price * 0.97,
            stop_loss=current_price * 1.02,
            take_profit=current_price * 0.92,
            additional_signals=["成交量确认", "关注持续性"],
            metadata={"volume_ratio": current_volume / avg_volume},
        )

    return None


async def main():
    """主函数"""
    print("🚀 启动币价监控规则系统...")

    # 创建规则引擎
    rule_engine = RuleEngine()

    # 创建价格监控器
    monitor = PriceMonitor(rule_engine)

    # 设置监控的交易对
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    # 为每个交易对添加规则
    for symbol in symbols:
        print(f"📊 为 {symbol} 添加监控规则...")

        # 添加价格波动规则
        volatility_rule = RuleFactory.create_price_volatility_rule(
            symbol=symbol, interval="1h", volatility_threshold=0.03  # 3%波动阈值
        )
        rule_engine.add_rule(volatility_rule)

        # 添加突破规则
        breakout_rule = RuleFactory.create_breakout_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(breakout_rule)

        # 添加MACD规则
        macd_rule = RuleFactory.create_macd_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(macd_rule)

        # 添加RSI规则
        rsi_rule = RuleFactory.create_rsi_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(rsi_rule)

        # 添加趋势分析规则
        trend_rule = RuleFactory.create_trend_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(trend_rule)

        # 添加自定义成交量规则
        volume_rule = RuleFactory.create_custom_rule(
            symbol=symbol, interval="1h", name="volume_analysis", evaluator=custom_volume_rule
        )
        rule_engine.add_rule(volume_rule)

        # 添加交易对到监控器
        monitor.add_symbol_interval(symbol, "1h", custom_signal_callback)

    print(f"✅ 已添加 {len(symbols)} 个交易对的监控规则")
    print("📋 监控规则包括:")
    print("   - 价格波动监控 (3%阈值)")
    print("   - 价格突破监控")
    print("   - MACD金叉死叉")
    print("   - RSI超买超卖")
    print("   - 趋势分析")
    print("   - 自定义成交量分析")
    print("\n🔄 开始监控... (按 Ctrl+C 停止)")

    try:
        # 启动监控
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  停止监控...")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"监控过程中发生错误: {e}")
        monitor.stop_monitoring()


def demo_single_symbol():
    """演示单个交易对的监控"""
    print("🎯 演示单个交易对监控...")

    # 创建规则引擎
    rule_engine = RuleEngine()

    # 创建价格监控器
    monitor = PriceMonitor(rule_engine)

    # 为BTCUSDT添加规则
    symbol = "BTCUSDT"

    # 添加所有规则
    rules = [
        RuleFactory.create_price_volatility_rule(symbol, "1h", 0.02),  # 2%波动
        RuleFactory.create_breakout_rule(symbol, "1h"),
        RuleFactory.create_macd_rule(symbol, "1h"),
        RuleFactory.create_rsi_rule(symbol, "1h"),
        RuleFactory.create_trend_rule(symbol, "1h"),
        RuleFactory.create_custom_rule(symbol, "1h", "成交量分析", custom_volume_rule),
    ]

    for rule in rules:
        rule_engine.add_rule(rule)

    # 添加交易对到监控器
    monitor.add_symbol_interval(symbol, "1h", custom_signal_callback)

    print(f"✅ 已为 {symbol} 添加 {len(rules)} 个监控规则")
    print("🔄 开始监控... (按 Ctrl+C 停止)")

    try:
        # 启动监控
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        print("\n⏹️  停止监控...")
        monitor.stop_monitoring()


def demo_rule_management():
    """演示规则管理功能"""
    print("🔧 演示规则管理功能...")

    # 创建规则引擎
    rule_engine = RuleEngine()

    # 添加规则
    symbol = "ETHUSDT"
    volatility_rule = RuleFactory.create_price_volatility_rule(symbol, "1h")
    breakout_rule = RuleFactory.create_breakout_rule(symbol, "1h")

    rule_engine.add_rule(volatility_rule)
    rule_engine.add_rule(breakout_rule)

    # 获取规则信息
    rules = rule_engine.get_rules_for_symbol_interval(symbol, "1h")
    print(f"📋 {symbol} 的规则列表:")
    for rule in rules:
        info = rule.get_rule_info()
        print(f"   - {info['name']} ({info['type']})")

    # 移除规则
    removed = rule_engine.remove_rule(symbol, "1h", RuleNames.PRICE_VOLATILITY)
    print(f"🗑️  移除规则: {'成功' if removed else '失败'}")

    # 再次获取规则信息
    rules = rule_engine.get_rules_for_symbol_interval(symbol, "1h")
    print("📋 移除后的规则列表:")
    for rule in rules:
        info = rule.get_rule_info()
        print(f"   - {info['name']} ({info['type']})")


def run_once_monitor():
    """单轮信号检测示例"""
    print("🚀 单轮信号检测示例...")

    # 创建规则引擎
    rule_engine = RuleEngine()

    # 获取数据并计算指标
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    df_dict = {}

    for symbol in symbols:
        df = get_futures_market_data(symbol, "1h", limit=100)
        if df is not None and not df.empty:
            # 计算所有需要的技术指标
            df = build_quantitative_analysis(df, indicators=None)
            df_dict[symbol] = df

    # 添加规则
    for symbol in symbols:
        rule_engine.add_rule(
            RuleFactory.create_price_volatility_rule(symbol, "1h", volatility_threshold=0.03)
        )
        rule_engine.add_rule(RuleFactory.create_breakout_rule(symbol, "1h"))
        rule_engine.add_rule(RuleFactory.create_macd_rule(symbol, "1h"))
        rule_engine.add_rule(RuleFactory.create_rsi_rule(symbol, "1h"))
        rule_engine.add_rule(RuleFactory.create_trend_rule(symbol, "1h"))

    # 检测信号并打印
    for symbol in symbols:
        df = df_dict.get(symbol)
        if df is not None:
            results = rule_engine.evaluate_rules(symbol, "1h", df)
            for signal in results:
                custom_signal_callback(signal)

    print("✅ 单轮信号检测完成")


async def example_using_config():
    """使用配置文件的示例"""
    print("🚀 使用配置文件启动监控...")

    from technical_index.config import ConfigManager, load_rules_from_config

    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config()

    # 创建规则引擎
    rule_engine = RuleEngine()

    # 从配置加载规则
    rules = load_rules_from_config(config)
    for rule in rules:
        rule_engine.add_rule(rule)

    # 创建监控器
    monitor = PriceMonitor(rule_engine)

    # 为每个交易对添加监控
    for symbol_config in config.symbols:
        monitor.add_symbol_interval(
            symbol_config.symbol, symbol_config.interval, custom_signal_callback
        )

    print(f"✅ 已加载 {len(rules)} 个规则")
    print(f"✅ 已配置 {len(config.symbols)} 个交易对")
    print("🔄 开始监控... (按 Ctrl+C 停止)")

    try:
        # 启动监控
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  停止监控...")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"监控过程中发生错误: {e}")
        monitor.stop_monitoring()


def demo_config_usage():
    """演示配置文件的使用"""
    print("🔧 演示配置文件使用...")

    from technical_index.config import ConfigManager, load_rules_from_config

    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config()

    print(f"📋 配置文件包含 {len(config.global_rules)} 个全局规则")
    print(f"📋 配置文件包含 {len(config.symbols)} 个交易对配置")

    # 显示配置信息
    for symbol_config in config.symbols:
        print(f"\n🔍 {symbol_config.symbol} ({symbol_config.interval}):")
        print(f"   使用全局规则: {'✅' if symbol_config.use_global_rules else '❌'}")
        print(f"   本地规则数量: {len(symbol_config.rules)}")

    # 加载规则
    rules = load_rules_from_config(config)
    print(f"\n✅ 从配置加载了 {len(rules)} 个规则实例")

    # 按 symbol_interval 分组显示
    rule_groups = {}
    for rule in rules:
        key = f"{rule.symbol}_{rule.interval}"
        if key not in rule_groups:
            rule_groups[key] = []
        rule_groups[key].append(rule)

    for key, group_rules in rule_groups.items():
        print(f"\n📊 {key}:")
        for rule in group_rules:
            print(f"   - {rule.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="币价监控规则系统示例")
    parser.add_argument(
        "--mode",
        choices=["multi", "single", "demo", "once", "config", "config_demo"],
        default="once",
        help="运行模式",
    )

    args = parser.parse_args()

    if args.mode == "multi":
        asyncio.run(main())
    elif args.mode == "single":
        demo_single_symbol()
    elif args.mode == "demo":
        demo_rule_management()
    elif args.mode == "once":
        run_once_monitor()
    elif args.mode == "config":
        asyncio.run(example_using_config())
    elif args.mode == "config_demo":
        demo_config_usage()
