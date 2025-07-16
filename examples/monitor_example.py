#!/usr/bin/env python3
"""
币价监控规则系统使用示例
演示如何设置规则、启动监控和自定义回调函数
"""

import asyncio
import logging

from technical_index.monitor import (PriceMonitor, RuleEngine, SignalResult,
                                     SignalType, create_breakout_rule,
                                     create_custom_rule, create_macd_rule,
                                     create_price_volatility_rule,
                                     create_rsi_rule, create_trend_rule)

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
        print(f"🎯 目标价格: {signal.target_price:.4f}")

    if signal.stop_loss:
        print(f"🛑 止损价格: {signal.stop_loss:.4f}")

    if signal.take_profit:
        print(f"✅ 止盈价格: {signal.take_profit:.4f}")

    if signal.resistance_level:
        print(f"🔺 阻力位: {signal.resistance_level:.4f}")

    if signal.support_level:
        print(f"🔻 支撑位: {signal.support_level:.4f}")

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
        volatility_rule = create_price_volatility_rule(
            symbol=symbol, interval="1h", volatility_threshold=0.03  # 3%波动阈值
        )
        rule_engine.add_rule(volatility_rule)

        # 添加突破规则
        breakout_rule = create_breakout_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(breakout_rule)

        # 添加MACD规则
        macd_rule = create_macd_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(macd_rule)

        # 添加RSI规则
        rsi_rule = create_rsi_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(rsi_rule)

        # 添加趋势分析规则
        trend_rule = create_trend_rule(symbol=symbol, interval="1h")
        rule_engine.add_rule(trend_rule)

        # 添加自定义成交量规则
        volume_rule = create_custom_rule(
            symbol=symbol, interval="1h", name="成交量分析", evaluator=custom_volume_rule
        )
        rule_engine.add_rule(volume_rule)

        # 添加交易对到监控器
        monitor.add_symbol(symbol, custom_signal_callback)

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
        create_price_volatility_rule(symbol, "1h", 0.02),  # 2%波动
        create_breakout_rule(symbol, "1h"),
        create_macd_rule(symbol, "1h"),
        create_rsi_rule(symbol, "1h"),
        create_trend_rule(symbol, "1h"),
        create_custom_rule(symbol, "1h", "成交量分析", custom_volume_rule),
    ]

    for rule in rules:
        rule_engine.add_rule(rule)

    # 添加交易对到监控器
    monitor.add_symbol(symbol, custom_signal_callback)

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
    volatility_rule = create_price_volatility_rule(symbol, "1h")
    breakout_rule = create_breakout_rule(symbol, "1h")

    rule_engine.add_rule(volatility_rule)
    rule_engine.add_rule(breakout_rule)

    # 获取规则信息
    rules = rule_engine.get_rules_for_symbol(symbol)
    print(f"📋 {symbol} 的规则列表:")
    for rule in rules:
        info = rule.get_rule_info()
        print(f"   - {info['name']} ({info['type']})")

    # 移除规则
    removed = rule_engine.remove_rule(symbol, "价格波动监控")
    print(f"🗑️  移除规则: {'成功' if removed else '失败'}")

    # 再次获取规则信息
    rules = rule_engine.get_rules_for_symbol(symbol)
    print("📋 移除后的规则列表:")
    for rule in rules:
        info = rule.get_rule_info()
        print(f"   - {info['name']} ({info['type']})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="币价监控规则系统示例")
    parser.add_argument(
        "--mode",
        choices=["multi", "single", "demo"],
        default="multi",
        help="运行模式: multi(多交易对), single(单交易对), demo(规则管理演示)",
    )

    args = parser.parse_args()

    if args.mode == "multi":
        asyncio.run(main())
    elif args.mode == "single":
        demo_single_symbol()
    elif args.mode == "demo":
        demo_rule_management()
