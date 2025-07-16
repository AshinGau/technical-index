#!/usr/bin/env python3
"""
币价监控命令行工具
支持从配置文件启动监控，管理规则和交易对
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Optional

from .config import (ConfigManager, create_default_config,
                     load_rules_from_config)
from .monitor import PriceMonitor, RuleEngine, SignalResult

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_signal_to_file(signal: SignalResult, filename: str = "log/signals.json") -> None:
    """保存信号到文件"""
    try:
        # 读取现有信号
        signals = []
        try:
            with open(filename, "r", encoding="utf-8") as f:
                signals = json.load(f)
        except FileNotFoundError:
            pass

        # 添加新信号
        signal_data = {
            "timestamp": signal.timestamp.isoformat(),
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
        signals.append(signal_data)

        # 保存到文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"保存信号到文件失败: {e}")


def create_signal_callback(save_to_file: bool = True, filename: str = "log/signals.json"):
    """创建信号回调函数"""

    def callback(signal: SignalResult) -> None:
        # 打印信号信息
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

        print("=" * 60 + "\n")

        # 保存到文件
        if save_to_file:
            save_signal_to_file(signal, filename)

    return callback


async def start_monitoring(config_file: str = "config/monitor_config.json") -> None:
    """启动监控"""
    print("🚀 启动币价监控系统...")

    # 加载配置
    config_manager = ConfigManager(config_file)
    config = config_manager.load_config()

    if not config.monitor.enabled:
        print("❌ 监控已禁用，请检查配置文件")
        return

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, config.monitor.log_level))

    # 创建规则引擎
    rule_engine = RuleEngine()

    # 从配置加载规则
    rules = load_rules_from_config(config)
    for rule, is_global in rules:
        rule_engine.add_rule(rule, is_global)

    print(f"✅ 已加载 {len(rules)} 个规则")

    # 创建价格监控器
    monitor = PriceMonitor(rule_engine)

    # 创建回调函数
    callback = create_signal_callback(
        save_to_file=config.monitor.save_signals, filename=config.monitor.signal_file
    )

    # 添加交易对
    for symbol_config in config.symbols:
        monitor.add_symbol(symbol_config.symbol, callback)
        print(f"📊 添加监控交易对: {symbol_config.symbol}")

    print(f"✅ 已添加 {len(config.symbols)} 个交易对的监控")
    print("🔄 开始监控... (按 Ctrl+C 停止)")

    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n⏹️  停止监控...")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"监控过程中发生错误: {e}")
        monitor.stop_monitoring()


def init_config(config_file: str = "config/monitor_config.json") -> None:
    """初始化配置文件"""
    print("🔧 初始化配置文件...")

    config_manager = ConfigManager(config_file)
    config = create_default_config()
    config_manager.config = config
    config_manager.save_config()

    print(f"✅ 配置文件已创建: {config_file}")
    print("📋 默认配置包括:")
    print(f"   - 监控交易对: {', '.join(config.monitor.symbols)}")
    print(f"   - 时间间隔: {config.monitor.interval}")
    print(f"   - 检查间隔: {config.monitor.check_interval_minutes} 分钟")
    print("   - 默认规则: 价格波动、突破、MACD、RSI、趋势分析")


def show_config(config_file: str = "config/monitor_config.json") -> None:
    """显示配置信息"""
    config_manager = ConfigManager(config_file)
    config = config_manager.load_config()

    print("📋 当前配置信息:")
    print(f"配置文件: {config_file}")
    print(f"监控状态: {'启用' if config.monitor.enabled else '禁用'}")
    print(f"时间间隔: {config.monitor.interval}")
    print(f"检查间隔: {config.monitor.check_interval_minutes} 分钟")
    print(f"日志级别: {config.monitor.log_level}")
    print(f"保存信号: {'是' if config.monitor.save_signals else '否'}")
    print(f"信号文件: {config.monitor.signal_file}")

    print(f"\n📊 监控交易对 ({len(config.symbols)} 个):")
    for symbol_config in config.symbols:
        print(f"   - {symbol_config.symbol} ({symbol_config.interval})")
        for rule in symbol_config.rules:
            status = "✅" if rule.enabled else "❌"
            print(f"     {status} {rule.name}")

    if config.global_rules:
        print(f"\n🌐 全局规则 ({len(config.global_rules)} 个):")
        for rule in config.global_rules:
            status = "✅" if rule.enabled else "❌"
            print(f"   {status} {rule.name}")


def add_symbol(
    symbol: str, interval: str = "1h", config_file: str = "config/monitor_config.json"
) -> None:
    """添加交易对"""
    print(f"➕ 添加交易对: {symbol}")

    config_manager = ConfigManager(config_file)
    config_manager.load_config()
    config_manager.add_symbol_config(symbol, interval)
    config_manager.save_config()

    print(f"✅ 已添加交易对: {symbol}")


def remove_symbol(symbol: str, config_file: str = "config/monitor_config.json") -> None:
    """移除交易对"""
    print(f"➖ 移除交易对: {symbol}")

    config_manager = ConfigManager(config_file)
    config_manager.load_config()

    if config_manager.remove_symbol_config(symbol):
        config_manager.save_config()
        print(f"✅ 已移除交易对: {symbol}")
    else:
        print(f"❌ 交易对 {symbol} 不存在")


def list_signals(filename: str = "log/signals.json") -> None:
    """列出历史信号"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            signals = json.load(f)

        print(f"📋 历史信号 ({len(signals)} 个):")
        print("-" * 80)

        for i, signal in enumerate(signals[-10:], 1):  # 显示最近10个信号
            timestamp = datetime.fromisoformat(signal["timestamp"])
            print(f"{i}. {timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {signal['symbol']}")
            print(f"   规则: {signal['rule_name']} | 类型: {signal['signal_type']}")
            print(f"   价格: {signal['current_price']:.4f} | 置信度: {signal['confidence']:.2f}")
            if signal.get("target_price"):
                print(
                    f"   目标: {signal['target_price']:.4f} | 止损: {signal.get('stop_loss', 'N/A')}"
                )
            print()

        if len(signals) > 10:
            print(f"... 还有 {len(signals) - 10} 个更早的信号")

    except FileNotFoundError:
        print(f"❌ 信号文件 {filename} 不存在")
    except Exception as e:
        print(f"❌ 读取信号文件失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="币价监控命令行工具")
    parser.add_argument(
        "--config",
        default="config/monitor_config.json",
        help="配置文件路径 (默认: config/monitor_config.json)",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 启动监控命令
    subparsers.add_parser("start", help="启动监控")

    # 初始化配置命令
    subparsers.add_parser("init", help="初始化配置文件")

    # 显示配置命令
    subparsers.add_parser("show", help="显示配置信息")

    # 添加交易对命令
    add_parser = subparsers.add_parser("add", help="添加交易对")
    add_parser.add_argument("symbol", help="交易对名称 (如: BTCUSDT)")
    add_parser.add_argument("--interval", default="1h", help="时间间隔 (默认: 1h)")

    # 移除交易对命令
    remove_parser = subparsers.add_parser("remove", help="移除交易对")
    remove_parser.add_argument("symbol", help="交易对名称")

    # 列出信号命令
    list_parser = subparsers.add_parser("signals", help="列出历史信号")
    list_parser.add_argument("--file", default="log/signals.json", help="信号文件路径")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "start":
            asyncio.run(start_monitoring(args.config))
        elif args.command == "init":
            init_config(args.config)
        elif args.command == "show":
            show_config(args.config)
        elif args.command == "add":
            add_symbol(args.symbol, args.interval, args.config)
        elif args.command == "remove":
            remove_symbol(args.symbol, args.config)
        elif args.command == "signals":
            list_signals(args.file)
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
    except Exception as e:
        logger.error(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
