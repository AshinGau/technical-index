#!/usr/bin/env python3
"""
监控系统命令行接口
"""

import argparse
import asyncio
import logging
import sys

from .config import ConfigManager, load_rules_from_config
from .constants import DEFAULT_CONFIG_FILE
from .monitor import PriceMonitor, RuleEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def show_config(config_file: str = DEFAULT_CONFIG_FILE) -> None:
    """显示当前配置"""
    try:
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()

        print("\n" + "=" * 60)
        print("📋 当前监控配置")
        print("=" * 60)

        # 显示全局规则
        print(f"\n🌍 全局规则 ({len(config.global_rules)} 个):")
        for i, rule in enumerate(config.global_rules, 1):
            print(f"  {i}. {rule.name} ({rule.rule_type})")
            print(f"     描述: {rule.description}")
            print(f"     启用: {'✅' if rule.enabled else '❌'}")
            if rule.parameters:
                print(f"     参数: {rule.parameters}")
            print()

        # 显示交易对配置
        print(f"📊 交易对配置 ({len(config.symbols)} 个):")
        for i, symbol_config in enumerate(config.symbols, 1):
            print(f"  {i}. {symbol_config.symbol} ({symbol_config.interval})")
            print(f"     使用全局规则: {'✅' if symbol_config.use_global_rules else '❌'}")
            print(f"     本地规则数量: {len(symbol_config.rules)}")

            if symbol_config.rules:
                print("     本地规则:")
                for j, rule in enumerate(symbol_config.rules, 1):
                    print(f"       {j}. {rule.name} ({rule.rule_type})")
                    print(f"           描述: {rule.description}")
                    print(f"           启用: {'✅' if rule.enabled else '❌'}")
                    if rule.parameters:
                        print(f"           参数: {rule.parameters}")

            if symbol_config.callback_module or symbol_config.callback_function:
                print(
                    f"     回调: {symbol_config.callback_module}.{symbol_config.callback_function}"
                )
            print()

        print("=" * 60)

    except Exception as e:
        logger.error(f"显示配置失败: {e}")
        sys.exit(1)


def start_monitoring(config_file: str = DEFAULT_CONFIG_FILE) -> None:
    """启动监控"""
    try:
        print("🚀 启动币价监控系统...")

        # 加载配置
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()

        if not config.symbols:
            print("❌ 没有配置任何交易对，请先修改配置文件")
            return

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
            # 设置回调函数
            callback = None
            if symbol_config.callback_module and symbol_config.callback_function:
                try:
                    import importlib

                    module = importlib.import_module(symbol_config.callback_module)
                    callback = getattr(module, symbol_config.callback_function)
                except Exception as e:
                    callback_name = (
                        f"{symbol_config.callback_module}.{symbol_config.callback_function}"
                    )
                    logger.warning(f"加载回调函数失败 {callback_name}: {e}")

            monitor.add_symbol_interval(symbol_config.symbol, symbol_config.interval, callback)

        print(f"✅ 已加载 {len(rules)} 个规则")
        print(f"✅ 已配置 {len(config.symbols)} 个交易对")
        print("🔄 开始监控... (按 Ctrl+C 停止)")

        # 运行监控（使用 asyncio.run 运行异步函数）
        asyncio.run(monitor.start_monitoring())

    except KeyboardInterrupt:
        print("\n⏹️  停止监控...")
    except Exception as e:
        logger.error(f"启动监控失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="币价监控系统")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help=f"配置文件路径 (默认: {DEFAULT_CONFIG_FILE})"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 命令
    start_parser = subparsers.add_parser("start", help="启动监控")
    start_parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help=f"配置文件路径 (默认: {DEFAULT_CONFIG_FILE})"
    )

    # show 命令
    show_parser = subparsers.add_parser("show", help="显示当前配置")
    show_parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help=f"配置文件路径 (默认: {DEFAULT_CONFIG_FILE})"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "start":
            config_file = getattr(args, "config", DEFAULT_CONFIG_FILE)
            start_monitoring(config_file)
        elif args.command == "show":
            config_file = getattr(args, "config", DEFAULT_CONFIG_FILE)
            show_config(config_file)
        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"执行命令失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
