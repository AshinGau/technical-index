#!/usr/bin/env python3
"""
币价监控规则系统
提供基于价格变化和技术指标的动态规则监控功能
"""

import asyncio
import logging
import threading
from typing import Callable, Dict, List, Optional

import pandas as pd

from .binance import get_futures_market_data
from .constants import RuleNames
from .rules import (
    BaseRule,
    RuleConfig,
    RuleType,
    SignalResult,
    PriceVolatilityRule,
    PriceBreakoutRule,
    NewHighLowRule,
    MACDGoldenCrossRule,
    RSISignalRule,
    TrendAnalysisRule,
    CustomRule,
)

# 配置日志
logger = logging.getLogger(__name__)


class RuleEngine:
    """规则引擎"""

    def __init__(self):
        self.rules: Dict[str, List[BaseRule]] = {}  # key: "symbol_interval"

    def add_rule(self, rule: BaseRule) -> None:
        """添加规则到指定的symbol_interval"""
        key = f"{rule.symbol}_{rule.interval}"
        if key not in self.rules:
            self.rules[key] = []
        self.rules[key].append(rule)

    def remove_rule(self, symbol: str, interval: str, rule_name: str) -> bool:
        """移除规则"""
        key = f"{symbol}_{interval}"
        if key in self.rules:
            for i, rule in enumerate(self.rules[key]):
                if rule.name == rule_name:
                    del self.rules[key][i]
                    return True
        return False

    def get_rules_for_symbol_interval(self, symbol: str, interval: str) -> List[BaseRule]:
        """获取指定交易对和间隔的规则"""
        key = f"{symbol}_{interval}"
        return self.rules.get(key, [])

    def get_all_symbol_intervals(self) -> List[tuple]:
        """获取所有交易对和间隔组合"""
        result = []
        for key in self.rules.keys():
            if self.rules[key]:  # 只返回有规则的组合
                symbol, interval = key.split("_", 1)
                result.append((symbol, interval))
        return result

    def evaluate_rules(self, symbol: str, interval: str, df: pd.DataFrame) -> List[SignalResult]:
        """评估指定交易对和间隔的规则"""
        results = []
        rules = self.get_rules_for_symbol_interval(symbol, interval)
        for rule in rules:
            if rule.enabled:
                try:
                    result = rule.evaluate(df)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"规则[{rule.name}] 评估失败: {e}")
        return results

    def add_rule_to_symbol_interval(self, symbol: str, interval: str, rule: BaseRule) -> None:
        """为指定的symbol_interval添加规则"""
        key = f"{symbol}_{interval}"
        if key not in self.rules:
            self.rules[key] = []
        self.rules[key].append(rule)

    def remove_all_rules_for_symbol_interval(self, symbol: str, interval: str) -> bool:
        """移除指定symbol_interval的所有规则"""
        key = f"{symbol}_{interval}"
        if key in self.rules:
            del self.rules[key]
            return True
        return False

    def get_rule_count_for_symbol_interval(self, symbol: str, interval: str) -> int:
        """获取指定symbol_interval的规则数量"""
        key = f"{symbol}_{interval}"
        return len(self.rules.get(key, []))


class PriceMonitor:
    """价格监控器"""

    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.callbacks: Dict[str, Callable[[SignalResult], None]] = {}
        self.running = False
        self._lock = threading.Lock()

    def add_symbol_interval(
        self, symbol: str, interval: str, callback: Optional[Callable[[SignalResult], None]] = None
    ) -> None:
        """添加交易对和间隔的监控，并启动监控任务"""
        with self._lock:
            key = f"{symbol}_{interval}"

            # 设置回调函数
            if callback:
                self.callbacks[key] = callback
            else:
                self.callbacks[key] = self.default_callback

            # 如果监控器正在运行，立即启动监控任务
            if self.running:
                self._start_monitoring_task(symbol, interval)

    def remove_symbol_interval(self, symbol: str, interval: str) -> None:
        """移除交易对和间隔的监控，并停止监控任务"""
        with self._lock:
            key = f"{symbol}_{interval}"

            # 停止监控任务
            if key in self.monitoring_tasks:
                task = self.monitoring_tasks[key]
                task.cancel()
                del self.monitoring_tasks[key]
                logger.info(f"已停止监控任务: {symbol} ({interval})")

            # 移除回调函数
            if key in self.callbacks:
                del self.callbacks[key]

    def _start_monitoring_task(self, symbol: str, interval: str) -> None:
        """启动单个监控任务"""
        key = f"{symbol}_{interval}"

        # 如果任务已存在，先取消
        if key in self.monitoring_tasks:
            self.monitoring_tasks[key].cancel()

        # 创建新的监控任务
        task = asyncio.create_task(self.monitor_symbol_interval(symbol, interval))
        self.monitoring_tasks[key] = task
        logger.info(f"已启动监控任务: {symbol} ({interval})")

    def set_callback(
        self, symbol: str, interval: str, callback: Callable[[SignalResult], None]
    ) -> None:
        """设置回调函数"""
        with self._lock:
            key = f"{symbol}_{interval}"
            self.callbacks[key] = callback

    def default_callback(self, signal: SignalResult) -> None:
        signal_type_str = signal.signal_type.value.upper()
        timestamp_str = signal.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 50)
        print(f"🚨 {signal_type_str} 信号 - {signal.symbol}")
        print(f"📅时间: {timestamp_str}")
        print(f"📊 规则: {signal.rule_name} ({signal.interval})")
        print(f"💰 当前价格: {signal.current_price:.4f}")

        def print_item(item):
            return signal.format_price_change(item) if item else "N/A"

        print(f"🎯 目标价格: {print_item(signal.target_price)}")
        print(f"🛑 止损价格: {print_item(signal.stop_loss)}")
        print(f"✅ 止盈价格: {print_item(signal.take_profit)}")
        print(f"🏔️ 阻力位: {print_item(signal.resistance_level)}")
        print(f"🛟 支撑位: {print_item(signal.support_level)}")
        print(f"📊 置信度: {signal.confidence:.2f}")
        print(f"⏱️ 预期持续: {signal.duration} 周期")
        if signal.additional_signals:
            print(f"🔍 额外关注: {', '.join(signal.additional_signals)}")
        print("=" * 50)

    def get_check_interval(self, interval: str) -> int:
        """根据时间间隔确定检查频率（秒）"""
        interval_map = {
            "1d": 3600,  # 1天间隔，1小时检查一次
            "4h": 1800,  # 4小时间隔，30分钟检查一次
            "1h": 900,  # 1小时间隔，15分钟检查一次
            "15m": 300,  # 15分钟间隔，5分钟检查一次
        }
        return interval_map.get(interval, 300)  # 默认5分钟

    async def monitor_symbol_interval(self, symbol: str, interval: str) -> None:
        """监控指定交易对和间隔"""
        check_interval = self.get_check_interval(interval)
        while self.running:
            try:
                df = get_futures_market_data(symbol, interval, limit=500)
                if df is not None and not df.empty:
                    # 评估该 symbol+interval 的所有规则
                    results = self.rule_engine.evaluate_rules(symbol, interval, df)
                    key = f"{symbol}_{interval}"
                    if key in self.callbacks:
                        for result in results:
                            self.callbacks[key](result)
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                logger.info(f"监控任务被取消: {symbol} ({interval})")
                break
            except Exception as e:
                logger.error(f"监控 {symbol} ({interval}) 时发生错误: {e}")
                await asyncio.sleep(check_interval)

    async def start_monitoring(self) -> None:
        """启动监控器"""
        with self._lock:
            self.running = True
            # 为每个已配置的 symbol+interval 组合启动监控任务
            symbol_intervals = self.rule_engine.get_all_symbol_intervals()
            for symbol, interval in symbol_intervals:
                self._start_monitoring_task(symbol, interval)

        logger.info("价格监控已启动")
        try:
            while True:
                with self._lock:
                    if not self.running:
                        break
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("收到停止信号，正在停止监控...")
            self.stop_monitoring()
        except Exception as e:
            logger.error(f"监控过程中发生错误: {e}")
            self.stop_monitoring()
            raise

    def stop_monitoring(self) -> None:
        """停止监控器"""
        with self._lock:
            self.running = False
            for task in self.monitoring_tasks.values():
                task.cancel()
            self.monitoring_tasks.clear()
        logger.info("价格监控已停止")

    def is_running(self) -> bool:
        """检查监控器是否正在运行"""
        with self._lock:
            return self.running


class RuleFactory:
    @staticmethod
    def create_price_volatility_rule(
        symbol: str,
        interval: str,
        volatility_threshold: float = 0.05,
        amplitude_multiplier: float = 2.0,
        change_multiplier: float = 2.0,
    ) -> PriceVolatilityRule:
        config = RuleConfig(
            name=RuleNames.PRICE_VOLATILITY,
            rule_type=RuleType.PRICE_BASED,
            symbol=symbol,
            interval=interval,
            parameters={
                "volatility_threshold": volatility_threshold,
                "amplitude_multiplier": amplitude_multiplier,
                "change_multiplier": change_multiplier,
            },
        )
        return PriceVolatilityRule(config)

    @staticmethod
    def create_breakout_rule(symbol: str, interval: str) -> PriceBreakoutRule:
        config = RuleConfig(
            name=RuleNames.PRICE_BREAKOUT,
            rule_type=RuleType.PRICE_BASED,
            symbol=symbol,
            interval=interval,
        )
        return PriceBreakoutRule(config)

    @staticmethod
    def create_rsi_rule(symbol: str, interval: str) -> RSISignalRule:
        config = RuleConfig(
            name=RuleNames.RSI_SIGNAL,
            rule_type=RuleType.TECHNICAL_INDICATOR,
            symbol=symbol,
            interval=interval,
        )
        return RSISignalRule(config)

    @staticmethod
    def create_trend_rule(symbol: str, interval: str) -> TrendAnalysisRule:
        config = RuleConfig(
            name=RuleNames.TREND_ANALYSIS,
            rule_type=RuleType.TECHNICAL_INDICATOR,
            symbol=symbol,
            interval=interval,
        )
        return TrendAnalysisRule(config)

    @staticmethod
    def create_new_high_low_rule(symbol: str, interval: str) -> NewHighLowRule:
        config = RuleConfig(
            name=RuleNames.NEW_HIGH_LOW,
            rule_type=RuleType.PRICE_BASED,
            symbol=symbol,
            interval=interval,
        )
        return NewHighLowRule(config)

    @staticmethod
    def create_macd_rule(symbol: str, interval: str) -> MACDGoldenCrossRule:
        config = RuleConfig(
            name=RuleNames.MACD_GOLDEN_CROSS,
            rule_type=RuleType.TECHNICAL_INDICATOR,
            symbol=symbol,
            interval=interval,
        )
        return MACDGoldenCrossRule(config)

    @staticmethod
    def create_custom_rule(
        symbol: str, interval: str, name: str, evaluator: Callable
    ) -> CustomRule:
        config = RuleConfig(
            name=name,
            rule_type=RuleType.CUSTOM,
            symbol=symbol,
            interval=interval,
            parameters={"evaluator": evaluator},
        )
        return CustomRule(config)
