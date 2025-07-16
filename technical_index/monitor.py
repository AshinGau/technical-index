#!/usr/bin/env python3
"""
币价监控规则系统
提供基于价格变化和技术指标的动态规则监控功能
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from .binance import get_futures_market_data
from .index import build_quantitative_analysis

# 配置日志
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型枚举"""

    BULLISH = "bullish"  # 看涨信号
    BEARISH = "bearish"  # 看跌信号
    NEUTRAL = "neutral"  # 中性信号
    ALERT = "alert"  # 预警信号


class RuleType(Enum):
    """规则类型枚举"""

    PRICE_BASED = "price_based"  # 基于价格的规则
    TECHNICAL_INDICATOR = "technical"  # 基于技术指标的规则
    CUSTOM = "custom"  # 自定义规则


@dataclass
class SignalResult:
    """信号结果数据类"""

    symbol: str
    rule_name: str
    signal_type: SignalType
    timestamp: datetime
    current_price: float
    confidence: float = 0.0
    duration: Optional[int] = None  # 预期持续时间（interval数量）
    target_price: Optional[float] = None  # 目标价格
    stop_loss: Optional[float] = None  # 止损价格
    take_profit: Optional[float] = None  # 止盈价格
    resistance_level: Optional[float] = None  # 阻力位
    support_level: Optional[float] = None  # 支撑位
    additional_signals: List[str] = field(default_factory=list)  # 额外关注信号
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class RuleConfig:
    """规则配置数据类"""

    name: str
    rule_type: RuleType
    symbol: str
    interval: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable[[SignalResult], None]] = None


class BaseRule(ABC):
    """规则基类"""

    def __init__(self, config: RuleConfig):
        self.config = config
        self.name = config.name
        self.symbol = config.symbol
        self.interval = config.interval
        self.parameters = config.parameters
        self.enabled = config.enabled

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        """评估规则并返回信号结果"""
        pass

    def get_rule_info(self) -> Dict[str, Any]:
        """获取规则信息"""
        return {
            "name": self.name,
            "type": self.config.rule_type.value,
            "symbol": self.symbol,
            "interval": self.interval,
            "enabled": self.enabled,
            "parameters": self.parameters,
        }


class PriceVolatilityRule(BaseRule):
    """价格波动规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.volatility_threshold = self.parameters.get("volatility_threshold", 0.05)  # 5%
        self.lookback_periods = self.parameters.get("lookback_periods", 20)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.lookback_periods:
            return None

        current_price = df["Close"].iloc[-1]
        price_change = abs(current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]

        if price_change > self.volatility_threshold:
            signal_type = SignalType.ALERT
            confidence = min(price_change / self.volatility_threshold, 1.0)

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=confidence,
                metadata={"price_change": price_change, "threshold": self.volatility_threshold},
            )
        return None


class PriceBreakoutRule(BaseRule):
    """价格突破规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.resistance_periods = self.parameters.get("resistance_periods", 20)
        self.support_periods = self.parameters.get("support_periods", 20)
        self.breakout_threshold = self.parameters.get("breakout_threshold", 0.02)  # 2%

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < max(self.resistance_periods, self.support_periods):
            return None

        current_price = df["Close"].iloc[-1]

        # 计算阻力位和支撑位
        resistance_level = df["High"].rolling(window=self.resistance_periods).max().iloc[-2]
        support_level = df["Low"].rolling(window=self.support_periods).min().iloc[-2]

        # 检查突破
        if current_price > resistance_level * (1 + self.breakout_threshold):
            # 向上突破
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.05  # 目标价格设为当前价格的5%
            stop_loss = support_level
            take_profit = current_price * 1.10  # 止盈设为当前价格的10%

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.8,
                duration=5,  # 预期持续5个interval
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                resistance_level=resistance_level,
                support_level=support_level,
                additional_signals=["关注成交量确认", "观察回踩支撑"],
                metadata={"breakout_type": "resistance", "breakout_level": resistance_level},
            )

        elif current_price < support_level * (1 - self.breakout_threshold):
            # 向下突破
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.95  # 目标价格设为当前价格的95%
            stop_loss = resistance_level
            take_profit = current_price * 0.90  # 止盈设为当前价格的90%

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.8,
                duration=5,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                resistance_level=resistance_level,
                support_level=support_level,
                additional_signals=["关注成交量确认", "观察反弹阻力"],
                metadata={"breakout_type": "support", "breakout_level": support_level},
            )

        return None


class NewHighLowRule(BaseRule):
    """新高新低规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.lookback_periods = self.parameters.get("lookback_periods", 100)
        self.confirmation_periods = self.parameters.get("confirmation_periods", 3)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.lookback_periods:
            return None

        current_price = df["Close"].iloc[-1]
        lookback_high = df["High"].rolling(window=self.lookback_periods).max().iloc[-2]
        lookback_low = df["Low"].rolling(window=self.lookback_periods).min().iloc[-2]

        # 检查新高
        if current_price > lookback_high:
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.08  # 目标价格设为当前价格的8%
            stop_loss = lookback_high * 0.98  # 止损设为突破位的98%

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.9,
                duration=10,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 1.15,
                resistance_level=current_price,
                support_level=lookback_high,
                additional_signals=["关注成交量放大", "观察是否形成新趋势"],
                metadata={"new_level": "high", "previous_high": lookback_high},
            )

        # 检查新低
        elif current_price < lookback_low:
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.92  # 目标价格设为当前价格的92%
            stop_loss = lookback_low * 1.02  # 止损设为突破位的102%

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.9,
                duration=10,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 0.85,
                resistance_level=lookback_low,
                support_level=current_price,
                additional_signals=["关注成交量放大", "观察是否形成新趋势"],
                metadata={"new_level": "low", "previous_low": lookback_low},
            )

        return None


class MACDGoldenCrossRule(BaseRule):
    """MACD金叉死叉规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.fast_period = self.parameters.get("fast_period", 12)
        self.slow_period = self.parameters.get("slow_period", 26)
        self.signal_period = self.parameters.get("signal_period", 9)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.slow_period + self.signal_period:
            return None

        # 计算MACD
        df_with_indicators = build_quantitative_analysis(df.copy())

        if (
            "MACD_12_26_9" not in df_with_indicators.columns
            or "MACDs_12_26_9" not in df_with_indicators.columns
        ):
            return None

        macd_line = df_with_indicators["MACD_12_26_9"].iloc[-1]
        signal_line = df_with_indicators["MACDs_12_26_9"].iloc[-1]
        prev_macd_line = df_with_indicators["MACD_12_26_9"].iloc[-2]
        prev_signal_line = df_with_indicators["MACDs_12_26_9"].iloc[-2]

        current_price = df["Close"].iloc[-1]

        # 金叉：MACD线从下方穿越信号线
        if prev_macd_line < prev_signal_line and macd_line > signal_line:
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.06
            stop_loss = current_price * 0.97

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.75,
                duration=8,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 1.12,
                additional_signals=["关注RSI确认", "观察成交量配合"],
                metadata={
                    "cross_type": "golden",
                    "macd_value": macd_line,
                    "signal_value": signal_line,
                },
            )

        # 死叉：MACD线从上方穿越信号线
        elif prev_macd_line > prev_signal_line and macd_line < signal_line:
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.94
            stop_loss = current_price * 1.03

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.75,
                duration=8,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 0.88,
                additional_signals=["关注RSI确认", "观察成交量配合"],
                metadata={
                    "cross_type": "death",
                    "macd_value": macd_line,
                    "signal_value": signal_line,
                },
            )

        return None


class RSISignalRule(BaseRule):
    """RSI信号规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.oversold_threshold = self.parameters.get("oversold_threshold", 30)
        self.overbought_threshold = self.parameters.get("overbought_threshold", 70)
        self.rsi_period = self.parameters.get("rsi_period", 14)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.rsi_period + 1:
            return None

        # 计算RSI
        df_with_indicators = build_quantitative_analysis(df.copy())

        if "RSI_14" not in df_with_indicators.columns:
            return None

        current_rsi = df_with_indicators["RSI_14"].iloc[-1]
        prev_rsi = df_with_indicators["RSI_14"].iloc[-2]
        current_price = df["Close"].iloc[-1]

        # RSI超卖反弹
        if prev_rsi < self.oversold_threshold and current_rsi > self.oversold_threshold:
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.05
            stop_loss = current_price * 0.98

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.7,
                duration=6,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 1.10,
                additional_signals=["关注MACD确认", "观察价格突破"],
                metadata={"rsi_value": current_rsi, "signal_type": "oversold_bounce"},
            )

        # RSI超买回落
        elif prev_rsi > self.overbought_threshold and current_rsi < self.overbought_threshold:
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.95
            stop_loss = current_price * 1.02

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.7,
                duration=6,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 0.90,
                additional_signals=["关注MACD确认", "观察价格突破"],
                metadata={"rsi_value": current_rsi, "signal_type": "overbought_fall"},
            )

        return None


class TrendAnalysisRule(BaseRule):
    """趋势分析规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.short_ma = self.parameters.get("short_ma", 7)
        self.long_ma = self.parameters.get("long_ma", 25)
        self.trend_periods = self.parameters.get("trend_periods", 10)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.long_ma + self.trend_periods:
            return None

        # 计算移动平均线
        df_with_indicators = build_quantitative_analysis(df.copy())

        short_ma_col = f"SMA_{self.short_ma}"
        long_ma_col = f"SMA_{self.long_ma}"

        if (
            short_ma_col not in df_with_indicators.columns
            or long_ma_col not in df_with_indicators.columns
        ):
            return None

        current_price = df["Close"].iloc[-1]
        current_short_ma = df_with_indicators[short_ma_col].iloc[-1]
        current_long_ma = df_with_indicators[long_ma_col].iloc[-1]

        # 计算趋势强度
        price_trend = (current_price - df["Close"].iloc[-self.trend_periods]) / df["Close"].iloc[
            -self.trend_periods
        ]
        ma_trend = (current_short_ma - current_long_ma) / current_long_ma

        # 上涨趋势
        if price_trend > 0.05 and ma_trend > 0.02 and current_short_ma > current_long_ma:
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.08
            stop_loss = current_long_ma * 0.98

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.8,
                duration=12,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 1.15,
                resistance_level=current_price * 1.05,
                support_level=current_long_ma,
                additional_signals=["关注成交量", "观察回调支撑"],
                metadata={
                    "trend_type": "uptrend",
                    "price_trend": price_trend,
                    "ma_trend": ma_trend,
                },
            )

        # 下跌趋势
        elif price_trend < -0.05 and ma_trend < -0.02 and current_short_ma < current_long_ma:
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.92
            stop_loss = current_long_ma * 1.02

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.8,
                duration=12,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 0.85,
                resistance_level=current_long_ma,
                support_level=current_price * 0.95,
                additional_signals=["关注成交量", "观察反弹阻力"],
                metadata={
                    "trend_type": "downtrend",
                    "price_trend": price_trend,
                    "ma_trend": ma_trend,
                },
            )

        # 震荡市场
        elif abs(price_trend) < 0.03 and abs(ma_trend) < 0.01:
            signal_type = SignalType.NEUTRAL

            return SignalResult(
                symbol=self.symbol,
                rule_name=self.name,
                signal_type=signal_type,
                timestamp=df.index[-1],
                current_price=current_price,
                confidence=0.6,
                duration=5,
                resistance_level=current_price * 1.03,
                support_level=current_price * 0.97,
                additional_signals=["等待突破信号", "关注区间边界"],
                metadata={
                    "trend_type": "sideways",
                    "price_trend": price_trend,
                    "ma_trend": ma_trend,
                },
            )

        return None


class CustomRule(BaseRule):
    """自定义规则基类"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.custom_evaluator = self.parameters.get("evaluator")

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if self.custom_evaluator and callable(self.custom_evaluator):
            return self.custom_evaluator(df, self.config)
        return None


class RuleEngine:
    """规则引擎"""

    def __init__(self):
        self.rules: Dict[str, BaseRule] = {}
        self.global_rules: Dict[str, BaseRule] = {}

    def add_rule(self, rule: BaseRule, is_global: bool = False) -> None:
        """添加规则"""
        rule_key = f"{rule.symbol}_{rule.name}"
        if is_global:
            self.global_rules[rule.name] = rule
        else:
            self.rules[rule_key] = rule

    def remove_rule(self, symbol: str, rule_name: str) -> bool:
        """移除规则"""
        rule_key = f"{symbol}_{rule_name}"
        if rule_key in self.rules:
            del self.rules[rule_key]
            return True
        return False

    def get_rules_for_symbol(self, symbol: str) -> List[BaseRule]:
        """获取指定交易对的所有规则"""
        symbol_rules = []

        # 添加全局规则
        symbol_rules.extend(self.global_rules.values())

        # 添加特定交易对的规则
        for rule_key, rule in self.rules.items():
            if rule_key.startswith(f"{symbol}_"):
                symbol_rules.append(rule)

        return symbol_rules

    def evaluate_rules(self, symbol: str, df: pd.DataFrame) -> List[SignalResult]:
        """评估所有规则"""
        results = []
        rules = self.get_rules_for_symbol(symbol)

        for rule in rules:
            if rule.enabled:
                try:
                    result = rule.evaluate(df)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"规则 {rule.name} 评估失败: {e}")

        return results


class PriceMonitor:
    """价格监控器"""

    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
        self.monitoring_symbols: Set[str] = set()
        self.callbacks: Dict[str, Callable[[SignalResult], None]] = {}
        self.is_running = False
        self.monitor_task = None

    def add_symbol(
        self, symbol: str, callback: Optional[Callable[[SignalResult], None]] = None
    ) -> None:
        """添加监控的交易对"""
        self.monitoring_symbols.add(symbol)
        if callback:
            self.callbacks[symbol] = callback

    def remove_symbol(self, symbol: str) -> None:
        """移除监控的交易对"""
        self.monitoring_symbols.discard(symbol)
        self.callbacks.pop(symbol, None)

    def set_callback(self, symbol: str, callback: Callable[[SignalResult], None]) -> None:
        """设置回调函数"""
        self.callbacks[symbol] = callback

    def default_callback(self, signal: SignalResult) -> None:
        """默认回调函数"""
        logger.info(f"信号触发 - {signal.symbol}: {signal.rule_name}")
        logger.info(f"信号类型: {signal.signal_type.value}")
        logger.info(f"当前价格: {signal.current_price}")
        logger.info(f"置信度: {signal.confidence}")

        if signal.target_price:
            logger.info(f"目标价格: {signal.target_price}")
        if signal.stop_loss:
            logger.info(f"止损价格: {signal.stop_loss}")
        if signal.take_profit:
            logger.info(f"止盈价格: {signal.take_profit}")
        if signal.resistance_level:
            logger.info(f"阻力位: {signal.resistance_level}")
        if signal.support_level:
            logger.info(f"支撑位: {signal.support_level}")
        if signal.additional_signals:
            logger.info(f"额外信号: {', '.join(signal.additional_signals)}")

        logger.info("-" * 50)

    async def monitor_symbol(self, symbol: str, interval: str) -> None:
        """监控单个交易对"""
        while self.is_running and symbol in self.monitoring_symbols:
            try:
                # 获取市场数据
                df = get_futures_market_data(symbol, interval, limit=200)
                if df is None or df.empty:
                    logger.warning(f"无法获取 {symbol} 的市场数据")
                    await asyncio.sleep(60)
                    continue

                # 评估规则
                signals = self.rule_engine.evaluate_rules(symbol, df)

                # 处理信号
                for signal in signals:
                    callback = self.callbacks.get(symbol, self.default_callback)
                    try:
                        callback(signal)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")

                # 等待下次检查
                if interval == "1d":
                    await asyncio.sleep(3600)  # 1小时
                elif interval == "1h":
                    await asyncio.sleep(900)  # 15分钟
                else:
                    await asyncio.sleep(300)  # 5分钟

            except Exception as e:
                logger.error(f"监控 {symbol} 时发生错误: {e}")
                await asyncio.sleep(60)

    async def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("开始价格监控...")

        # 为每个交易对创建监控任务
        tasks = []
        for symbol in self.monitoring_symbols:
            # 获取该交易对的规则来确定interval
            rules = self.rule_engine.get_rules_for_symbol(symbol)
            if rules:
                interval = rules[0].interval  # 使用第一个规则的interval
                task = asyncio.create_task(self.monitor_symbol(symbol, interval))
                tasks.append(task)

        # 等待所有任务
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop_monitoring(self) -> None:
        """停止监控"""
        self.is_running = False
        logger.info("停止价格监控")


# 便捷函数
def create_price_volatility_rule(
    symbol: str, interval: str, volatility_threshold: float = 0.05
) -> PriceVolatilityRule:
    """创建价格波动规则"""
    config = RuleConfig(
        name="价格波动监控",
        rule_type=RuleType.PRICE_BASED,
        symbol=symbol,
        interval=interval,
        parameters={"volatility_threshold": volatility_threshold},
    )
    return PriceVolatilityRule(config)


def create_breakout_rule(symbol: str, interval: str) -> PriceBreakoutRule:
    """创建突破规则"""
    config = RuleConfig(
        name="价格突破监控", rule_type=RuleType.PRICE_BASED, symbol=symbol, interval=interval
    )
    return PriceBreakoutRule(config)


def create_macd_rule(symbol: str, interval: str) -> MACDGoldenCrossRule:
    """创建MACD规则"""
    config = RuleConfig(
        name="MACD金叉死叉",
        rule_type=RuleType.TECHNICAL_INDICATOR,
        symbol=symbol,
        interval=interval,
    )
    return MACDGoldenCrossRule(config)


def create_rsi_rule(symbol: str, interval: str) -> RSISignalRule:
    """创建RSI规则"""
    config = RuleConfig(
        name="RSI超买超卖", rule_type=RuleType.TECHNICAL_INDICATOR, symbol=symbol, interval=interval
    )
    return RSISignalRule(config)


def create_trend_rule(symbol: str, interval: str) -> TrendAnalysisRule:
    """创建趋势分析规则"""
    config = RuleConfig(
        name="趋势分析", rule_type=RuleType.TECHNICAL_INDICATOR, symbol=symbol, interval=interval
    )
    return TrendAnalysisRule(config)


def create_custom_rule(symbol: str, interval: str, name: str, evaluator: Callable) -> CustomRule:
    """创建自定义规则"""
    config = RuleConfig(
        name=name,
        rule_type=RuleType.CUSTOM,
        symbol=symbol,
        interval=interval,
        parameters={"evaluator": evaluator},
    )
    return CustomRule(config)
