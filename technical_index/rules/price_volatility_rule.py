from typing import Optional

import pandas as pd

from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class PriceVolatilityRule(BaseRule):
    """
    价格波动规则

    该规则用于检测价格异常波动, 包括三种检测方式:
    1. 价格变化检测: 检测当前价格相对于前一收盘价的异常变化
    2. 振幅异常检测: 检测当前K线振幅相对于历史平均振幅的异常
    3. 涨跌幅异常检测: 检测当前价格相对于开盘价的异常涨跌幅

    参数说明:
    - volatility_threshold: 价格变化阈值, 默认5%
    - lookback_periods: 回看周期数, 用于计算历史平均值, 默认20
    - amplitude_multiplier: 振幅倍数阈值, 默认2.0倍
    - change_multiplier: 涨跌幅倍数阈值, 默认2.0倍
    """

    def __init__(self, config: RuleConfig):
        """
        初始化价格波动规则

        Args:
            config: 规则配置对象，包含规则参数
        """
        super().__init__(config)
        # 价格变化阈值，超过此阈值触发信号
        self.volatility_threshold = self.parameters.get("volatility_threshold", 0.05)
        # 回看周期数，用于计算历史统计值
        self.lookback_periods = self.parameters.get("lookback_periods", 20)
        # 振幅倍数阈值，当前振幅超过历史平均振幅的此倍数时触发信号
        self.amplitude_multiplier = self.parameters.get("amplitude_multiplier", 2.0)
        # 涨跌幅倍数阈值，当前涨跌幅超过历史平均涨跌幅的此倍数时触发信号
        self.change_multiplier = self.parameters.get("change_multiplier", 2.0)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        """
        评估价格数据，检测异常波动
        """
        # 检查数据量是否足够
        if len(df) < self.lookback_periods:
            return None

        # 获取当前价格数据
        current_price = df["Close"].iloc[-1]
        current_high = df["High"].iloc[-1]
        current_low = df["Low"].iloc[-1]

        # 计算价格变化率（当前收盘价相对于前一收盘价）
        price_change = abs(current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]

        # 检测1：价格变化异常
        if price_change > self.volatility_threshold:
            signal_type = SignalType.ALERT
            # 置信度基于变化率与阈值的比值，最大为1.0
            confidence = min(price_change / self.volatility_threshold, 1.0)
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=confidence,
                metadata={
                    "detection_type": "price_change",
                    "price_change": (current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2],
                    "threshold": self.volatility_threshold,
                },
            )

        # 检测2：振幅异常
        amplitude_signal = self._check_amplitude_anomaly(df, current_high, current_low)
        if amplitude_signal:
            return amplitude_signal

        # 检测3：涨跌幅异常
        change_signal = self._check_change_anomaly(df, current_price)
        if change_signal:
            return change_signal

        return None

    def _check_amplitude_anomaly(
        self, df: pd.DataFrame, current_high: float, current_low: float
    ) -> Optional[SignalResult]:
        """
        检测振幅异常
        """
        # 获取历史数据（不包括当前K线）
        lookback_data = df.iloc[-self.lookback_periods : -1]
        if len(lookback_data) < 5:
            return None

        # 计算历史每日振幅
        daily_amplitudes = (lookback_data["High"] - lookback_data["Low"]) / lookback_data["Low"]
        # 计算历史平均振幅
        avg_amplitude = daily_amplitudes.mean()
        # 计算当前K线振幅
        current_amplitude = (current_high - current_low) / current_low

        # 检查当前振幅是否异常
        if current_amplitude > avg_amplitude * self.amplitude_multiplier:
            signal_type = SignalType.ALERT
            # 置信度基于当前振幅与阈值的比值
            confidence = min(current_amplitude / (avg_amplitude * self.amplitude_multiplier), 1.0)
            return self.create_signal(
                signal_type=signal_type,
                current_price=df["Close"].iloc[-1],
                confidence=confidence,
                metadata={
                    "detection_type": "amplitude_anomaly",
                    "current_amplitude": current_amplitude,
                    "avg_amplitude": avg_amplitude,
                    "multiplier": self.amplitude_multiplier,
                    "current_high": current_high,
                    "current_low": current_low,
                },
            )
        return None

    def _check_change_anomaly(
        self, df: pd.DataFrame, current_price: float
    ) -> Optional[SignalResult]:
        """
        检测涨跌幅异常
        """
        # 获取历史数据（不包括当前K线）
        lookback_data = df.iloc[-self.lookback_periods : -1]
        if len(lookback_data) < 5:
            return None

        # 计算历史每日涨跌幅
        daily_changes = (lookback_data["Close"] - lookback_data["Open"]) / lookback_data["Open"]
        # 计算历史平均涨跌幅（取绝对值）
        avg_change = daily_changes.abs().mean()
        # 计算当前涨跌幅
        current_change = abs(current_price - df["Open"].iloc[-1]) / df["Open"].iloc[-1]

        # 检查当前涨跌幅是否异常
        if current_change > avg_change * self.change_multiplier:
            signal_type = SignalType.ALERT
            # 置信度基于当前涨跌幅与阈值的比值
            confidence = min(current_change / (avg_change * self.change_multiplier), 1.0)
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=confidence,
                metadata={
                    "detection_type": "change_anomaly",
                    "current_change": (current_price - df["Open"].iloc[-1]) / df["Open"].iloc[-1],
                    "avg_change": avg_change,
                    "multiplier": self.change_multiplier,
                },
            )
        return None
