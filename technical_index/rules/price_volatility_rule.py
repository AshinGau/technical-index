from typing import Optional

import pandas as pd

from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class PriceVolatilityRule(BaseRule):
    """价格波动规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.volatility_threshold = self.parameters.get("volatility_threshold", 0.05)
        self.lookback_periods = self.parameters.get("lookback_periods", 20)
        self.amplitude_multiplier = self.parameters.get("amplitude_multiplier", 2.0)
        self.change_multiplier = self.parameters.get("change_multiplier", 2.0)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.lookback_periods:
            return None
        current_price = df["Close"].iloc[-1]
        current_high = df["High"].iloc[-1]
        current_low = df["Low"].iloc[-1]
        price_change = abs(current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]
        if price_change > self.volatility_threshold:
            signal_type = SignalType.ALERT
            confidence = min(price_change / self.volatility_threshold, 1.0)
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=confidence,
                metadata={
                    "detection_type": "price_change",
                    "price_change": price_change,
                    "threshold": self.volatility_threshold,
                },
            )
        amplitude_signal = self._check_amplitude_anomaly(df, current_high, current_low)
        if amplitude_signal:
            return amplitude_signal
        change_signal = self._check_change_anomaly(df, current_price)
        if change_signal:
            return change_signal
        return None

    def _check_amplitude_anomaly(
        self, df: pd.DataFrame, current_high: float, current_low: float
    ) -> Optional[SignalResult]:
        lookback_data = df.iloc[-self.lookback_periods : -1]
        if len(lookback_data) < 5:
            return None
        daily_amplitudes = (lookback_data["High"] - lookback_data["Low"]) / lookback_data["Low"]
        avg_amplitude = daily_amplitudes.mean()
        current_amplitude = (current_high - current_low) / current_low
        if current_amplitude > avg_amplitude * self.amplitude_multiplier:
            signal_type = SignalType.ALERT
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
        lookback_data = df.iloc[-self.lookback_periods : -1]
        if len(lookback_data) < 5:
            return None
        daily_changes = (lookback_data["Close"] - lookback_data["Open"]) / lookback_data["Open"]
        avg_change = daily_changes.abs().mean()
        current_change = abs(current_price - df["Open"].iloc[-1]) / df["Open"].iloc[-1]
        if current_change > avg_change * self.change_multiplier:
            signal_type = SignalType.ALERT
            confidence = min(current_change / (avg_change * self.change_multiplier), 1.0)
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=confidence,
                metadata={
                    "detection_type": "change_anomaly",
                    "current_change": current_change,
                    "avg_change": avg_change,
                    "multiplier": self.change_multiplier,
                },
            )
        return None
