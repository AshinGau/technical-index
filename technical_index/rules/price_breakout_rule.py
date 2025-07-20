from typing import Optional

import pandas as pd

from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class PriceBreakoutRule(BaseRule):
    """价格突破规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.lookback_periods = self.parameters.get("lookback_periods", 20)
        self.breakout_threshold = self.parameters.get("breakout_threshold", 0.03)
        self.pullback_threshold = self.parameters.get("pullback_threshold", 0.01)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.lookback_periods:
            return None
        current_price = df["Close"].iloc[-1]
        recent_high = df["High"].iloc[-self.lookback_periods : -1].max()
        recent_low = df["Low"].iloc[-self.lookback_periods : -1].min()
        if current_price > recent_high * (1 + self.breakout_threshold):
            return self.create_signal(
                signal_type=SignalType.BULLISH,
                current_price=current_price,
                confidence=0.8,
                metadata={"breakout_type": "up", "recent_high": recent_high},
            )
        elif current_price < recent_low * (1 - self.breakout_threshold):
            return self.create_signal(
                signal_type=SignalType.BEARISH,
                current_price=current_price,
                confidence=0.8,
                metadata={"breakout_type": "down", "recent_low": recent_low},
            )
        return None
