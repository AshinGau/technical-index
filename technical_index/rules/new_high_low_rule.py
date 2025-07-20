from typing import Optional

import pandas as pd

from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class NewHighLowRule(BaseRule):
    """新高新低规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.lookback_periods = self.parameters.get("lookback_periods", 50)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.lookback_periods:
            return None
        current_price = df["Close"].iloc[-1]
        historical_high = df["High"].iloc[-self.lookback_periods : -1].max()
        historical_low = df["Low"].iloc[-self.lookback_periods : -1].min()
        if current_price > historical_high:
            return self.create_signal(
                signal_type=SignalType.BULLISH,
                current_price=current_price,
                confidence=0.9,
                metadata={"new_high": True, "historical_high": historical_high},
            )
        elif current_price < historical_low:
            return self.create_signal(
                signal_type=SignalType.BEARISH,
                current_price=current_price,
                confidence=0.9,
                metadata={"new_low": True, "historical_low": historical_low},
            )
        return None
