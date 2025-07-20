from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class RSISignalRule(BaseRule):
    """RSI信号规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.oversold_threshold = self.parameters.get("oversold_threshold", 20)
        self.overbought_threshold = self.parameters.get("overbought_threshold", 80)
        self.rsi_period = self.parameters.get("rsi_period", 14)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.rsi_period + 1:
            return None
        indicator_params = {"rsi_length": self.rsi_period}
        df_with_indicators = build_quantitative_analysis(
            df.copy(), indicators=["rsi"], **indicator_params
        )
        rsi_col = f"RSI_{self.rsi_period}"
        if rsi_col not in df_with_indicators.columns:
            return None
        current_rsi = df_with_indicators[rsi_col].iloc[-1]
        prev_rsi = df_with_indicators[rsi_col].iloc[-2]
        current_price = df["Close"].iloc[-1]
        if prev_rsi < self.oversold_threshold and current_rsi > self.oversold_threshold:
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.05
            stop_loss = current_price * 0.98
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=0.7,
                duration=6,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 1.10,
                additional_signals=["关注MACD确认", "观察价格突破"],
                metadata={"rsi_value": current_rsi, "signal_type": "oversold_bounce"},
            )
        elif prev_rsi > self.overbought_threshold and current_rsi < self.overbought_threshold:
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.95
            stop_loss = current_price * 1.02
            return self.create_signal(
                signal_type=signal_type,
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
