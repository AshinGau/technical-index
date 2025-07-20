from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


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
        indicator_params = {"ma_periods": (self.short_ma, self.long_ma)}
        df_with_indicators = build_quantitative_analysis(
            df.copy(), indicators=["sma"], **indicator_params
        )
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
        price_trend = (current_price - df["Close"].iloc[-self.trend_periods]) / df["Close"].iloc[
            -self.trend_periods
        ]
        ma_trend = (current_short_ma - current_long_ma) / current_long_ma
        if price_trend > 0.05 and ma_trend > 0.02 and current_short_ma > current_long_ma:
            signal_type = SignalType.BULLISH
            target_price = current_price * 1.08
            stop_loss = current_price * 0.95
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=0.8,
                duration=12,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 1.15,
                additional_signals=["关注成交量配合", "观察RSI确认"],
                metadata={
                    "trend_type": "uptrend",
                    "price_trend": price_trend,
                    "ma_trend": ma_trend,
                    "short_ma": current_short_ma,
                    "long_ma": current_long_ma,
                },
            )
        elif price_trend < -0.05 and ma_trend < -0.02 and current_short_ma < current_long_ma:
            signal_type = SignalType.BEARISH
            target_price = current_price * 0.92
            stop_loss = current_price * 1.05
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=0.8,
                duration=12,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=current_price * 0.85,
                additional_signals=["关注成交量配合", "观察RSI确认"],
                metadata={
                    "trend_type": "downtrend",
                    "price_trend": price_trend,
                    "ma_trend": ma_trend,
                    "short_ma": current_short_ma,
                    "long_ma": current_long_ma,
                },
            )
        return None
