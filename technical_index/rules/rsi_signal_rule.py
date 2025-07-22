from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class RSISignalRule(BaseRule):
    """
    RSI超买超卖信号规则（业界标准版）
    - 多头信号：RSI从下向上突破20，判定为超卖反弹。
    - 空头信号：RSI从上向下跌破80，判定为超买回落。
    - 可选：MACD柱辅助、ATR极端波动过滤。
    - 参数：oversold_threshold=20, overbought_threshold=80, rsi_period=14。
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.oversold_threshold = self.parameters.get("oversold_threshold", 20)
        self.overbought_threshold = self.parameters.get("overbought_threshold", 80)
        self.rsi_period = self.parameters.get("rsi_period", 14)
        # 可选辅助
        self.macd_fast = self.parameters.get("macd_fast", 12)
        self.macd_slow = self.parameters.get("macd_slow", 26)
        self.macd_signal = self.parameters.get("macd_signal", 9)
        self.atr_length = self.parameters.get("atr_length", 14)
        self.max_atr_ratio = self.parameters.get("max_atr_ratio", 0.5)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        min_len = max(self.rsi_period, self.macd_slow, self.atr_length) + 2
        if len(df) < min_len:
            return None
        indicator_params = {
            "rsi_length": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "atr_length": self.atr_length,
        }
        df_ind = build_quantitative_analysis(
            df.copy(), indicators=["rsi", "macd", "atr"], **indicator_params
        )
        if df_ind is None:
            return None
        current = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        current_rsi = current[f"RSI_{self.rsi_period}"]
        prev_rsi = prev[f"RSI_{self.rsi_period}"]
        current_price = current["Close"]
        macd_hist = current[f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"]
        atr = current[f"ATRr_{self.atr_length}"]
        atr_ratio = atr / current_price if current_price > 0 else 0
        # 多头信号：RSI从下向上突破20
        bullish = (
            prev_rsi < self.oversold_threshold
            and current_rsi >= self.oversold_threshold
            and atr_ratio < self.max_atr_ratio
        )
        # 空头信号：RSI从上向下跌破80
        bearish = (
            prev_rsi > self.overbought_threshold
            and current_rsi <= self.overbought_threshold
            and atr_ratio < self.max_atr_ratio
        )
        if bullish:
            return self.create_signal(
                signal_type=SignalType.BULLISH,
                current_price=current_price,
                confidence=0.7,
                duration=8,
                target_price=current_price * 1.04,
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.08,
                additional_signals=["RSI超卖反弹"],
                metadata={
                    "rsi": current_rsi,
                    "macd_hist": macd_hist,
                    "atr": atr,
                    "atr_ratio": atr_ratio,
                },
            )
        elif bearish:
            return self.create_signal(
                signal_type=SignalType.BEARISH,
                current_price=current_price,
                confidence=0.7,
                duration=8,
                target_price=current_price * 0.96,
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.92,
                additional_signals=["RSI超买回落"],
                metadata={
                    "rsi": current_rsi,
                    "macd_hist": macd_hist,
                    "atr": atr,
                    "atr_ratio": atr_ratio,
                },
            )
        return None
