from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class MACDGoldenCrossRule(BaseRule):
    """增强版MACD金叉死叉规则"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.fast_period = self.parameters.get("fast_period", 12)
        self.slow_period = self.parameters.get("slow_period", 26)
        self.signal_period = self.parameters.get("signal_period", 9)
        self.angle_threshold = self.parameters.get("angle_threshold", 0.1)  # 角度阈值(绝对斜率)
        self.cross_gap = self.parameters.get("cross_gap", 3)  # 前后无交叉周期数

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.slow_period + self.signal_period + self.cross_gap:
            return None
        indicator_params = {
            "macd_fast": self.fast_period,
            "macd_slow": self.slow_period,
            "macd_signal": self.signal_period,
        }
        df_with_ind = build_quantitative_analysis(
            df.copy(), indicators=["macd"], **indicator_params
        )
        macd_col = f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        signal_col = f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        if macd_col not in df_with_ind.columns or signal_col not in df_with_ind.columns:
            return None
        macd = df_with_ind[macd_col]
        signal = df_with_ind[signal_col]
        idx = -1

        # 检查前后cross_gap周期内无交叉
        def is_cross(i):
            return (macd.iloc[i - 1] < signal.iloc[i - 1] and macd.iloc[i] > signal.iloc[i]) or (
                macd.iloc[i - 1] > signal.iloc[i - 1] and macd.iloc[i] < signal.iloc[i]
            )

        # 当前点是否金叉
        is_golden = macd.iloc[idx - 1] < signal.iloc[idx - 1] and macd.iloc[idx] > signal.iloc[idx]
        is_death = macd.iloc[idx - 1] > signal.iloc[idx - 1] and macd.iloc[idx] < signal.iloc[idx]
        if not (is_golden or is_death):
            return None
        # 检查前后cross_gap周期内无交叉
        for i in range(idx - self.cross_gap, idx):
            if i == idx - 1:
                continue
            if is_cross(i):
                return None
        for i in range(idx + 1, idx + 1 + self.cross_gap):
            if i >= len(macd):
                break
            if is_cross(i):
                return None
        # 计算交叉角度（斜率）
        macd_slope = macd.iloc[idx] - macd.iloc[idx - 1]
        signal_slope = signal.iloc[idx] - signal.iloc[idx - 1]
        angle = abs(macd_slope - signal_slope)
        if angle < self.angle_threshold:
            return None
        current_price = df["Close"].iloc[idx]
        if is_golden:
            signal_type = SignalType.BULLISH
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=min(angle / self.angle_threshold, 1.0),
                duration=8,
                target_price=current_price * 1.06,
                stop_loss=current_price * 0.97,
                take_profit=current_price * 1.12,
                additional_signals=["关注RSI确认", "观察成交量配合"],
                metadata={
                    "cross_type": "golden",
                    "macd_value": macd.iloc[idx],
                    "signal_value": signal.iloc[idx],
                    "angle": angle,
                },
            )
        elif is_death:
            signal_type = SignalType.BEARISH
            return self.create_signal(
                signal_type=signal_type,
                current_price=current_price,
                confidence=min(angle / self.angle_threshold, 1.0),
                duration=8,
                target_price=current_price * 0.94,
                stop_loss=current_price * 1.03,
                take_profit=current_price * 0.88,
                additional_signals=["关注RSI确认", "观察成交量配合"],
                metadata={
                    "cross_type": "death",
                    "macd_value": macd.iloc[idx],
                    "signal_value": signal.iloc[idx],
                    "angle": angle,
                },
            )
        return None
