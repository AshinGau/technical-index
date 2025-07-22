from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class PriceBreakoutRule(BaseRule):
    """
    价格突破规则（业界标准版，斐波那契辅助）：
    - 多头：收盘价突破N日高点（阻力），成交量放大，短期均线>长期均线。
    - 空头：收盘价跌破N日低点（支撑），成交量放大，短期均线<长期均线。
    - 斐波那契关键位辅助：突破后站稳0.618（多头）/0.382（空头）信号更强，目标/止损参考1.382/1.618。
    - ATR过滤极端波动。
    - 参数：lookback_high/low=20, volume_ratio=1.5, ma_short=7, ma_long=21。
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.lookback_high = self.parameters.get("lookback_high", 20)
        self.lookback_low = self.parameters.get("lookback_low", 20)
        self.volume_ratio = self.parameters.get("volume_ratio", 1.5)
        self.ma_short = self.parameters.get("ma_short", 7)
        self.ma_long = self.parameters.get("ma_long", 21)
        self.atr_length = self.parameters.get("atr_length", 14)
        self.max_atr_ratio = self.parameters.get("max_atr_ratio", 0.5)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        min_len = max(self.lookback_high, self.lookback_low, self.ma_long, self.atr_length) + 2
        if len(df) < min_len:
            return None
        # 计算指标
        indicator_params = {
            "ma_periods": (self.ma_short, self.ma_long),
            "atr_length": self.atr_length,
        }
        df_ind = build_quantitative_analysis(
            df.copy(), indicators=["sma", "atr"], **indicator_params
        )
        if df_ind is None:
            return None
        current = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]
        current_close = current["Close"]
        current_volume = current["Volume"]
        avg_volume = df_ind["Volume"].iloc[-self.lookback_high - 1 : -1].mean()
        sma_short = current[f"SMA_{self.ma_short}"]
        sma_long = current[f"SMA_{self.ma_long}"]
        atr = current[f"ATRr_{self.atr_length}"]
        atr_ratio = atr / current_close if current_close > 0 else 0
        recent_high = df_ind["High"].iloc[-self.lookback_high - 1 : -1].max()
        recent_low = df_ind["Low"].iloc[-self.lookback_low - 1 : -1].min()
        # 斐波那契关键位
        fib_range = recent_high - recent_low
        fib_382 = recent_high - 0.382 * fib_range
        fib_500 = recent_high - 0.5 * fib_range
        fib_618 = recent_high - 0.618 * fib_range
        fib_1382 = recent_high + 0.382 * fib_range
        fib_1618 = recent_high + 0.618 * fib_range
        # 多头突破判据
        bullish = (
            prev["Close"] <= recent_high
            and current_close > recent_high
            and current_volume > avg_volume * self.volume_ratio
            and sma_short > sma_long
            and atr_ratio < self.max_atr_ratio
            and current_close > fib_618
        )
        # 空头突破判据
        bearish = (
            prev["Close"] >= recent_low
            and current_close < recent_low
            and current_volume > avg_volume * self.volume_ratio
            and sma_short < sma_long
            and atr_ratio < self.max_atr_ratio
            and current_close < fib_382
        )
        if bullish:
            return self.create_signal(
                signal_type=SignalType.BULLISH,
                current_price=current_close,
                confidence=0.75,
                duration=8,
                target_price=fib_1382,
                stop_loss=fib_500,
                take_profit=fib_1618,
                additional_signals=["突破阻力位", "成交量放大", "均线多头", "斐波那契0.618上方"],
                metadata={
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "fib_382": fib_382,
                    "fib_500": fib_500,
                    "fib_618": fib_618,
                    "fib_1382": fib_1382,
                    "fib_1618": fib_1618,
                    "atr": atr,
                    "atr_ratio": atr_ratio,
                    "avg_volume": avg_volume,
                    "current_volume": current_volume,
                },
            )
        elif bearish:
            return self.create_signal(
                signal_type=SignalType.BEARISH,
                current_price=current_close,
                confidence=0.75,
                duration=8,
                target_price=fib_1382,
                stop_loss=fib_500,
                take_profit=fib_1618,
                additional_signals=["跌破支撑位", "成交量放大", "均线空头", "斐波那契0.382下方"],
                metadata={
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "fib_382": fib_382,
                    "fib_500": fib_500,
                    "fib_618": fib_618,
                    "fib_1382": fib_1382,
                    "fib_1618": fib_1618,
                    "atr": atr,
                    "atr_ratio": atr_ratio,
                    "avg_volume": avg_volume,
                    "current_volume": current_volume,
                },
            )
        return None
