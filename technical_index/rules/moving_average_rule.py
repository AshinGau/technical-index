from typing import Optional, Tuple

import pandas as pd
import numpy as np

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class MovingAverageRule(BaseRule):
    """专业级多均线量化分析规则

    业界最佳实践整合:
    1. 多均线系统: 快线、中线和慢线的组合分析
    2. 均线排列: 多头排列和空头排列的识别
    3. 均线交叉: 快线穿越中慢线的信号识别
    4. 趋势强度: 基于均线斜率和间距的趋势强度评估
    5. 成交量确认: 均线突破时的成交量配合验证
    6. 动量指标: 结合RSI、MACD等指标确认趋势
    7. 支撑阻力: 均线作为动态支撑阻力位分析
    8. 多重时间框架: 不同周期均线的共振分析

    参数说明:
    - sma_periods: 简单移动平均线周期列表，默认(5, 10, 20, 50)
    - ema_periods: 指数移动平均线周期列表，默认None
    - lookback_period: 检查交叉的回顾周期数，默认3
    - volume_ratio_threshold: 成交量比率阈值，默认1.5
    - trend_strength_threshold: 趋势强度阈值，默认0.6
    - momentum_rsi_period: RSI周期，默认14
    - momentum_macd_fast: MACD快线周期，默认12
    - momentum_macd_slow: MACD慢线周期，默认26
    - momentum_macd_signal: MACD信号线周期，默认9
    - crossover_weight: 均线交叉权重，默认0.3
    - trend_weight: 趋势强度权重，默认0.25
    - volume_weight: 成交量权重，默认0.2
    - momentum_weight: 动量指标权重，默认0.25

    信号特征:
    - 综合评分: 多因子加权评分系统
    - 动态止损: 基于ATR和均线支撑阻力的动态止损
    - 风险收益比: 根据趋势强度调整目标价位

    Metadata返回值说明:
    - ma_type: 均线类型 ("sma"=简单移动平均, "ema"=指数移动平均)
    - crossover_type: 交叉类型 ("bullish"=多头交叉, "bearish"=空头交叉)
    - composite_score: 综合评分 (0.0-1.0, 越高信号越强)
    - trend_strength: 趋势强度 (0.0-1.0, 越高趋势越强)
    - ma_alignment: 均线排列状态 ("bullish"=多头排列, "bearish"=空头排列, "mixed"=混合排列)
    - fast_ma_value: 快线当前值
    - slow_ma_values: 所有慢线当前值字典 {周期: 值}
    - crossover_slow: 快线穿过的慢线周期列表
    - atr_value: ATR波动率值

    强度判断标准:
    - composite_score:
      * 0.6-0.7: 弱信号，建议观望
      * 0.7-0.8: 中等信号，可考虑轻仓
      * 0.8-0.9: 强信号，建议正常仓位
      * 0.9-1.0: 极强信号，可考虑重仓
    - trend_strength:
      * 0.0-0.3: 趋势不明显
      * 0.3-0.6: 趋势初现
      * 0.6-0.8: 趋势确立
      * 0.8-1.0: 趋势强劲
    - ma_alignment:
      * "bullish": 多头排列，均线向上发散
      * "bearish": 空头排列，均线向下发散
      * "mixed": 混合排列，均线交错
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)

        # 均线参数
        self.sma_periods = self.parameters.get("sma_periods", (5, 10, 20, 50))
        self.ema_periods = self.parameters.get("ema_periods", None)

        # 确定均线类型和周期
        if self.ema_periods is not None:
            self.ma_type = "ema"
            self.periods = self.ema_periods
        else:
            self.ma_type = "sma"
            self.periods = self.sma_periods

        # 快线和慢线定义
        self.fast_period = min(self.periods)
        self.slow_periods = [p for p in self.periods if p > self.fast_period]

        # 分析参数
        self.lookback_period = self.parameters.get("lookback_period", 3)
        self.volume_ratio_threshold = self.parameters.get("volume_ratio_threshold", 1.5)
        self.trend_strength_threshold = self.parameters.get("trend_strength_threshold", 0.6)

        # 动量指标参数
        self.momentum_rsi_period = self.parameters.get("momentum_rsi_period", 14)
        self.momentum_macd_fast = self.parameters.get("momentum_macd_fast", 12)
        self.momentum_macd_slow = self.parameters.get("momentum_macd_slow", 26)
        self.momentum_macd_signal = self.parameters.get("momentum_macd_signal", 9)

        # 权重参数
        self.crossover_weight = self.parameters.get("crossover_weight", 0.3)
        self.trend_weight = self.parameters.get("trend_weight", 0.25)
        self.volume_weight = self.parameters.get("volume_weight", 0.2)
        self.momentum_weight = self.parameters.get("momentum_weight", 0.25)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < max(self.periods) + 10:
            return None

        # 计算技术指标
        df_with_ind = self._calculate_indicators(df)
        if df_with_ind is None:
            return None

        # 检测均线交叉信号
        crossover_signal = self._detect_crossover(df_with_ind)
        if crossover_signal is None:
            return None

        crossover_type, crossover_strength, crossover_slow_periods = crossover_signal
        current_price = df["Close"].iloc[-1]

        # 分析趋势强度
        trend_strength = self._analyze_trend_strength(df_with_ind)

        # 分析均线排列
        ma_alignment = self._analyze_ma_alignment(df_with_ind)

        # 分析成交量确认
        volume_score = self._analyze_volume(df_with_ind)

        # 分析动量指标
        momentum_score = self._analyze_momentum(df_with_ind)

        # 计算综合评分
        composite_score = self._calculate_composite_score(
            crossover_strength,
            trend_strength,
            volume_score,
            momentum_score,
            crossover_type,
            ma_alignment,
        )

        if composite_score < 0.6:
            return None

        # 生成信号
        return self._generate_signal(
            df_with_ind,
            current_price,
            crossover_type,
            composite_score,
            trend_strength,
            ma_alignment,
            crossover_slow_periods,
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算所需的技术指标"""
        try:
            df_with_ind = df.copy()

            # 计算均线
            for period in self.periods:
                if self.ma_type == "sma":
                    df_with_ind[f"MA_{period}"] = df_with_ind["Close"].rolling(period).mean()
                else:  # ema
                    df_with_ind[f"MA_{period}"] = df_with_ind["Close"].ewm(span=period).mean()

            # 计算动量指标
            indicator_params = {
                "rsi_length": self.momentum_rsi_period,
                "macd_fast": self.momentum_macd_fast,
                "macd_slow": self.momentum_macd_slow,
                "macd_signal": self.momentum_macd_signal,
                "atr_length": 14,
            }

            df_with_ind = build_quantitative_analysis(
                df_with_ind, indicators=["rsi", "macd", "atr"], **indicator_params
            )

            if df_with_ind is None:
                return None

            return df_with_ind

        except Exception:
            return None

    def _detect_crossover(self, df: pd.DataFrame) -> Optional[Tuple[str, float, list]]:
        """检测均线交叉信号"""
        fast_ma = df[f"MA_{self.fast_period}"]

        crossover_signals = []
        crossover_slow_periods = []

        for slow_period in self.slow_periods:
            slow_ma = df[f"MA_{slow_period}"]

            # 检测快线穿越慢线
            for i in range(-self.lookback_period, 0):  # 检查最近lookback_period个周期
                if i == 0 or abs(i) > len(fast_ma):
                    continue

                # 多头交叉：快线从下方穿越慢线
                bullish_cross = (
                    fast_ma.iloc[i - 1] < slow_ma.iloc[i - 1] and fast_ma.iloc[i] > slow_ma.iloc[i]
                )

                # 空头交叉：快线从上方穿越慢线
                bearish_cross = (
                    fast_ma.iloc[i - 1] > slow_ma.iloc[i - 1] and fast_ma.iloc[i] < slow_ma.iloc[i]
                )

                if bullish_cross:
                    crossover_signals.append(("bullish", i, slow_period))
                    if slow_period not in crossover_slow_periods:
                        crossover_slow_periods.append(slow_period)
                elif bearish_cross:
                    crossover_signals.append(("bearish", i, slow_period))
                    if slow_period not in crossover_slow_periods:
                        crossover_slow_periods.append(slow_period)

        if not crossover_signals:
            return None

        # 选择最近的交叉信号
        latest_signal = min(crossover_signals, key=lambda x: abs(x[1]))
        crossover_type, idx, slow_period = latest_signal

        # 计算交叉强度（基于穿越的均线数量）
        crossover_count = len([s for s in crossover_signals if s[0] == crossover_type])
        crossover_strength = min(crossover_count / len(self.slow_periods), 1.0)

        return crossover_type, crossover_strength, crossover_slow_periods

    def _analyze_trend_strength(self, df: pd.DataFrame) -> float:
        """分析趋势强度"""
        # 计算均线斜率
        slope_scores = []

        for period in self.periods:
            ma_col = f"MA_{period}"
            ma_values = df[ma_col]

            if len(ma_values) < 5:
                continue

            # 计算最近5个周期的斜率
            recent_ma = ma_values.iloc[-5:]
            if len(recent_ma) < 2:
                continue

            # 使用线性回归计算斜率
            x = np.arange(len(recent_ma))
            y = recent_ma.values

            if len(y) < 2:
                continue

            slope = np.polyfit(x, y, 1)[0]

            # 标准化斜率
            price_range = df["Close"].iloc[-20:].max() - df["Close"].iloc[-20:].min()
            if price_range > 0:
                normalized_slope = abs(slope) / price_range
                slope_scores.append(min(normalized_slope * 10, 1.0))

        if not slope_scores:
            return 0.0

        return np.mean(slope_scores)

    def _analyze_ma_alignment(self, df: pd.DataFrame) -> str:
        """分析均线排列状态"""
        ma_values = []
        for period in self.periods:
            ma_col = f"MA_{period}"
            if ma_col in df.columns:
                ma_values.append(df[ma_col].iloc[-1])

        if len(ma_values) < 2:
            return "mixed"

        # 检查是否按顺序排列
        is_bullish = all(ma_values[i] >= ma_values[i + 1] for i in range(len(ma_values) - 1))
        is_bearish = all(ma_values[i] <= ma_values[i + 1] for i in range(len(ma_values) - 1))

        if is_bullish:
            return "bullish"
        elif is_bearish:
            return "bearish"
        else:
            return "mixed"

    def _analyze_volume(self, df: pd.DataFrame) -> float:
        """分析成交量确认"""
        if "Volume" not in df.columns:
            return 0.5

        current_volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume"].iloc[-20:].mean()

        if avg_volume == 0:
            return 0.5

        volume_ratio = current_volume / avg_volume

        if volume_ratio > self.volume_ratio_threshold:
            return 1.0
        elif volume_ratio > 1.0:
            return 0.7
        else:
            return 0.3

    def _analyze_momentum(self, df: pd.DataFrame) -> float:
        """分析动量指标"""
        momentum_scores = []

        # RSI分析
        rsi_col = f"RSI_{self.momentum_rsi_period}"
        if rsi_col in df.columns:
            rsi_value = df[rsi_col].iloc[-1]
            if rsi_value > 70:
                momentum_scores.append(0.3)  # 超买
            elif rsi_value < 30:
                momentum_scores.append(0.3)  # 超卖
            elif 40 < rsi_value < 60:
                momentum_scores.append(0.8)  # 中性偏强
            else:
                momentum_scores.append(0.6)  # 一般

        # MACD分析
        macd_col = (
            f"MACD_{self.momentum_macd_fast}_{self.momentum_macd_slow}_{self.momentum_macd_signal}"
        )
        macd_signal_col = (
            f"MACDs_{self.momentum_macd_fast}_{self.momentum_macd_slow}_{self.momentum_macd_signal}"
        )

        if macd_col in df.columns and macd_signal_col in df.columns:
            macd_value = df[macd_col].iloc[-1]
            macd_signal = df[macd_signal_col].iloc[-1]

            if macd_value > macd_signal and macd_value > 0:
                momentum_scores.append(0.9)  # 强势多头
            elif macd_value < macd_signal and macd_value < 0:
                momentum_scores.append(0.1)  # 强势空头
            elif macd_value > macd_signal:
                momentum_scores.append(0.7)  # 弱多头
            else:
                momentum_scores.append(0.3)  # 弱空头

        if not momentum_scores:
            return 0.5

        return np.mean(momentum_scores)

    def _calculate_composite_score(
        self,
        crossover_strength: float,
        trend_strength: float,
        volume_score: float,
        momentum_score: float,
        crossover_type: str,
        ma_alignment: str,
    ) -> float:
        """计算综合评分"""
        base_score = 0.5

        # 加权计算
        total_score = (
            base_score
            + crossover_strength * self.crossover_weight
            + trend_strength * self.trend_weight
            + volume_score * self.volume_weight
            + momentum_score * self.momentum_weight
        )

        # 根据交叉类型调整
        if crossover_type == "bullish":
            total_score *= 1.1
        else:
            total_score *= 0.9

        # 根据均线排列调整
        if ma_alignment == "bullish" and crossover_type == "bullish":
            total_score *= 1.2
        elif ma_alignment == "bearish" and crossover_type == "bearish":
            total_score *= 1.2
        elif ma_alignment == "mixed":
            total_score *= 0.9

        return min(total_score, 1.0)

    def _generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        crossover_type: str,
        composite_score: float,
        trend_strength: float,
        ma_alignment: str,
        crossover_slow_periods: list,
    ) -> SignalResult:
        """生成交易信号"""
        # 计算动态止损
        atr_col = "ATR_14"
        atr_value = df[atr_col].iloc[-1] if atr_col in df.columns else current_price * 0.02

        # 获取均线值
        fast_ma_value = df[f"MA_{self.fast_period}"].iloc[-1]

        # 获取所有慢线的值
        slow_ma_values = {}
        for period in self.slow_periods:
            slow_ma_values[period] = df[f"MA_{period}"].iloc[-1]

        if crossover_type == "bullish":
            signal_type = SignalType.BULLISH
            stop_loss = min(fast_ma_value, current_price - 2 * atr_value)
            target_price = current_price + 3 * atr_value
            take_profit = current_price + 5 * atr_value
            duration = max(5, int(10 * composite_score))
        else:
            signal_type = SignalType.BEARISH
            stop_loss = max(fast_ma_value, current_price + 2 * atr_value)
            target_price = current_price - 3 * atr_value
            take_profit = current_price - 5 * atr_value
            duration = max(5, int(8 * composite_score))

        return self.create_signal(
            signal_type=signal_type,
            current_price=current_price,
            confidence=composite_score,
            duration=duration,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            additional_signals=[
                f"均线类型: {self.ma_type.upper()}",
                f"快线MA({self.fast_period}): {fast_ma_value:.4f}",
                f"慢线值: {', '.join([f'MA({p}): {v:.4f}' for p, v in slow_ma_values.items()])}",
                f"趋势强度: {trend_strength:.2f}",
                f"均线排列: {ma_alignment}",
                self._get_volume_ratio_text(df),
                self._get_momentum_status_text(df),
            ],
            metadata={
                "ma_type": self.ma_type,
                "crossover_type": crossover_type,
                "composite_score": composite_score,
                "trend_strength": trend_strength,
                "ma_alignment": ma_alignment,
                "fast_ma_value": fast_ma_value,
                "slow_ma_values": slow_ma_values,
                "crossover_slow": crossover_slow_periods,
                "atr_value": atr_value,
                "ma_periods": self.periods,
            },
        )

    def _get_volume_ratio_text(self, df: pd.DataFrame) -> str:
        """获取成交量比率文本"""
        if "Volume" not in df.columns:
            return "无成交量数据"

        current_volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume"].iloc[-20:].mean()

        if avg_volume == 0:
            return "成交量数据异常"

        volume_ratio = current_volume / avg_volume
        return f"成交量比率: {volume_ratio:.2f}"

    def _get_momentum_status_text(self, df: pd.DataFrame) -> str:
        """获取动量状态文本"""
        momentum_info = []

        # RSI状态
        rsi_col = f"RSI_{self.momentum_rsi_period}"
        if rsi_col in df.columns:
            rsi_value = df[rsi_col].iloc[-1]
            if rsi_value > 70:
                momentum_info.append("RSI超买")
            elif rsi_value < 30:
                momentum_info.append("RSI超卖")
            else:
                momentum_info.append(f"RSI: {rsi_value:.1f}")

        # MACD状态
        macd_col = (
            f"MACD_{self.momentum_macd_fast}_{self.momentum_macd_slow}_{self.momentum_macd_signal}"
        )
        if macd_col in df.columns:
            macd_value = df[macd_col].iloc[-1]
            if macd_value > 0:
                momentum_info.append("MACD多头")
            else:
                momentum_info.append("MACD空头")

        return " | ".join(momentum_info) if momentum_info else "动量状态: 未知"
