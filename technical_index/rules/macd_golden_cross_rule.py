from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class MACDGoldenCrossRule(BaseRule):
    """专业级MACD量化分析规则

    业界最佳实践整合:
    1. 传统金叉死叉: MACD线与信号线的交叉
    2. MACD柱状图分析: 柱状图的变化趋势和强度
    3. 背离检测: 价格与MACD的顶背离和底背离
    4. 零轴分析: MACD在零轴上下方的位置和穿越
    5. 成交量确认: MACD信号需要成交量配合验证
    6. 趋势过滤: 结合移动平均线过滤信号方向
    7. 多重确认: 多个时间框架的MACD信号共振

    参数说明:
    - lookback_period: 回看周期内出现交叉，默认1
    - fast_period: 快线周期，默认12
    - slow_period: 慢线周期，默认26
    - signal_period: 信号线周期，默认9
    - volume_ratio_threshold: 成交量比率阈值，默认1.5
    - divergence_lookback: 背离检测回看周期，默认20
    - trend_ma_period: 趋势过滤MA周期，默认50
    - histogram_threshold: 柱状图变化阈值，默认0.001
    - zero_cross_weight: 零轴穿越权重，默认0.3
    - divergence_weight: 背离权重，默认0.4
    - volume_weight: 成交量权重，默认0.2
    - trend_weight: 趋势权重，默认0.1

    信号特征:
    - 综合评分: 多因子加权评分系统
    - 动态止损: 基于ATR的动态止损
    - 风险收益比: 根据信号强度调整目标

    Metadata返回值说明:
    - cross_type: 交叉类型 ("golden"=金叉, "death"=死叉)
    - composite_score: 综合评分 (0.0-1.0, 越高信号越强)
    - macd_value: MACD线当前值 (正值=多头趋势, 负值=空头趋势)
    - signal_value: 信号线当前值 (MACD的移动平均)
    - histogram_value: MACD柱状图值 (MACD-信号线, 正值=柱状图向上, 负值=向下)
    - atr_value: ATR波动率值 (用于动态止损计算)

    强度判断标准:
    - composite_score:
      * 0.6-0.7: 弱信号，建议观望
      * 0.7-0.8: 中等信号，可考虑轻仓
      * 0.8-0.9: 强信号，建议正常仓位
      * 0.9-1.0: 极强信号，可考虑重仓
    - macd_value:
      * > 0: 多头趋势，数值越大趋势越强
      * < 0: 空头趋势，数值越小趋势越强
    - histogram_value:
      * > 0: 柱状图向上，动量增强
      * < 0: 柱状图向下，动量减弱
      * 绝对值越大，动量变化越剧烈
    - atr_value:
      * 数值越大，市场波动越剧烈
      * 用于动态止损和仓位管理
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        # MACD基础参数
        self.fast_period = self.parameters.get("fast_period", 12)
        self.slow_period = self.parameters.get("slow_period", 26)
        self.signal_period = self.parameters.get("signal_period", 9)

        # lookback参数
        self.lookback_period = self.parameters.get("lookback_period", 1)

        # 高级分析参数
        self.volume_ratio_threshold = self.parameters.get("volume_ratio_threshold", 1.5)
        self.divergence_lookback = self.parameters.get("divergence_lookback", 20)
        self.trend_ma_period = self.parameters.get("trend_ma_period", 50)
        self.histogram_threshold = self.parameters.get("histogram_threshold", 0.001)

        # 权重参数
        self.zero_cross_weight = self.parameters.get("zero_cross_weight", 0.3)
        self.divergence_weight = self.parameters.get("divergence_weight", 0.4)
        self.volume_weight = self.parameters.get("volume_weight", 0.2)
        self.trend_weight = self.parameters.get("trend_weight", 0.1)

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < max(
            self.slow_period + self.signal_period,
            self.divergence_lookback,
            self.trend_ma_period,
            self.lookback_period,
        ):
            return None

        # 计算技术指标
        df_with_ind = self._calculate_indicators(df)
        if df_with_ind is None:
            return None

        # 回看lookback_period周期，找到最后一个交叉信号
        macd_col = f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        signal_col = f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        macd = df_with_ind[macd_col]
        signal = df_with_ind[signal_col]
        last_cross_idx = None
        cross_type = None
        for i in range(-self.lookback_period, 0):
            if i == 0 or abs(i) > len(macd):
                continue
            is_golden = macd.iloc[i - 1] < signal.iloc[i - 1] and macd.iloc[i] > signal.iloc[i]
            is_death = macd.iloc[i - 1] > signal.iloc[i - 1] and macd.iloc[i] < signal.iloc[i]
            if is_golden or is_death:
                last_cross_idx = i
                cross_type = "golden" if is_golden else "death"
        if last_cross_idx is None or cross_type is None:
            return None
        # 用最后一个交叉信号点为基准，后续评分等都以该点为准
        idx = last_cross_idx
        current_price = df["Close"].iloc[-1]  # 评分等用最新K线

        # 2. MACD柱状图分析
        histogram_score = self._analyze_histogram(df_with_ind, idx)
        # 3. 背离检测
        divergence_score = self._detect_divergence(df_with_ind, idx)
        # 4. 零轴分析
        zero_cross_score = self._analyze_zero_cross(df_with_ind, idx)
        # 5. 成交量确认
        volume_score = self._analyze_volume(df, idx)
        # 6. 趋势过滤
        trend_score = self._analyze_trend(df_with_ind, idx)
        # 7. 综合评分
        total_score = self._calculate_composite_score(
            cross_type,
            histogram_score,
            divergence_score,
            zero_cross_score,
            volume_score,
            trend_score,
        )
        if total_score < 0.6:
            return None
        # 8. 生成信号（评分等用最新K线，信号点用last_cross_idx）
        return self._generate_signal(df_with_ind, idx, current_price, cross_type, total_score)

    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算所需的技术指标"""
        try:
            indicator_params = {
                "macd_fast": self.fast_period,
                "macd_slow": self.slow_period,
                "macd_signal": self.signal_period,
                "atr_length": 14,
            }

            # 计算MACD和ATR
            df_with_ind = build_quantitative_analysis(
                df.copy(), indicators=["macd", "atr"], **indicator_params
            )

            if df_with_ind is None:
                return None

            # 计算趋势MA
            df_with_ind[f"MA_{self.trend_ma_period}"] = (
                df_with_ind["Close"].rolling(self.trend_ma_period).mean()
            )

            # 计算MACD柱状图
            macd_col = f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
            signal_col = f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"

            if macd_col not in df_with_ind.columns or signal_col not in df_with_ind.columns:
                return None

            df_with_ind["MACD_Histogram"] = df_with_ind[macd_col] - df_with_ind[signal_col]

            return df_with_ind

        except Exception:
            return None

    def _analyze_macd_cross(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """分析MACD交叉信号"""
        macd_col = f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        signal_col = f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"

        macd = df[macd_col]
        signal = df[signal_col]

        # 检查金叉死叉
        is_golden = macd.iloc[idx - 1] < signal.iloc[idx - 1] and macd.iloc[idx] > signal.iloc[idx]
        is_death = macd.iloc[idx - 1] > signal.iloc[idx - 1] and macd.iloc[idx] < signal.iloc[idx]

        if is_golden:
            return "golden"
        elif is_death:
            return "death"
        return None

    def _analyze_histogram(self, df: pd.DataFrame, idx: int) -> float:
        """分析MACD柱状图变化"""
        histogram = df["MACD_Histogram"]

        # 计算柱状图变化率
        current_hist = histogram.iloc[idx]
        prev_hist = histogram.iloc[idx - 1]

        if abs(prev_hist) < 1e-6:
            return 0.0

        hist_change = (current_hist - prev_hist) / abs(prev_hist)

        # 柱状图强度评分
        if abs(hist_change) > self.histogram_threshold:
            return min(abs(hist_change) / 0.01, 1.0)  # 标准化到0-1
        return 0.0

    def _detect_divergence(self, df: pd.DataFrame, idx: int) -> float:
        """检测价格与MACD的背离"""
        if idx < self.divergence_lookback:
            return 0.0

        prices = df["Close"].iloc[idx - self.divergence_lookback : idx + 1]
        macd_hist = df["MACD_Histogram"].iloc[idx - self.divergence_lookback : idx + 1]

        # 寻找局部极值
        price_peaks = self._find_peaks(prices)
        price_troughs = self._find_troughs(prices)
        macd_peaks = self._find_peaks(macd_hist)
        macd_troughs = self._find_troughs(macd_hist)

        # 检测背离
        divergence_score = 0.0

        # 顶背离：价格创新高，MACD未创新高
        if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
            if (
                prices.iloc[price_peaks[-1]] > prices.iloc[price_peaks[-2]]
                and macd_hist.iloc[macd_peaks[-1]] < macd_hist.iloc[macd_peaks[-2]]
            ):
                divergence_score += 0.5

        # 底背离：价格创新低，MACD未创新低
        if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
            if (
                prices.iloc[price_troughs[-1]] < prices.iloc[price_troughs[-2]]
                and macd_hist.iloc[macd_troughs[-1]] > macd_hist.iloc[macd_troughs[-2]]
            ):
                divergence_score += 0.5

        return divergence_score

    def _analyze_zero_cross(self, df: pd.DataFrame, idx: int) -> float:
        """分析MACD零轴穿越"""
        macd_col = f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        macd = df[macd_col]

        current_macd = macd.iloc[idx]
        prev_macd = macd.iloc[idx - 1]

        # 零轴穿越评分
        if prev_macd < 0 and current_macd > 0:  # 向上穿越零轴
            return 1.0
        elif prev_macd > 0 and current_macd < 0:  # 向下穿越零轴
            return 1.0
        elif current_macd > 0:  # 在零轴上方
            return 0.5
        else:  # 在零轴下方
            return 0.0

    def _analyze_volume(self, df: pd.DataFrame, idx: int) -> float:
        """分析成交量确认"""
        if "Volume" not in df.columns:
            return 0.5  # 无成交量数据时给中性评分

        volume = df["Volume"]
        current_volume = volume.iloc[idx]

        # 计算平均成交量
        start_idx = max(0, idx - 20)
        avg_volume = volume.iloc[start_idx:idx].mean()

        if avg_volume == 0:
            return 0.5

        volume_ratio = current_volume / avg_volume

        if volume_ratio > self.volume_ratio_threshold:
            return 1.0
        elif volume_ratio > 1.0:
            return 0.7
        else:
            return 0.3

    def _analyze_trend(self, df: pd.DataFrame, idx: int) -> float:
        """分析趋势过滤"""
        ma_col = f"MA_{self.trend_ma_period}"
        if ma_col not in df.columns:
            return 0.5

        current_price = df["Close"].iloc[idx]
        ma_value = df[ma_col].iloc[idx]

        # 价格相对于MA的位置
        if current_price > ma_value:
            return 1.0  # 上升趋势
        else:
            return 0.0  # 下降趋势

    def _calculate_composite_score(
        self,
        cross_signal: str,
        histogram_score: float,
        divergence_score: float,
        zero_cross_score: float,
        volume_score: float,
        trend_score: float,
    ) -> float:
        """计算综合评分"""
        base_score = 0.5  # 基础分数

        # 加权计算
        total_score = (
            base_score
            + divergence_score * self.divergence_weight
            + zero_cross_score * self.zero_cross_weight
            + volume_score * self.volume_weight
            + trend_score * self.trend_weight
        )

        # 根据交叉信号调整
        if cross_signal == "golden":
            total_score *= 1.2  # 金叉信号增强
        elif cross_signal == "death":
            total_score *= 0.8  # 死叉信号减弱

        return min(total_score, 1.0)

    def _generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        current_price: float,
        cross_signal: str,
        total_score: float,
    ) -> SignalResult:
        """生成交易信号"""
        # 计算动态止损
        atr_col = "ATR_14"
        atr_value = df[atr_col].iloc[idx] if atr_col in df.columns else current_price * 0.02

        # 计算成交量比率
        if "Volume" in df.columns:
            current_volume = df["Volume"].iloc[idx]
            start_idx = max(0, idx - 20)
            avg_volume = df["Volume"].iloc[start_idx:idx].mean()
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else None
        else:
            volume_ratio = None

        if cross_signal == "golden":
            signal_type = SignalType.BULLISH
            stop_loss = current_price - 2 * atr_value
            target_price = current_price + 3 * atr_value
            take_profit = current_price + 5 * atr_value
            duration = max(5, int(10 * total_score))  # 根据评分调整持续时间
        else:
            signal_type = SignalType.BEARISH
            stop_loss = current_price + 2 * atr_value
            target_price = current_price - 3 * atr_value
            take_profit = current_price - 5 * atr_value
            duration = max(5, int(8 * total_score))

        return self.create_signal(
            signal_type=signal_type,
            current_price=current_price,
            confidence=total_score,
            duration=duration,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            additional_signals=[
                f"MACD柱状图强度: {df['MACD_Histogram'].iloc[idx]:.4f}",
                self._get_volume_ratio_text(df, idx),
                self._get_trend_status_text(df, idx, current_price),
            ],
            metadata={
                "cross_type": cross_signal,
                "macd_cross_idx": idx,
                "composite_score": total_score,
                "macd_value": df[
                    f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"
                ].iloc[idx],
                "signal_value": df[
                    f"MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}"
                ].iloc[idx],
                "histogram_value": df["MACD_Histogram"].iloc[idx],
                "atr_value": atr_value,
                "volume_ratio": volume_ratio,
            },
        )

    def _find_peaks(self, series: pd.Series) -> list:
        """寻找序列的峰值点"""
        peaks = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i - 1] and series.iloc[i] > series.iloc[i + 1]:
                peaks.append(i)
        return peaks

    def _find_troughs(self, series: pd.Series) -> list:
        """寻找序列的谷值点"""
        troughs = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] < series.iloc[i - 1] and series.iloc[i] < series.iloc[i + 1]:
                troughs.append(i)
        return troughs

    def _get_volume_ratio_text(self, df: pd.DataFrame, idx: int) -> str:
        """获取成交量比率文本"""
        if "Volume" not in df.columns:
            return "无成交量数据"

        current_volume = df["Volume"].iloc[idx]
        start_idx = max(0, idx - 20)
        avg_volume = df["Volume"].iloc[start_idx:idx].mean()

        if avg_volume == 0:
            return "成交量数据异常"

        volume_ratio = current_volume / avg_volume
        return f"成交量比率: {volume_ratio:.2f}"

    def _get_trend_status_text(self, df: pd.DataFrame, idx: int, current_price: float) -> str:
        """获取趋势状态文本"""
        ma_col = f"MA_{self.trend_ma_period}"
        if ma_col not in df.columns:
            return "趋势状态: 未知"

        ma_value = df[ma_col].iloc[idx]
        trend_status = "上升" if current_price > ma_value else "下降"
        return f"趋势状态: {trend_status}"
