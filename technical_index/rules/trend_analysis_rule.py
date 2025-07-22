from typing import Optional

import pandas as pd

from ..index import build_quantitative_analysis
from .base_rule import BaseRule, RuleConfig, SignalResult, SignalType


class TrendAnalysisRule(BaseRule):
    """
    专业量化趋势分析规则 - 优化版

    该规则通过多维度技术分析来判断市场趋势方向和强度，采用动态参数调整和自适应机制：

    核心改进：
    1. 动态Duration计算：基于ATR和波动率自动调整信号持续时间
    2. 多时间框架分析：结合短期、中期、长期趋势判断
    3. 市场适应性：根据市场状态自动调整参数
    4. 趋势强度量化：更精确的趋势强度计算
    5. 风险控制：集成波动率和回撤分析

    技术指标组合：
    1. 趋势指标：ADX, Aroon, Vortex, VHF, Choppiness
    2. 动量指标：RSI, MACD, ROC, TSI
    3. 波动率指标：ATR, Bollinger Bands
    4. 支撑阻力：移动平均线, 布林带

    信号生成逻辑：
    1. 市场环境判断：趋势/震荡/混合市场
    2. 趋势强度评估：多指标综合评分
    3. 趋势方向确认：多时间框架验证
    4. 风险收益评估：基于波动率和回撤
    5. 动态参数调整：根据市场状态优化

    参数说明：
        - adx_length: ADX计算周期, 默认14
        - aroon_length: Aroon计算周期, 默认25
        - vhf_length: VHF计算周期, 默认28
        - vortex_length: Vortex计算周期, 默认14
        - rsi_length: RSI计算周期, 默认14
        - macd_fast/slow/signal: MACD参数, 默认12/26/9
        - atr_length: ATR计算周期, 默认14
        - bb_length: 布林带周期, 默认20
        - short_ma: 短期移动平均线周期, 默认7
        - medium_ma: 中期移动平均线周期, 默认21
        - long_ma: 长期移动平均线周期, 默认50
        - trend_periods: 趋势判断周期, 默认10
        - volatility_lookback: 波动率回看周期, 默认20
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        # 趋势指标参数
        self.adx_length = self.parameters.get("adx_length", 14)
        self.aroon_length = self.parameters.get("aroon_length", 25)
        self.vhf_length = self.parameters.get("vhf_length", 28)
        self.vortex_length = self.parameters.get("vortex_length", 14)

        # 动量指标参数
        self.rsi_length = self.parameters.get("rsi_length", 14)
        self.macd_fast = self.parameters.get("macd_fast", 12)
        self.macd_slow = self.parameters.get("macd_slow", 26)
        self.macd_signal = self.parameters.get("macd_signal", 9)

        # 波动率指标参数
        self.atr_length = self.parameters.get("atr_length", 14)
        self.bb_length = self.parameters.get("bb_length", 20)

        # 移动平均线参数
        self.short_ma = self.parameters.get("short_ma", 7)
        self.medium_ma = self.parameters.get("medium_ma", 21)
        self.long_ma = self.parameters.get("long_ma", 50)
        self.trend_periods = self.parameters.get("trend_periods", 10)
        self.volatility_lookback = self.parameters.get("volatility_lookback", 20)

        # 计算最大所需周期
        self.max_period = max(
            self.adx_length,
            self.aroon_length,
            self.vhf_length,
            self.vortex_length,
            self.long_ma,
            self.volatility_lookback,
        )

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if len(df) < self.max_period + self.trend_periods:
            return None

        # 计算所有技术指标
        indicator_params = {
            "ma_periods": (self.short_ma, self.medium_ma, self.long_ma),
            "adx_length": self.adx_length,
            "aroon_length": self.aroon_length,
            "vhf_length": self.vhf_length,
            "vortex_length": self.vortex_length,
            "rsi_length": self.rsi_length,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "atr_length": self.atr_length,
            "bb_length": self.bb_length,
        }

        df_with_indicators = build_quantitative_analysis(
            df.copy(),
            indicators=["adx", "aroon", "vhf", "vortex", "rsi", "macd", "atr", "bbands", "sma"],
            **indicator_params,
        )

        if df_with_indicators is None:
            return None

        # 获取当前价格和指标值
        current_price = df["Close"].iloc[-1]

        # 检查必要的指标列是否存在
        required_columns = [
            f"ADX_{self.adx_length}",
            f"AROONU_{self.aroon_length}",
            f"AROOND_{self.aroon_length}",
            f"VHF_{self.vhf_length}",
            f"VTXP_{self.vortex_length}",
            f"VTXM_{self.vortex_length}",
            f"RSI_{self.rsi_length}",
            f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}",
            f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}",
            f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}",
            f"ATRr_{self.atr_length}",
            f"BBL_{self.bb_length}_2.0",
            f"BBM_{self.bb_length}_2.0",
            f"BBU_{self.bb_length}_2.0",
            f"SMA_{self.short_ma}",
            f"SMA_{self.medium_ma}",
            f"SMA_{self.long_ma}",
        ]

        for col in required_columns:
            if col not in df_with_indicators.columns:
                return None

        # 获取指标值
        adx = df_with_indicators[f"ADX_{self.adx_length}"].iloc[-1]
        aroon_up = df_with_indicators[f"AROONU_{self.aroon_length}"].iloc[-1]
        aroon_down = df_with_indicators[f"AROOND_{self.aroon_length}"].iloc[-1]
        vhf = df_with_indicators[f"VHF_{self.vhf_length}"].iloc[-1]
        vortex_plus = df_with_indicators[f"VTXP_{self.vortex_length}"].iloc[-1]
        vortex_minus = df_with_indicators[f"VTXM_{self.vortex_length}"].iloc[-1]
        rsi = df_with_indicators[f"RSI_{self.rsi_length}"].iloc[-1]
        macd = df_with_indicators[
            f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        ].iloc[-1]
        macd_hist = df_with_indicators[
            f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        ].iloc[-1]
        macd_signal = df_with_indicators[
            f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
        ].iloc[-1]
        atr = df_with_indicators[f"ATRr_{self.atr_length}"].iloc[-1]
        bb_lower = df_with_indicators[f"BBL_{self.bb_length}_2.0"].iloc[-1]
        bb_upper = df_with_indicators[f"BBU_{self.bb_length}_2.0"].iloc[-1]
        short_ma = df_with_indicators[f"SMA_{self.short_ma}"].iloc[-1]
        medium_ma = df_with_indicators[f"SMA_{self.medium_ma}"].iloc[-1]
        long_ma = df_with_indicators[f"SMA_{self.long_ma}"].iloc[-1]

        # 计算动态duration
        dynamic_duration = self._calculate_dynamic_duration(df, atr, current_price)

        # 分析市场环境
        market_environment = self._analyze_market_environment(vhf, adx, atr, current_price)

        # 分析趋势强度
        trend_strength = self._analyze_trend_strength_enhanced(adx, vhf, atr, current_price, df)

        # 分析趋势方向
        trend_direction = self._analyze_trend_direction_enhanced(
            aroon_up,
            aroon_down,
            vortex_plus,
            vortex_minus,
            short_ma,
            medium_ma,
            long_ma,
            rsi,
            macd,
            macd_hist,
            macd_signal,
            current_price,
            df,
        )

        # 分析风险收益
        risk_reward = self._analyze_risk_reward(current_price, atr, bb_lower, bb_upper, df)

        # 综合判断
        signal_type, confidence = self._generate_signal_enhanced(
            trend_strength, trend_direction, risk_reward, market_environment
        )

        if signal_type is None:
            return None

        # 生成信号
        result = self._create_comprehensive_signal_enhanced(
            signal_type,
            current_price,
            confidence,
            trend_strength,
            trend_direction,
            risk_reward,
            market_environment,
            dynamic_duration,
            {
                "adx": adx,
                "aroon_up": aroon_up,
                "aroon_down": aroon_down,
                "vhf": vhf,
                "vortex_plus": vortex_plus,
                "vortex_minus": vortex_minus,
                "rsi": rsi,
                "macd": macd,
                "macd_hist": macd_hist,
                "atr": atr,
                "bb_lower": bb_lower,
                "bb_upper": bb_upper,
                "short_ma": short_ma,
                "medium_ma": medium_ma,
                "long_ma": long_ma,
            },
        )

        return result

    def _calculate_dynamic_duration(
        self, df: pd.DataFrame, atr: float, current_price: float
    ) -> int:
        """
        动态计算信号持续时间

        基于以下因素：
        1. ATR波动率：波动率越大，duration越短
        2. 价格位置：相对于布林带的位置
        3. 历史波动率：与历史波动率比较
        """
        # 计算历史波动率
        returns = df["Close"].pct_change().dropna()
        if len(returns) >= self.volatility_lookback:
            historical_vol = returns.tail(self.volatility_lookback).std()
            current_vol = atr / current_price

            # 波动率比率
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

            # 基础duration
            base_duration = 12

            # 根据波动率调整
            if vol_ratio > 1.5:  # 高波动率
                duration = max(6, int(base_duration * 0.5))
            elif vol_ratio > 1.2:  # 中等波动率
                duration = max(8, int(base_duration * 0.7))
            elif vol_ratio < 0.8:  # 低波动率
                duration = min(24, int(base_duration * 1.5))
            else:  # 正常波动率
                duration = base_duration

            return duration
        else:
            return 12

    def _analyze_market_environment(
        self, vhf: float, adx: float, atr: float, current_price: float
    ) -> dict:
        """
        分析市场环境

        市场环境分类：
        1. Trending: 强趋势市场
        2. Choppy: 震荡市场
        3. Mixed: 混合市场
        4. Volatile: 高波动市场
        """
        # VHF分析
        if vhf > 0.5:
            vhf_env = "trending"
            vhf_score = min((vhf - 0.5) / 0.5, 1.0)
        elif vhf > 0.3:
            vhf_env = "mixed"
            vhf_score = (vhf - 0.3) / 0.2
        else:
            vhf_env = "choppy"
            vhf_score = 0.0

        # ADX分析
        if adx > 25:
            adx_env = "strong_trend"
            adx_score = min((adx - 25) / 25, 1.0)
        elif adx > 20:
            adx_env = "medium_trend"
            adx_score = (adx - 20) / 5
        else:
            adx_env = "weak_trend"
            adx_score = 0.0

        # 波动率分析
        vol_ratio = atr / current_price
        if vol_ratio > 0.03:  # 3%以上波动率
            vol_env = "high_volatility"
            vol_score = min((vol_ratio - 0.03) / 0.02, 1.0)
        elif vol_ratio > 0.02:  # 2-3%波动率
            vol_env = "medium_volatility"
            vol_score = (vol_ratio - 0.02) / 0.01
        else:
            vol_env = "low_volatility"
            vol_score = 0.0

        # 综合环境判断
        trend_score = (vhf_score + adx_score) / 2
        if trend_score > 0.6:
            environment = "trending"
        elif trend_score < 0.3:
            environment = "choppy"
        else:
            environment = "mixed"

        return {
            "environment": environment,
            "trend_score": trend_score,
            "volatility_score": vol_score,
            "vhf_environment": vhf_env,
            "adx_environment": adx_env,
            "volatility_environment": vol_env,
        }

    def _analyze_trend_strength_enhanced(
        self, adx: float, vhf: float, atr: float, current_price: float, df: pd.DataFrame
    ) -> dict:
        """
        增强版趋势强度分析

        考虑因素：
        1. ADX趋势强度
        2. VHF趋势/震荡判断
        3. 价格动量
        4. 趋势一致性
        5. 波动率稳定性
        """
        # ADX趋势强度分析
        if adx > 30:
            adx_strength = "very_strong"
            adx_score = min((adx - 30) / 20, 1.0)
        elif adx > 25:
            adx_strength = "strong"
            adx_score = (adx - 25) / 5
        elif adx > 20:
            adx_strength = "medium"
            adx_score = (adx - 20) / 5
        else:
            adx_strength = "weak"
            adx_score = 0.0

        # VHF趋势分析
        if vhf > 0.6:
            vhf_strength = "strong_trend"
            vhf_score = min((vhf - 0.6) / 0.4, 1.0)
        elif vhf > 0.5:
            vhf_strength = "trend"
            vhf_score = (vhf - 0.5) / 0.1
        elif vhf > 0.3:
            vhf_strength = "mixed"
            vhf_score = (vhf - 0.3) / 0.2
        else:
            vhf_strength = "choppy"
            vhf_score = 0.0

        # 价格动量分析
        price_momentum = self._calculate_price_momentum(df)

        # 趋势一致性分析
        trend_consistency = self._calculate_trend_consistency(df)

        # 波动率稳定性分析
        volatility_stability = self._calculate_volatility_stability(df, atr)

        # 综合趋势强度
        trend_score = (
            adx_score * 0.3
            + vhf_score * 0.25
            + price_momentum * 0.2
            + trend_consistency * 0.15
            + volatility_stability * 0.1
        )

        return {
            "overall_strength": trend_score,
            "adx_strength": adx_strength,
            "adx_score": adx_score,
            "vhf_strength": vhf_strength,
            "vhf_score": vhf_score,
            "price_momentum": price_momentum,
            "trend_consistency": trend_consistency,
            "volatility_stability": volatility_stability,
        }

    def _analyze_trend_direction_enhanced(
        self,
        aroon_up: float,
        aroon_down: float,
        vortex_plus: float,
        vortex_minus: float,
        short_ma: float,
        medium_ma: float,
        long_ma: float,
        rsi: float,
        macd: float,
        macd_hist: float,
        macd_signal: float,
        current_price: float,
        df: pd.DataFrame,
    ) -> dict:
        """
        增强版趋势方向分析

        多时间框架分析：
        1. 短期趋势：短期MA, RSI, MACD
        2. 中期趋势：中期MA, Aroon, Vortex
        3. 长期趋势：长期MA, 价格结构
        """
        # 短期趋势分析
        short_trend = self._analyze_short_term_trend(
            short_ma, medium_ma, rsi, macd, macd_hist, macd_signal, current_price
        )

        # 中期趋势分析
        medium_trend = self._analyze_medium_term_trend(
            aroon_up, aroon_down, vortex_plus, vortex_minus, medium_ma, long_ma
        )

        # 长期趋势分析
        long_trend = self._analyze_long_term_trend(long_ma, current_price, df)

        # 综合方向判断
        bullish_signals = sum(
            [short_trend["bullish"], medium_trend["bullish"], long_trend["bullish"]]
        )
        bearish_signals = sum(
            [short_trend["bearish"], medium_trend["bearish"], long_trend["bearish"]]
        )

        if bullish_signals >= 2:
            direction = "bullish"
            direction_score = (
                (bullish_signals / 3)
                * (short_trend["strength"] + medium_trend["strength"] + long_trend["strength"])
                / 3
            )
        elif bearish_signals >= 2:
            direction = "bearish"
            direction_score = (
                (bearish_signals / 3)
                * (short_trend["strength"] + medium_trend["strength"] + long_trend["strength"])
                / 3
            )
        else:
            direction = "sideways"
            direction_score = 0.0

        return {
            "direction": direction,
            "direction_score": direction_score,
            "short_term": short_trend,
            "medium_term": medium_trend,
            "long_term": long_trend,
        }

    def _analyze_risk_reward(
        self, current_price: float, atr: float, bb_lower: float, bb_upper: float, df: pd.DataFrame
    ) -> dict:
        """
        风险收益分析

        考虑因素：
        1. 当前价格位置
        2. 支撑阻力位
        3. 波动率风险
        4. 历史回撤
        """
        # 价格位置分析
        bb_position = (
            (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        )

        # 支撑阻力分析
        support_resistance = self._calculate_support_resistance(current_price, df)

        # 波动率风险
        volatility_risk = atr / current_price

        # 历史回撤分析
        drawdown_risk = self._calculate_drawdown_risk(df)

        # 风险收益比
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            current_price, atr, support_resistance
        )

        return {
            "bb_position": bb_position,
            "support_resistance": support_resistance,
            "volatility_risk": volatility_risk,
            "drawdown_risk": drawdown_risk,
            "risk_reward_ratio": risk_reward_ratio,
        }

    def _generate_signal_enhanced(
        self,
        trend_strength: dict,
        trend_direction: dict,
        risk_reward: dict,
        market_environment: dict,
    ) -> tuple:
        """
        增强版信号生成

        信号生成条件：
        1. 趋势信号：趋势强度 > 0.4 且 方向强度 > 0.6
        2. 强趋势信号：趋势强度 > 0.6 且 方向强度 > 0.7
        3. 震荡信号：市场环境为震荡且震荡强度 > 0.7
        """
        trend_score = trend_strength["overall_strength"]
        direction = trend_direction["direction"]
        direction_score = trend_direction["direction_score"]
        environment = market_environment["environment"]
        risk_ratio = risk_reward["risk_reward_ratio"]

        # 强趋势信号
        if trend_score > 0.6 and direction_score > 0.7 and risk_ratio > 1.5:
            if direction == "bullish":
                return SignalType.BULLISH, min(trend_score * direction_score, 0.95)
            else:
                return SignalType.BEARISH, min(trend_score * direction_score, 0.95)

        # 中等趋势信号
        elif trend_score > 0.4 and direction_score > 0.6 and risk_ratio > 1.2:
            if direction == "bullish":
                return SignalType.BULLISH, min(trend_score * direction_score * 0.8, 0.8)
            else:
                return SignalType.BEARISH, min(trend_score * direction_score * 0.8, 0.8)

        # 震荡市场信号
        elif environment == "choppy" and trend_score < 0.3:
            return SignalType.NEUTRAL, min(0.7, 1.0 - trend_score)

        # 无明确信号
        else:
            return None, 0.0

    def _create_comprehensive_signal_enhanced(
        self,
        signal_type: SignalType,
        current_price: float,
        confidence: float,
        trend_strength: dict,
        trend_direction: dict,
        risk_reward: dict,
        market_environment: dict,
        duration: int,
        raw_indicators: dict,
    ) -> Optional[SignalResult]:
        """创建增强版综合信号"""
        # 根据信号类型和市场环境设置目标价格和止损
        atr = raw_indicators["atr"]

        if signal_type == SignalType.BULLISH:
            # 看涨信号：基于ATR和风险收益比设置目标
            target_price = current_price + (atr * 2.0)  # 2倍ATR目标
            stop_loss = current_price - (atr * 1.5)  # 1.5倍ATR止损
            take_profit = current_price + (atr * 3.0)  # 3倍ATR止盈
            additional_signals = [
                f"趋势强度: {trend_strength['overall_strength']:.2f}",
                f"方向强度: {trend_direction['direction_score']:.2f}",
                f"风险收益比: {risk_reward['risk_reward_ratio']:.2f}",
                "关注成交量配合",
                "观察RSI确认",
                "注意回调风险",
            ]
        elif signal_type == SignalType.BEARISH:
            # 看跌信号
            target_price = current_price - (atr * 2.0)
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)
            additional_signals = [
                f"趋势强度: {trend_strength['overall_strength']:.2f}",
                f"方向强度: {trend_direction['direction_score']:.2f}",
                f"风险收益比: {risk_reward['risk_reward_ratio']:.2f}",
                "关注成交量配合",
                "观察RSI确认",
                "注意反弹风险",
            ]
        elif signal_type == SignalType.NEUTRAL:
            # 震荡市场信号
            target_price = current_price
            stop_loss = current_price - (atr * 1.0)
            take_profit = current_price + (atr * 1.0)
            additional_signals = [
                f"市场环境: {market_environment['environment']}",
                f"震荡强度: {1.0 - trend_strength['overall_strength']:.2f}",
                "建议区间交易",
                "关注突破方向",
                "控制仓位",
            ]
        else:
            target_price = current_price
            stop_loss = current_price - (atr * 1.0)
            take_profit = current_price + (atr * 1.0)
            additional_signals = ["市场震荡，建议观望", "等待明确信号", "关注突破方向"]

        # 构建元数据
        metadata = {
            "trend_direction": trend_direction["direction"],
            "trend_strength": trend_strength["overall_strength"],
            "market_environment": market_environment["environment"],
            "risk_reward_ratio": risk_reward["risk_reward_ratio"],
            "dynamic_duration": duration,
        }

        return self.create_signal(
            signal_type=signal_type,
            current_price=current_price,
            confidence=confidence,
            duration=duration,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            additional_signals=additional_signals,
            metadata=metadata,
        )

    # 辅助计算方法
    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        """计算价格动量"""
        if len(df) < 10:
            return 0.0

        # 计算最近10个周期的价格变化
        recent_prices = df["Close"].tail(10)
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

        # 归一化到0-1
        return min(max(abs(price_change) * 10, 0.0), 1.0)

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """计算趋势一致性"""
        if len(df) < 20:
            return 0.0

        # 计算最近20个周期的趋势一致性
        recent_prices = df["Close"].tail(20)
        up_days = sum(
            1
            for i in range(1, len(recent_prices))
            if recent_prices.iloc[i] > recent_prices.iloc[i - 1]
        )
        consistency = up_days / (len(recent_prices) - 1)

        # 转换为0-1的强度
        return abs(consistency - 0.5) * 2

    def _calculate_volatility_stability(self, df: pd.DataFrame, current_atr: float) -> float:
        """计算波动率稳定性"""
        if len(df) < 20:
            return 0.5

        # 计算历史ATR的稳定性
        returns = df["Close"].pct_change().dropna().tail(20)
        historical_vol = returns.std()
        current_vol = current_atr / df["Close"].iloc[-1]

        # 波动率稳定性：越稳定分数越高
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        if 0.8 <= vol_ratio <= 1.2:
            return 1.0
        elif 0.6 <= vol_ratio <= 1.4:
            return 0.7
        else:
            return 0.3

    def _analyze_short_term_trend(
        self,
        short_ma: float,
        medium_ma: float,
        rsi: float,
        macd: float,
        macd_hist: float,
        macd_signal: float,
        current_price: float,
    ) -> dict:
        """分析短期趋势"""
        bullish_signals = 0
        bearish_signals = 0
        strength = 0.0

        # MA分析
        if current_price > short_ma > medium_ma:
            bullish_signals += 1
            strength += 0.3
        elif current_price < short_ma < medium_ma:
            bearish_signals += 1
            strength += 0.3

        # RSI分析
        if 40 < rsi < 70:
            if rsi > 55:
                bullish_signals += 1
                strength += 0.2
            else:
                bearish_signals += 1
                strength += 0.2

        # MACD分析
        if macd > macd_signal and macd_hist > 0:
            bullish_signals += 1
            strength += 0.2
        elif macd < macd_signal and macd_hist < 0:
            bearish_signals += 1
            strength += 0.2

        return {
            "bullish": bullish_signals > bearish_signals,
            "bearish": bearish_signals > bullish_signals,
            "strength": min(strength, 1.0),
        }

    def _analyze_medium_term_trend(
        self,
        aroon_up: float,
        aroon_down: float,
        vortex_plus: float,
        vortex_minus: float,
        medium_ma: float,
        long_ma: float,
    ) -> dict:
        """分析中期趋势"""
        bullish_signals = 0
        bearish_signals = 0
        strength = 0.0

        # Aroon分析
        if aroon_up > aroon_down:
            bullish_signals += 1
            strength += abs(aroon_up - aroon_down) / 100
        else:
            bearish_signals += 1
            strength += abs(aroon_down - aroon_up) / 100

        # Vortex分析
        if vortex_plus > vortex_minus:
            bullish_signals += 1
            strength += (
                abs(vortex_plus - vortex_minus) / max(vortex_plus, vortex_minus)
                if max(vortex_plus, vortex_minus) > 0
                else 0
            )
        else:
            bearish_signals += 1
            strength += (
                abs(vortex_minus - vortex_plus) / max(vortex_plus, vortex_minus)
                if max(vortex_plus, vortex_minus) > 0
                else 0
            )

        # MA分析
        if medium_ma > long_ma:
            bullish_signals += 1
            strength += abs(medium_ma - long_ma) / long_ma
        else:
            bearish_signals += 1
            strength += abs(long_ma - medium_ma) / long_ma

        return {
            "bullish": bullish_signals > bearish_signals,
            "bearish": bearish_signals > bullish_signals,
            "strength": min(strength, 1.0),
        }

    def _analyze_long_term_trend(
        self, long_ma: float, current_price: float, df: pd.DataFrame
    ) -> dict:
        """分析长期趋势"""
        if len(df) < 50:
            return {"bullish": False, "bearish": False, "strength": 0.0}

        # 长期价格结构分析
        long_term_prices = df["Close"].tail(50)
        price_trend = (
            long_term_prices.iloc[-1] - long_term_prices.iloc[0]
        ) / long_term_prices.iloc[0]

        # 相对于长期MA的位置
        ma_position = (current_price - long_ma) / long_ma

        if price_trend > 0.1 and ma_position > 0.05:  # 10%以上涨幅且高于MA 5%
            return {"bullish": True, "bearish": False, "strength": min(abs(price_trend), 1.0)}
        elif price_trend < -0.1 and ma_position < -0.05:  # 10%以上跌幅且低于MA 5%
            return {"bullish": False, "bearish": True, "strength": min(abs(price_trend), 1.0)}
        else:
            return {"bullish": False, "bearish": False, "strength": 0.0}

    def _calculate_support_resistance(self, current_price: float, df: pd.DataFrame) -> dict:
        """计算支撑阻力位"""
        if len(df) < 20:
            return {"support": current_price * 0.95, "resistance": current_price * 1.05}

        recent_data = df.tail(20)
        recent_highs = recent_data["High"].max()
        recent_lows = recent_data["Low"].min()

        resistance = recent_highs
        support = recent_lows

        return {
            "support": support,
            "resistance": resistance,
            "support_distance": (current_price - support) / current_price,
            "resistance_distance": (resistance - current_price) / current_price,
        }

    def _calculate_drawdown_risk(self, df: pd.DataFrame) -> float:
        """计算回撤风险"""
        if len(df) < 20:
            return 0.5

        # 计算最近20个周期的最大回撤
        recent_prices = df["Close"].tail(20)
        peak = recent_prices.expanding().max()
        drawdown = (recent_prices - peak) / peak

        max_drawdown = abs(drawdown.min())
        return min(max_drawdown * 5, 1.0)  # 放大5倍并限制在0-1

    def _calculate_risk_reward_ratio(
        self, current_price: float, atr: float, support_resistance: dict
    ) -> float:
        """计算风险收益比"""
        support_distance = support_resistance["support_distance"]
        resistance_distance = support_resistance["resistance_distance"]

        # 基于ATR的风险
        atr_risk = atr / current_price

        # 基于支撑阻力的收益潜力
        potential_reward = max(support_distance, resistance_distance)

        if atr_risk > 0:
            return potential_reward / atr_risk
        else:
            return 1.0
