#!/usr/bin/env python3
"""
技术指标计算模块 - 重构版
提供高性能、精确的技术分析指标计算功能
"""

import logging
import warnings
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import pandas_ta as ta

# 过滤pandas警告
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 指标映射配置
INDICATOR_MAPPING = {
    # 动量指标
    "macd": {"function": "macd", "params": ["macd_fast", "macd_slow", "macd_signal"]},
    "rsi": {"function": "rsi", "params": ["rsi_length"]},
    "stoch": {"function": "stoch", "params": ["stoch_k", "stoch_d", "stoch_smooth_k"]},
    "willr": {"function": "willr", "params": ["willr_length"]},
    "cci": {"function": "cci", "params": ["cci_length"]},
    "roc": {"function": "roc", "params": ["roc_length"]},
    "mom": {"function": "mom", "params": ["mom_length"]},
    "trix": {"function": "trix", "params": ["trix_length"]},
    "tsi": {"function": "tsi", "params": ["tsi_fast", "tsi_slow"]},
    "kdj": {"function": "kdj", "params": ["kdj_length"]},
    "fisher": {"function": "fisher", "params": ["fisher_length"]},
    "coppock": {"function": "coppock", "params": ["coppock_length"]},
    "uo": {"function": "uo", "params": ["uo_length1", "uo_length2", "uo_length3"]},
    # 重叠指标
    "sma": {"function": "sma", "params": ["ma_periods"], "custom": True},
    "ema": {"function": "ema", "params": ["ma_periods"], "custom": True},
    "dema": {"function": "dema", "params": ["dema_length"]},
    "tema": {"function": "tema", "params": ["tema_length"]},
    "hma": {"function": "hma", "params": ["hma_length"]},
    "wma": {"function": "wma", "params": ["wma_length"]},
    "kama": {"function": "kama", "params": ["kama_length", "kama_pow1", "kama_pow2"]},
    "vwap": {"function": "vwap", "params": []},
    "ichimoku": {"function": "ichimoku", "params": []},
    "supertrend": {
        "function": "supertrend",
        "params": ["supertrend_length", "supertrend_multiplier"],
    },
    # 趋势指标
    "adx": {"function": "adx", "params": ["adx_length"]},
    "aroon": {"function": "aroon", "params": ["aroon_length"]},
    "psar": {"function": "psar", "params": ["psar_af0", "psar_af", "psar_max_af"]},
    "vortex": {"function": "vortex", "params": ["vortex_length"]},
    "vhf": {"function": "vhf", "params": ["vhf_length"]},
    "chop": {"function": "chop", "params": ["chop_length"]},
    "ttm_trend": {"function": "ttm_trend", "params": ["ttm_length"]},
    # 波动率指标
    "bbands": {"function": "bbands", "params": ["bb_length", "bb_std"]},
    "atr": {"function": "atr", "params": ["atr_length"]},
    "natr": {"function": "natr", "params": ["natr_length"]},
    "keltner": {"function": "keltner", "params": ["kc_length", "kc_std"]},
    "donchian": {"function": "donchian", "params": ["dc_length"]},
    "massi": {"function": "massi", "params": ["mi_length"]},
    "ui": {"function": "ui", "params": ["ui_length"]},
    # 成交量指标
    "obv": {"function": "obv", "params": []},
    "mfi": {"function": "mfi", "params": ["mfi_length"]},
    "ad": {"function": "ad", "params": []},
    "adosc": {"function": "adosc", "params": ["adosc_fast", "adosc_slow"]},
    "cmf": {"function": "cmf", "params": ["cmf_length"]},
    "eom": {"function": "eom", "params": ["eom_length"]},
    "pvi": {"function": "pvi", "params": []},
    "nvi": {"function": "nvi", "params": []},
    # 统计指标
    "zscore": {"function": "zscore", "params": ["zscore_length"]},
    "kurtosis": {"function": "kurtosis", "params": ["kurtosis_length"]},
    "skew": {"function": "skew", "params": ["skew_length"]},
    "variance": {"function": "variance", "params": ["variance_length"]},
    "entropy": {"function": "entropy", "params": ["entropy_length"]},
    "quantile": {"function": "quantile", "params": ["quantile_length", "quantile_q"]},
    # K线模式
    "cdl_pattern": {"function": "cdl_pattern", "params": ["cdl_pattern_name"]},
    "cdl_doji": {"function": "cdl_doji", "params": []},
    "cdl_hammer": {"function": "cdl_hammer", "params": []},
    "cdl_shooting_star": {"function": "cdl_shooting_star", "params": []},
    "cdl_engulfing": {"function": "cdl_engulfing", "params": []},
    "cdl_harami": {"function": "cdl_harami", "params": []},
    "cdl_marubozu": {"function": "cdl_marubozu", "params": []},
    "cdl_morning_star": {"function": "cdl_morning_star", "params": []},
    "cdl_evening_star": {"function": "cdl_evening_star", "params": []},
    "cdl_three_white_soldiers": {"function": "cdl_three_white_soldiers", "params": []},
    "cdl_three_black_crows": {"function": "cdl_three_black_crows", "params": []},
}


def build_quantitative_analysis(
    df: pd.DataFrame, indicators: Union[str, List[str]], **kwargs
) -> Optional[pd.DataFrame]:
    """
    高性能技术指标计算函数

    Args:
        df: 包含OHLCV数据的DataFrame
        indicators: 要计算的指标列表，支持具体指标名称
        **kwargs: 技术指标参数

    Returns:
        添加了技术指标的DataFrame，如果计算失败返回None

    Examples:
        # 计算单个指标
        df = build_quantitative_analysis(df, "rsi", rsi_length=14)

        # 计算多个指标
        df = build_quantitative_analysis(df, ["rsi", "macd", "atr"],
                                       rsi_length=14, macd_fast=12, macd_slow=26, atr_length=14)

        # 计算移动平均线
        df = build_quantitative_analysis(df, ["sma", "ema"], ma_periods=(7, 21, 50))
    """
    if df is None or df.empty:
        logger.warning("输入数据为空")
        return None

    # 标准化指标列表
    if isinstance(indicators, str):
        indicators = [indicators]

    if not indicators:
        logger.warning("未指定要计算的指标")
        return df

    result_df = df.copy()
    try:
        # 计算移动平均线（特殊处理）
        if "sma" in indicators or "ema" in indicators:
            result_df = _calculate_moving_averages(result_df, indicators, **kwargs)

        # 计算其他指标
        for indicator in indicators:
            if indicator in ["sma", "ema"]:
                continue

            if indicator in INDICATOR_MAPPING:
                result_df = _calculate_single_indicator(result_df, indicator, **kwargs)
            else:
                logger.warning(f"未知指标: {indicator}")

        return result_df

    except Exception as e:
        logger.error(f"计算技术指标时出错: {e}")
        return None


def _calculate_moving_averages(df: pd.DataFrame, indicators: List[str], **kwargs) -> pd.DataFrame:
    """计算移动平均线"""
    ma_periods = kwargs.get("ma_periods", (7, 25, 99))

    if not isinstance(ma_periods, (list, tuple)):
        ma_periods = [ma_periods]

    for period in ma_periods:
        if "sma" in indicators:
            df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()

        if "ema" in indicators:
            df[f"EMA_{period}"] = ta.ema(df["Close"], length=period)

    return df


def _calculate_single_indicator(df: pd.DataFrame, indicator: str, **kwargs) -> pd.DataFrame:
    """计算单个技术指标"""
    if indicator not in INDICATOR_MAPPING:
        logger.warning(f"未知指标: {indicator}")
        return df

    config = INDICATOR_MAPPING[indicator]
    function_name = config["function"]
    params = config["params"]

    try:
        # 获取参数值
        param_values = {}
        for param in params:
            if param in kwargs:
                param_values[param] = kwargs[param]
            else:
                # 使用默认值
                param_values[param] = _get_default_param_value(param)

        # 执行指标计算
        if function_name == "macd":
            df.ta.macd(
                fast=param_values.get("macd_fast", 12),
                slow=param_values.get("macd_slow", 26),
                signal=param_values.get("macd_signal", 9),
                append=True,
            )

        elif function_name == "rsi":
            df.ta.rsi(length=param_values.get("rsi_length", 14), append=True)

        elif function_name == "stoch":
            df.ta.stoch(
                k=param_values.get("stoch_k", 14),
                d=param_values.get("stoch_d", 3),
                smooth_k=param_values.get("stoch_smooth_k", 3),
                append=True,
            )

        elif function_name == "willr":
            df.ta.willr(length=param_values.get("willr_length", 14), append=True)

        elif function_name == "cci":
            df.ta.cci(length=param_values.get("cci_length", 20), append=True)

        elif function_name == "roc":
            df.ta.roc(length=param_values.get("roc_length", 10), append=True)

        elif function_name == "mom":
            df.ta.mom(length=param_values.get("mom_length", 10), append=True)

        elif function_name == "trix":
            df.ta.trix(length=param_values.get("trix_length", 18), append=True)

        elif function_name == "tsi":
            df.ta.tsi(
                fast=param_values.get("tsi_fast", 13),
                slow=param_values.get("tsi_slow", 25),
                append=True,
            )

        elif function_name == "kdj":
            df.ta.kdj(length=param_values.get("kdj_length", 9), append=True)

        elif function_name == "fisher":
            df.ta.fisher(length=param_values.get("fisher_length", 9), append=True)

        elif function_name == "coppock":
            df.ta.coppock(length=param_values.get("coppock_length", 10), append=True)

        elif function_name == "uo":
            df.ta.uo(
                length1=param_values.get("uo_length1", 7),
                length2=param_values.get("uo_length2", 14),
                length3=param_values.get("uo_length3", 28),
                append=True,
            )

        elif function_name == "dema":
            df.ta.dema(length=param_values.get("dema_length", 20), append=True)

        elif function_name == "tema":
            df.ta.tema(length=param_values.get("tema_length", 20), append=True)

        elif function_name == "hma":
            df.ta.hma(length=param_values.get("hma_length", 20), append=True)

        elif function_name == "wma":
            df.ta.wma(length=param_values.get("wma_length", 20), append=True)

        elif function_name == "kama":
            df.ta.kama(
                length=param_values.get("kama_length", 10),
                pow1=param_values.get("kama_pow1", 2),
                pow2=param_values.get("kama_pow2", 30),
                append=True,
            )

        elif function_name == "vwap":
            df.ta.vwap(append=True)

        elif function_name == "ichimoku":
            df.ta.ichimoku(append=True)

        elif function_name == "supertrend":
            df.ta.supertrend(
                length=param_values.get("supertrend_length", 7),
                multiplier=param_values.get("supertrend_multiplier", 3.0),
                append=True,
            )

        elif function_name == "adx":
            df.ta.adx(length=param_values.get("adx_length", 14), append=True)

        elif function_name == "aroon":
            df.ta.aroon(length=param_values.get("aroon_length", 25), append=True)

        elif function_name == "psar":
            df.ta.psar(
                af0=param_values.get("psar_af0", 0.02),
                af=param_values.get("psar_af", 0.02),
                max_af=param_values.get("psar_max_af", 0.2),
                append=True,
            )

        elif function_name == "vortex":
            df.ta.vortex(length=param_values.get("vortex_length", 14), append=True)

        elif function_name == "vhf":
            df.ta.vhf(length=param_values.get("vhf_length", 28), append=True)

        elif function_name == "chop":
            df.ta.chop(length=param_values.get("chop_length", 14), append=True)

        elif function_name == "ttm_trend":
            df.ta.ttm_trend(length=param_values.get("ttm_length", 5), append=True)

        elif function_name == "bbands":
            df.ta.bbands(
                length=param_values.get("bb_length", 20),
                std=param_values.get("bb_std", 2),
                append=True,
            )

        elif function_name == "atr":
            df.ta.atr(length=param_values.get("atr_length", 14), append=True)

        elif function_name == "natr":
            df.ta.natr(length=param_values.get("natr_length", 14), append=True)

        elif function_name == "keltner":
            df.ta.kc(
                length=param_values.get("kc_length", 20),
                std=param_values.get("kc_std", 2),
                append=True,
            )

        elif function_name == "donchian":
            df.ta.donchian(length=param_values.get("dc_length", 20), append=True)

        elif function_name == "massi":
            df.ta.massi(length=param_values.get("mi_length", 9), append=True)

        elif function_name == "ui":
            df.ta.ui(length=param_values.get("ui_length", 14), append=True)

        elif function_name == "obv":
            df.ta.obv(append=True)

        elif function_name == "mfi":
            df.ta.mfi(length=param_values.get("mfi_length", 14), append=True)

        elif function_name == "ad":
            df.ta.ad(append=True)

        elif function_name == "adosc":
            df.ta.adosc(
                fast=param_values.get("adosc_fast", 3),
                slow=param_values.get("adosc_slow", 10),
                append=True,
            )

        elif function_name == "cmf":
            df.ta.cmf(length=param_values.get("cmf_length", 20), append=True)

        elif function_name == "eom":
            df.ta.eom(length=param_values.get("eom_length", 14), append=True)

        elif function_name == "pvi":
            df.ta.pvi(append=True)

        elif function_name == "nvi":
            df.ta.nvi(append=True)

        elif function_name == "zscore":
            df.ta.zscore(length=param_values.get("zscore_length", 20), append=True)

        elif function_name == "kurtosis":
            df.ta.kurtosis(length=param_values.get("kurtosis_length", 20), append=True)

        elif function_name == "skew":
            df.ta.skew(length=param_values.get("skew_length", 20), append=True)

        elif function_name == "variance":
            df.ta.variance(length=param_values.get("variance_length", 20), append=True)

        elif function_name == "entropy":
            df.ta.entropy(length=param_values.get("entropy_length", 20), append=True)

        elif function_name == "quantile":
            df.ta.quantile(
                length=param_values.get("quantile_length", 20),
                q=param_values.get("quantile_q", 0.5),
                append=True,
            )

        elif function_name.startswith("cdl_"):
            _calculate_candlestick_pattern(df, function_name)

        else:
            logger.warning(f"未实现的指标: {indicator}")

        return df

    except Exception as e:
        logger.error(f"计算指标 {indicator} 时出错: {e}")
        return df


def _get_default_param_value(param: str) -> Any:
    """获取参数默认值"""
    defaults = {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_length": 14,
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_smooth_k": 3,
        "willr_length": 14,
        "cci_length": 20,
        "roc_length": 10,
        "mom_length": 10,
        "trix_length": 18,
        "tsi_fast": 13,
        "tsi_slow": 25,
        "kdj_length": 9,
        "fisher_length": 9,
        "coppock_length": 10,
        "uo_length1": 7,
        "uo_length2": 14,
        "uo_length3": 28,
        "dema_length": 20,
        "tema_length": 20,
        "hma_length": 20,
        "wma_length": 20,
        "kama_length": 10,
        "kama_pow1": 2,
        "kama_pow2": 30,
        "supertrend_length": 7,
        "supertrend_multiplier": 3.0,
        "adx_length": 14,
        "aroon_length": 25,
        "psar_af0": 0.02,
        "psar_af": 0.02,
        "psar_max_af": 0.2,
        "vortex_length": 14,
        "vhf_length": 28,
        "chop_length": 14,
        "ttm_length": 5,
        "bb_length": 20,
        "bb_std": 2,
        "atr_length": 14,
        "natr_length": 14,
        "kc_length": 20,
        "kc_std": 2,
        "dc_length": 20,
        "mi_length": 9,
        "ui_length": 14,
        "mfi_length": 14,
        "adosc_fast": 3,
        "adosc_slow": 10,
        "cmf_length": 20,
        "eom_length": 14,
        "zscore_length": 20,
        "kurtosis_length": 20,
        "skew_length": 20,
        "variance_length": 20,
        "entropy_length": 20,
        "quantile_length": 20,
        "quantile_q": 0.5,
    }

    return defaults.get(param, None)


def _calculate_candlestick_pattern(df: pd.DataFrame, pattern_name: str) -> None:
    """计算K线模式"""
    try:
        if pattern_name == "cdl_doji":
            df.ta.cdl_doji(append=True)
        elif pattern_name == "cdl_hammer":
            df.ta.cdl_hammer(append=True)
        elif pattern_name == "cdl_shooting_star":
            df.ta.cdl_shooting_star(append=True)
        elif pattern_name == "cdl_engulfing":
            df.ta.cdl_engulfing(append=True)
        elif pattern_name == "cdl_harami":
            df.ta.cdl_harami(append=True)
        elif pattern_name == "cdl_marubozu":
            df.ta.cdl_marubozu(append=True)
        elif pattern_name == "cdl_morning_star":
            df.ta.cdl_morning_star(append=True)
        elif pattern_name == "cdl_evening_star":
            df.ta.cdl_evening_star(append=True)
        elif pattern_name == "cdl_three_white_soldiers":
            df.ta.cdl_three_white_soldiers(append=True)
        elif pattern_name == "cdl_three_black_crows":
            df.ta.cdl_three_black_crows(append=True)
        else:
            logger.warning(f"未实现的K线模式: {pattern_name}")
    except Exception as e:
        logger.error(f"计算K线模式 {pattern_name} 时出错: {e}")


def get_available_indicators() -> Dict[str, List[str]]:
    """
    获取所有可用的技术指标

    Returns:
        按类型分组的指标字典
    """
    return {
        "momentum": [
            "macd",
            "rsi",
            "stoch",
            "willr",
            "cci",
            "roc",
            "mom",
            "trix",
            "tsi",
            "kdj",
            "fisher",
            "coppock",
            "uo",
        ],
        "overlap": [
            "sma",
            "ema",
            "dema",
            "tema",
            "hma",
            "wma",
            "kama",
            "vwap",
            "ichimoku",
            "supertrend",
        ],
        "trend": ["adx", "aroon", "psar", "vortex", "vhf", "chop", "ttm_trend"],
        "volatility": ["bbands", "atr", "natr", "keltner", "donchian", "massi", "ui"],
        "volume": ["obv", "mfi", "ad", "adosc", "cmf", "eom", "pvi", "nvi"],
        "statistics": ["zscore", "kurtosis", "skew", "variance", "entropy", "quantile"],
        "candlestick": [
            "cdl_doji",
            "cdl_hammer",
            "cdl_shooting_star",
            "cdl_engulfing",
            "cdl_harami",
            "cdl_marubozu",
            "cdl_morning_star",
            "cdl_evening_star",
            "cdl_three_white_soldiers",
            "cdl_three_black_crows",
        ],
    }


def get_indicator_info(indicator: str) -> Optional[Dict[str, Any]]:
    """
    获取指标信息

    Args:
        indicator: 指标名称

    Returns:
        指标信息字典，包含参数和描述
    """
    if indicator in INDICATOR_MAPPING:
        config = INDICATOR_MAPPING[indicator]
        return {
            "name": indicator,
            "function": config["function"],
            "parameters": config["params"],
            "description": _get_indicator_description(indicator),
        }
    return None


def _get_indicator_description(indicator: str) -> str:
    """获取指标描述"""
    descriptions = {
        "macd": "移动平均收敛发散 - 趋势跟踪动量指标",
        "rsi": "相对强弱指数 - 超买超卖指标",
        "stoch": "随机指标 - 动量振荡器",
        "willr": "威廉指标 - 超买超卖指标",
        "cci": "商品通道指数 - 价格偏离度指标",
        "roc": "变化率 - 价格变化速度指标",
        "mom": "动量 - 价格动量指标",
        "trix": "三重指数平滑移动平均 - 过滤价格噪音",
        "tsi": "真实强度指数 - 双重平滑动量指标",
        "kdj": "KDJ随机指标组合",
        "fisher": "Fisher变换 - 价格正态化指标",
        "coppock": "Coppock曲线 - 长期动量指标",
        "uo": "终极振荡器 - 综合多个时间框架的动量",
        "sma": "简单移动平均线 - 基础趋势指标",
        "ema": "指数移动平均线 - 对近期价格更敏感",
        "dema": "双重指数移动平均线 - 减少滞后",
        "tema": "三重指数移动平均线 - 进一步减少滞后",
        "hma": "赫尔移动平均线 - 加权移动平均",
        "wma": "加权移动平均线 - 线性加权",
        "kama": "考夫曼自适应移动平均线 - 自适应平滑",
        "vwap": "成交量加权平均价格 - 日内交易参考",
        "ichimoku": "一目均衡表 - 综合趋势指标",
        "supertrend": "超级趋势 - 趋势跟踪指标",
        "adx": "平均方向指数 - 趋势强度指标",
        "aroon": "Aroon指标 - 趋势强度和方向",
        "psar": "抛物线转向 - 动态止损指标",
        "vortex": "涡旋指标 - 识别趋势开始和结束",
        "vhf": "垂直水平过滤器 - 趋势/震荡市场识别",
        "chop": "震荡指标 - 市场震荡程度",
        "ttm_trend": "TTM趋势 - 简化趋势判断",
        "bbands": "布林带 - 波动率通道",
        "atr": "平均真实波幅 - 波动率指标",
        "natr": "归一化平均真实波幅 - 标准化波动率",
        "keltner": "肯特纳通道 - 基于ATR的通道",
        "donchian": "唐奇安通道 - 极值通道",
        "massi": "质量指数 - 识别反转信号",
        "ui": "溃疡指数 - 衡量下行风险",
        "obv": "能量潮 - 成交量累积指标",
        "mfi": "资金流量指数 - 成交量加权RSI",
        "ad": "累积/派发线 - 资金流向指标",
        "adosc": "累积/派发振荡器 - 资金流向振荡器",
        "cmf": "钱德动量流量 - 资金流量指标",
        "eom": "易变动量 - 价格和成交量关系",
        "pvi": "正成交量指数 - 成交量增加时累积",
        "nvi": "负成交量指数 - 成交量减少时累积",
        "zscore": "Z分数 - 标准化分数",
        "kurtosis": "峰度 - 分布尖峭程度",
        "skew": "偏度 - 分布不对称程度",
        "variance": "方差 - 价格波动程度",
        "entropy": "熵 - 价格序列复杂度",
        "quantile": "分位数 - 价格分布特征",
    }

    return descriptions.get(indicator, "技术分析指标")


def calculate_momentum_indicators(df, **kwargs):
    """计算动量指标"""
    return build_quantitative_analysis(
        df,
        [
            "macd",
            "rsi",
            "stoch",
            "willr",
            "cci",
            "roc",
            "mom",
            "trix",
            "tsi",
            "kdj",
            "fisher",
            "coppock",
            "uo",
        ],
        **kwargs,
    )


def calculate_overlap_indicators(df, **kwargs):
    """计算重叠指标"""
    return build_quantitative_analysis(
        df,
        ["sma", "ema", "dema", "tema", "hma", "wma", "kama", "vwap", "ichimoku", "supertrend"],
        **kwargs,
    )


def calculate_trend_indicators(df, **kwargs):
    """计算趋势指标"""
    return build_quantitative_analysis(
        df, ["adx", "aroon", "psar", "vortex", "vhf", "chop", "ttm_trend"], **kwargs
    )


def calculate_volatility_indicators(df, **kwargs):
    """计算波动率指标"""
    return build_quantitative_analysis(
        df, ["bbands", "atr", "natr", "keltner", "donchian", "massi", "ui"], **kwargs
    )


def calculate_volume_indicators(df, **kwargs):
    """计算成交量指标"""
    return build_quantitative_analysis(
        df, ["obv", "mfi", "ad", "adosc", "cmf", "eom", "pvi", "nvi"], **kwargs
    )


def calculate_statistics_indicators(df, **kwargs):
    """计算统计指标"""
    return build_quantitative_analysis(
        df, ["zscore", "kurtosis", "skew", "variance", "entropy", "quantile"], **kwargs
    )


def calculate_candlestick_patterns(df, **kwargs):
    """计算K线模式"""
    return build_quantitative_analysis(
        df,
        [
            "cdl_doji",
            "cdl_hammer",
            "cdl_shooting_star",
            "cdl_engulfing",
            "cdl_harami",
            "cdl_marubozu",
            "cdl_morning_star",
            "cdl_evening_star",
            "cdl_three_white_soldiers",
            "cdl_three_black_crows",
        ],
        **kwargs,
    )
