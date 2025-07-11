#!/usr/bin/env python3
"""
技术指标计算模块
提供全面的技术分析指标计算功能
"""

import logging
import warnings

import pandas as pd
import pandas_ta as ta

# 过滤pandas警告
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_quantitative_analysis(df, **kwargs):
    """
    在DataFrame上计算技术指标

    Args:
        df: 包含OHLCV数据的DataFrame
        **kwargs: 技术指标参数，包括ma_periods等

    Returns:
        添加了技术指标的DataFrame
    """
    if df is None or df.empty:
        return None

    # 计算所有技术指标
    df = calculate_momentum_indicators(df, **kwargs)
    df = calculate_overlap_indicators(df, **kwargs)
    df = calculate_trend_indicators(df, **kwargs)
    df = calculate_volatility_indicators(df, **kwargs)
    df = calculate_volume_indicators(df, **kwargs)
    df = calculate_statistics_indicators(df, **kwargs)
    df = calculate_candlestick_patterns(df, **kwargs)

    return df


def calculate_momentum_indicators(df, **kwargs):
    """
    计算动量指标 - 用于衡量价格变化的速度和强度
    """
    logger.info("计算动量指标...")

    # MACD (移动平均收敛发散) - 趋势跟踪动量指标
    # 用途：识别趋势变化、买卖信号
    macd_fast = kwargs.get("macd_fast", 12)
    macd_slow = kwargs.get("macd_slow", 26)
    macd_signal = kwargs.get("macd_signal", 9)
    df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)

    # RSI (相对强弱指数) - 超买超卖指标
    # 用途：识别超买超卖区域，寻找反转信号
    rsi_length = kwargs.get("rsi_length", 14)
    df.ta.rsi(length=rsi_length, append=True)

    # Stochastic (随机指标) - 动量振荡器
    # 用途：识别超买超卖，寻找背离信号
    stoch_k = kwargs.get("stoch_k", 14)
    stoch_d = kwargs.get("stoch_d", 3)
    stoch_smooth_k = kwargs.get("stoch_smooth_k", 3)
    df.ta.stoch(k=stoch_k, d=stoch_d, smooth_k=stoch_smooth_k, append=True)

    # Williams %R - 超买超卖指标
    # 用途：识别超买超卖区域
    willr_length = kwargs.get("willr_length", 14)
    df.ta.willr(length=willr_length, append=True)

    # CCI (商品通道指数) - 价格偏离度指标
    # 用途：识别超买超卖，寻找反转信号
    cci_length = kwargs.get("cci_length", 20)
    df.ta.cci(length=cci_length, append=True)

    # ROC (变化率) - 价格变化速度指标
    # 用途：衡量价格变化速度，识别动量
    roc_length = kwargs.get("roc_length", 10)
    df.ta.roc(length=roc_length, append=True)

    # Momentum (动量) - 价格动量指标
    # 用途：衡量价格变化幅度
    mom_length = kwargs.get("mom_length", 10)
    df.ta.mom(length=mom_length, append=True)

    # TRIX - 三重指数平滑移动平均
    # 用途：过滤价格噪音，识别趋势
    trix_length = kwargs.get("trix_length", 18)
    df.ta.trix(length=trix_length, append=True)

    # TSI (真实强度指数) - 双重平滑动量指标
    # 用途：识别趋势变化，减少噪音
    tsi_fast = kwargs.get("tsi_fast", 13)
    tsi_slow = kwargs.get("tsi_slow", 25)
    df.ta.tsi(fast=tsi_fast, slow=tsi_slow, append=True)

    # KDJ - 随机指标组合
    # 用途：综合超买超卖和趋势信号
    kdj_length = kwargs.get("kdj_length", 9)
    df.ta.kdj(length=kdj_length, append=True)

    # Fisher Transform - 价格正态化指标
    # 用途：将价格转换为正态分布，便于分析
    fisher_length = kwargs.get("fisher_length", 9)
    df.ta.fisher(length=fisher_length, append=True)

    # Coppock Curve - 长期动量指标
    # 用途：识别长期底部，适用于股票市场
    coppock_length = kwargs.get("coppock_length", 10)
    df.ta.coppock(length=coppock_length, append=True)

    # Ultimate Oscillator - 终极振荡器
    # 用途：综合多个时间框架的动量
    uo_length1 = kwargs.get("uo_length1", 7)
    uo_length2 = kwargs.get("uo_length2", 14)
    uo_length3 = kwargs.get("uo_length3", 28)
    df.ta.uo(length1=uo_length1, length2=uo_length2, length3=uo_length3, append=True)

    return df


def calculate_overlap_indicators(df, **kwargs):
    """
    计算重叠指标 - 移动平均线和价格重叠指标
    """
    logger.info("计算重叠指标...")

    # 获取ma_periods参数，默认为(7, 25, 99)
    ma_periods = kwargs.get("ma_periods", (7, 25, 99))

    # 计算SMA (简单移动平均线)
    # 用途：识别趋势方向，支撑阻力位
    for period in ma_periods:
        df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()

    # 计算EMA (指数移动平均线)
    # 用途：对近期价格更敏感的趋势指标
    for period in ma_periods:
        df[f"EMA_{period}"] = ta.ema(df["Close"], length=period)

    # DEMA (双重指数移动平均线)
    # 用途：减少滞后，更快响应价格变化
    dema_length = kwargs.get("dema_length", 20)
    df.ta.dema(length=dema_length, append=True)

    # TEMA (三重指数移动平均线)
    # 用途：进一步减少滞后，更平滑的趋势线
    tema_length = kwargs.get("tema_length", 20)
    df.ta.tema(length=tema_length, append=True)

    # HMA (赫尔移动平均线)
    # 用途：减少滞后同时保持平滑
    hma_length = kwargs.get("hma_length", 20)
    df.ta.hma(length=hma_length, append=True)

    # WMA (加权移动平均线)
    # 用途：对近期数据给予更高权重
    wma_length = kwargs.get("wma_length", 20)
    df.ta.wma(length=wma_length, append=True)

    # KAMA (考夫曼自适应移动平均线)
    # 用途：根据市场波动性自适应调整
    kama_length = kwargs.get("kama_length", 10)
    df.ta.kama(length=kama_length, append=True)

    # VWAP (成交量加权平均价格)
    # 用途：日内交易的重要参考价格
    df.ta.vwap(append=True)

    # Ichimoku (一目均衡表)
    # 用途：综合趋势、支撑阻力、动量指标
    ichimoku_tenkan = kwargs.get("ichimoku_tenkan", 9)
    ichimoku_kijun = kwargs.get("ichimoku_kijun", 26)
    ichimoku_senkou = kwargs.get("ichimoku_senkou", 52)
    df.ta.ichimoku(
        tenkan=ichimoku_tenkan,
        kijun=ichimoku_kijun,
        senkou=ichimoku_senkou,
        append=True,
    )

    # SuperTrend - 趋势跟踪指标
    # 用途：动态支撑阻力，趋势跟踪
    supertrend_period = kwargs.get("supertrend_period", 10)
    supertrend_multiplier = kwargs.get("supertrend_multiplier", 3.0)
    df.ta.supertrend(period=supertrend_period, multiplier=supertrend_multiplier, append=True)

    return df


def calculate_trend_indicators(df, **kwargs):
    """
    计算趋势指标 - 用于识别和确认趋势方向
    """
    logger.info("计算趋势指标...")

    # ADX (平均方向指数) - 趋势强度指标
    # 用途：衡量趋势强度，判断是否适合趋势交易
    adx_length = kwargs.get("adx_length", 14)
    df.ta.adx(length=adx_length, append=True)

    # Aroon - 趋势强度和方向指标
    # 用途：识别趋势开始和结束，衡量趋势强度
    aroon_length = kwargs.get("aroon_length", 25)
    df.ta.aroon(length=aroon_length, append=True)

    # PSAR (抛物线转向) - 趋势跟踪指标
    # 用途：动态止损，趋势跟踪
    psar_af0 = kwargs.get("psar_af0", 0.02)
    psar_af = kwargs.get("psar_af", 0.02)
    psar_max_af = kwargs.get("psar_max_af", 0.2)
    df.ta.psar(af0=psar_af0, af=psar_af, max_af=psar_max_af, append=True)

    # Vortex - 涡旋指标
    # 用途：识别趋势开始和结束
    vortex_length = kwargs.get("vortex_length", 14)
    df.ta.vortex(length=vortex_length, append=True)

    # VHF (垂直水平过滤器) - 趋势/震荡市场识别
    # 用途：判断市场是趋势还是震荡
    vhf_length = kwargs.get("vhf_length", 28)
    df.ta.vhf(length=vhf_length, append=True)

    # Choppiness - 市场震荡程度指标
    # 用途：判断市场是否处于震荡状态
    chop_length = kwargs.get("chop_length", 14)
    df.ta.chop(length=chop_length, append=True)

    # TTM Trend - TTM趋势指标
    # 用途：简化趋势判断
    ttm_length = kwargs.get("ttm_length", 5)
    df.ta.ttm_trend(length=ttm_length, append=True)

    return df


def calculate_volatility_indicators(df, **kwargs):
    """
    计算波动率指标 - 用于衡量价格波动程度
    """
    logger.info("计算波动率指标...")

    # Bollinger Bands (布林带) - 波动率通道
    # 用途：识别超买超卖，判断波动率
    bb_length = kwargs.get("bb_length", 20)
    bb_std = kwargs.get("bb_std", 2)
    df.ta.bbands(length=bb_length, std=bb_std, append=True)

    # ATR (平均真实波幅) - 波动率指标
    # 用途：设置止损，判断市场波动性
    atr_length = kwargs.get("atr_length", 14)
    df.ta.atr(length=atr_length, append=True)

    # NATR (归一化平均真实波幅)
    # 用途：标准化波动率，便于比较
    natr_length = kwargs.get("natr_length", 14)
    df.ta.natr(length=natr_length, append=True)

    # Keltner Channel - 基于ATR的通道
    # 用途：动态支撑阻力，波动率通道
    kc_length = kwargs.get("kc_length", 20)
    kc_std = kwargs.get("kc_std", 2)
    df.ta.kc(length=kc_length, std=kc_std, append=True)

    # Donchian Channel - 极值通道
    # 用途：识别支撑阻力，突破交易
    dc_length = kwargs.get("dc_length", 20)
    df.ta.donchian(length=dc_length, append=True)

    # Mass Index - 质量指数
    # 用途：识别反转信号
    mi_length = kwargs.get("mi_length", 9)
    df.ta.massi(length=mi_length, append=True)

    # Ulcer Index - 溃疡指数
    # 用途：衡量下行风险
    ui_length = kwargs.get("ui_length", 14)
    df.ta.ui(length=ui_length, append=True)

    return df


def calculate_volume_indicators(df, **kwargs):
    """
    计算成交量指标 - 用于分析成交量与价格的关系
    """
    logger.info("计算成交量指标...")

    # OBV (能量潮) - 成交量累积指标
    # 用途：确认价格趋势，识别背离
    df.ta.obv(append=True)

    # MFI (资金流量指数) - 成交量加权RSI
    # 用途：结合价格和成交量的超买超卖指标
    mfi_length = kwargs.get("mfi_length", 14)
    df.ta.mfi(length=mfi_length, append=True)

    # AD (累积/派发线) - 资金流向指标
    # 用途：判断资金流入流出
    df.ta.ad(append=True)

    # ADOSC (累积/派发振荡器)
    # 用途：资金流向的振荡器版本
    adosc_fast = kwargs.get("adosc_fast", 3)
    adosc_slow = kwargs.get("adosc_slow", 10)
    df.ta.adosc(fast=adosc_fast, slow=adosc_slow, append=True)

    # CMF (钱德动量流量) - 资金流量指标
    # 用途：衡量资金流入流出的强度
    cmf_length = kwargs.get("cmf_length", 20)
    df.ta.cmf(length=cmf_length, append=True)

    # EOM (易变动量) - 价格和成交量的关系
    # 用途：判断价格变动与成交量的关系
    eom_length = kwargs.get("eom_length", 14)
    df.ta.eom(length=eom_length, append=True)

    # VWAP (成交量加权平均价格)
    # 用途：日内交易的重要参考价格
    df.ta.vwap(append=True)

    # PVI (正成交量指数)
    # 用途：只在成交量增加时累积价格变化
    df.ta.pvi(append=True)

    # NVI (负成交量指数)
    # 用途：只在成交量减少时累积价格变化
    df.ta.nvi(append=True)

    return df


def calculate_statistics_indicators(df, **kwargs):
    """
    计算统计指标 - 用于价格数据的统计分析
    """
    logger.info("计算统计指标...")

    # Z-Score - 标准化分数
    # 用途：判断价格偏离程度
    zscore_length = kwargs.get("zscore_length", 20)
    df.ta.zscore(length=zscore_length, append=True)

    # Kurtosis - 峰度
    # 用途：衡量价格分布的尖峭程度
    kurtosis_length = kwargs.get("kurtosis_length", 20)
    df.ta.kurtosis(length=kurtosis_length, append=True)

    # Skew - 偏度
    # 用途：衡量价格分布的偏斜程度
    skew_length = kwargs.get("skew_length", 20)
    df.ta.skew(length=skew_length, append=True)

    # Variance - 方差
    # 用途：衡量价格波动性
    variance_length = kwargs.get("variance_length", 20)
    df.ta.variance(length=variance_length, append=True)

    # Standard Deviation - 标准差
    # 用途：衡量价格分散程度
    stdev_length = kwargs.get("stdev_length", 20)
    df.ta.stdev(length=stdev_length, append=True)

    # Median - 中位数
    # 用途：价格的中心趋势
    median_length = kwargs.get("median_length", 20)
    df.ta.median(length=median_length, append=True)

    # MAD (平均绝对偏差)
    # 用途：衡量价格偏离中位数的程度
    mad_length = kwargs.get("mad_length", 20)
    df.ta.mad(length=mad_length, append=True)

    return df


def calculate_candlestick_patterns(df, **kwargs):
    """
    计算K线形态 - 用于识别经典的K线形态
    """
    logger.info("计算K线形态...")

    # 计算所有K线形态
    # 用途：识别经典的反转和持续形态
    df.ta.cdl_pattern(name="all", append=True)

    # 计算Heikin-Ashi K线
    # 用途：平滑价格数据，更容易识别趋势
    df.ta.ha(append=True)

    return df


def get_available_indicators():
    """
    获取所有可用的技术指标列表

    Returns:
        dict: 按分类组织的技术指标字典
    """
    return {
        "momentum": [
            "MACD",
            "RSI",
            "Stochastic",
            "Williams %R",
            "CCI",
            "ROC",
            "Momentum",
            "TRIX",
            "TSI",
            "KDJ",
            "Fisher Transform",
            "Coppock Curve",
            "Ultimate Oscillator",
        ],
        "overlap": [
            "SMA",
            "EMA",
            "DEMA",
            "TEMA",
            "HMA",
            "WMA",
            "KAMA",
            "VWAP",
            "Ichimoku",
            "SuperTrend",
        ],
        "trend": ["ADX", "Aroon", "PSAR", "Vortex", "VHF", "Choppiness", "TTM Trend"],
        "volatility": [
            "Bollinger Bands",
            "ATR",
            "NATR",
            "Keltner Channel",
            "Donchian Channel",
            "Mass Index",
            "Ulcer Index",
        ],
        "volume": ["OBV", "MFI", "AD", "ADOSC", "CMF", "EOM", "PVI", "NVI"],
        "statistics": [
            "Z-Score",
            "Kurtosis",
            "Skew",
            "Variance",
            "Standard Deviation",
            "Median",
            "MAD",
        ],
        "candlestick": ["CDL Patterns", "Heikin-Ashi"],
    }


def build_indicator_parameters(ma_periods=(7, 25, 99), **kwargs):
    """
    根据用户自定义参数构建技术指标参数字典

    Args:
        ma_periods: 移动平均线周期列表
        **kwargs: 其他技术指标参数

    Returns:
        dict: 技术指标参数字典，包含所有可配置的参数及其默认值
    """
    params = {
        # 移动平均线周期参数
        "ma_periods": ma_periods,  # 移动平均线周期列表，用于计算SMA和EMA
        # MACD参数
        "macd_fast": 12,  # MACD快线周期，通常为12
        "macd_slow": 26,  # MACD慢线周期，通常为26
        "macd_signal": 9,  # MACD信号线周期，通常为9
        # RSI参数
        "rsi_length": 14,  # RSI计算周期，标准为14
        # 随机指标参数
        "stoch_k": 14,  # 随机指标%K周期
        "stoch_d": 3,  # 随机指标%D周期
        "stoch_smooth_k": 3,  # 随机指标%K平滑周期
        # Williams %R参数
        "willr_length": 14,  # Williams %R计算周期
        # CCI参数
        "cci_length": 20,  # 商品通道指数计算周期
        # ROC参数
        "roc_length": 10,  # 变化率计算周期
        # 动量参数
        "mom_length": 10,  # 动量指标计算周期
        # TRIX参数
        "trix_length": 18,  # TRIX计算周期
        # TSI参数
        "tsi_fast": 13,  # TSI快线周期
        "tsi_slow": 25,  # TSI慢线周期
        # KDJ参数
        "kdj_length": 9,  # KDJ计算周期
        # Fisher Transform参数
        "fisher_length": 9,  # Fisher Transform计算周期
        # Coppock Curve参数
        "coppock_length": 10,  # Coppock Curve计算周期
        # Ultimate Oscillator参数
        "uo_length1": 7,  # 终极振荡器第一周期
        "uo_length2": 14,  # 终极振荡器第二周期
        "uo_length3": 28,  # 终极振荡器第三周期
        # 移动平均线参数
        "dema_length": 20,  # 双重指数移动平均线周期
        "tema_length": 20,  # 三重指数移动平均线周期
        "hma_length": 20,  # 赫尔移动平均线周期
        "wma_length": 20,  # 加权移动平均线周期
        "kama_length": 10,  # 考夫曼自适应移动平均线周期
        # 一目均衡表参数
        "ichimoku_tenkan": 9,  # 一目均衡表转换线周期
        "ichimoku_kijun": 26,  # 一目均衡表基准线周期
        "ichimoku_senkou": 52,  # 一目均衡表先行带周期
        # SuperTrend参数
        "supertrend_period": 10,  # SuperTrend周期
        "supertrend_multiplier": 3.0,  # SuperTrend倍数
        # 趋势指标参数
        "adx_length": 14,  # 平均方向指数周期
        "aroon_length": 25,  # Aroon指标周期
        # PSAR参数
        "psar_af0": 0.02,  # PSAR初始加速因子
        "psar_af": 0.02,  # PSAR加速因子
        "psar_max_af": 0.2,  # PSAR最大加速因子
        # 其他趋势指标参数
        "vortex_length": 14,  # 涡旋指标周期
        "vhf_length": 28,  # 垂直水平过滤器周期
        "chop_length": 14,  # 震荡程度指标周期
        "ttm_length": 5,  # TTM趋势指标周期
        # 波动率指标参数
        "bb_length": 20,  # 布林带周期
        "bb_std": 2,  # 布林带标准差倍数
        "atr_length": 14,  # 平均真实波幅周期
        "natr_length": 14,  # 归一化平均真实波幅周期
        "kc_length": 20,  # Keltner Channel周期
        "kc_std": 2,  # Keltner Channel标准差倍数
        "dc_length": 20,  # Donchian Channel周期
        "mi_length": 9,  # 质量指数周期
        "ui_length": 14,  # 溃疡指数周期
        # 成交量指标参数
        "mfi_length": 14,  # 资金流量指数周期
        "adosc_fast": 3,  # 累积/派发振荡器快线周期
        "adosc_slow": 10,  # 累积/派发振荡器慢线周期
        "cmf_length": 20,  # 钱德动量流量周期
        "eom_length": 14,  # 易变动量周期
        # 统计指标参数
        "zscore_length": 20,  # Z-Score计算周期
        "kurtosis_length": 20,  # 峰度计算周期
        "skew_length": 20,  # 偏度计算周期
        "variance_length": 20,  # 方差计算周期
        "stdev_length": 20,  # 标准差计算周期
        "median_length": 20,  # 中位数计算周期
        "mad_length": 20,  # 平均绝对偏差计算周期
    }

    # 更新或添加用户自定义参数
    params.update(kwargs)
    return params
