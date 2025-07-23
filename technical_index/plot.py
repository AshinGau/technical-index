#!/usr/bin/env python3
"""
绘制技术指标
"""

import logging
import warnings
from typing import Optional

import mplfinance as mpf  # 引入绘图库
import pandas as pd

# 忽略 pandas 的一些警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlotConfig:
    """
    绘图配置类
    """

    def __init__(self, **kwargs):
        # 要绘制的技术指标列表
        self.indicators = kwargs.get("indicators", ["taker_buy"])

        # 移动平均线配置
        self.sma_periods = kwargs.get("sma_periods", None)  # 如果提供则绘制SMA
        self.ema_periods = kwargs.get("ema_periods", None)  # 如果提供则绘制EMA

        # 图表基本配置
        self.limit = kwargs.get("limit", 150)
        self.figsize = kwargs.get("figsize", (32, 28))
        self.figscale = kwargs.get("figscale", 1.0)
        self.style = kwargs.get("style", "yahoo")
        self.title = kwargs.get("title", "Technical Indicators")

        # 技术指标参数
        # MACD参数
        self.macd_fast = kwargs.get("macd_fast", 12)
        self.macd_slow = kwargs.get("macd_slow", 26)
        self.macd_signal = kwargs.get("macd_signal", 9)

        # RSI参数
        self.rsi_length = kwargs.get("rsi_length", 14)

        # Stochastic参数
        self.stoch_k = kwargs.get("stoch_k", 14)
        self.stoch_d = kwargs.get("stoch_d", 3)
        self.stoch_smooth_k = kwargs.get("stoch_smooth_k", 3)

        # Williams %R参数
        self.willr_length = kwargs.get("willr_length", 14)

        # CCI参数
        self.cci_length = kwargs.get("cci_length", 20)

        # ROC参数
        self.roc_length = kwargs.get("roc_length", 10)

        # Momentum参数
        self.mom_length = kwargs.get("mom_length", 10)

        # TRIX参数
        self.trix_length = kwargs.get("trix_length", 18)

        # TSI参数
        self.tsi_fast = kwargs.get("tsi_fast", 13)
        self.tsi_slow = kwargs.get("tsi_slow", 25)

        # KDJ参数
        self.kdj_length = kwargs.get("kdj_length", 9)

        # Fisher Transform参数
        self.fisher_length = kwargs.get("fisher_length", 9)

        # Coppock Curve参数
        self.coppock_length = kwargs.get("coppock_length", 10)

        # Ultimate Oscillator参数
        self.uo_length1 = kwargs.get("uo_length1", 7)
        self.uo_length2 = kwargs.get("uo_length2", 14)
        self.uo_length3 = kwargs.get("uo_length3", 28)

        # 趋势指标参数
        self.adx_length = kwargs.get("adx_length", 14)
        self.aroon_length = kwargs.get("aroon_length", 25)
        self.psar_af0 = kwargs.get("psar_af0", 0.02)
        self.psar_af = kwargs.get("psar_af", 0.02)
        self.psar_max_af = kwargs.get("psar_max_af", 0.2)
        self.vortex_length = kwargs.get("vortex_length", 14)
        self.vhf_length = kwargs.get("vhf_length", 28)
        self.chop_length = kwargs.get("chop_length", 14)
        self.ttm_length = kwargs.get("ttm_length", 5)

        # 波动率指标参数
        self.bb_length = kwargs.get("bb_length", 20)
        self.bb_std = kwargs.get("bb_std", 2)
        self.atr_length = kwargs.get("atr_length", 14)
        self.natr_length = kwargs.get("natr_length", 14)
        self.kc_length = kwargs.get("kc_length", 20)
        self.kc_std = kwargs.get("kc_std", 2)
        self.dc_length = kwargs.get("dc_length", 20)
        self.mi_length = kwargs.get("mi_length", 9)
        self.ui_length = kwargs.get("ui_length", 14)

        # 成交量指标参数
        self.mfi_length = kwargs.get("mfi_length", 14)
        self.adosc_fast = kwargs.get("adosc_fast", 3)
        self.adosc_slow = kwargs.get("adosc_slow", 10)
        self.cmf_length = kwargs.get("cmf_length", 20)
        self.eom_length = kwargs.get("eom_length", 14)

        # 统计指标参数
        self.zscore_length = kwargs.get("zscore_length", 20)
        self.kurtosis_length = kwargs.get("kurtosis_length", 20)
        self.skew_length = kwargs.get("skew_length", 20)
        self.variance_length = kwargs.get("variance_length", 20)
        self.stdev_length = kwargs.get("stdev_length", 20)
        self.median_length = kwargs.get("median_length", 20)
        self.mad_length = kwargs.get("mad_length", 20)

        # 面板比例配置
        self.panel_ratios = kwargs.get("panel_ratios", None)


def plot_candlestick_with_indicators(df, config: Optional[PlotConfig] = None, **kwargs):
    """
    绘制K线图，并附带交易量和多个技术指标子图。

    :param df: 包含OHLCV和技术指标的DataFrame
    :param config: 绘图配置对象
    :param **kwargs: 技术指标参数，包括ma_periods等
    """
    if df is None or df.empty:
        logger.error("数据为空，无法绘图。")
        return

    # 创建配置对象
    if config is None:
        config = PlotConfig(**kwargs)
    else:
        # 更新配置对象中的参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # mplfinance需要大写开头的列名
    df_plot = df.tail(config.limit).copy()

    # 1. 准备所有附图 (Additional Plots)
    ap = []

    # 添加移动平均线到主图
    if config.sma_periods:
        for period in config.sma_periods:
            column_name = f"SMA_{period}"
            if column_name in df_plot.columns:
                ap.append(mpf.make_addplot(df_plot[column_name], panel=0))

    if config.ema_periods:
        for period in config.ema_periods:
            column_name = f"EMA_{period}"
            if column_name in df_plot.columns:
                ap.append(mpf.make_addplot(df_plot[column_name], panel=0))

    # 根据indicators配置添加技术指标
    indicators_lower = [ind.lower() for ind in config.indicators]
    panel_index = 2  # 从第2个面板开始（第0个是主图，第1个是成交量）

    # MACD 指标
    if "macd" in indicators_lower:
        macd_col = f"MACD_{config.macd_fast}_{config.macd_slow}_{config.macd_signal}"
        macd_signal_col = f"MACDs_{config.macd_fast}_{config.macd_slow}_{config.macd_signal}"
        macd_hist_col = f"MACDh_{config.macd_fast}_{config.macd_slow}_{config.macd_signal}"

        if macd_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[macd_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="MACD",
                    ),
                    mpf.make_addplot(df_plot[macd_signal_col], panel=panel_index, color="orange"),
                    mpf.make_addplot(
                        df_plot[macd_hist_col],
                        type="bar",
                        panel=panel_index,
                        color="grey",
                        alpha=0.5,
                    ),
                ]
            )
            panel_index += 1

    # RSI 指标
    if "rsi" in indicators_lower:
        rsi_col = f"RSI_{config.rsi_length}"
        if rsi_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[rsi_col], panel=panel_index, color="blue", ylabel="RSI"
                    ),
                    mpf.make_addplot(
                        pd.Series(70, index=df_plot.index),
                        panel=panel_index,
                        color="r",
                        linestyle="--",
                        secondary_y=False,
                    ),
                    mpf.make_addplot(
                        pd.Series(30, index=df_plot.index),
                        panel=panel_index,
                        color="g",
                        linestyle="--",
                        secondary_y=False,
                    ),
                ]
            )
            panel_index += 1

    # Stochastic 指标
    if "stochastic" in indicators_lower or "stoch" in indicators_lower:
        stoch_k_col = f"STOCHk_{config.stoch_k}_{config.stoch_d}_{config.stoch_smooth_k}"
        stoch_d_col = f"STOCHd_{config.stoch_k}_{config.stoch_d}_{config.stoch_smooth_k}"

        if stoch_k_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[stoch_k_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="Stochastic",
                    ),
                    mpf.make_addplot(df_plot[stoch_d_col], panel=panel_index, color="red"),
                    mpf.make_addplot(
                        pd.Series(80, index=df_plot.index),
                        panel=panel_index,
                        color="r",
                        linestyle="--",
                        secondary_y=False,
                    ),
                    mpf.make_addplot(
                        pd.Series(20, index=df_plot.index),
                        panel=panel_index,
                        color="g",
                        linestyle="--",
                        secondary_y=False,
                    ),
                ]
            )
            panel_index += 1

    # Williams %R 指标
    if "williams_r" in indicators_lower or "willr" in indicators_lower:
        willr_col = f"WILLR_{config.willr_length}"
        if willr_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[willr_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="Williams %R",
                    ),
                    mpf.make_addplot(
                        pd.Series(-20, index=df_plot.index),
                        panel=panel_index,
                        color="r",
                        linestyle="--",
                        secondary_y=False,
                    ),
                    mpf.make_addplot(
                        pd.Series(-80, index=df_plot.index),
                        panel=panel_index,
                        color="g",
                        linestyle="--",
                        secondary_y=False,
                    ),
                ]
            )
            panel_index += 1

    # CCI 指标
    if "cci" in indicators_lower:
        cci_col = f"CCI_{config.cci_length}"
        if cci_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[cci_col],
                        panel=panel_index,
                        color="orange",
                        ylabel="CCI",
                    ),
                    mpf.make_addplot(
                        pd.Series(100, index=df_plot.index),
                        panel=panel_index,
                        color="r",
                        linestyle="--",
                        secondary_y=False,
                    ),
                    mpf.make_addplot(
                        pd.Series(-100, index=df_plot.index),
                        panel=panel_index,
                        color="g",
                        linestyle="--",
                        secondary_y=False,
                    ),
                ]
            )
            panel_index += 1

    # ROC 指标
    if "roc" in indicators_lower:
        roc_col = f"ROC_{config.roc_length}"
        if roc_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[roc_col], panel=panel_index, color="brown", ylabel="ROC"
                    ),
                ]
            )
            panel_index += 1

    # Momentum 指标
    if "momentum" in indicators_lower or "mom" in indicators_lower:
        mom_col = f"MOM_{config.mom_length}"
        if mom_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[mom_col],
                        panel=panel_index,
                        color="teal",
                        ylabel="Momentum",
                    ),
                ]
            )
            panel_index += 1

    # TRIX 指标
    if "trix" in indicators_lower:
        trix_col = f"TRIX_{config.trix_length}"
        if trix_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[trix_col],
                        panel=panel_index,
                        color="navy",
                        ylabel="TRIX",
                    ),
                ]
            )
            panel_index += 1

    # TSI 指标
    if "tsi" in indicators_lower:
        tsi_col = f"TSI_{config.tsi_fast}_{config.tsi_slow}"
        tsi_signal_col = f"TSIs_{config.tsi_fast}_{config.tsi_slow}"

        if tsi_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[tsi_col], panel=panel_index, color="blue", ylabel="TSI"
                    ),
                    mpf.make_addplot(df_plot[tsi_signal_col], panel=panel_index, color="red"),
                ]
            )
            panel_index += 1

    # KDJ 指标
    if "kdj" in indicators_lower:
        kdj_k_col = f"KDJk_{config.kdj_length}"
        kdj_d_col = f"KDJd_{config.kdj_length}"
        kdj_j_col = f"KDJj_{config.kdj_length}"

        if kdj_k_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[kdj_k_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="KDJ",
                    ),
                    mpf.make_addplot(df_plot[kdj_d_col], panel=panel_index, color="red"),
                    mpf.make_addplot(df_plot[kdj_j_col], panel=panel_index, color="green"),
                ]
            )
            panel_index += 1

    # Fisher Transform 指标
    if "fisher" in indicators_lower:
        fisher_col = f"FISHERT_{config.fisher_length}"
        fisher_signal_col = f"FISHERTs_{config.fisher_length}"

        if fisher_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[fisher_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="Fisher",
                    ),
                    mpf.make_addplot(df_plot[fisher_signal_col], panel=panel_index, color="red"),
                ]
            )
            panel_index += 1

    # Coppock Curve 指标
    if "coppock" in indicators_lower:
        coppock_col = f"COPP_{config.coppock_length}"
        if coppock_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[coppock_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="Coppock",
                    ),
                ]
            )
            panel_index += 1

    # Ultimate Oscillator 指标
    if "uo" in indicators_lower or "ultimate" in indicators_lower:
        uo_col = f"UO_{config.uo_length1}_{config.uo_length2}_{config.uo_length3}"
        if uo_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[uo_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="Ultimate Oscillator",
                    ),
                    mpf.make_addplot(
                        pd.Series(70, index=df_plot.index),
                        panel=panel_index,
                        color="r",
                        linestyle="--",
                        secondary_y=False,
                    ),
                    mpf.make_addplot(
                        pd.Series(30, index=df_plot.index),
                        panel=panel_index,
                        color="g",
                        linestyle="--",
                        secondary_y=False,
                    ),
                ]
            )
            panel_index += 1

    # DMI 指标
    if "adx" in indicators_lower:
        adx_col = f"ADX_{config.adx_length}"
        dmp_col = f"DMP_{config.adx_length}"
        dmn_col = f"DMN_{config.adx_length}"

        if adx_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[adx_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="ADX/DMI",
                    ),
                    mpf.make_addplot(df_plot[dmp_col], panel=panel_index, color="green"),
                    mpf.make_addplot(df_plot[dmn_col], panel=panel_index, color="red"),
                ]
            )
            panel_index += 1

    # Aroon 指标
    if "aroon" in indicators_lower:
        aroon_up_col = f"AROONU_{config.aroon_length}"
        aroon_down_col = f"AROOND_{config.aroon_length}"
        aroon_osc_col = f"AROONOSC_{config.aroon_length}"

        if aroon_up_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[aroon_up_col],
                        panel=panel_index,
                        color="green",
                        ylabel="Aroon",
                    ),
                    mpf.make_addplot(df_plot[aroon_down_col], panel=panel_index, color="red"),
                    mpf.make_addplot(df_plot[aroon_osc_col], panel=panel_index, color="blue"),
                ]
            )
            panel_index += 1

    # PSAR 指标
    if "psar" in indicators_lower:
        psar_col = f"PSAR_{config.psar_af0}_{config.psar_af}_{config.psar_max_af}"
        if psar_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[psar_col],
                        panel=0,
                        type="scatter",
                        markersize=50,
                        color="red",
                    ),
                ]
            )

    # Vortex 指标
    if "vortex" in indicators_lower:
        vortex_plus_col = f"VTXP_{config.vortex_length}"
        vortex_minus_col = f"VTXM_{config.vortex_length}"

        if vortex_plus_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[vortex_plus_col],
                        panel=panel_index,
                        color="green",
                        ylabel="Vortex",
                    ),
                    mpf.make_addplot(df_plot[vortex_minus_col], panel=panel_index, color="red"),
                ]
            )
            panel_index += 1

    # VHF 指标
    if "vhf" in indicators_lower:
        vhf_col = f"VHF_{config.vhf_length}"
        if vhf_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[vhf_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="VHF",
                    ),
                ]
            )
            panel_index += 1

    # Choppiness 指标
    if "chop" in indicators_lower or "choppiness" in indicators_lower:
        chop_col = f"CHOP_{config.chop_length}"
        if chop_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[chop_col],
                        panel=panel_index,
                        color="brown",
                        ylabel="Choppiness",
                    ),
                ]
            )
            panel_index += 1

    # TTM Trend 指标
    if "ttm_trend" in indicators_lower or "ttm" in indicators_lower:
        ttm_col = f"TTM_TREND_{config.ttm_length}"
        if ttm_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[ttm_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="TTM Trend",
                    ),
                ]
            )
            panel_index += 1

    # ATR 指标
    if "atr" in indicators_lower:
        atr_col = f"ATRr_{config.atr_length}"
        if atr_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[atr_col],
                        panel=panel_index,
                        color="orange",
                        ylabel="ATR",
                    ),
                ]
            )
            panel_index += 1

    # NATR 指标
    if "natr" in indicators_lower:
        natr_col = f"NATR_{config.natr_length}"
        if natr_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[natr_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="NATR",
                    ),
                ]
            )
            panel_index += 1

    # Keltner Channel 指标
    if "keltner" in indicators_lower or "kc" in indicators_lower:
        kc_upper_col = f"KCUe_{config.kc_length}_{config.kc_std}"
        kc_middle_col = f"KCBe_{config.kc_length}_{config.kc_std}"
        kc_lower_col = f"KCLe_{config.kc_length}_{config.kc_std}"

        if kc_upper_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(df_plot[kc_upper_col], panel=0, color="red", linestyle="--"),
                    mpf.make_addplot(df_plot[kc_middle_col], panel=0, color="blue", linestyle="--"),
                    mpf.make_addplot(df_plot[kc_lower_col], panel=0, color="red", linestyle="--"),
                ]
            )

    # Donchian Channel 指标
    if "donchian" in indicators_lower or "dc" in indicators_lower:
        dc_upper_col = f"DCU_{config.dc_length}"
        dc_middle_col = f"DCB_{config.dc_length}"
        dc_lower_col = f"DCL_{config.dc_length}"

        if dc_upper_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(df_plot[dc_upper_col], panel=0, color="red", linestyle="--"),
                    mpf.make_addplot(df_plot[dc_middle_col], panel=0, color="blue", linestyle="--"),
                    mpf.make_addplot(df_plot[dc_lower_col], panel=0, color="red", linestyle="--"),
                ]
            )

    # Mass Index 指标
    if "massi" in indicators_lower or "mass" in indicators_lower:
        massi_col = f"MASI_{config.mi_length}"
        if massi_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[massi_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="Mass Index",
                    ),
                ]
            )
            panel_index += 1

    # Ulcer Index 指标
    if "ui" in indicators_lower or "ulcer" in indicators_lower:
        ui_col = f"UI_{config.ui_length}"
        if ui_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[ui_col],
                        panel=panel_index,
                        color="red",
                        ylabel="Ulcer Index",
                    ),
                ]
            )
            panel_index += 1

    # MFI 指标
    if "mfi" in indicators_lower:
        mfi_col = f"MFI_{config.mfi_length}"
        if mfi_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[mfi_col], panel=panel_index, color="cyan", ylabel="MFI"
                    ),
                    mpf.make_addplot(
                        pd.Series(80, index=df_plot.index),
                        panel=panel_index,
                        color="r",
                        linestyle="--",
                        secondary_y=False,
                    ),
                    mpf.make_addplot(
                        pd.Series(20, index=df_plot.index),
                        panel=panel_index,
                        color="g",
                        linestyle="--",
                        secondary_y=False,
                    ),
                ]
            )
            panel_index += 1

    # OBV 指标
    if "obv" in indicators_lower:
        if "OBV" in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot["OBV"], panel=panel_index, color="magenta", ylabel="OBV"
                    ),
                ]
            )
            panel_index += 1

    # AD 指标
    if "ad" in indicators_lower:
        if "AD" in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(df_plot["AD"], panel=panel_index, color="blue", ylabel="AD"),
                ]
            )
            panel_index += 1

    # ADOSC 指标
    if "adosc" in indicators_lower:
        adosc_col = f"ADOSC_{config.adosc_fast}_{config.adosc_slow}"
        if adosc_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[adosc_col],
                        panel=panel_index,
                        color="orange",
                        ylabel="ADOSC",
                    ),
                ]
            )
            panel_index += 1

    # CMF 指标
    if "cmf" in indicators_lower:
        cmf_col = f"CMF_{config.cmf_length}"
        if cmf_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[cmf_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="CMF",
                    ),
                ]
            )
            panel_index += 1

    # EOM 指标
    if "eom" in indicators_lower:
        eom_col = f"EOM_{config.eom_length}"
        if eom_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[eom_col], panel=panel_index, color="brown", ylabel="EOM"
                    ),
                ]
            )
            panel_index += 1

    # PVI 指标
    if "pvi" in indicators_lower:
        if "PVI" in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot["PVI"], panel=panel_index, color="green", ylabel="PVI"
                    ),
                ]
            )
            panel_index += 1

    # NVI 指标
    if "nvi" in indicators_lower:
        if "NVI" in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(df_plot["NVI"], panel=panel_index, color="red", ylabel="NVI"),
                ]
            )
            panel_index += 1

    # 统计指标
    # Z-Score 指标
    if "zscore" in indicators_lower:
        zscore_col = f"ZS_{config.zscore_length}"
        if zscore_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[zscore_col],
                        panel=panel_index,
                        color="blue",
                        ylabel="Z-Score",
                    ),
                ]
            )
            panel_index += 1

    # Kurtosis 指标
    if "kurtosis" in indicators_lower:
        kurtosis_col = f"KURT_{config.kurtosis_length}"
        if kurtosis_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[kurtosis_col],
                        panel=panel_index,
                        color="purple",
                        ylabel="Kurtosis",
                    ),
                ]
            )
            panel_index += 1

    # Skew 指标
    if "skew" in indicators_lower:
        skew_col = f"SKEW_{config.skew_length}"
        if skew_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[skew_col],
                        panel=panel_index,
                        color="orange",
                        ylabel="Skew",
                    ),
                ]
            )
            panel_index += 1

    # Variance 指标
    if "variance" in indicators_lower:
        variance_col = f"VAR_{config.variance_length}"
        if variance_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[variance_col],
                        panel=panel_index,
                        color="brown",
                        ylabel="Variance",
                    ),
                ]
            )
            panel_index += 1

    # Standard Deviation 指标
    if "stdev" in indicators_lower or "std" in indicators_lower:
        stdev_col = f"STDEV_{config.stdev_length}"
        if stdev_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[stdev_col],
                        panel=panel_index,
                        color="navy",
                        ylabel="Std Dev",
                    ),
                ]
            )
            panel_index += 1

    # Median 指标
    if "median" in indicators_lower:
        median_col = f"MEDIAN_{config.median_length}"
        if median_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(df_plot[median_col], panel=0, color="green", linestyle="--"),
                ]
            )

    # MAD 指标
    if "mad" in indicators_lower:
        mad_col = f"MAD_{config.mad_length}"
        if mad_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot[mad_col], panel=panel_index, color="red", ylabel="MAD"
                    ),
                ]
            )
            panel_index += 1

    # 布林带
    if "bbands" in indicators_lower or "bollinger" in indicators_lower:
        bb_upper_col = f"BBU_{config.bb_length}_{config.bb_std}"
        bb_middle_col = f"BBM_{config.bb_length}_{config.bb_std}"
        bb_lower_col = f"BBL_{config.bb_length}_{config.bb_std}"

        if bb_upper_col in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(df_plot[bb_upper_col], panel=0, color="red", linestyle="--"),
                    mpf.make_addplot(df_plot[bb_middle_col], panel=0, color="blue", linestyle="--"),
                    mpf.make_addplot(df_plot[bb_lower_col], panel=0, color="red", linestyle="--"),
                ]
            )

    # 主动买量
    if "taker_buy" in indicators_lower:
        if "Taker_Buy_Base_Asset_Volume" in df_plot.columns:
            ap.extend(
                [
                    mpf.make_addplot(
                        df_plot["Taker_Buy_Base_Asset_Volume"],
                        panel=panel_index,
                        color="green",
                        ylabel="Taker Buy Volume",
                    ),
                ]
            )
            panel_index += 1

    # 2. 执行绘图
    # 动态调整面板比例
    # 计算实际的面板数量：主图(0) + 成交量(1) + 指标面板(从2开始)
    num_panels = panel_index  # panel_index 已经是下一个面板的索引，所以实际面板数就是 panel_index
    panel_ratios = [6, 2] + [3] * (num_panels - 2)  # 主图6，成交量2，其他指标各3
    if config.panel_ratios is not None:
        panel_ratios = config.panel_ratios

    mpf.plot(
        df_plot,
        type="candle",  # K线图样式
        style=config.style,  # 使用配置的样式
        title=config.title,
        ylabel="Price ($)",
        volume=True,  # 显示成交量（默认在第1个附图面板）
        ylabel_lower="Volume",
        addplot=ap,  # 添加我们定义的所有附图
        panel_ratios=tuple(panel_ratios),  # 动态面板比例
        figscale=config.figscale,  # 放大整个图表
        figsize=config.figsize,  # 设置图像大小
        tight_layout=True,
    )
