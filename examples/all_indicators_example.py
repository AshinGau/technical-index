#!/usr/bin/env python3
"""
所有技术指标示例
展示如何使用 PlotConfig 配置所有可用的技术指标
"""

from technical_index.binance import get_futures_market_data
from technical_index.index import (build_indicator_parameters,
                                   build_quantitative_analysis)
from technical_index.plot import PlotConfig, plot_candlestick_with_indicators


def get_all_available_indicators():
    """
    获取所有可用的技术指标列表
    """
    return {
        "动量指标": [
            "macd",
            "rsi",
            "stochastic",
            "williams_r",
            "cci",
            "roc",
            "momentum",
            "trix",
            "tsi",
            "kdj",
            "fisher",
            "coppock",
            "uo",
        ],
        "趋势指标": ["dmi", "aroon", "psar", "vortex", "vhf", "chop", "ttm_trend"],
        "波动率指标": ["bbands", "atr", "natr", "keltner", "donchian", "massi", "ui"],
        "成交量指标": [
            "mfi",
            "obv",
            "ad",
            "adosc",
            "cmf",
            "eom",
            "pvi",
            "nvi",
            "taker_buy",
        ],
        "统计指标": [
            "zscore",
            "kurtosis",
            "skew",
            "variance",
            "stdev",
            "median",
            "mad",
        ],
    }


def example_momentum_indicators():
    """
    示例：动量指标组合
    """
    print("=== 动量指标组合示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 动量指标配置
    config = PlotConfig(
        indicators=["macd", "rsi", "stochastic", "cci"],
        sma_periods=(7, 25),
        limit=150,
        title="ETHUSDT - Momentum Indicators",
        # 自定义参数
        macd_fast=8,
        macd_slow=21,
        rsi_length=10,
        stoch_k=10,
        stoch_d=3,
        cci_length=15,
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_trend_indicators():
    """
    示例：趋势指标组合
    """
    print("=== 趋势指标组合示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 趋势指标配置
    config = PlotConfig(
        indicators=["dmi", "aroon", "psar", "vortex"],
        ema_periods=(12, 26),
        limit=150,
        title="ETHUSDT - Trend Indicators",
        # 自定义参数
        adx_length=10,
        aroon_length=20,
        vortex_length=10,
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_volatility_indicators():
    """
    示例：波动率指标组合
    """
    print("=== 波动率指标组合示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 波动率指标配置
    config = PlotConfig(
        indicators=["bbands", "atr", "keltner", "donchian"],
        sma_periods=(7, 25),
        limit=150,
        title="ETHUSDT - Volatility Indicators",
        # 自定义参数
        bb_length=15,
        bb_std=2.5,
        atr_length=10,
        kc_length=15,
        kc_std=2.5,
        dc_length=15,
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_volume_indicators():
    """
    示例：成交量指标组合
    """
    print("=== 成交量指标组合示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 成交量指标配置
    config = PlotConfig(
        indicators=["mfi", "obv", "ad", "adosc", "taker_buy"],
        sma_periods=(7, 25),
        limit=150,
        title="ETHUSDT - Volume Indicators",
        # 自定义参数
        mfi_length=10,
        adosc_fast=3,
        adosc_slow=10,
        cmf_length=15,
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_statistical_indicators():
    """
    示例：统计指标组合
    """
    print("=== 统计指标组合示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 统计指标配置
    config = PlotConfig(
        indicators=["zscore", "kurtosis", "skew", "variance", "stdev"],
        sma_periods=(7, 25),
        limit=150,
        title="ETHUSDT - Statistical Indicators",
        # 自定义参数
        zscore_length=15,
        kurtosis_length=15,
        skew_length=15,
        variance_length=15,
        stdev_length=15,
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_comprehensive_analysis():
    """
    示例：综合分析 - 选择最重要的指标
    """
    print("=== 综合分析示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 综合分析配置
    config = PlotConfig(
        indicators=["macd", "rsi", "bbands", "mfi", "atr", "obv"],
        sma_periods=(7, 25),
        limit=200,
        title="ETHUSDT - Comprehensive Analysis",
        figsize=(40, 30),
        figscale=1.2,
        # 自定义参数
        macd_fast=8,
        macd_slow=21,
        rsi_length=10,
        bb_length=15,
        bb_std=2.5,
        mfi_length=10,
        atr_length=10,
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def print_available_indicators():
    """
    打印所有可用的技术指标
    """
    print("=== 所有可用的技术指标 ===")
    indicators = get_all_available_indicators()

    for category, ind_list in indicators.items():
        print(f"\n{category}:")
        for i, indicator in enumerate(ind_list, 1):
            print(f"  {i:2d}. {indicator}")

    print(f"\n总计: {sum(len(ind_list) for ind_list in indicators.values())} 个指标")


def main():
    """
    主函数：运行所有示例
    """
    print("开始运行所有技术指标示例...\n")

    # 打印可用指标
    print_available_indicators()

    print("\n" + "=" * 50)
    print("开始绘制图表示例...")

    # 运行各种示例
    example_momentum_indicators()
    example_trend_indicators()
    example_volatility_indicators()
    example_volume_indicators()
    example_statistical_indicators()
    example_comprehensive_analysis()

    print("\n所有示例运行完成！")
    print("\n使用说明:")
    print("1. 可以通过 indicators 参数选择要绘制的指标")
    print("2. 指标名称不区分大小写")
    print("3. 可以通过相应的参数自定义指标的计算周期")
    print("4. 支持 sma_periods 和 ema_periods 来控制移动平均线")
    print("5. 支持 taker_buy 来显示主动买量")


if __name__ == "__main__":
    main()
