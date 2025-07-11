#!/usr/bin/env python3
"""
PlotConfig 配置示例
展示如何使用 PlotConfig 类来配置技术指标图表
"""

from technical_index.binance import get_futures_market_data
from technical_index.index import (build_indicator_parameters,
                                   build_quantitative_analysis)
from technical_index.plot import PlotConfig, plot_candlestick_with_indicators


def example_basic_config():
    """
    示例1：基本配置
    """
    print("=== 基本配置示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 基本配置
    config = PlotConfig(
        indicators=["macd", "rsi"],  # 只绘制MACD和RSI
        sma_periods=(7, 25),  # 在主图绘制SMA线
        limit=150,  # 显示最近150条数据
        title="ETHUSDT - Basic Analysis",
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_advanced_config():
    """
    示例2：高级配置
    """
    print("=== 高级配置示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 高级配置
    config = PlotConfig(
        indicators=["macd", "dmi", "mfi", "obv", "rsi", "bbands"],  # 多个指标
        ema_periods=(12, 26),  # 绘制EMA线
        limit=200,  # 显示更多数据
        figsize=(40, 30),  # 更大的图表
        figscale=2.0,  # 更大的缩放
        title="ETHUSDT - Advanced Technical Analysis",
        # 自定义技术指标参数
        macd_fast=8,  # 更快的MACD
        macd_slow=21,  # 更慢的MACD
        adx_length=10,  # 更短的ADX
        mfi_length=10,  # 更短的MFI
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config)


def example_custom_indicators():
    """
    示例3：自定义指标组合
    """
    print("=== 自定义指标组合示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 不同的指标组合
    configs = [
        {
            "name": "趋势分析",
            "config": PlotConfig(
                indicators=["macd", "dmi"],
                sma_periods=(7, 25, 99),
                title="ETHUSDT - Trend Analysis",
            ),
        },
        {
            "name": "动量分析",
            "config": PlotConfig(
                indicators=["rsi", "mfi"],
                ema_periods=(12, 26),
                title="ETHUSDT - Momentum Analysis",
            ),
        },
        {
            "name": "成交量分析",
            "config": PlotConfig(
                indicators=["obv", "mfi"],
                sma_periods=(7, 25),
                title="ETHUSDT - Volume Analysis",
            ),
        },
    ]

    for config_info in configs:
        print(f"绘制 {config_info['name']} 图表...")
        plot_candlestick_with_indicators(df_with_indicators, config=config_info["config"])


def example_kwargs_usage():
    """
    示例4：使用kwargs方式
    """
    print("=== kwargs使用示例 ===")

    # 获取数据
    df = get_futures_market_data("ETHUSDT", "1h", limit=300)
    if df is None or df.empty:
        return

    # 计算技术指标
    params = build_indicator_parameters(ma_periods=(7, 25, 99))
    df_with_indicators = build_quantitative_analysis(df, **params)

    # 直接使用kwargs参数
    plot_candlestick_with_indicators(
        df_with_indicators,
        indicators=["macd", "rsi", "bbands"],
        sma_periods=(7, 25),
        limit=100,
        title="ETHUSDT - kwargs Example",
        figsize=(30, 20),
    )


def main():
    """
    主函数：运行所有示例
    """
    print("开始运行PlotConfig配置示例...\n")

    # 运行各种示例
    example_basic_config()
    example_advanced_config()
    example_custom_indicators()
    example_kwargs_usage()

    print("\n所有示例运行完成！")


if __name__ == "__main__":
    main()
