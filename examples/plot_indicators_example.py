#!/usr/bin/env python3
"""
绘制技术指标示例
演示如何使用 plot_candlestick_with_indicators 函数和 PlotConfig 配置对象
"""

from technical_index.binance import get_futures_market_data
from technical_index.index import (build_indicator_parameters,
                                   build_quantitative_analysis)
from technical_index.plot import PlotConfig, plot_candlestick_with_indicators


def main():
    """
    主函数：获取ETHUSDT数据并绘制技术指标图表
    """
    print("正在获取ETHUSDT期货市场数据...")

    # 获取ETHUSDT的1小时K线数据，限制500条
    df = get_futures_market_data("ETHUSDT", "1h", limit=500)

    if df is None or df.empty:
        print("获取数据失败或数据为空")
        return

    print(f"成功获取 {len(df)} 条数据")
    print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")

    # 构建技术指标参数
    indicator_params = build_indicator_parameters(ma_periods=(7, 25, 99))

    # 计算技术指标
    print("正在计算技术指标...")
    df_with_indicators = build_quantitative_analysis(df, **indicator_params)

    if df_with_indicators is None:
        print("计算技术指标失败")
        return

    print("技术指标计算完成")
    print(f"数据列数: {len(df_with_indicators.columns)}")

    # 示例1：使用默认配置绘制图表
    print("\n=== 示例1：使用默认配置 ===")
    config1 = PlotConfig(
        indicators=["macd", "dmi", "mfi", "obv"],
        sma_periods=(7, 25, 99),
        limit=150,  # 绘制SMA线
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config1)

    # 示例2：自定义配置 - 只绘制MACD和RSI
    print("\n=== 示例2：自定义配置 - 只绘制MACD和RSI ===")
    config2 = PlotConfig(
        indicators=["macd", "rsi"],
        ema_periods=(12, 26),  # 绘制EMA线
        limit=200,
        title="ETHUSDT - MACD & RSI Analysis",
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config2)

    # 示例3：包含主动买量指标
    print("\n=== 示例3：包含主动买量指标 ===")
    config3 = PlotConfig(
        indicators=["macd", "rsi", "taker_buy"],
        sma_periods=(7, 25),
        limit=150,
        title="ETHUSDT - With Taker Buy Volume",
    )

    plot_candlestick_with_indicators(df_with_indicators, config=config3)

    # 示例4：使用kwargs方式传递配置
    print("\n=== 示例4：使用kwargs方式 ===")
    plot_candlestick_with_indicators(
        df_with_indicators,
        indicators=["macd", "dmi", "mfi", "taker_buy"],
        sma_periods=(7, 25),
        limit=100,
        title="ETHUSDT - Custom Analysis with Taker Buy",
    )

    print("所有图表绘制完成！")


if __name__ == "__main__":
    main()
