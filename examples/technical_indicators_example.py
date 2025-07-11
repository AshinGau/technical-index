#!/usr/bin/env python3
"""
技术指标使用示例
展示如何使用technical_index模块计算各种技术指标
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# 过滤pandas警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from technical_index.index import (
    build_quantitative_analysis,
    get_available_indicators,
    build_indicator_parameters,
)
from technical_index.binance import get_futures_market_data


def get_ethusdt_data():
    """获取ETHUSDT的期货市场数据"""
    print("正在获取ETHUSDT期货市场数据...")
    df = get_futures_market_data("ETHUSDT", "1h", limit=500)

    if df is None or df.empty:
        print("无法获取ETHUSDT数据，使用模拟数据...")
        return create_sample_data()

    # 此时df已经确定不为None，使用类型断言
    df_typed = df  # type: pd.DataFrame
    print(f"成功获取ETHUSDT数据，形状: {df_typed.shape}")
    print(f"数据时间范围: {df_typed.index[0]} 到 {df_typed.index[-1]}")
    print(f"数据列: {list(df_typed.columns)}")

    return df_typed


def create_sample_data():
    """创建示例OHLCV数据（备用方案）"""
    print("创建模拟OHLCV数据...")
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.normal(0, 2, 100)
    close_prices = base_price + trend + noise

    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        if i == 0:
            open_price = close + np.random.normal(0, 1)
        else:
            open_price = data[-1]["Close"] + np.random.normal(0, 1)

        daily_range = abs(close - open_price) + np.random.uniform(0.5, 2)
        high = max(open_price, close) + daily_range * np.random.uniform(0.3, 0.7)
        low = min(open_price, close) - daily_range * np.random.uniform(0.3, 0.7)
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "Date": date,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    return df


def demonstrate_basic_usage():
    """演示基本使用方法"""
    print("=== 技术指标基本使用示例 ===")

    # 获取ETHUSDT数据
    df = get_ethusdt_data()
    print(f"原始数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())

    # 计算技术指标（使用默认参数）
    df_with_indicators = build_quantitative_analysis(df)
    print(f"\n添加技术指标后数据形状: {df_with_indicators.shape}")

    # 显示新增的技术指标列
    original_columns = set(df.columns)
    new_columns = set(df_with_indicators.columns) - original_columns
    print(f"\n新增的技术指标列数量: {len(new_columns)}")
    print("部分新增指标列:")
    for i, col in enumerate(sorted(new_columns)):
        if i < 20:
            print(f"  {col}")
        else:
            print(f"  ... 还有 {len(new_columns) - 20} 个指标")
            break

    return df_with_indicators


def demonstrate_custom_parameters():
    """演示自定义参数使用"""
    print("\n=== 自定义参数使用示例 ===")

    # 获取ETHUSDT数据
    df = get_ethusdt_data()

    # 获取默认参数
    default_params = build_indicator_parameters()
    print("默认参数示例:")
    print(f"  移动平均线周期: {default_params['ma_periods']}")
    print(f"  RSI周期: {default_params['rsi_length']}")
    print(f"  MACD快线周期: {default_params['macd_fast']}")
    print(f"  布林带周期: {default_params['bb_length']}")

    # 自定义参数
    custom_params = {
        "ma_periods": (5, 13, 21, 55),  # 自定义移动平均线周期
        "rsi_length": 21,  # 自定义RSI周期
        "macd_fast": 8,  # 自定义MACD快线周期
        "macd_slow": 21,  # 自定义MACD慢线周期
        "bb_length": 15,  # 自定义布林带周期
        "bb_std": 2.5,  # 自定义布林带标准差倍数
        "atr_length": 10,  # 自定义ATR周期
    }

    # 构建最终参数（用户参数覆盖默认参数）
    final_params = build_indicator_parameters(**custom_params)
    print(f"\n最终参数示例:")
    print(f"  移动平均线周期: {final_params['ma_periods']}")
    print(f"  RSI周期: {final_params['rsi_length']}")
    print(f"  MACD快线周期: {final_params['macd_fast']}")

    # 计算技术指标（使用自定义参数）
    df_custom = build_quantitative_analysis(df, **custom_params)

    # 显示自定义参数的效果
    print("\n使用自定义参数计算的技术指标:")
    if "RSI_21" in df_custom.columns:
        print(f"RSI_21 最新值: {df_custom['RSI_21'].iloc[-1]:.2f}")
    if "MACD_8_21_9" in df_custom.columns:
        print(f"MACD 最新值: {df_custom['MACD_8_21_9'].iloc[-1]:.4f}")
    if "BBL_15_2.5" in df_custom.columns:
        print(f"布林带下轨 最新值: {df_custom['BBL_15_2.5'].iloc[-1]:.2f}")

    # 检查自定义移动平均线
    for period in custom_params["ma_periods"]:
        sma_col = f"SMA_{period}"
        ema_col = f"EMA_{period}"
        if sma_col in df_custom.columns:
            print(f"{sma_col} 最新值: {df_custom[sma_col].iloc[-1]:.2f}")
        if ema_col in df_custom.columns:
            print(f"{ema_col} 最新值: {df_custom[ema_col].iloc[-1]:.2f}")

    return df_custom


def demonstrate_indicator_categories():
    """演示不同类别的技术指标"""
    print("\n=== 技术指标分类展示 ===")

    # 获取所有可用指标
    indicators = get_available_indicators()

    for category, indicator_list in indicators.items():
        print(f"\n{category.upper()} 指标 ({len(indicator_list)}个):")
        for indicator in indicator_list:
            print(f"  - {indicator}")

    # 获取默认参数
    params = build_indicator_parameters()
    print(f"\n可配置的参数数量: {len(params)}")
    print("部分参数示例:")
    for i, (param, value) in enumerate(params.items()):
        if i < 10:
            print(f"  {param}: {value}")


def main():
    """主函数"""
    print("技术指标计算模块使用示例")
    print("=" * 50)

    # 1. 基本使用示例
    df_with_indicators = demonstrate_basic_usage()

    # 2. 自定义参数示例
    demonstrate_custom_parameters()

    # 3. 指标分类展示
    demonstrate_indicator_categories()

    print("\n" + "=" * 50)
    print("示例完成！")
    print("\n使用说明:")
    print("1. 基本使用: build_quantitative_analysis(df)")
    print("2. 自定义参数: build_quantitative_analysis(df, ma_periods=(5,13,21), rsi_length=21)")
    print("3. 获取可用指标: get_available_indicators()")
    print("4. 构建参数: build_indicator_parameters(ma_periods=(5,13,21), rsi_length=21)")


if __name__ == "__main__":
    main()
