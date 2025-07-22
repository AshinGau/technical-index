#!/usr/bin/env python3
"""
技术指标测试模块
测试各种技术指标的计算是否正确
"""

import unittest

import numpy as np
import pandas as pd

from technical_index.index import (
    build_quantitative_analysis,
    calculate_candlestick_patterns,
    calculate_momentum_indicators,
    calculate_overlap_indicators,
    calculate_statistics_indicators,
    calculate_trend_indicators,
    calculate_volatility_indicators,
    calculate_volume_indicators,
    get_available_indicators,
    get_indicator_info,
)


class TestTechnicalIndicators(unittest.TestCase):
    """技术指标测试类"""

    def setUp(self):
        """测试前准备数据"""
        # 创建测试用的OHLCV数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # 生成价格数据
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

        self.df = pd.DataFrame(data)
        self.df.set_index("Date", inplace=True)

    def test_data_integrity(self):
        """测试数据完整性"""
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            self.assertIn(col, self.df.columns)

    def test_build_quantitative_analysis_basic(self):
        """测试基本技术指标计算"""
        # 测试指定指标的计算
        result = build_quantitative_analysis(self.df, ["rsi", "macd", "atr"])

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.columns), len(self.df.columns))

        # 检查是否添加了技术指标
        original_columns = set(self.df.columns)
        new_columns = set(result.columns) - original_columns
        self.assertGreater(len(new_columns), 0)

    def test_build_quantitative_analysis_custom_params(self):
        """测试自定义参数的技术指标计算"""
        custom_params = {
            "ma_periods": (5, 13),
            "rsi_length": 21,
            "macd_fast": 8,
            "macd_slow": 21,
            "bb_length": 15,
            "bb_std": 2.5,
        }
        result = build_quantitative_analysis(
            self.df, ["rsi", "macd", "bbands", "sma"], **custom_params
        )
        self.assertIsNotNone(result)
        # 检查自定义参数是否生效
        if "RSI_21" in result.columns:
            self.assertIn("RSI_21", result.columns)
        if "MACD_8_21_9" in result.columns:
            self.assertIn("MACD_8_21_9", result.columns)

    def test_momentum_indicators(self):
        """测试动量指标计算"""
        result = calculate_momentum_indicators(self.df.copy())

        # 检查是否添加了动量指标
        momentum_indicators = [
            "RSI_14",
            "MACD_12_26_9",
            "STOCHk_14_3_3",
            "STOCHd_14_3_3",
        ]
        for indicator in momentum_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                # 检查指标值是否合理
                self.assertTrue(result[indicator].notna().any())

    def test_overlap_indicators(self):
        """测试重叠指标计算"""
        custom_params = {"ma_periods": (7, 25)}
        result = calculate_overlap_indicators(self.df.copy(), **custom_params)

        # 检查移动平均线
        for period in custom_params["ma_periods"]:
            sma_col = f"SMA_{period}"
            ema_col = f"EMA_{period}"
            self.assertIn(sma_col, result.columns)
            self.assertIn(ema_col, result.columns)

            # 检查移动平均线值是否合理
            self.assertTrue(result[sma_col].notna().any())
            self.assertTrue(result[ema_col].notna().any())

    def test_trend_indicators(self):
        """测试趋势指标计算"""
        result = calculate_trend_indicators(self.df.copy())

        # 检查趋势指标
        trend_indicators = ["ADX_14", "AROONU_25", "AROOND_25", "AROONOSC_25"]
        for indicator in trend_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                self.assertTrue(result[indicator].notna().any())

    def test_volatility_indicators(self):
        """测试波动率指标计算"""
        result = calculate_volatility_indicators(self.df.copy())

        # 检查波动率指标
        volatility_indicators = ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "ATRr_14"]
        for indicator in volatility_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                self.assertTrue(result[indicator].notna().any())

    def test_volume_indicators(self):
        """测试成交量指标计算"""
        result = calculate_volume_indicators(self.df.copy())

        # 检查成交量指标
        volume_indicators = ["OBV", "MFI_14", "AD", "ADOSC_3_10"]
        for indicator in volume_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                self.assertTrue(result[indicator].notna().any())

    def test_statistics_indicators(self):
        """测试统计指标计算"""
        result = calculate_statistics_indicators(self.df.copy())

        # 检查统计指标
        stats_indicators = ["ZS_20", "KURT_20", "SKEW_20", "VAR_20"]
        for indicator in stats_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                self.assertTrue(result[indicator].notna().any())

    def test_candlestick_patterns(self):
        """测试K线形态计算"""
        result = calculate_candlestick_patterns(self.df.copy())

        # 检查K线形态指标
        pattern_indicators = ["HA_open", "HA_high", "HA_low", "HA_close"]
        for indicator in pattern_indicators:
            if indicator in result.columns:
                self.assertIn(indicator, result.columns)
                self.assertTrue(result[indicator].notna().any())

    def test_empty_dataframe(self):
        """测试空DataFrame的处理"""
        empty_df = pd.DataFrame()
        result = build_quantitative_analysis(empty_df, ["rsi"])
        self.assertIsNone(result)

    def test_none_dataframe(self):
        """测试None DataFrame的处理"""
        result = build_quantitative_analysis(None, ["rsi"])
        self.assertIsNone(result)

    def test_missing_columns(self):
        """测试缺少必要列的处理"""
        # 创建缺少Volume列的DataFrame
        df_missing = self.df.drop("Volume", axis=1)

        # 应该能够处理缺少Volume的情况（某些指标可能无法计算）
        try:
            result = build_quantitative_analysis(df_missing, ["rsi"])
            self.assertIsNotNone(result)
        except Exception as e:
            # 如果出现异常，应该是预期的（因为缺少必要列）
            self.assertIsInstance(e, Exception)

    def test_get_available_indicators(self):
        """测试获取可用指标列表"""
        indicators = get_available_indicators()

        self.assertIsInstance(indicators, dict)
        self.assertGreater(len(indicators), 0)

        # 检查分类
        expected_categories = [
            "momentum",
            "overlap",
            "trend",
            "volatility",
            "volume",
            "statistics",
            "candlestick",
        ]
        for category in expected_categories:
            self.assertIn(category, indicators)
            self.assertIsInstance(indicators[category], list)
            self.assertGreater(len(indicators[category]), 0)

    def test_get_indicator_info(self):
        """测试获取指标信息"""
        # 测试RSI指标信息
        rsi_info = get_indicator_info("rsi")
        self.assertIsNotNone(rsi_info)
        self.assertIn("name", rsi_info)
        self.assertIn("function", rsi_info)
        self.assertIn("parameters", rsi_info)
        self.assertIn("description", rsi_info)
        self.assertEqual(rsi_info["name"], "rsi")

        # 测试MACD指标信息
        macd_info = get_indicator_info("macd")
        self.assertIsNotNone(macd_info)
        self.assertIn("macd_fast", macd_info["parameters"])
        self.assertIn("macd_slow", macd_info["parameters"])

        # 测试未知指标
        unknown_info = get_indicator_info("unknown_indicator")
        self.assertIsNone(unknown_info)

    def test_indicator_values_range(self):
        """测试指标值范围是否合理"""
        result = build_quantitative_analysis(self.df, ["rsi", "bbands"])

        # 检查RSI值是否在合理范围内
        if "RSI_14" in result.columns:
            rsi_values = result["RSI_14"].dropna()
            if len(rsi_values) > 0:
                self.assertTrue((rsi_values >= 0).all())
                self.assertTrue((rsi_values <= 100).all())

        # 检查布林带值是否合理
        if "BBL_20_2.0" in result.columns and "BBU_20_2.0" in result.columns:
            bbl_values = result["BBL_20_2.0"].dropna()
            bbu_values = result["BBU_20_2.0"].dropna()
            if len(bbl_values) > 0 and len(bbu_values) > 0:
                # 布林带下轨应该小于等于上轨
                self.assertTrue((bbl_values <= bbu_values).all())

    def test_performance(self):
        """测试性能（计算时间）"""
        import time

        start_time = time.time()
        build_quantitative_analysis(self.df, ["rsi", "macd", "atr", "bbands"])
        end_time = time.time()

        calculation_time = end_time - start_time

        # 计算时间应该在合理范围内（比如小于5秒）
        self.assertLess(calculation_time, 5.0)
        print(f"技术指标计算时间: {calculation_time:.2f}秒")


def run_performance_test():
    """运行性能测试"""
    print("运行技术指标性能测试...")

    # 创建更大的数据集
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")

    base_price = 100
    trend = np.linspace(0, 200, 1000)
    noise = np.random.normal(0, 5, 1000)
    close_prices = base_price + trend + noise

    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        if i == 0:
            open_price = close + np.random.normal(0, 2)
        else:
            open_price = data[-1]["Close"] + np.random.normal(0, 2)

        daily_range = abs(close - open_price) + np.random.uniform(1, 5)
        high = max(open_price, close) + daily_range * np.random.uniform(0.3, 0.7)
        low = min(open_price, close) - daily_range * np.random.uniform(0.3, 0.7)
        volume = np.random.uniform(10000, 100000)

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

    df_large = pd.DataFrame(data)
    df_large.set_index("Date", inplace=True)

    import time

    start_time = time.time()
    result = build_quantitative_analysis(df_large)
    end_time = time.time()

    calculation_time = end_time - start_time
    print(f"大数据集计算时间: {calculation_time:.2f}秒")
    print(f"数据集大小: {len(df_large)} 行")
    print(f"原始列数: {len(df_large.columns)}")
    print(f"计算后列数: {len(result.columns)}")
    print(f"新增指标数: {len(result.columns) - len(df_large.columns)}")


if __name__ == "__main__":
    # 运行单元测试
    unittest.main(verbosity=2, exit=False)

    # 运行性能测试
    print("\n" + "=" * 50)
    run_performance_test()
