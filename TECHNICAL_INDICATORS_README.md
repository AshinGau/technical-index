 # 技术指标计算模块使用指南

## 概述

本模块基于 `pandas_ta` 库提供了全面的技术指标计算功能，包含以下7大类技术指标：

1. **动量指标** - 衡量价格变化速度和强度
2. **重叠指标** - 移动平均线和价格重叠指标
3. **趋势指标** - 识别和确认趋势方向
4. **波动率指标** - 衡量价格波动程度
5. **成交量指标** - 分析成交量与价格关系
6. **统计指标** - 价格数据的统计分析
7. **K线形态** - 识别经典K线形态

## 快速开始

### 基本使用

```python
import pandas as pd
from technical_index.index import build_quantitative_analysis

# 准备OHLCV数据
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# 计算所有技术指标（使用默认参数）
df_with_indicators = build_quantitative_analysis(df)
```

### 自定义参数使用

```python
# 自定义技术指标参数
custom_params = {
    'ma_periods': (5, 13, 21, 55),  # 移动平均线周期
    'rsi_length': 21,               # RSI周期
    'macd_fast': 8,                 # MACD快线周期
    'macd_slow': 21,                # MACD慢线周期
    'bb_length': 15,                # 布林带周期
    'bb_std': 2.5,                  # 布林带标准差倍数
    'atr_length': 10,               # ATR周期
}

# 计算技术指标（使用自定义参数）
df_with_indicators = build_quantitative_analysis(df, **custom_params)
```

## 参数配置

### 获取默认参数

```python
from technical_index.index import get_indicator_parameters

params = get_indicator_parameters()
print(f"移动平均线周期: {params['ma_periods']}")
print(f"RSI周期: {params['rsi_length']}")
print(f"MACD快线周期: {params['macd_fast']}")
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ma_periods` | (7, 25, 99) | 移动平均线周期列表，用于计算SMA和EMA |
| `rsi_length` | 14 | RSI计算周期，标准为14 |
| `macd_fast` | 12 | MACD快线周期，通常为12 |
| `macd_slow` | 26 | MACD慢线周期，通常为26 |
| `macd_signal` | 9 | MACD信号线周期，通常为9 |
| `bb_length` | 20 | 布林带周期 |
| `bb_std` | 2 | 布林带标准差倍数 |
| `atr_length` | 14 | 平均真实波幅周期 |

## 技术指标详细说明

### 1. 动量指标 (Momentum Indicators)

| 指标 | 默认参数 | 用途 |
|------|----------|------|
| **MACD** | fast=12, slow=26, signal=9 | 趋势跟踪动量指标，识别趋势变化和买卖信号 |
| **RSI** | length=14 | 相对强弱指数，识别超买超卖区域 |
| **Stochastic** | k=14, d=3, smooth_k=3 | 随机指标，识别超买超卖和背离信号 |
| **Williams %R** | length=14 | 威廉指标，识别超买超卖区域 |
| **CCI** | length=20 | 商品通道指数，识别超买超卖 |
| **ROC** | length=10 | 变化率，衡量价格变化速度 |
| **Momentum** | length=10 | 动量指标，衡量价格变化幅度 |
| **TRIX** | length=18 | 三重指数平滑移动平均，过滤噪音 |
| **TSI** | fast=13, slow=25 | 真实强度指数，识别趋势变化 |
| **KDJ** | length=9 | 随机指标组合，综合超买超卖信号 |
| **Fisher Transform** | length=9 | 价格正态化指标 |
| **Coppock Curve** | length=10 | 长期动量指标，识别长期底部 |
| **Ultimate Oscillator** | length1=7, length2=14, length3=28 | 终极振荡器，综合多时间框架 |

### 2. 重叠指标 (Overlap Indicators)

| 指标 | 默认参数 | 用途 |
|------|----------|------|
| **SMA** | ma_periods=(7,25,99) | 简单移动平均线，识别趋势方向 |
| **EMA** | ma_periods=(7,25,99) | 指数移动平均线，对近期价格更敏感 |
| **DEMA** | length=20 | 双重指数移动平均线，减少滞后 |
| **TEMA** | length=20 | 三重指数移动平均线，更平滑的趋势线 |
| **HMA** | length=20 | 赫尔移动平均线，减少滞后保持平滑 |
| **WMA** | length=20 | 加权移动平均线，近期数据权重更高 |
| **KAMA** | length=10 | 考夫曼自适应移动平均线，自适应调整 |
| **VWAP** | - | 成交量加权平均价格，日内交易参考 |
| **Ichimoku** | tenkan=9, kijun=26, senkou=52 | 一目均衡表，综合趋势指标 |
| **SuperTrend** | period=10, multiplier=3.0 | 趋势跟踪指标，动态支撑阻力 |

### 3. 趋势指标 (Trend Indicators)

| 指标 | 默认参数 | 用途 |
|------|----------|------|
| **ADX** | length=14 | 平均方向指数，衡量趋势强度 |
| **Aroon** | length=25 | 趋势强度和方向指标 |
| **PSAR** | af0=0.02, af=0.02, max_af=0.2 | 抛物线转向，动态止损 |
| **Vortex** | length=14 | 涡旋指标，识别趋势开始结束 |
| **VHF** | length=28 | 垂直水平过滤器，判断趋势/震荡 |
| **Choppiness** | length=14 | 市场震荡程度指标 |
| **TTM Trend** | length=5 | TTM趋势指标，简化趋势判断 |

### 4. 波动率指标 (Volatility Indicators)

| 指标 | 默认参数 | 用途 |
|------|----------|------|
| **Bollinger Bands** | length=20, std=2 | 布林带，识别超买超卖和波动率 |
| **ATR** | length=14 | 平均真实波幅，设置止损 |
| **NATR** | length=14 | 归一化平均真实波幅，标准化波动率 |
| **Keltner Channel** | length=20, std=2 | 基于ATR的通道，动态支撑阻力 |
| **Donchian Channel** | length=20 | 极值通道，识别支撑阻力 |
| **Mass Index** | length=9 | 质量指数，识别反转信号 |
| **Ulcer Index** | length=14 | 溃疡指数，衡量下行风险 |

### 5. 成交量指标 (Volume Indicators)

| 指标 | 默认参数 | 用途 |
|------|----------|------|
| **OBV** | - | 能量潮，确认价格趋势 |
| **MFI** | length=14 | 资金流量指数，成交量加权RSI |
| **AD** | - | 累积/派发线，判断资金流向 |
| **ADOSC** | fast=3, slow=10 | 累积/派发振荡器 |
| **CMF** | length=20
