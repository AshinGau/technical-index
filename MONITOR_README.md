# 币价监控规则系统

一个优雅的币价监控规则系统，支持动态扩展规则、多交易对监控、基于价格变化和技术指标的信号生成。

## 功能特性

### 🎯 核心功能
- **动态规则系统**: 支持运行时添加、移除、修改监控规则
- **多交易对监控**: 同时监控多个交易对，每个交易对可设置独立规则
- **全局规则**: 支持跨交易对的通用规则
- **智能信号**: 基于技术指标和价格变化的智能信号生成
- **风险控制**: 自动计算止损、止盈、目标价格和支撑阻力位

### 📊 监控规则类型

#### 价格变化规则
- **价格波动监控**: 检测价格大幅波动（可配置阈值）
- **价格突破监控**: 识别支撑阻力位突破
- **新高新低监控**: 检测历史新高或新低

#### 技术指标规则
- **MACD金叉死叉**: 移动平均收敛发散指标信号
- **RSI超买超卖**: 相对强弱指数信号
- **趋势分析**: 基于移动平均线的趋势识别
- **自定义规则**: 支持用户自定义技术指标规则

### ⚙️ 系统特性
- **异步监控**: 基于asyncio的高效异步监控
- **配置管理**: JSON配置文件支持
- **信号持久化**: 自动保存信号到文件
- **灵活回调**: 支持自定义信号处理回调函数
- **日志记录**: 完整的日志记录和错误处理

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化配置

```bash
python -m technical_index.monitor_cli init
```

这将创建默认的配置文件 `config/monitor_config.json`。

### 3. 启动监控

```bash
python -m technical_index.monitor_cli start
```

### 4. 查看配置

```bash
python -m technical_index.monitor_cli show
```

## 详细使用指南

### 配置文件结构

```json
{
  "monitor": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "interval": "1h",
    "check_interval_minutes": 15,
    "enabled": true,
    "log_level": "INFO",
    "save_signals": true,
    "signal_file": "signals.json"
  },
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "interval": "1h",
      "rules": [
        {
          "name": "价格波动监控",
          "rule_type": "price_based",
          "enabled": true,
          "parameters": {
            "volatility_threshold": 0.03
          }
        }
      ]
    }
  ],
  "global_rules": []
}
```

### 命令行工具

#### 初始化配置
```bash
python -m technical_index.monitor_cli init --config config/my_config.json
```

#### 启动监控
```bash
python -m technical_index.monitor_cli start --config config/my_config.json
```

#### 添加交易对
```bash
python -m technical_index.monitor_cli add BTCUSDT --interval 1h
```

#### 移除交易对
```bash
python -m technical_index.monitor_cli remove BTCUSDT
```

#### 查看历史信号
```bash
python -m technical_index.monitor_cli signals --file log/signals.json
```

### 编程接口

#### 基本使用

```python
import asyncio
from technical_index import (
    RuleEngine, PriceMonitor, create_price_volatility_rule,
    create_macd_rule, SignalResult
)

async def main():
    # 创建规则引擎
    rule_engine = RuleEngine()
    
    # 添加规则
    volatility_rule = create_price_volatility_rule("BTCUSDT", "1h", 0.03)
    macd_rule = create_macd_rule("BTCUSDT", "1h")
    
    rule_engine.add_rule(volatility_rule)
    rule_engine.add_rule(macd_rule)
    
    # 创建监控器
    monitor = PriceMonitor(rule_engine)
    
    # 自定义回调函数
    def my_callback(signal: SignalResult):
        print(f"信号触发: {signal.symbol} - {signal.rule_name}")
        print(f"信号类型: {signal.signal_type.value}")
        print(f"当前价格: {signal.current_price}")
        if signal.target_price:
            print(f"目标价格: {signal.target_price}")
        if signal.stop_loss:
            print(f"止损价格: {signal.stop_loss}")
    
    # 添加交易对
    monitor.add_symbol("BTCUSDT", my_callback)
    
    # 启动监控
    await monitor.start_monitoring()

# 运行
asyncio.run(main())
```

#### 使用配置文件

```python
from technical_index import ConfigManager, load_rules_from_config

# 加载配置
config_manager = ConfigManager("config/monitor_config.json")
config = config_manager.load_config()

# 从配置创建规则
rules = load_rules_from_config(config)
for rule, is_global in rules:
    rule_engine.add_rule(rule, is_global)
```

#### 创建自定义规则

```python
from technical_index import CustomRule, RuleConfig, RuleType, SignalResult, SignalType

def my_custom_rule(df, config):
    """自定义规则示例"""
    if len(df) < 20:
        return None
    
    current_price = df['Close'].iloc[-1]
    avg_price = df['Close'].rolling(window=20).mean().iloc[-1]
    
    # 价格突破20日均线
    if current_price > avg_price * 1.02:
        return SignalResult(
            symbol=config.symbol,
            rule_name=config.name,
            signal_type=SignalType.BULLISH,
            timestamp=df.index[-1],
            current_price=current_price,
            confidence=0.8,
            target_price=current_price * 1.05,
            stop_loss=avg_price,
            additional_signals=["关注成交量确认"]
        )
    return None

# 创建自定义规则
custom_rule = CustomRule(RuleConfig(
    name="自定义突破规则",
    rule_type=RuleType.CUSTOM,
    symbol="BTCUSDT",
    interval="1h",
    parameters={"evaluator": my_custom_rule}
))
```

## 信号类型说明

### SignalType 枚举
- `BULLISH`: 看涨信号 - 建议做多
- `BEARISH`: 看跌信号 - 建议做空  
- `NEUTRAL`: 中性信号 - 观望为主
- `ALERT`: 预警信号 - 需要关注

### SignalResult 字段
- `symbol`: 交易对名称
- `rule_name`: 触发规则的名称
- `signal_type`: 信号类型
- `timestamp`: 信号时间戳
- `current_price`: 当前价格
- `confidence`: 置信度 (0-1)
- `duration`: 预期持续时间（周期数）
- `target_price`: 目标价格
- `stop_loss`: 止损价格
- `take_profit`: 止盈价格
- `resistance_level`: 阻力位
- `support_level`: 支撑位
- `additional_signals`: 额外关注信号列表
- `metadata`: 额外元数据

## 规则参数说明

### 价格波动规则 (PriceVolatilityRule)
- `volatility_threshold`: 波动阈值，默认0.05 (5%)
- `lookback_periods`: 回看周期数，默认20

### 价格突破规则 (PriceBreakoutRule)
- `resistance_periods`: 阻力位计算周期，默认20
- `support_periods`: 支撑位计算周期，默认20
- `breakout_threshold`: 突破阈值，默认0.02 (2%)

### MACD规则 (MACDGoldenCrossRule)
- `fast_period`: 快线周期，默认12
- `slow_period`: 慢线周期，默认26
- `signal_period`: 信号线周期，默认9

### RSI规则 (RSISignalRule)
- `oversold_threshold`: 超卖阈值，默认30
- `overbought_threshold`: 超买阈值，默认70
- `rsi_period`: RSI计算周期，默认14

### 趋势分析规则 (TrendAnalysisRule)
- `short_ma`: 短期移动平均线周期，默认7
- `long_ma`: 长期移动平均线周期，默认25
- `trend_periods`: 趋势判断周期，默认10

## 监控间隔说明

系统根据不同的时间间隔设置不同的检查频率：

- `interval="1d"`: 每小时检查一次
- `interval="1h"`: 每15分钟检查一次
- `interval="15m"`: 每5分钟检查一次
- 其他间隔: 每5分钟检查一次

## 最佳实践

### 1. 规则组合
建议组合使用不同类型的规则：
- 价格规则 + 技术指标规则
- 短期规则 + 长期规则
- 趋势规则 + 反转规则

### 2. 参数调优
- 根据交易对特性调整参数
- 考虑市场波动性设置阈值
- 定期回测和优化规则参数

### 3. 风险控制
- 设置合理的止损止盈
- 关注信号置信度
- 结合多个信号确认

### 4. 性能优化
- 合理设置检查间隔
- 避免过多交易对同时监控
- 定期清理历史数据

## 故障排除

### 常见问题

1. **无法获取市场数据**
   - 检查网络连接
   - 确认交易对名称正确
   - 检查币安API状态

2. **规则未触发**
   - 检查规则参数设置
   - 确认规则已启用
   - 查看日志输出

3. **监控停止**
   - 检查配置文件
   - 查看错误日志
   - 确认交易对存在

### 日志级别
- `DEBUG`: 详细调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息

## 扩展开发

### 添加新规则类型

1. 继承 `BaseRule` 类
2. 实现 `evaluate` 方法
3. 在 `load_rules_from_config` 中添加支持

### 自定义回调函数

```python
def advanced_callback(signal: SignalResult):
    # 发送邮件通知
    send_email_notification(signal)
    
    # 记录到数据库
    save_to_database(signal)
    
    # 触发交易信号
    if signal.confidence > 0.8:
        execute_trade_signal(signal)
```

### 集成其他数据源

可以扩展 `binance.py` 模块或创建新的数据源模块来支持其他交易所。

## 许可证

本项目采用 MIT 许可证。 