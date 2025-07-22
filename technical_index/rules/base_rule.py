from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


class SignalType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    ALERT = "alert"


class RuleType(Enum):
    PRICE_BASED = "price_based"
    TECHNICAL_INDICATOR = "technical"
    CUSTOM = "custom"


@dataclass
class SignalResult:
    symbol: str  # 交易对符号，如 "BTCUSDT"
    rule_name: str  # 触发信号的规则名称
    signal_type: SignalType  # 信号类型：看涨(bullish)、看跌(bearish)、中性(neutral)、警报(alert)
    timestamp: datetime  # 信号生成的时间戳
    current_price: float  # 当前价格
    interval: str  # 时间周期，如 "1h", "4h", "1d"
    confidence: float = 0.0  # 信号置信度，0.0-1.0，表示信号的可信程度
    duration: Optional[int] = None  # 信号持续时间（分钟），用于判断信号的持续性
    target_price: Optional[float] = None  # 目标价格，预期达到的价格水平
    stop_loss: Optional[float] = None  # 止损价格，建议的止损位
    take_profit: Optional[float] = None  # 止盈价格，建议的止盈位
    resistance_level: Optional[float] = None  # 阻力位价格，重要的阻力水平
    support_level: Optional[float] = None  # 支撑位价格，重要的支撑水平
    additional_signals: List[str] = field(
        default_factory=list
    )  # 额外的信号信息列表，包含其他相关信号
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据字典，存储规则特定的额外信息

    def format_price_change(self, compare: float, origin: Optional[float] = None) -> str:
        if origin is None:
            origin = self.current_price

        change_percent = ((compare - origin) / origin) * 100
        change_sign = "+" if change_percent >= 0 else ""
        return f"{compare:.{4}f} ({change_sign}{change_percent:.2f}%)"


@dataclass
class RuleConfig:
    name: str
    rule_type: RuleType
    symbol: str
    interval: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable[[SignalResult], None]] = None


class BaseRule(ABC):
    def __init__(self, config: RuleConfig):
        self.config = config
        self.name = config.name
        self.symbol = config.symbol
        self.interval = config.interval
        self.parameters = config.parameters
        self.enabled = config.enabled

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        pass

    def get_rule_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.config.rule_type.value,
            "symbol": self.symbol,
            "interval": self.interval,
            "enabled": self.enabled,
            "parameters": self.parameters,
        }

    def create_signal(
        self, signal_type: SignalType, current_price: float, **kwargs
    ) -> SignalResult:
        """创建信号结果，自动包含 interval 信息"""
        return SignalResult(
            symbol=self.symbol,
            rule_name=self.name,
            signal_type=signal_type,
            timestamp=datetime.now(),
            current_price=current_price,
            interval=self.interval,
            **kwargs,
        )
