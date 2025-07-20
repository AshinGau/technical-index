from typing import Optional

import pandas as pd

from .base_rule import BaseRule, RuleConfig, SignalResult


class CustomRule(BaseRule):
    """自定义规则基类"""

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.config = config
        self.evaluator = self.parameters.get("evaluator")

    def evaluate(self, df: pd.DataFrame) -> Optional[SignalResult]:
        if self.evaluator and callable(self.evaluator):
            return self.evaluator(df, self.config)
        return None
