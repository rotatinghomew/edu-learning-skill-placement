"""运行参数与统计口径。

所有阈值均为“相对同一可比评分组均值”的 logit 值。能力 logit 为 0
表示该技能可比评分组中学员的平均水平；正数表示高于组内平均。
"""

from dataclasses import dataclass, fields
from statistics import NormalDist


@dataclass(frozen=True)
class Settings:
    score_min: float = 0.0
    score_max: float = 100.0
    confidence: float = 0.90
    min_judgments: int = 2
    min_works: int = 2
    min_component_raters: int = 2
    min_component_students: int = 2
    residual_prior_sd: float = 0.35
    residual_prior_weight: int = 2
    max_iterations: int = 500
    tolerance: float = 1e-9

    def __post_init__(self):
        if not self.score_max > self.score_min:
            raise ValueError("score_max 必须大于 score_min")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence 必须在 0.5 与 1.0 之间")
        for name in ("min_judgments", "min_works", "min_component_raters",
                     "min_component_students", "residual_prior_weight"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须为正整数")
        if self.residual_prior_sd <= 0:
            raise ValueError("residual_prior_sd 必须为正数")

    @property
    def z(self):
        """双侧置信区间的正态分位数。"""
        return NormalDist().inv_cdf(0.5 + self.confidence / 2)

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values):
        values = values or {}
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"未知配置项: {', '.join(sorted(unknown))}")
        return cls(**values)
