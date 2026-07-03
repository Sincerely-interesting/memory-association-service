"""
三级标签域管理系统

提供货品(Commodity)、人群(Crowd)、平台(Platform)三个标签域的:
- 预置标签初始化
- 标签CRUD操作
- 层级查询（祖先/后代/兄弟）
- 域内筛选与统计

5C联想引擎:
- 单域查询 / 双域交叉 / 全链路5C匹配
- 质心向量计算 / 近邻标签查找
"""
from __future__ import annotations

from .commodity_domain import CommodityDomain
from .crowd_domain import CrowdDomain
from .platform_domain import PlatformDomain
from .centroid_calculator import CentroidCalculator
from .association_engine import (
    AssociationEngine, TagQuery, AssociationResult, Full5CResult,
)

__all__ = [
    "CommodityDomain", "CrowdDomain", "PlatformDomain",
    "CentroidCalculator",
    "AssociationEngine", "TagQuery", "AssociationResult", "Full5CResult",
]
