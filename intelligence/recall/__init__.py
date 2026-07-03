"""
标签联想召回模块

提供基于标签图谱的联想召回策略:
- TagAssociationRecall: 基于标签图谱的联想召回
- CrossDomainRecall: 跨域联想召回

这些策略可与现有5种记忆召回策略并行运行，扩展记忆系统的联想能力。
"""
from __future__ import annotations

from .tag_association_recall import TagAssociationRecall
from .cross_domain_recall import CrossDomainRecall

__all__ = ["TagAssociationRecall", "CrossDomainRecall"]
