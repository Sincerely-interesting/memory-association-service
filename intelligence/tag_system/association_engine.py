"""
5C联想引擎 (AssociationEngine)

核心联想匹配算法，支持:
- 单域查询: 从一个域的标签联想匹配其他域
- 双域交叉: 两个域的标签交叉匹配
- 全链路5C匹配: 补全缺失的5C维度
- 多策略融合: 语义/共现/规则/时间衰减
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.models import (
        TagNode, TagAssociation, CrossDomainLink,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
        ALL_DOMAINS, ASSOC_SEMANTIC, ASSOC_COOCCURRENCE,
        ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
        RULE_CONSTRAINT, RULE_BOOST, RULE_FILTER,
    )
    from core.tag_graph import TagAssociationGraph
    from infrastructure.tag_embedding import TagEmbeddingService
    from .centroid_calculator import CentroidCalculator
except ImportError:
    from models import (
        TagNode, TagAssociation, CrossDomainLink,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
        ALL_DOMAINS, ASSOC_SEMANTIC, ASSOC_COOCCURRENCE,
        ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
        RULE_CONSTRAINT, RULE_BOOST, RULE_FILTER,
    )
    from tag_graph import TagAssociationGraph
    from tag_embedding import TagEmbeddingService
    from centroid_calculator import CentroidCalculator


# ── 数据结构 ──────────────────────────────────────────────────────

@dataclass
class TagQuery:
    """标签查询"""
    # 单域查询
    source_domain: str | None = None
    source_tag_names: list[str] = field(default_factory=list)

    # 双域交叉
    target_domain: str | None = None
    target_tag_names: list[str] = field(default_factory=list)

    # 过滤条件
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # 确保列表
        if isinstance(self.source_tag_names, str):
            self.source_tag_names = [self.source_tag_names]
        if isinstance(self.target_tag_names, str):
            self.target_tag_names = [self.target_tag_names]


@dataclass
class AssociationResult:
    """联想结果"""
    tag: TagNode                         # 匹配的标签
    score: float                         # 综合得分 0-1
    strategy: str                        # 来源策略
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tag_id": self.tag.id,
            "tag_name": self.tag.name,
            "tag_domain": self.tag.domain,
            "score": round(self.score, 4),
            "strategy": self.strategy,
            "context": self.context,
        }


@dataclass
class Full5CResult:
    """完整5C匹配结果"""
    content: list[AssociationResult] = field(default_factory=list)
    channel: list[AssociationResult] = field(default_factory=list)
    crowd: list[AssociationResult] = field(default_factory=list)
    commodity: list[AssociationResult] = field(default_factory=list)
    conversion_score: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "content": [r.to_dict() for r in self.content],
            "channel": [r.to_dict() for r in self.channel],
            "crowd": [r.to_dict() for r in self.crowd],
            "commodity": [r.to_dict() for r in self.commodity],
            "conversion_score": round(self.conversion_score, 4),
            "confidence": round(self.confidence, 4),
        }


# ── 5C维度映射 ──────────────────────────────────────────────────

# 5C维度 → 标签域映射
DIMENSION_DOMAIN_MAP = {
    "content": DOMAIN_CROWD,      # 内容偏好在人群域
    "channel": DOMAIN_PLATFORM,   # 渠道在平台域
    "crowd": DOMAIN_CROWD,        # 人群在人群域
    "commodity": DOMAIN_COMMODITY, # 货品在货品域
}

# 5C维度 → 子域映射
DIMENSION_SUBDOMAIN_MAP = {
    "content": ["content_preference"],
    "channel": ["platform", "content_feature"],
    "crowd": ["demographics", "interest", "behavior", "lifecycle"],
    "commodity": ["category", "price_band", "style", "scenario", "efficacy"],
}


class AssociationEngine:
    """
    5C联想引擎

    核心匹配算法流程:
        1. 解析查询 → 获取源标签
        2. 多策略并行匹配
        3. 融合去重排序
        4. 返回Top-K结果
    """

    def __init__(
        self,
        graph: TagAssociationGraph,
        embedding_service: TagEmbeddingService,
        centroid_calculator: CentroidCalculator | None = None,
    ):
        self.graph = graph
        self.embedding_service = embedding_service
        self.centroid_calculator = centroid_calculator

        # 策略权重
        self.strategy_weights = {
            "graph_walk": 0.35,       # 图遍历（邻接表）
            "semantic": 0.30,         # 语义相似度
            "cooccurrence": 0.20,     # 共现关系
            "rule": 0.15,             # 跨域规则
        }

        # 配置
        self.config = {
            "min_score": 0.1,
            "max_results": 20,
            "time_decay_half_life_days": 30.0,
        }

    # ── 单域查询 ──────────────────────────────────────────────────

    def single_domain_query(
        self,
        query: TagQuery,
        top_k: int = 10,
    ) -> list[AssociationResult]:
        """
        单域查询: 从源域标签联想匹配其他域

        例: 输入货品标签 → 返回匹配的人群+平台
        """
        if not query.source_tag_names:
            return []

        # 1. 解析源标签
        source_tags = self._resolve_tags(query.source_tag_names, query.source_domain)
        if not source_tags:
            return []

        # 2. 确定目标域
        target_domains = self._get_other_domains(query.source_domain)

        # 3. 多策略匹配
        results = []
        for target_domain in target_domains:
            candidates = self._get_domain_tags(target_domain, query.filters)
            for strategy_name, strategy_func in [
                ("graph_walk", self._strategy_graph_walk),
                ("semantic", self._strategy_semantic),
                ("cooccurrence", self._strategy_cooccurrence),
            ]:
                strategy_results = strategy_func(source_tags, candidates)
                for tag, score in strategy_results:
                    results.append(AssociationResult(
                        tag=tag,
                        score=score * self.strategy_weights.get(strategy_name, 0.1),
                        strategy=strategy_name,
                        context={"source_domain": query.source_domain, "target_domain": target_domain},
                    ))

        # 4. 融合去重排序
        results = self._fuse_and_rank(results)
        return results[:top_k]

    # ── 双域交叉 ──────────────────────────────────────────────────

    def cross_domain_query(
        self,
        query: TagQuery,
        top_k: int = 10,
    ) -> list[AssociationResult]:
        """
        双域交叉: 两个域的标签交叉匹配

        例: 输入人群+平台 → 返回匹配的货品+内容
        """
        if not query.source_tag_names or not query.target_tag_names:
            return []

        source_tags = self._resolve_tags(query.source_tag_names, query.source_domain)
        target_tags = self._resolve_tags(query.target_tag_names, query.target_domain)

        if not source_tags or not target_tags:
            return []

        # 查找同时匹配源和目标的标签
        results = []

        # 通过源标签的邻接找到中间标签
        intermediate_tags = set()
        for src_tag in source_tags:
            for neighbor_id, strength in self.graph.get_neighbors(src_tag.id):
                neighbor = self.graph.tags.get(neighbor_id)
                if neighbor:
                    intermediate_tags.add(neighbor)

        # 通过目标标签的邻接找到中间标签
        for tgt_tag in target_tags:
            for neighbor_id, strength in self.graph.get_neighbors(tgt_tag.id):
                neighbor = self.graph.tags.get(neighbor_id)
                if neighbor:
                    intermediate_tags.add(neighbor)

        # 对中间标签计算综合得分
        for tag in intermediate_tags:
            score = 0.0
            # 源标签相似度
            for src_tag in source_tags:
                sim = self.embedding_service.tag_similarity(src_tag, tag)
                score = max(score, sim)
            # 目标标签相似度
            for tgt_tag in target_tags:
                sim = self.embedding_service.tag_similarity(tgt_tag, tag)
                score = max(score, sim)
            # 与源域和目标域的关联强度
            for src_tag in source_tags:
                assoc = self.graph.get_association(src_tag.id, tag.id)
                if assoc:
                    score = max(score, assoc.strength)

            if score >= self.config["min_score"]:
                results.append(AssociationResult(
                    tag=tag,
                    score=score,
                    strategy="cross_domain",
                    context={
                        "source_domain": query.source_domain,
                        "target_domain": query.target_domain,
                    },
                ))

        results = self._fuse_and_rank(results)
        return results[:top_k]

    # ── 5C匹配 ───────────────────────────────────────────────────

    def match_5c(
        self,
        partial_5c: dict[str, list[str] | None],
        top_k: int = 5,
    ) -> list[Full5CResult]:
        """
        全链路5C匹配

        输入部分5C维度，补全缺失维度。

        Args:
            partial_5c: {"content": [...], "channel": None, "crowd": [...], "commodity": [...]}
            top_k: 返回Top-K组合
        """
        # 识别已提供和缺失的维度
        provided = {k: v for k, v in partial_5c.items() if v is not None}
        missing = [k for k, v in partial_5c.items() if v is None]

        if not missing:
            # 已完整，直接构建结果
            return [self._build_5c_result(provided, 1.0)]

        # 收集所有已知标签
        known_tags = []
        for dim, tag_names in provided.items():
            domain = DIMENSION_DOMAIN_MAP.get(dim)
            tags = self._resolve_tags(tag_names, domain)
            known_tags.extend(tags)

        if not known_tags:
            return []

        # 对每个缺失维度进行联想填充
        filled: dict[str, list[AssociationResult]] = {}
        for dim in missing:
            domain = DIMENSION_DOMAIN_MAP.get(dim)
            candidates = self._get_domain_tags(domain)
            dim_results = []

            for src_tag in known_tags:
                # 图遍历
                for neighbor_id, strength in self.graph.get_neighbors(src_tag.id):
                    neighbor = self.graph.tags.get(neighbor_id)
                    if neighbor and neighbor.domain == domain:
                        dim_results.append(AssociationResult(
                            tag=neighbor,
                            score=strength,
                            strategy="graph_walk",
                        ))
                # 语义相似度
                for cand in candidates:
                    sim = self.embedding_service.tag_similarity(src_tag, cand)
                    if sim >= self.config["min_score"]:
                        dim_results.append(AssociationResult(
                            tag=cand,
                            score=sim,
                            strategy="semantic",
                        ))

            dim_results = self._fuse_and_rank(dim_results)
            filled[dim] = dim_results[:top_k]

        # 构建5C组合
        results = self._build_5c_combinations(provided, filled, top_k)
        return results

    # ── 策略实现 ──────────────────────────────────────────────────

    def _strategy_graph_walk(
        self,
        source_tags: list[TagNode],
        candidates: list[TagNode],
    ) -> list[tuple[TagNode, float]]:
        """图遍历策略: 通过邻接表找到关联标签"""
        results = []
        candidate_ids = {c.id for c in candidates}

        for src_tag in source_tags:
            for neighbor_id, strength in self.graph.get_neighbors(src_tag.id):
                if neighbor_id in candidate_ids:
                    neighbor = self.graph.tags.get(neighbor_id)
                    if neighbor:
                        results.append((neighbor, strength))
        return results

    def _strategy_semantic(
        self,
        source_tags: list[TagNode],
        candidates: list[TagNode],
    ) -> list[tuple[TagNode, float]]:
        """语义相似度策略"""
        results = []
        for src_tag in source_tags:
            src_vec = self.embedding_service.encode_tag(src_tag)
            for cand in candidates:
                if cand.id == src_tag.id:
                    continue
                cand_vec = self.embedding_service.encode_tag(cand)
                sim = self.embedding_service.cosine_similarity(src_vec, cand_vec)
                if sim >= self.config["min_score"]:
                    results.append((cand, sim))
        return results

    def _strategy_cooccurrence(
        self,
        source_tags: list[TagNode],
        candidates: list[TagNode],
    ) -> list[tuple[TagNode, float]]:
        """共现关系策略"""
        results = []
        candidate_ids = {c.id for c in candidates}

        for assoc in self.graph.associations:
            src_match = None
            tgt_match = None

            if assoc.source_tag_id in {t.id for t in source_tags}:
                src_match = assoc.source_tag_id
                tgt_match = assoc.target_tag_id
            elif assoc.target_tag_id in {t.id for t in source_tags}:
                src_match = assoc.target_tag_id
                tgt_match = assoc.source_tag_id

            if tgt_match in candidate_ids:
                tag = self.graph.tags.get(tgt_match)
                if tag:
                    results.append((tag, assoc.strength))
        return results

    # ── 辅助方法 ──────────────────────────────────────────────────

    def _resolve_tags(self, tag_names: list[str], domain: str | None = None) -> list[TagNode]:
        """解析标签名称为标签节点"""
        tags = []
        for name in tag_names:
            for tag_id, tag in self.graph.tags.items():
                if tag.name == name:
                    if domain is None or tag.domain == domain:
                        tags.append(tag)
                        break
        return tags

    def _get_domain_tags(self, domain: str, filters: dict | None = None) -> list[TagNode]:
        """获取某个域的标签"""
        tags = self.graph.get_tags_by_domain(domain)
        if filters:
            level = filters.get("level")
            if level is not None:
                tags = [t for t in tags if t.level == level]
        return tags

    def _get_other_domains(self, exclude_domain: str | None = None) -> list[str]:
        """获取其他域"""
        if exclude_domain:
            return [d for d in ALL_DOMAINS if d != exclude_domain]
        return list(ALL_DOMAINS)

    def _fuse_and_rank(self, results: list[AssociationResult]) -> list[AssociationResult]:
        """融合去重排序"""
        # 按tag_id去重，保留最高分
        best: dict[str, AssociationResult] = {}
        for r in results:
            key = r.tag.id
            if key not in best or r.score > best[key].score:
                best[key] = r

        # 按得分排序
        ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return ranked

    def _build_5c_result(
        self,
        provided: dict[str, list[str]],
        confidence: float,
    ) -> Full5CResult:
        """构建完整5C结果"""
        result = Full5CResult(confidence=confidence)
        for dim, tag_names in provided.items():
            domain = DIMENSION_DOMAIN_MAP.get(dim)
            tags = self._resolve_tags(tag_names, domain)
            assoc_results = [
                AssociationResult(tag=t, score=1.0, strategy="provided")
                for t in tags
            ]
            setattr(result, dim, assoc_results)
        return result

    def _build_5c_combinations(
        self,
        provided: dict[str, list[str]],
        filled: dict[str, list[AssociationResult]],
        top_k: int,
    ) -> list[Full5CResult]:
        """构建5C组合"""
        results = []

        # 简化: 取每个维度的Top-1组合
        all_dims = list(DIMENSION_DOMAIN_MAP.keys())
        dim_options: dict[str, list[AssociationResult]] = {}
        for dim in all_dims:
            if dim in provided:
                domain = DIMENSION_DOMAIN_MAP.get(dim)
                tags = self._resolve_tags(provided[dim], domain)
                dim_options[dim] = [
                    AssociationResult(tag=t, score=1.0, strategy="provided")
                    for t in tags
                ] or [AssociationResult(
                    tag=TagNode(id="empty", name="", name_en=""),
                    score=0.0, strategy="empty"
                )]
            elif dim in filled and filled[dim]:
                dim_options[dim] = filled[dim][:3]  # 每维度取Top-3
            else:
                dim_options[dim] = [AssociationResult(
                    tag=TagNode(id="empty", name="", name_en=""),
                    score=0.0, strategy="empty"
                )]

        # 笛卡尔积（限制组合数）
        import itertools
        dim_names = list(dim_options.keys())
        dim_lists = [dim_options[d] for d in dim_names]

        for combo in itertools.product(*dim_lists):
            if len(results) >= top_k:
                break

            # 计算组合得分
            scores = [r.score for r in combo if r.tag.id != "empty"]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            confidence = min(scores) if scores else 0.0

            result = Full5CResult(
                conversion_score=avg_score,
                confidence=confidence,
            )
            for dim, assoc in zip(dim_names, combo):
                if assoc.tag.id != "empty":
                    getattr(result, dim).append(assoc)

            results.append(result)

        # 按转化得分排序
        results.sort(key=lambda x: x.conversion_score, reverse=True)
        return results[:top_k]
