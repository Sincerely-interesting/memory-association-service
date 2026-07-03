"""
跨域联想召回策略 (CrossDomainRecall)

基于货品↔人群↔平台的跨域关联进行记忆召回:
1. 识别查询中的域特征
2. 通过跨域规则和关联找到相关域的标签
3. 返回跨域联想结果

支持:
- 货品↔人群 跨域联想
- 人群↔平台 跨域联想
- 货品↔平台 跨域联想
- 三域联合联想
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.models import (
        TagNode, CrossDomainLink,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
        ALL_DOMAINS, RULE_CONSTRAINT, RULE_BOOST, RULE_FILTER,
    )
    from core.tag_graph import TagAssociationGraph
    from infrastructure.tag_embedding import TagEmbeddingService
except ImportError:
    from models import (
        TagNode, CrossDomainLink,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
        ALL_DOMAINS, RULE_CONSTRAINT, RULE_BOOST, RULE_FILTER,
    )
    from tag_graph import TagAssociationGraph
    from tag_embedding import TagEmbeddingService


@dataclass
class CrossDomainResult:
    """跨域联想结果"""
    tag: TagNode
    domain: str
    score: float
    source_domain: str       # 来源域
    rule_applied: str = ""   # 应用的规则
    path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossDomainRecall:
    """
    跨域联想召回策略

    基于标签图谱的跨域关联，从一个域的标签联想到其他域的标签。
    """

    def __init__(
        self,
        tag_graph: TagAssociationGraph,
        embedding_service: TagEmbeddingService,
    ):
        self.graph = tag_graph
        self.embedding_service = embedding_service

        # 配置
        self.config = {
            "direct_weight": 0.4,
            "rule_weight": 0.3,
            "semantic_weight": 0.2,
            "association_weight": 0.1,
            "min_score": 0.1,
            "max_results": 20,
        }

    # ── 主召回入口 ────────────────────────────────────────────────

    def recall(
        self,
        source_domain: str,
        source_tags: list[str],
        target_domain: str | None = None,
        top_k: int = 10,
    ) -> list[CrossDomainResult]:
        """
        跨域联想召回

        Args:
            source_domain: 来源域
            source_tags: 来源标签名称列表
            target_domain: 目标域（None=所有其他域）
            top_k: 返回数量

        Returns:
            跨域联想结果
        """
        if not source_tags:
            return []

        # 1. 解析源标签
        resolved_tags = self._resolve_tags(source_tags, source_domain)
        if not resolved_tags:
            return []

        # 2. 确定目标域
        if target_domain:
            target_domains = [target_domain]
        else:
            target_domains = [d for d in ALL_DOMAINS if d != source_domain]

        # 3. 多策略跨域联想
        results = []

        for tgt_domain in target_domains:
            # 3.1 直接跨域关联
            direct = self._direct_cross_domain(resolved_tags, tgt_domain)
            results.extend(direct)

            # 3.2 规则驱动联想
            rule_based = self._rule_based联想(resolved_tags, source_domain, tgt_domain)
            results.extend(rule_based)

            # 3.3 语义相似度跨域
            semantic = self._semantic_cross_domain(resolved_tags, tgt_domain)
            results.extend(semantic)

            # 3.4 图遍历跨域
            graph_walk = self._graph_walk_cross_domain(resolved_tags, tgt_domain)
            results.extend(graph_walk)

        # 4. 合并去重排序
        results = self._fuse_and_rank(results)
        return results[:top_k]

    # ── 直接跨域关联 ──────────────────────────────────────────────

    def _direct_cross_domain(
        self,
        source_tags: list[TagNode],
        target_domain: str,
    ) -> list[CrossDomainResult]:
        """通过联想关系直接跨域"""
        results = []
        for src_tag in source_tags:
            for neighbor_id, strength in self.graph.get_neighbors(src_tag.id):
                neighbor = self.graph.tags.get(neighbor_id)
                if neighbor and neighbor.domain == target_domain:
                    score = strength * self.config["direct_weight"]
                    results.append(CrossDomainResult(
                        tag=neighbor,
                        domain=target_domain,
                        score=score,
                        source_domain=src_tag.domain,
                        path=[src_tag.name, neighbor.name],
                        metadata={"strength": strength, "method": "direct"},
                    ))
        return results

    # ── 规则驱动联想 ──────────────────────────────────────────────

    def _rule_based联想(
        self,
        source_tags: list[TagNode],
        source_domain: str,
        target_domain: str,
    ) -> list[CrossDomainResult]:
        """通过跨域规则进行联想"""
        results = []

        # 查找适用的跨域规则
        applicable_rules = self.graph.get_cross_domain_links(
            source_domain=source_domain,
            target_domain=target_domain,
        )

        for rule in applicable_rules:
            # 简化的规则评估（实际应用中需要更复杂的规则引擎）
            if rule.rule_type == RULE_BOOST:
                # 加成规则：找到目标域的高权重标签
                target_tags = self.graph.get_tags_by_domain(target_domain)
                for tgt_tag in target_tags[:10]:  # 限制数量
                    score = rule.weight * self.config["rule_weight"]
                    results.append(CrossDomainResult(
                        tag=tgt_tag,
                        domain=target_domain,
                        score=score,
                        source_domain=source_domain,
                        rule_applied=rule.rule_expression,
                        path=[rule.rule_expression, tgt_tag.name],
                        metadata={"rule_id": rule.id, "method": "rule"},
                    ))
            elif rule.rule_type == RULE_FILTER:
                # 过滤规则：只保留符合条件的标签
                # 简化处理：返回所有标签，标记应用了过滤规则
                target_tags = self.graph.get_tags_by_domain(target_domain)
                for tgt_tag in target_tags[:5]:
                    score = rule.weight * self.config["rule_weight"] * 0.8
                    results.append(CrossDomainResult(
                        tag=tgt_tag,
                        domain=target_domain,
                        score=score,
                        source_domain=source_domain,
                        rule_applied=rule.rule_expression,
                        path=[rule.rule_expression, tgt_tag.name],
                        metadata={"rule_id": rule.id, "method": "filter"},
                    ))

        return results

    # ── 语义相似度跨域 ────────────────────────────────────────────

    def _semantic_cross_domain(
        self,
        source_tags: list[TagNode],
        target_domain: str,
    ) -> list[CrossDomainResult]:
        """通过语义相似度进行跨域联想"""
        results = []
        target_tags = self.graph.get_tags_by_domain(target_domain)

        for src_tag in source_tags:
            src_vec = self.embedding_service.encode_tag(src_tag)
            for tgt_tag in target_tags:
                tgt_vec = self.embedding_service.encode_tag(tgt_tag)
                sim = self.embedding_service.cosine_similarity(src_vec, tgt_vec)
                score = sim * self.config["semantic_weight"]
                if score >= self.config["min_score"]:
                    results.append(CrossDomainResult(
                        tag=tgt_tag,
                        domain=target_domain,
                        score=score,
                        source_domain=src_tag.domain,
                        path=[src_tag.name, tgt_tag.name],
                        metadata={"similarity": sim, "method": "semantic"},
                    ))

        return results

    # ── 图遍历跨域 ────────────────────────────────────────────────

    def _graph_walk_cross_domain(
        self,
        source_tags: list[TagNode],
        target_domain: str,
    ) -> list[CrossDomainResult]:
        """通过图遍历进行跨域联想"""
        results = []
        visited = set()

        for src_tag in source_tags:
            # BFS 遍历
            queue = [(src_tag.id, [src_tag.name], 0)]
            while queue:
                current_id, path, depth = queue.pop(0)
                if depth >= 2:  # 最多2跳
                    continue

                for neighbor_id, strength in self.graph.get_neighbors(current_id):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)

                    neighbor = self.graph.tags.get(neighbor_id)
                    if not neighbor:
                        continue

                    # 如果到达目标域，记录结果
                    if neighbor.domain == target_domain:
                        score = strength * self.config["association_weight"] * (0.7 ** depth)
                        if score >= self.config["min_score"]:
                            results.append(CrossDomainResult(
                                tag=neighbor,
                                domain=target_domain,
                                score=score,
                                source_domain=src_tag.domain,
                                path=path + [neighbor.name],
                                metadata={"depth": depth, "method": "graph_walk"},
                            ))

                    # 继续遍历
                    queue.append((neighbor_id, path + [neighbor.name], depth + 1))

        return results

    # ── 辅助方法 ──────────────────────────────────────────────────

    def _resolve_tags(self, tag_names: list[str], domain: str) -> list[TagNode]:
        """解析标签名称"""
        tags = []
        for name in tag_names:
            for tag_id, tag in self.graph.tags.items():
                if tag.name == name and tag.domain == domain:
                    tags.append(tag)
                    break
        return tags

    def _fuse_and_rank(self, results: list[CrossDomainResult]) -> list[CrossDomainResult]:
        """合并去重排序"""
        best: dict[str, CrossDomainResult] = {}
        for r in results:
            key = f"{r.tag.id}_{r.source_domain}"
            if key not in best or r.score > best[key].score:
                best[key] = r

        ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return ranked

    # ── 便捷查询 ──────────────────────────────────────────────────

    def commodity_to_crowd(
        self,
        commodity_tags: list[str],
        top_k: int = 10,
    ) -> list[CrossDomainResult]:
        """货品→人群 联想"""
        return self.recall(DOMAIN_COMMODITY, commodity_tags, DOMAIN_CROWD, top_k)

    def crowd_to_platform(
        self,
        crowd_tags: list[str],
        top_k: int = 10,
    ) -> list[CrossDomainResult]:
        """人群→平台 联想"""
        return self.recall(DOMAIN_CROWD, crowd_tags, DOMAIN_PLATFORM, top_k)

    def commodity_to_platform(
        self,
        commodity_tags: list[str],
        top_k: int = 10,
    ) -> list[CrossDomainResult]:
        """货品→平台 联想"""
        return self.recall(DOMAIN_COMMODITY, commodity_tags, DOMAIN_PLATFORM, top_k)

    def full_5c联想(
        self,
        commodity_tags: list[str],
        crowd_tags: list[str],
        top_k: int = 5,
    ) -> dict[str, list[CrossDomainResult]]:
        """全链路5C联想"""
        results = {}

        # 货品→人群
        results["commodity_to_crowd"] = self.commodity_to_crowd(commodity_tags, top_k)

        # 人群→平台
        if crowd_tags:
            results["crowd_to_platform"] = self.crowd_to_platform(crowd_tags, top_k)

        # 货品→平台
        results["commodity_to_platform"] = self.commodity_to_platform(commodity_tags, top_k)

        return results
