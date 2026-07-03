"""
标签联想召回策略 (TagAssociationRecall)

基于标签图谱的联想召回:
1. 从查询文本提取关键词
2. 在标签图中匹配标签
3. 通过联想关系找到相关标签
4. 返回关联的记忆/内容

支持:
- 直接标签匹配
- 图遍历联想（邻接表）
- 语义相似度联想
- 层级联想（父/子标签）
"""
from __future__ import annotations

import re
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
        TagNode, TagAssociation,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
        ALL_DOMAINS, ASSOC_SEMANTIC, ASSOC_COOCCURRENCE,
        ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
    )
    from core.tag_graph import TagAssociationGraph
    from infrastructure.tag_embedding import TagEmbeddingService
except ImportError:
    from models import (
        TagNode, TagAssociation,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
        ALL_DOMAINS, ASSOC_SEMANTIC, ASSOC_COOCCURRENCE,
        ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
    )
    from tag_graph import TagAssociationGraph
    from tag_embedding import TagEmbeddingService


@dataclass
class TagRecallResult:
    """标签召回结果"""
    tag: TagNode
    score: float
    source: str              # "direct" / "graph_walk" / "semantic" / "hierarchical"
    path: list[str] = field(default_factory=list)  # 联想路径
    metadata: dict[str, Any] = field(default_factory=dict)


class TagAssociationRecall:
    """
    标签联想召回策略

    从查询文本中提取标签，通过图谱联想找到相关标签，
    并返回带有联想路径的结果。
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
            "direct_match_weight": 0.5,
            "graph_walk_weight": 0.3,
            "semantic_weight": 0.2,
            "max联想跳数": 2,
            "min_score": 0.1,
            "max_results": 20,
        }

        # 停用词
        self._stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "吗", "什么", "那", "这个", "那个", "可以", "怎么", "如何",
        }

    # ── 主召回入口 ────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        top_k: int = 10,
        domains: list[str] | None = None,
        include_path: bool = True,
    ) -> list[TagRecallResult]:
        """
        从查询文本召回相关标签

        Args:
            query: 查询文本
            top_k: 返回数量
            domains: 限制查询的域
            include_path: 是否包含联想路径

        Returns:
            召回结果列表
        """
        if not query or not query.strip():
            return []

        # 1. 提取关键词
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # 2. 直接匹配
        direct_results = self._direct_match(keywords, domains)

        # 3. 图遍历联想
        graph_results = self._graph_walk联想(direct_results, domains)

        # 4. 语义相似度联想
        semantic_results = self._semantic联想(keywords, domains)

        # 5. 合并去重排序
        all_results = direct_results + graph_results + semantic_results
        results = self._fuse_and_rank(all_results)

        return results[:top_k]

    # ── 关键词提取 ────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本中提取关键词"""
        # 简单分词（基于标签名称匹配）
        keywords = []
        text_lower = text.lower()

        # 按标签名称匹配
        for tag_id, tag in self.graph.tags.items():
            if tag.name in text and tag.name not in keywords:
                keywords.append(tag.name)
            elif tag.name_en and tag.name_en.lower() in text_lower:
                if tag.name not in keywords:
                    keywords.append(tag.name)

        # 如果没有匹配到标签，尝试简单的字符级匹配
        if not keywords:
            # 提取中文词（2-4字）
            chinese_words = re.findall(r'[一-鿿]{2,4}', text)
            for word in chinese_words:
                if word not in self._stop_words and word not in keywords:
                    keywords.append(word)

        return keywords[:10]  # 限制数量

    # ── 直接匹配 ──────────────────────────────────────────────────

    def _direct_match(
        self,
        keywords: list[str],
        domains: list[str] | None = None,
    ) -> list[TagRecallResult]:
        """直接标签匹配"""
        results = []
        for keyword in keywords:
            for tag_id, tag in self.graph.tags.items():
                if tag.name == keyword:
                    if domains and tag.domain not in domains:
                        continue
                    results.append(TagRecallResult(
                        tag=tag,
                        score=self.config["direct_match_weight"],
                        source="direct",
                        path=[tag.name],
                        metadata={"keyword": keyword},
                    ))
        return results

    # ── 图遍历联想 ────────────────────────────────────────────────

    def _graph_walk联想(
        self,
        source_results: list[TagRecallResult],
        domains: list[str] | None = None,
    ) -> list[TagRecallResult]:
        """通过图遍历进行联想"""
        results = []
        visited = set()

        for source in source_results:
            tag_id = source.tag.id
            if tag_id in visited:
                continue
            visited.add(tag_id)

            # BFS 遍历邻接表
            queue = [(tag_id, source.path, 0)]
            while queue:
                current_id, path, depth = queue.pop(0)
                if depth >= self.config["max联想跳数"]:
                    continue

                for neighbor_id, strength in self.graph.get_neighbors(current_id):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)

                    neighbor = self.graph.tags.get(neighbor_id)
                    if not neighbor:
                        continue
                    if domains and neighbor.domain not in domains:
                        continue

                    # 计算得分（距离衰减）
                    score = strength * self.config["graph_walk_weight"] * (0.7 ** depth)
                    if score >= self.config["min_score"]:
                        new_path = path + [neighbor.name]
                        results.append(TagRecallResult(
                            tag=neighbor,
                            score=score,
                            source="graph_walk",
                            path=new_path,
                            metadata={"depth": depth, "strength": strength},
                        ))
                        queue.append((neighbor_id, new_path, depth + 1))

        return results

    # ── 语义相似度联想 ────────────────────────────────────────────

    def _semantic联想(
        self,
        keywords: list[str],
        domains: list[str] | None = None,
    ) -> list[TagRecallResult]:
        """通过语义相似度进行联想"""
        results = []

        # 为关键词创建临时标签
        for keyword in keywords:
            query_tag = TagNode(
                id=f"query_{keyword}",
                name=keyword,
                name_en=keyword,
            )

            # 查找相似标签
            candidates = list(self.graph.tags.values())
            if domains:
                candidates = [t for t in candidates if t.domain in domains]

            similar = self.embedding_service.find_similar_tags(
                query_tag, candidates, top_k=5, min_score=self.config["min_score"]
            )

            for tag, sim in similar:
                score = sim * self.config["semantic_weight"]
                results.append(TagRecallResult(
                    tag=tag,
                    score=score,
                    source="semantic",
                    path=[keyword, tag.name],
                    metadata={"similarity": sim, "keyword": keyword},
                ))

        return results

    # ── 合并排序 ──────────────────────────────────────────────────

    def _fuse_and_rank(self, results: list[TagRecallResult]) -> list[TagRecallResult]:
        """合并去重排序"""
        # 按 tag.id 去重，保留最高分
        best: dict[str, TagRecallResult] = {}
        for r in results:
            key = r.tag.id
            if key not in best or r.score > best[key].score:
                best[key] = r

        # 按得分排序
        ranked = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return ranked

    # ── 辅助方法 ──────────────────────────────────────────────────

    def get联想路径(
        self,
        source_tag: TagNode,
        target_tag: TagNode,
        max_depth: int = 3,
    ) -> list[str] | None:
        """查找两个标签之间的联想路径（BFS）"""
        if source_tag.id == target_tag.id:
            return [source_tag.name]

        visited = {source_tag.id}
        queue = [(source_tag.id, [source_tag.name])]

        while queue:
            current_id, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            for neighbor_id, _ in self.graph.get_neighbors(current_id):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor = self.graph.tags.get(neighbor_id)
                if not neighbor:
                    continue

                new_path = path + [neighbor.name]
                if neighbor_id == target_tag.id:
                    return new_path

                queue.append((neighbor_id, new_path))

        return None
