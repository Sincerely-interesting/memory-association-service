"""
质心计算器 (CentroidCalculator)

计算用户兴趣质心向量，支持:
- 加权平均 + 时间衰减的质心计算
- 近邻标签查找（基于嵌入相似度）
- 质心偏移追踪
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.models import TagNode, SignalFeature
    from core.tag_graph import TagAssociationGraph
    from infrastructure.tag_embedding import TagEmbeddingService
except ImportError:
    from models import TagNode, SignalFeature
    from tag_graph import TagAssociationGraph
    from tag_embedding import TagEmbeddingService


@dataclass
class CentroidResult:
    """质心计算结果"""
    vector: list[float]                    # 质心向量
    dimension: int                         # 向量维度
    signal_count: int                      # 参与计算的信号数
    total_weight: float                    # 总权重
    nearest_tags: list[tuple[str, float]] = field(default_factory=list)
    # [(tag_id, similarity), ...]

    @property
    def is_valid(self) -> bool:
        """质心是否有效"""
        return self.dimension > 0 and self.signal_count > 0


@dataclass
class CentroidShift:
    """质心偏移记录"""
    old_centroid: list[float]
    new_centroid: list[float]
    shift_magnitude: float                 # 偏移量（欧氏距离）
    timestamp: float
    reason: str = ""


class CentroidCalculator:
    """
    人群兴趣质心计算器

    基于用户的信号特征（社媒互动、商单反馈、内容偏好），
    计算加权平均的兴趣质心向量。
    """

    def __init__(
        self,
        graph: TagAssociationGraph,
        embedding_service: TagEmbeddingService,
    ):
        self.graph = graph
        self.embedding_service = embedding_service

        # 质心缓存: user_id -> CentroidResult
        self._centroid_cache: dict[str, CentroidResult] = {}

        # 偏移历史: user_id -> list[CentroidShift]
        self._shift_history: dict[str, list[CentroidShift]] = {}

        # 配置
        self.config = {
            "half_life_days": 30.0,          # 时间衰减半衰期
            "max_shift_history": 100,        # 最大偏移历史
            "min_signals_for_centroid": 2,   # 最少信号数
        }

    # ── 质心计算 ──────────────────────────────────────────────────

    def compute_centroid(
        self,
        signals: list[SignalFeature],
        user_id: str | None = None,
        weights: list[float] | None = None,
    ) -> CentroidResult:
        """
        计算用户兴趣质心

        Args:
            signals: 用户的信号特征列表
            user_id: 用户ID（用于缓存和偏移追踪）
            weights: 自定义权重（None=使用时间衰减权重）

        Returns:
            CentroidResult
        """
        if not signals:
            return CentroidResult(vector=[], dimension=0, signal_count=0, total_weight=0)

        # 过滤过期信号
        valid_signals = [s for s in signals if not s.is_expired and s.embedding]
        if not valid_signals:
            return CentroidResult(vector=[], dimension=0, signal_count=0, total_weight=0)

        # 计算权重
        if weights is None:
            weights = self._compute_time_decay_weights(valid_signals)

        # 加权平均
        dim = len(valid_signals[0].embedding)
        centroid = [0.0] * dim
        total_weight = 0.0

        for signal, weight in zip(valid_signals, weights):
            if signal.embedding and len(signal.embedding) == dim:
                for i in range(dim):
                    centroid[i] += signal.embedding[i] * weight
                total_weight += weight

        if total_weight > 0:
            centroid = [x / total_weight for x in centroid]

        # L2 归一化
        norm = math.sqrt(sum(x * x for x in centroid))
        if norm > 0:
            centroid = [x / norm for x in centroid]

        result = CentroidResult(
            vector=centroid,
            dimension=dim,
            signal_count=len(valid_signals),
            total_weight=total_weight,
        )

        # 缓存和偏移追踪
        if user_id:
            old = self._centroid_cache.get(user_id)
            self._centroid_cache[user_id] = result
            if old and old.is_valid:
                self._track_shift(user_id, old.vector, centroid)

        return result

    def _compute_time_decay_weights(self, signals: list[SignalFeature]) -> list[float]:
        """计算时间衰减权重"""
        now = time.time()
        half_life = self.config["half_life_days"] * 86400  # 转换为秒
        weights = []
        for signal in signals:
            elapsed = now - signal.captured_at
            weight = 0.5 ** (elapsed / half_life)
            weights.append(weight)
        return weights

    def _track_shift(self, user_id: str, old_vec: list[float], new_vec: list[float]):
        """追踪质心偏移"""
        shift_mag = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(old_vec, new_vec))
        )
        shift = CentroidShift(
            old_centroid=old_vec,
            new_centroid=new_vec,
            shift_magnitude=shift_mag,
            timestamp=time.time(),
        )
        history = self._shift_history.setdefault(user_id, [])
        history.append(shift)
        if len(history) > self.config["max_shift_history"]:
            history.pop(0)

    # ── 近邻查找 ──────────────────────────────────────────────────

    def find_nearest_tags(
        self,
        centroid: list[float],
        domain: str | None = None,
        level: int | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[TagNode, float]]:
        """
        找到与质心最近的标签

        Args:
            centroid: 质心向量
            domain: 限制查询的域
            level: 限制查询的层级
            top_k: 返回数量
            min_score: 最低相似度

        Returns:
            [(tag, similarity), ...]
        """
        if not centroid:
            return []

        # 筛选候选标签
        candidates = list(self.graph.tags.values())
        if domain:
            candidates = [t for t in candidates if t.domain == domain]
        if level is not None:
            candidates = [t for t in candidates if t.level == level]

        scores = []
        for tag in candidates:
            if tag.embedding:
                sim = self.embedding_service.cosine_similarity(centroid, tag.embedding)
                if sim >= min_score:
                    scores.append((tag, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_nearest_across_domains(
        self,
        centroid: list[float],
        top_k_per_domain: int = 5,
        min_score: float = 0.0,
    ) -> dict[str, list[tuple[TagNode, float]]]:
        """在每个域中分别查找最近标签"""
        from core.models import ALL_DOMAINS
        results = {}
        for domain in ALL_DOMAINS:
            results[domain] = self.find_nearest_tags(
                centroid, domain=domain,
                top_k=top_k_per_domain, min_score=min_score,
            )
        return results

    # ── 缓存管理 ──────────────────────────────────────────────────

    def get_cached_centroid(self, user_id: str) -> CentroidResult | None:
        """获取缓存的质心"""
        return self._centroid_cache.get(user_id)

    def get_shift_history(self, user_id: str) -> list[CentroidShift]:
        """获取质心偏移历史"""
        return self._shift_history.get(user_id, [])

    def clear_cache(self, user_id: str | None = None):
        """清空缓存"""
        if user_id:
            self._centroid_cache.pop(user_id, None)
        else:
            self._centroid_cache.clear()
