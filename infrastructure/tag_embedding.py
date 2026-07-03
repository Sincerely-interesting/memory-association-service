"""
标签语义嵌入服务

提供标签→向量映射、向量相似度计算、批量编码。
支持两种模式:
- RuleMode: 基于同义词表和标签属性的规则匹配（无需外部模型）
- EmbeddingMode: 调用外部嵌入模型（OpenAI/本地模型）
"""
from __future__ import annotations

import math
import re
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
    from core.models import TagNode
except ImportError:
    from models import TagNode


# ── 同义词/相关词词典 ────────────────────────────────────────────

# 基于中文语义的相似度规则（无需外部模型）
# 格式: {tag_name: [related_names]}
SYNONYM_MAP: dict[str, list[str]] = {
    # 货品域
    "精华": ["面部精华", "脸部精华液", "serum"],
    "面霜": ["面霜乳液", "保湿霜", "cream"],
    "口红": ["唇膏", "唇釉", "lipstick"],
    "防晒": ["防晒霜", "防晒乳", "sunscreen"],
    "美白": ["提亮", "亮肤", "brightening"],
    "抗老": ["抗皱", "紧致", "anti_aging"],
    "保湿": ["补水", "滋润", "hydrating"],
    # 人群域
    "Z世代": ["95后", "00后", "年轻人", "gen_z"],
    "新锐白领": ["职场新人", "都市白领", "young_professional"],
    "成分党": ["配方党", "成分爱好者", "ingredient_focused"],
    "颜值党": ["外观控", "颜值控", "aesthetic_focused"],
    # 平台域
    "抖音": ["TikTok", "tiktok_douyin"],
    "小红书": ["RED", "instagram_xiaohongshu"],
    "短视频": ["短剧", "video"],
    "直播": ["带货直播", "live_stream"],
    "图文笔记": ["笔记", "图文", "image_text_note"],
}

# 基于类别的隐式相关性（同类别标签有基础相似度）
CATEGORY_SIMILARITY: dict[str, list[str]] = {
    "护肤": ["精华", "面霜", "防晒", "面膜", "洁面", "爽肤水"],
    "彩妆": ["口红", "粉底", "眼影", "腮红", "睫毛膏"],
    "年龄段": ["Z世代", "新锐白领", "精致妈妈", "银发族"],
    "消费频次": ["高频消费", "中频消费", "低频消费"],
    "内容形式": ["短视频", "图文笔记", "直播", "测评长文"],
    "互动模式": ["评论活跃", "点赞收藏", "分享传播"],
}


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    tag_id: str
    vector: list[float]
    method: str  # "rule" / "external"
    latency_ms: float = 0.0


class TagEmbeddingService:
    """
    标签语义嵌入服务

    提供标签→向量映射和相似度计算。

    模式:
        - rule: 基于同义词表和标签属性的规则匹配（默认，无外部依赖）
        - external: 调用外部嵌入模型（需要配置 embedding_provider）
    """

    def __init__(self, mode: str = "rule", embedding_provider: Any = None):
        """
        Args:
            mode: "rule" 或 "external"
            embedding_provider: 外部嵌入模型提供者（mode="external" 时需要）
        """
        self.mode = mode
        self.embedding_provider = embedding_provider

        # 缓存: tag_id -> embedding vector
        self._cache: dict[str, list[float]] = {}

        # 同义词索引: name -> set of related names
        self._synonym_index: dict[str, set[str]] = self._build_synonym_index()

        # 类别索引: tag_name -> parent category name
        self._category_index: dict[str, str] = self._build_category_index()

    def _build_synonym_index(self) -> dict[str, set[str]]:
        """构建同义词索引（双向）"""
        index: dict[str, set[str]] = defaultdict(set)
        for name, synonyms in SYNONYM_MAP.items():
            for syn in synonyms:
                index[name].add(syn)
                index[syn].add(name)
        return dict(index)

    def _build_category_index(self) -> dict[str, str]:
        """构建类别索引"""
        index: dict[str, str] = {}
        for category, members in CATEGORY_SIMILARITY.items():
            for member in members:
                index[member] = category
        return index

    # ── 向量计算 ──────────────────────────────────────────────────

    def encode_tag(self, tag: TagNode) -> list[float]:
        """
        将标签编码为向量

        rule模式: 基于标签属性生成稀疏特征向量
        external模式: 调用外部嵌入模型
        """
        # 检查缓存
        if tag.id in self._cache:
            return self._cache[tag.id]

        if self.mode == "external" and self.embedding_provider:
            vector = self._encode_external(tag)
        else:
            vector = self._encode_rule(tag)

        self._cache[tag.id] = vector
        return vector

    def _encode_rule(self, tag: TagNode) -> list[float]:
        """
        规则模式编码

        生成 128 维稀疏向量:
        [0-7]   域编码 (one-hot: commodity/crowd/platform)
        [8-15]  层级编码 (one-hot: L1/L2/L3)
        [16-31] 子域编码 (one-hot: category/price_band/style/...)
        [32-95] 语义特征 (同义词/类别匹配)
        [96-127] 名称哈希特征
        """
        vector = [0.0] * 128

        # 域编码 [0-7]
        domain_map = {"commodity": 0, "crowd": 1, "platform": 2}
        domain_idx = domain_map.get(tag.domain, 0)
        vector[domain_idx] = 1.0

        # 层级编码 [8-15]
        level_idx = 8 + (tag.level - 1)
        vector[level_idx] = 1.0

        # 子域编码 [16-31]
        sub_domain = tag.metadata.get("sub_domain", "")
        sub_domain_map = {
            "category": 16, "price_band": 17, "style": 18,
            "scenario": 19, "efficacy": 20, "demographics": 21,
            "interest": 22, "behavior": 23, "content_preference": 24,
            "lifecycle": 25, "content_feature": 26, "platform": 27,
        }
        sub_idx = sub_domain_map.get(sub_domain, 16)
        vector[sub_idx] = 1.0

        # 语义特征 [32-95]
        # 同义词匹配
        related = self._synonym_index.get(tag.name, set())
        for i, rel_name in enumerate(sorted(related)):
            if i < 32:
                hash_val = hash(rel_name) % 32
                vector[32 + hash_val] = max(vector[32 + hash_val], 0.8)

        # 类别匹配
        category = self._category_index.get(tag.name)
        if category:
            for member in CATEGORY_SIMILARITY.get(category, []):
                hash_val = hash(member) % 32
                vector[64 + hash_val] = max(vector[64 + hash_val], 0.6)

        # 名称哈希特征 [96-127]
        for char in tag.name:
            hash_val = ord(char) % 32
            vector[96 + hash_val] = max(vector[96 + hash_val], 0.5)

        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def _encode_external(self, tag: TagNode) -> list[float]:
        """外部模型编码"""
        try:
            text = f"{tag.name} {tag.name_en} {tag.description}"
            if hasattr(self.embedding_provider, "get_embedding"):
                return self.embedding_provider.get_embedding(text)
            elif callable(self.embedding_provider):
                return self.embedding_provider(text)
        except Exception as e:
            logger.warning(f"外部嵌入编码失败，回退到规则模式: {e}")
            return self._encode_rule(tag)
        return self._encode_rule(tag)

    def encode_tags_batch(self, tags: list[TagNode]) -> list[list[float]]:
        """批量编码标签"""
        return [self.encode_tag(tag) for tag in tags]

    # ── 相似度计算 ────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def tag_similarity(self, tag_a: TagNode, tag_b: TagNode) -> float:
        """计算两个标签的语义相似度"""
        vec_a = self.encode_tag(tag_a)
        vec_b = self.encode_tag(tag_b)
        return self.cosine_similarity(vec_a, vec_b)

    def find_similar_tags(
        self,
        query_tag: TagNode,
        candidates: list[TagNode],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[TagNode, float]]:
        """找到与查询标签最相似的标签"""
        query_vec = self.encode_tag(query_tag)
        scores = []
        for candidate in candidates:
            if candidate.id == query_tag.id:
                continue
            cand_vec = self.encode_tag(candidate)
            sim = self.cosine_similarity(query_vec, cand_vec)
            if sim >= min_score:
                scores.append((candidate, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── 缓存管理 ──────────────────────────────────────────────────

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def cache_size(self) -> int:
        """缓存大小"""
        return len(self._cache)
