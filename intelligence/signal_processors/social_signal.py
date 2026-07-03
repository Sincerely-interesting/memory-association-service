"""
社媒数据信号处理器 (SocialSignalProcessor)

解析 TikTok/Instagram/X 的数据，提取标准化特征:
- 互动率 (engagement_rate)
- 内容完成率 (completion_rate)
- 受众画像 (audience_profile)
- 内容类型分布 (content_type_distribution)
- 话题热度 (topic_heat)
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.models import SignalFeature, SIGNAL_SOCIAL
except ImportError:
    from models import SignalFeature, SIGNAL_SOCIAL


# ── 平台特征定义 ──────────────────────────────────────────────────

PLATFORM_CONFIGS: dict[str, dict] = {
    "tiktok_douyin": {
        "name": "抖音/TikTok",
        "engagement_weights": {
            "like": 0.25,
            "comment": 0.20,
            "share": 0.15,
            "save": 0.10,
            "completion_rate": 0.30,
        },
        "content_types": ["short_video", "live", "story", "long_video"],
        "audience_fields": ["age_range", "gender_ratio", "city_tier", "interests"],
    },
    "instagram_xiaohongshu": {
        "name": "小红书/Instagram",
        "engagement_weights": {
            "like": 0.20,
            "comment": 0.30,
            "share": 0.15,
            "save": 0.35,
        },
        "content_types": ["image_text", "short_video", "live", "note"],
        "audience_fields": ["age_range", "gender_ratio", "city_tier", "lifestyle"],
    },
    "x_twitter": {
        "name": "X/Twitter",
        "engagement_weights": {
            "like": 0.20,
            "reply": 0.30,
            "retweet": 0.40,
            "quote": 0.10,
        },
        "content_types": ["text", "image", "video", "poll", "thread"],
        "audience_fields": ["age_range", "gender_ratio", "interests", "location"],
    },
    "wechat_channels": {
        "name": "微信视频号",
        "engagement_weights": {
            "like": 0.20,
            "comment": 0.25,
            "share": 0.35,
            "save": 0.20,
        },
        "content_types": ["short_video", "live", "article"],
        "audience_fields": ["age_range", "city_tier", "social_graph"],
    },
}


@dataclass
class SocialMediaMetrics:
    """社媒数据指标"""
    platform: str
    content_id: str
    creator_id: str

    # 互动数据
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    views: int = 0
    completion_rate: float = 0.0

    # 内容信息
    content_type: str = "short_video"
    topic: str = ""
    hashtags: list[str] = field(default_factory=list)

    # 受众信息
    audience_age_range: str = ""
    audience_gender_ratio: str = ""
    audience_city_tier: str = ""

    # 时间
    published_at: float = 0.0
    collected_at: float = 0.0

    def __post_init__(self):
        if self.collected_at == 0.0:
            self.collected_at = time.time()


class SocialSignalProcessor:
    """
    社媒数据信号处理器

    将原始社媒数据转换为标准化的 SignalFeature。
    """

    def __init__(self):
        self._platform_configs = PLATFORM_CONFIGS

    # ── 主处理入口 ────────────────────────────────────────────────

    def process(
        self,
        metrics: SocialMediaMetrics,
        generate_embedding: bool = True,
    ) -> SignalFeature:
        """
        处理社媒数据，生成 SignalFeature

        Args:
            metrics: 社媒数据指标
            generate_embedding: 是否生成嵌入向量

        Returns:
            SignalFeature
        """
        features = {}

        # 1. 互动率计算
        engagement = self._compute_engagement(metrics)
        features.update(engagement)

        # 2. 内容特征
        content_features = self._extract_content_features(metrics)
        features.update(content_features)

        # 3. 受众特征
        audience_features = self._extract_audience_features(metrics)
        features.update(audience_features)

        # 4. 生成嵌入向量
        embedding = None
        if generate_embedding:
            embedding = self._generate_embedding(metrics, features)

        # 5. 构建 SignalFeature
        signal_id = self._make_signal_id(metrics)
        signal = SignalFeature(
            id=signal_id,
            source=SIGNAL_SOCIAL,
            features=features,
            embedding=embedding,
            platform=metrics.platform,
            creator_id=metrics.creator_id,
            content_id=metrics.content_id,
            captured_at=metrics.collected_at,
        )

        return signal

    def process_batch(
        self,
        metrics_list: list[SocialMediaMetrics],
        generate_embedding: bool = True,
    ) -> list[SignalFeature]:
        """批量处理"""
        return [self.process(m, generate_embedding) for m in metrics_list]

    # ── 互动率计算 ────────────────────────────────────────────────

    def _compute_engagement(self, m: SocialMediaMetrics) -> dict[str, float]:
        """计算互动率"""
        config = self._platform_configs.get(m.platform, {})
        weights = config.get("engagement_weights", {})

        total_engagement = m.likes + m.comments + m.shares + m.saves
        views = max(m.views, 1)

        features = {
            "engagement_rate": total_engagement / views if views > 0 else 0.0,
            "like_rate": m.likes / views if views > 0 else 0.0,
            "comment_rate": m.comments / views if views > 0 else 0.0,
            "share_rate": m.shares / views if views > 0 else 0.0,
            "save_rate": m.saves / views if views > 0 else 0.0,
            "completion_rate": m.completion_rate,
            "total_engagement": float(total_engagement),
            "views": float(m.views),
        }

        # 加权互动分
        weighted_score = 0.0
        for metric, weight in weights.items():
            if metric == "completion_rate":
                weighted_score += m.completion_rate * weight
            elif metric == "like":
                weighted_score += (m.likes / views if views > 0 else 0) * weight
            elif metric == "comment":
                weighted_score += (m.comments / views if views > 0 else 0) * weight
            elif metric == "share":
                weighted_score += (m.shares / views if views > 0 else 0) * weight
            elif metric == "save":
                weighted_score += (m.saves / views if views > 0 else 0) * weight
        features["weighted_engagement"] = weighted_score

        return features

    # ── 内容特征 ──────────────────────────────────────────────────

    def _extract_content_features(self, m: SocialMediaMetrics) -> dict[str, Any]:
        """提取内容特征"""
        features = {
            "content_type": m.content_type,
            "has_topic": 1.0 if m.topic else 0.0,
            "hashtag_count": float(len(m.hashtags)),
        }

        # 内容类型 one-hot
        config = self._platform_configs.get(m.platform, {})
        content_types = config.get("content_types", [])
        for ct in content_types:
            features[f"ct_{ct}"] = 1.0 if m.content_type == ct else 0.0

        # 话题热度（基于 hashtag 数量和互动）
        topic_heat = min(1.0, len(m.hashtags) / 5.0) * 0.3
        topic_heat += min(1.0, (m.likes + m.comments) / 1000) * 0.7
        features["topic_heat"] = topic_heat

        return features

    # ── 受众特征 ──────────────────────────────────────────────────

    def _extract_audience_features(self, m: SocialMediaMetrics) -> dict[str, Any]:
        """提取受众特征"""
        features = {}

        # 年龄段编码
        age_map = {
            "18-24": [1, 0, 0, 0],   # Z世代
            "25-34": [0, 1, 0, 0],   # 新锐白领
            "35-44": [0, 0, 1, 0],   # 中年
            "45+":   [0, 0, 0, 1],   # 银发
        }
        age_vec = age_map.get(m.audience_age_range, [0, 0, 0, 0])
        features["audience_age_gen_z"] = age_vec[0]
        features["audience_age_professional"] = age_vec[1]
        features["audience_age_middle"] = age_vec[2]
        features["audience_age_senior"] = age_vec[3]

        # 性别比例编码
        gender_map = {
            "female_dominant": 0.7,
            "female_slight": 0.6,
            "balanced": 0.5,
            "male_slight": 0.4,
            "male_dominant": 0.3,
        }
        features["female_ratio"] = gender_map.get(m.audience_gender_ratio, 0.5)

        # 城市层级编码
        city_map = {
            "tier_1": 1.0,
            "tier_1_2": 0.8,
            "tier_2": 0.6,
            "tier_3": 0.4,
            "lower_tier": 0.2,
        }
        features["city_tier_score"] = city_map.get(m.audience_city_tier, 0.5)

        return features

    # ── 嵌入向量 ──────────────────────────────────────────────────

    def _generate_embedding(
        self,
        m: SocialMediaMetrics,
        features: dict[str, float],
    ) -> list[float]:
        """生成特征嵌入向量"""
        # 64维特征向量
        vector = [0.0] * 64

        # [0-7] 互动特征
        vector[0] = features.get("engagement_rate", 0)
        vector[1] = features.get("like_rate", 0)
        vector[2] = features.get("comment_rate", 0)
        vector[3] = features.get("share_rate", 0)
        vector[4] = features.get("save_rate", 0)
        vector[5] = features.get("completion_rate", 0)
        vector[6] = features.get("weighted_engagement", 0)
        vector[7] = min(1.0, features.get("total_engagement", 0) / 10000)

        # [8-15] 内容类型
        content_types = ["short_video", "live", "image_text", "text", "thread"]
        for i, ct in enumerate(content_types[:8]):
            vector[8 + i] = features.get(f"ct_{ct}", 0)

        # [16-23] 受众特征
        vector[16] = features.get("audience_age_gen_z", 0)
        vector[17] = features.get("audience_age_professional", 0)
        vector[18] = features.get("audience_age_middle", 0)
        vector[19] = features.get("audience_age_senior", 0)
        vector[20] = features.get("female_ratio", 0.5)
        vector[21] = features.get("city_tier_score", 0.5)
        vector[22] = features.get("topic_heat", 0)
        vector[23] = features.get("hashtag_count", 0) / 10

        # [24-31] 平台编码
        platform_keys = ["tiktok_douyin", "instagram_xiaohongshu", "x_twitter", "wechat_channels"]
        for i, pk in enumerate(platform_keys):
            vector[24 + i] = 1.0 if m.platform == pk else 0.0

        # [32-63] 内容哈希特征
        content_str = f"{m.topic}{' '.join(m.hashtags)}"
        for i, char in enumerate(content_str[:32]):
            hash_val = ord(char) % 32
            vector[32 + hash_val] = max(vector[32 + hash_val], 0.5)

        # L2 归一化
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    # ── 辅助方法 ──────────────────────────────────────────────────

    def _make_signal_id(self, m: SocialMediaMetrics) -> str:
        """生成信号ID"""
        raw = f"{m.platform}:{m.creator_id}:{m.content_id}:{m.collected_at}"
        return f"social_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    # ── 便捷构造方法 ──────────────────────────────────────────────

    @classmethod
    def from_tiktok(cls, **kwargs) -> SocialMediaMetrics:
        """从TikTok数据构造"""
        return SocialMediaMetrics(platform="tiktok_douyin", **kwargs)

    @classmethod
    def from_xiaohongshu(cls, **kwargs) -> SocialMediaMetrics:
        """从小红书数据构造"""
        return SocialMediaMetrics(platform="instagram_xiaohongshu", **kwargs)

    @classmethod
    def from_x(cls, **kwargs) -> SocialMediaMetrics:
        """从X/Twitter数据构造"""
        return SocialMediaMetrics(platform="x_twitter", **kwargs)

    @classmethod
    def from_wechat(cls, **kwargs) -> SocialMediaMetrics:
        """从微信视频号数据构造"""
        return SocialMediaMetrics(platform="wechat_channels", **kwargs)
