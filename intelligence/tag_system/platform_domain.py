"""
平台域 (Platform Domain)

管理电商/社媒平台相关的标签体系:
- 平台标识: TikTok/抖音, Instagram/小红书, X/Twitter, 微信视频号
- 内容特征标签: 短视频/图文/直播/长视频/...
- 算法特征: 各平台的算法权重参数
- 人群特征: 各平台的用户画像
- 电商路径: 从内容到转化的链路
"""
from __future__ import annotations

import time

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from core.models import (
    TagNode,
    DOMAIN_PLATFORM,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2,
)
from core.tag_graph import TagAssociationGraph


# ── 预置平台定义 ────────────────────────────────────────────────

# 平台特征数据
PLATFORM_DATA: dict[str, dict] = {
    "tiktok_douyin": {
        "name": "抖音/TikTok",
        "name_en": "TikTok/Douyin",
        "content_features": [
            ("短视频15-60s", "short_form_video"),
            ("强视觉冲击", "visual_impact"),
            ("BGM驱动", "bgm_driven"),
            ("剧情化", "storytelling"),
            ("直播带货", "live_commerce"),
            ("切片二创", "clip_remix"),
        ],
        "algorithm_weights": {
            "completion_rate": 0.4,
            "like": 0.25,
            "comment": 0.2,
            "share": 0.15,
        },
        "audience_profile": {
            "age_range": "18-35",
            "gender_skew": "female_slight",
            "city_tier": "lower_tier_penetration",
            "spending_power": "medium_low",
        },
        "ecommerce_path": [
            "短视频挂车",
            "直播转化",
            "商城搜索",
            "品牌旗舰店",
        ],
    },
    "instagram_xiaohongshu": {
        "name": "小红书/Instagram",
        "name_en": "Xiaohongshu/RED",
        "content_features": [
            ("图文笔记", "image_text_note"),
            ("精美封面", "aesthetic_cover"),
            ("真实体验", "authentic_experience"),
            ("干货分享", "valuable_content"),
            ("测评种草", "review_seeding"),
            ("合集推荐", "collection"),
        ],
        "algorithm_weights": {
            "save": 0.35,
            "comment": 0.3,
            "like": 0.2,
            "share": 0.15,
        },
        "audience_profile": {
            "age_range": "20-35",
            "gender_skew": "female_dominant",
            "city_tier": "tier_1_2",
            "spending_power": "medium_high",
        },
        "ecommerce_path": [
            "种草笔记",
            "搜索比价",
            "平台内转化",
            "站外跳转",
        ],
    },
    "x_twitter": {
        "name": "X/Twitter",
        "name_en": "X/Twitter",
        "content_features": [
            ("观点输出", "opinion_output"),
            ("话题讨论", "topic_discussion"),
            ("热点追踪", "trending_tracking"),
            ("简短有力", "concise_powerful"),
            ("thread长推", "thread"),
            ("观点投票", "poll"),
        ],
        "algorithm_weights": {
            "retweet": 0.4,
            "reply": 0.3,
            "like": 0.2,
            "quote": 0.1,
        },
        "audience_profile": {
            "age_range": "22-40",
            "gender_skew": "male_slight",
            "interests": "tech,finance,general_knowledge",
            "spending_power": "medium_high",
        },
        "ecommerce_path": [
            "话题造势",
            "品牌认知",
            "搜索转化",
            "独立站导流",
        ],
    },
    "wechat_channels": {
        "name": "微信视频号",
        "name_en": "WeChat Channels",
        "content_features": [
            ("私域沉淀", "private_domain"),
            ("社交裂变", "social_viral"),
            ("长内容", "long_form_content"),
            ("信任背书", "trust_endorsement"),
            ("直播互动", "live_interaction"),
            ("朋友圈联动", "mom_link"),
        ],
        "algorithm_weights": {
            "social_recommend": 0.5,
            "algo_recommend": 0.3,
            "search": 0.2,
        },
        "audience_profile": {
            "age_range": "30+",
            "gender_skew": "balanced",
            "city_tier": "lower_tier_penetration",
            "spending_power": "medium",
        },
        "ecommerce_path": [
            "内容种草",
            "私域转化",
            "复购裂变",
            "社群运营",
        ],
    },
}


class PlatformDomain:
    """
    平台域标签管理器

    管理各平台的特征标签，包括内容特征、算法参数、人群画像和电商路径。
    """

    def __init__(self, graph: TagAssociationGraph):
        self.graph = graph

        # 内部索引
        self._name_index: dict[str, str] = {}
        self._platform_index: dict[str, dict[str, str]] = {}  # platform_key -> {feature_name: tag_id}

    # ── 初始化 ────────────────────────────────────────────────────

    def initialize(self):
        """初始化预置平台数据"""
        for platform_key, data in PLATFORM_DATA.items():
            self._init_platform(platform_key, data)

        total_features = sum(
            len(v) for v in self._platform_index.values()
        )
        logger.info(
            f"平台域初始化完成: {len(PLATFORM_DATA)}个平台, "
            f"{total_features}个内容特征标签"
        )

    def _init_platform(self, platform_key: str, data: dict):
        """初始化单个平台"""
        # 创建平台根节点
        platform_id = self.graph.add_tag(
            name=data["name"],
            name_en=data["name_en"],
            domain=DOMAIN_PLATFORM,
            level=LEVEL_CATEGORY_1,
            description=f"平台: {data['name']}",
            metadata={
                "sub_domain": "platform",
                "platform_key": platform_key,
                "algorithm_weights": data.get("algorithm_weights", {}),
                "audience_profile": data.get("audience_profile", {}),
                "ecommerce_path": data.get("ecommerce_path", []),
            },
        )
        self._name_index[data["name"]] = platform_id
        self._name_index[platform_key] = platform_id

        # 初始化内容特征标签
        feature_map: dict[str, str] = {}
        for feature_name, feature_en in data.get("content_features", []):
            tag_id = self.graph.add_tag(
                name=feature_name,
                name_en=feature_en,
                domain=DOMAIN_PLATFORM,
                level=LEVEL_CATEGORY_2,
                parent_id=platform_id,
                description=f"内容特征: {feature_name}",
                metadata={
                    "sub_domain": "content_feature",
                    "platform": platform_key,
                },
            )
            self._name_index[feature_name] = tag_id
            feature_map[feature_name] = tag_id

        self._platform_index[platform_key] = feature_map

    # ── 平台查询 ──────────────────────────────────────────────────

    def get_platform(self, platform_key: str) -> TagNode | None:
        """按平台key获取平台节点"""
        tag_id = self._name_index.get(platform_key)
        if tag_id:
            return self.graph.tags.get(tag_id)
        return None

    def get_all_platforms(self) -> list[TagNode]:
        """获取所有平台节点"""
        results = []
        for platform_key in PLATFORM_DATA:
            tag_id = self._name_index.get(platform_key)
            if tag_id and tag_id in self.graph.tags:
                results.append(self.graph.tags[tag_id])
        return results

    def get_content_features(self, platform_key: str) -> list[TagNode]:
        """获取某个平台的内容特征标签"""
        feature_map = self._platform_index.get(platform_key, {})
        return [
            self.graph.tags[tid]
            for tid in feature_map.values()
            if tid in self.graph.tags
        ]

    def get_algorithm_weights(self, platform_key: str) -> dict[str, float]:
        """获取某个平台的算法权重"""
        platform = self.get_platform(platform_key)
        if platform:
            return platform.metadata.get("algorithm_weights", {})
        return {}

    def get_audience_profile(self, platform_key: str) -> dict:
        """获取某个平台的人群画像"""
        platform = self.get_platform(platform_key)
        if platform:
            return platform.metadata.get("audience_profile", {})
        return {}

    def get_ecommerce_path(self, platform_key: str) -> list[str]:
        """获取某个平台的电商路径"""
        platform = self.get_platform(platform_key)
        if platform:
            return platform.metadata.get("ecommerce_path", [])
        return []

    # ── 搜索 ──────────────────────────────────────────────────────

    def get_tag_by_name(self, name: str) -> TagNode | None:
        """按名称获取标签"""
        tag_id = self._name_index.get(name)
        if tag_id:
            return self.graph.tags.get(tag_id)
        return None

    def search(self, keyword: str) -> list[TagNode]:
        """按关键词搜索平台标签"""
        results = []
        for name, tag_id in self._name_index.items():
            if keyword in name and tag_id in self.graph.tags:
                results.append(self.graph.tags[tag_id])
        return results

    def search_by_platform(self, platform_key: str, keyword: str) -> list[TagNode]:
        """在某个平台内搜索标签"""
        feature_map = self._platform_index.get(platform_key, {})
        results = []
        for name, tag_id in feature_map.items():
            if keyword in name and tag_id in self.graph.tags:
                results.append(self.graph.tags[tag_id])
        return results

    # ── CRUD ──────────────────────────────────────────────────────

    def add_content_feature(
        self,
        platform_key: str,
        feature_name: str,
        feature_en: str,
        description: str = "",
    ) -> str:
        """为某个平台添加内容特征标签"""
        platform_id = self._name_index.get(platform_key)
        if not platform_id:
            raise ValueError(f"平台 {platform_key} 不存在")

        tag_id = self.graph.add_tag(
            name=feature_name,
            name_en=feature_en,
            domain=DOMAIN_PLATFORM,
            level=LEVEL_CATEGORY_2,
            parent_id=platform_id,
            description=description or f"内容特征: {feature_name}",
            metadata={
                "sub_domain": "content_feature",
                "platform": platform_key,
            },
        )
        self._name_index[feature_name] = tag_id
        self._platform_index.setdefault(platform_key, {})[feature_name] = tag_id

        return tag_id

    def remove_tag(self, name: str) -> bool:
        """按名称删除标签"""
        tag_id = self._name_index.pop(name, None)
        if not tag_id:
            return False

        # 从平台索引移除
        for features in self._platform_index.values():
            features.pop(name, None)

        return self.graph.remove_tag(tag_id)

    def count(self) -> dict:
        """统计各平台标签数量"""
        return {
            platform_key: len(features)
            for platform_key, features in self._platform_index.items()
        }
