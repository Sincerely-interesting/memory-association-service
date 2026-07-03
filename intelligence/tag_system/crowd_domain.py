"""
人群域 (Crowd Domain)

管理目标用户相关的标签体系:
- 人口统计 (Demographics): 年龄段/性别倾向/城市层级
- 兴趣质心 (InterestCentroid): 主兴趣簇/次兴趣簇
- 消费行为 (ConsumptionBehavior): 消费频次/决策因子/渠道偏好
- 内容偏好 (ContentPreference): 内容形式/内容调性/互动模式
- 生命周期 (LifecycleStage): 新客/首购/复购/沉睡/流失
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
    DOMAIN_CROWD,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2,
)
from core.tag_graph import TagAssociationGraph


# ── 预置标签定义 ────────────────────────────────────────────────

# 子域常量
SUB_DOMAIN_DEMOGRAPHICS = "demographics"
SUB_DOMAIN_INTEREST = "interest"
SUB_DOMAIN_BEHAVIOR = "behavior"
SUB_DOMAIN_CONTENT_PREF = "content_preference"
SUB_DOMAIN_LIFECYCLE = "lifecycle"

# 人口统计标签: {子分类: [(name, name_en, metadata)]}
DEMOGRAPHICS_TAGS: dict[str, list[tuple[str, str, dict]]] = {
    "年龄段": [
        ("Z世代", "gen_z", {"age_min": 18, "age_max": 24}),
        ("新锐白领", "young_professional", {"age_min": 25, "age_max": 34}),
        ("精致妈妈", "quality_mom", {"age_min": 30, "age_max": 40}),
        ("银发族", "senior", {"age_min": 50, "age_max": 99}),
        ("少年", "teenager", {"age_min": 13, "age_max": 17}),
        ("中年", "middle_aged", {"age_min": 40, "age_max": 55}),
    ],
    "性别倾向": [
        ("女性主导", "female_oriented", {"gender_ratio": "70F_30M"}),
        ("男性主导", "male_oriented", {"gender_ratio": "30F_70M"}),
        ("中性", "neutral", {"gender_ratio": "50F_50M"}),
    ],
    "城市层级": [
        ("一线城市", "tier_1", {"tier": 1, "cities": "北京,上海,广州,深圳"}),
        ("新一线城市", "new_tier_1", {"tier": 1.5, "cities": "成都,杭州,重庆,武汉"}),
        ("二线城市", "tier_2", {"tier": 2}),
        ("三线城市", "tier_3", {"tier": 3}),
        ("下沉市场", "lower_tier", {"tier": 4, "cities": "四五线及以下"}),
    ],
}

# 兴趣质心标签
INTEREST_PRIMARY_TAGS: list[tuple[str, str]] = [
    ("美妆护肤", "beauty_interest"),
    ("穿搭时尚", "fashion_interest"),
    ("美食探店", "food_interest"),
    ("科技数码", "tech_interest"),
    ("家居家装", "home_interest"),
    ("母婴育儿", "parenting_interest"),
    ("健身运动", "fitness_interest"),
    ("旅行探店", "travel_interest"),
    ("汽车", "automotive_interest"),
    ("宠物", "pet_interest"),
]

INTEREST_SECONDARY_TAGS: list[tuple[str, str]] = [
    ("影视娱乐", "entertainment"),
    ("学习成长", "learning"),
    ("理财投资", "finance"),
    ("游戏", "gaming"),
    ("音乐", "music"),
    ("摄影", "photography"),
]

# 消费行为标签
BEHAVIOR_FREQ_TAGS: list[tuple[str, str, dict]] = [
    ("高频消费", "high_frequency", {"frequency": "monthly_3+", "desc": "月均购买3次以上"}),
    ("中频消费", "medium_frequency", {"frequency": "monthly_1_2", "desc": "月均购买1-2次"}),
    ("低频消费", "low_frequency", {"frequency": "quarterly_1", "desc": "季度购买1次"}),
]

BEHAVIOR_DECISION_TAGS: list[tuple[str, str]] = [
    ("成分党", "ingredient_focused"),
    ("颜值党", "aesthetic_focused"),
    ("性价比", "value_focused"),
    ("品牌忠诚", "brand_loyal"),
    ("KOL驱动", "kol_driven"),
    ("从众心理", "herd_mentality"),
    ("理性决策", "rational_decision"),
]

BEHAVIOR_CHANNEL_TAGS: list[tuple[str, str]] = [
    ("直播冲动型", "live_impulse"),
    ("搜索目的型", "search_intent"),
    ("种草拔草型", "seeding_harvesting"),
    ("社交分享型", "social_share"),
    ("比价型", "price_comparison"),
]

# 内容偏好标签
CONTENT_FORMAT_TAGS: list[tuple[str, str]] = [
    ("短视频", "short_video"),
    ("图文笔记", "image_text_note"),
    ("直播", "live_stream"),
    ("测评长文", "review_article"),
    ("Vlog", "vlog"),
    ("教程", "tutorial"),
]

CONTENT_TONE_TAGS: list[tuple[str, str]] = [
    ("专业硬核", "professional"),
    ("轻松日常", "casual"),
    ("种草安利", "recommendation"),
    ("避坑指南", "anti_pitfall"),
    ("搞笑幽默", "humorous"),
    ("真实测评", "authentic_review"),
]

CONTENT_INTERACTION_TAGS: list[tuple[str, str]] = [
    ("评论活跃", "comment_active"),
    ("点赞收藏", "like_save"),
    ("分享传播", "share_spread"),
    ("潜水观看", "lurker"),
    ("二次创作", "remix"),
]

# 生命周期标签
LIFECYCLE_TAGS: list[tuple[str, str, dict]] = [
    ("新客", "new_customer", {"days_since_first": 0, "purchases": 0}),
    ("首购", "first_purchase", {"purchases": 1}),
    ("复购", "repeat_purchase", {"purchases": "2+"}),
    ("活跃", "active", {"last_purchase_days": "<30"}),
    ("沉睡", "dormant", {"last_purchase_days": "30-90"}),
    ("流失", "churned", {"last_purchase_days": ">90"}),
]


class CrowdDomain:
    """
    人群域标签管理器

    管理目标人群相关的所有标签，包括人口统计、兴趣质心、
    消费行为、内容偏好和生命周期阶段。
    """

    def __init__(self, graph: TagAssociationGraph):
        self.graph = graph

        # 内部索引: tag_name -> tag_id
        self._name_index: dict[str, str] = {}

        # 子域索引
        self._sub_domains: dict[str, dict[str, str]] = {
            SUB_DOMAIN_DEMOGRAPHICS: {},
            SUB_DOMAIN_INTEREST: {},
            SUB_DOMAIN_BEHAVIOR: {},
            SUB_DOMAIN_CONTENT_PREF: {},
            SUB_DOMAIN_LIFECYCLE: {},
        }

    # ── 初始化 ────────────────────────────────────────────────────

    def initialize(self):
        """初始化预置标签数据"""
        self._init_demographics()
        self._init_interests()
        self._init_behavior()
        self._init_content_preference()
        self._init_lifecycle()
        logger.info(
            f"人群域初始化完成: 人口统计{sum(len(v) for v in DEMOGRAPHICS_TAGS.values())}个, "
            f"兴趣{len(INTEREST_PRIMARY_TAGS) + len(INTEREST_SECONDARY_TAGS)}个, "
            f"行为{len(BEHAVIOR_FREQ_TAGS) + len(BEHAVIOR_DECISION_TAGS) + len(BEHAVIOR_CHANNEL_TAGS)}个, "
            f"内容偏好{len(CONTENT_FORMAT_TAGS) + len(CONTENT_TONE_TAGS) + len(CONTENT_INTERACTION_TAGS)}个, "
            f"生命周期{len(LIFECYCLE_TAGS)}个"
        )

    def _init_demographics(self):
        """初始化人口统计标签"""
        for sub_cat, tags in DEMOGRAPHICS_TAGS.items():
            # 创建子分类节点
            sub_id = self.graph.add_tag(
                name=sub_cat,
                name_en=sub_cat,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_2,
                description=f"人口统计子分类: {sub_cat}",
                metadata={"sub_domain": SUB_DOMAIN_DEMOGRAPHICS},
            )
            self._name_index[sub_cat] = sub_id

            for name, name_en, meta in tags:
                tag_id = self.graph.add_tag(
                    name=name,
                    name_en=name_en,
                    domain=DOMAIN_CROWD,
                    level=LEVEL_CATEGORY_1,
                    parent_id=sub_id,
                    description=f"人口统计: {name}",
                    metadata={"sub_domain": SUB_DOMAIN_DEMOGRAPHICS, **meta},
                )
                self._name_index[name] = tag_id
                self._sub_domains[SUB_DOMAIN_DEMOGRAPHICS][name] = tag_id

    def _init_interests(self):
        """初始化兴趣质心标签"""
        # 主兴趣簇
        primary_id = self.graph.add_tag(
            name="主兴趣簇",
            name_en="primary_interests",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户主要兴趣方向",
            metadata={"sub_domain": SUB_DOMAIN_INTEREST, "type": "primary"},
        )
        self._name_index["主兴趣簇"] = primary_id

        for name, name_en in INTEREST_PRIMARY_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=primary_id,
                description=f"主兴趣: {name}",
                metadata={"sub_domain": SUB_DOMAIN_INTEREST, "type": "primary"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_INTEREST][name] = tag_id

        # 次兴趣簇
        secondary_id = self.graph.add_tag(
            name="次兴趣簇",
            name_en="secondary_interests",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户次要兴趣方向",
            metadata={"sub_domain": SUB_DOMAIN_INTEREST, "type": "secondary"},
        )
        self._name_index["次兴趣簇"] = secondary_id

        for name, name_en in INTEREST_SECONDARY_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=secondary_id,
                description=f"次兴趣: {name}",
                metadata={"sub_domain": SUB_DOMAIN_INTEREST, "type": "secondary"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_INTEREST][name] = tag_id

    def _init_behavior(self):
        """初始化消费行为标签"""
        # 消费频次
        freq_id = self.graph.add_tag(
            name="消费频次",
            name_en="purchase_frequency",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户消费频率特征",
            metadata={"sub_domain": SUB_DOMAIN_BEHAVIOR, "type": "frequency"},
        )
        self._name_index["消费频次"] = freq_id

        for name, name_en, meta in BEHAVIOR_FREQ_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=freq_id,
                description=f"消费频次: {name}",
                metadata={"sub_domain": SUB_DOMAIN_BEHAVIOR, **meta},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_BEHAVIOR][name] = tag_id

        # 决策因子
        decision_id = self.graph.add_tag(
            name="决策因子",
            name_en="decision_factors",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户购买决策的主要驱动因素",
            metadata={"sub_domain": SUB_DOMAIN_BEHAVIOR, "type": "decision"},
        )
        self._name_index["决策因子"] = decision_id

        for name, name_en in BEHAVIOR_DECISION_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=decision_id,
                description=f"决策因子: {name}",
                metadata={"sub_domain": SUB_DOMAIN_BEHAVIOR, "type": "decision"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_BEHAVIOR][name] = tag_id

        # 渠道偏好
        channel_id = self.graph.add_tag(
            name="渠道偏好",
            name_en="channel_preference",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户偏好的购物渠道",
            metadata={"sub_domain": SUB_DOMAIN_BEHAVIOR, "type": "channel"},
        )
        self._name_index["渠道偏好"] = channel_id

        for name, name_en in BEHAVIOR_CHANNEL_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=channel_id,
                description=f"渠道偏好: {name}",
                metadata={"sub_domain": SUB_DOMAIN_BEHAVIOR, "type": "channel"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_BEHAVIOR][name] = tag_id

    def _init_content_preference(self):
        """初始化内容偏好标签"""
        # 内容形式
        format_id = self.graph.add_tag(
            name="内容形式",
            name_en="content_format",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户偏好的内容形式",
            metadata={"sub_domain": SUB_DOMAIN_CONTENT_PREF, "type": "format"},
        )
        self._name_index["内容形式"] = format_id

        for name, name_en in CONTENT_FORMAT_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=format_id,
                description=f"内容形式: {name}",
                metadata={"sub_domain": SUB_DOMAIN_CONTENT_PREF, "type": "format"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_CONTENT_PREF][name] = tag_id

        # 内容调性
        tone_id = self.graph.add_tag(
            name="内容调性",
            name_en="content_tone",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户偏好的内容风格",
            metadata={"sub_domain": SUB_DOMAIN_CONTENT_PREF, "type": "tone"},
        )
        self._name_index["内容调性"] = tone_id

        for name, name_en in CONTENT_TONE_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=tone_id,
                description=f"内容调性: {name}",
                metadata={"sub_domain": SUB_DOMAIN_CONTENT_PREF, "type": "tone"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_CONTENT_PREF][name] = tag_id

        # 互动模式
        interaction_id = self.graph.add_tag(
            name="互动模式",
            name_en="interaction_mode",
            domain=DOMAIN_CROWD,
            level=LEVEL_CATEGORY_2,
            description="用户的内容互动习惯",
            metadata={"sub_domain": SUB_DOMAIN_CONTENT_PREF, "type": "interaction"},
        )
        self._name_index["互动模式"] = interaction_id

        for name, name_en in CONTENT_INTERACTION_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                parent_id=interaction_id,
                description=f"互动模式: {name}",
                metadata={"sub_domain": SUB_DOMAIN_CONTENT_PREF, "type": "interaction"},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_CONTENT_PREF][name] = tag_id

    def _init_lifecycle(self):
        """初始化生命周期标签"""
        for name, name_en, meta in LIFECYCLE_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_CROWD,
                level=LEVEL_CATEGORY_1,
                description=f"生命周期: {name}",
                metadata={"sub_domain": SUB_DOMAIN_LIFECYCLE, **meta},
            )
            self._name_index[name] = tag_id
            self._sub_domains[SUB_DOMAIN_LIFECYCLE][name] = tag_id

    # ── 查询 ──────────────────────────────────────────────────────

    def get_tag_by_name(self, name: str) -> TagNode | None:
        """按名称获取标签"""
        tag_id = self._name_index.get(name)
        if tag_id:
            return self.graph.tags.get(tag_id)
        return None

    def get_demographics_tags(self) -> list[TagNode]:
        """获取所有人人口统计标签"""
        return self._get_sub_domain_tags(SUB_DOMAIN_DEMOGRAPHICS)

    def get_interest_tags(self) -> list[TagNode]:
        """获取所有兴趣标签"""
        return self._get_sub_domain_tags(SUB_DOMAIN_INTEREST)

    def get_behavior_tags(self) -> list[TagNode]:
        """获取所有消费行为标签"""
        return self._get_sub_domain_tags(SUB_DOMAIN_BEHAVIOR)

    def get_content_preference_tags(self) -> list[TagNode]:
        """获取所有内容偏好标签"""
        return self._get_sub_domain_tags(SUB_DOMAIN_CONTENT_PREF)

    def get_lifecycle_tags(self) -> list[TagNode]:
        """获取所有生命周期标签"""
        return self._get_sub_domain_tags(SUB_DOMAIN_LIFECYCLE)

    def _get_sub_domain_tags(self, sub_domain: str) -> list[TagNode]:
        """获取某个子域的所有标签"""
        tag_ids = self._sub_domains.get(sub_domain, {})
        return [
            self.graph.tags[tid]
            for tid in tag_ids.values()
            if tid in self.graph.tags
        ]

    def get_tags_by_sub_category(self, sub_category: str) -> list[TagNode]:
        """获取某个子分类下的标签"""
        cat_id = self._name_index.get(sub_category)
        if not cat_id:
            return []
        children = self.graph.get_children(cat_id)
        return children

    def search(self, keyword: str) -> list[TagNode]:
        """按关键词搜索人群标签"""
        results = []
        for name, tag_id in self._name_index.items():
            if keyword in name and tag_id in self.graph.tags:
                results.append(self.graph.tags[tag_id])
        return results

    # ── CRUD ──────────────────────────────────────────────────────

    def add_tag(
        self,
        name: str,
        name_en: str,
        sub_domain: str = SUB_DOMAIN_DEMOGRAPHICS,
        parent_name: str | None = None,
        level: int = LEVEL_CATEGORY_1,
        description: str = "",
        metadata: dict | None = None,
    ) -> str:
        """添加自定义标签"""
        parent_id = None
        if parent_name:
            parent_id = self._name_index.get(parent_name)

        meta = {"sub_domain": sub_domain}
        if metadata:
            meta.update(metadata)

        tag_id = self.graph.add_tag(
            name=name,
            name_en=name_en,
            domain=DOMAIN_CROWD,
            level=level,
            parent_id=parent_id,
            description=description or f"{sub_domain}: {name}",
            metadata=meta,
        )
        self._name_index[name] = tag_id
        if sub_domain in self._sub_domains:
            self._sub_domains[sub_domain][name] = tag_id

        return tag_id

    def remove_tag(self, name: str) -> bool:
        """按名称删除标签"""
        tag_id = self._name_index.pop(name, None)
        if not tag_id:
            return False
        for sub_domain in self._sub_domains.values():
            sub_domain.pop(name, None)
        return self.graph.remove_tag(tag_id)

    def count(self) -> dict:
        """统计各子域标签数量"""
        return {
            "demographics": len(self._sub_domains.get(SUB_DOMAIN_DEMOGRAPHICS, {})),
            "interest": len(self._sub_domains.get(SUB_DOMAIN_INTEREST, {})),
            "behavior": len(self._sub_domains.get(SUB_DOMAIN_BEHAVIOR, {})),
            "content_preference": len(self._sub_domains.get(SUB_DOMAIN_CONTENT_PREF, {})),
            "lifecycle": len(self._sub_domains.get(SUB_DOMAIN_LIFECYCLE, {})),
            "total": len(self._name_index),
        }
