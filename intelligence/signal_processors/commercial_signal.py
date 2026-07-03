"""
商单机会信号处理器 (CommercialSignalProcessor)

解析品牌商单机会数据，提取标准化特征:
- 品牌调性 (brand_tone)
- 预算范围 (budget_range)
- KPI要求 (kpi_requirements)
- 目标人群 (target_audience)
- 内容要求 (content_requirements)
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
    from core.models import SignalFeature, SIGNAL_COMMERCIAL
except ImportError:
    from models import SignalFeature, SIGNAL_COMMERCIAL


# ── 品牌调性定义 ──────────────────────────────────────────────────

BRAND_TONE_MAP: dict[str, list[str]] = {
    "年轻时尚": ["gen_z", "fashion", "trendy", "youthful"],
    "高端奢华": ["premium", "luxury", "exclusive", "sophisticated"],
    "国潮文化": ["guochao", "traditional", "cultural", "heritage"],
    "科技极客": ["tech", "innovation", "geek", "futuristic"],
    "自然健康": ["natural", "organic", "healthy", "sustainable"],
    "可爱甜美": ["cute", "sweet", "kawaii", "playful"],
    "专业权威": ["professional", "authority", "expert", "trustworthy"],
    "运动活力": ["sporty", "active", "energetic", "dynamic"],
}

# 预算等级映射
BUDGET_LEVELS: dict[str, int] = {
    "micro": 1,       # <5k
    "small": 2,       # 5k-20k
    "medium": 3,      # 20k-100k
    "large": 4,       # 100k-500k
    "premium": 5,     # 500k+
}

# KPI 类型
KPI_TYPES = [
    "exposure",       # 曝光量
    "engagement",     # 互动量
    "conversion",     # 转化量
    "brand_awareness", # 品牌认知
    "direct_sales",   # 直接销售
]


@dataclass
class BrandOpportunity:
    """品牌商单机会"""
    opportunity_id: str
    brand_name: str

    # 品牌信息
    brand_tone: list[str] = field(default_factory=list)
    brand_category: str = ""
    brand_level: str = ""          # tier_1 / tier_2 / tier_3

    # 预算
    budget_range: str = ""         # "5k-20k" / "20k-100k" / ...
    budget_currency: str = "CNY"
    budget_amount_min: float = 0.0
    budget_amount_max: float = 0.0

    # KPI要求
    kpi_type: str = "engagement"
    kpi_target: float = 0.0        # 目标值
    kpi_unit: str = ""             # "impressions" / "likes" / "sales"

    # 目标人群
    target_age_range: str = ""
    target_gender: str = ""        # female / male / all
    target_city_tier: str = ""

    # 内容要求
    content_type: str = ""         # short_video / image_text / live
    content_count: int = 1
    content_deadline_days: int = 30

    # 平台
    platform: str = ""
    platforms: list[str] = field(default_factory=list)

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    # 时间
    created_at: float = 0.0
    expires_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class CommercialSignalProcessor:
    """
    商单机会信号处理器

    将品牌商单机会数据转换为标准化的 SignalFeature。
    """

    def __init__(self):
        self._tone_map = BRAND_TONE_MAP

    # ── 主处理入口 ────────────────────────────────────────────────

    def process(
        self,
        opportunity: BrandOpportunity,
        generate_embedding: bool = True,
    ) -> SignalFeature:
        """
        处理商单机会数据，生成 SignalFeature

        Args:
            opportunity: 品牌商单机会
            generate_embedding: 是否生成嵌入向量

        Returns:
            SignalFeature
        """
        features = {}

        # 1. 品牌调性特征
        brand_features = self._extract_brand_features(opportunity)
        features.update(brand_features)

        # 2. 预算特征
        budget_features = self._extract_budget_features(opportunity)
        features.update(budget_features)

        # 3. KPI特征
        kpi_features = self._extract_kpi_features(opportunity)
        features.update(kpi_features)

        # 4. 目标人群特征
        audience_features = self._extract_audience_features(opportunity)
        features.update(audience_features)

        # 5. 内容要求特征
        content_features = self._extract_content_features(opportunity)
        features.update(content_features)

        # 6. 生成嵌入向量
        embedding = None
        if generate_embedding:
            embedding = self._generate_embedding(opportunity, features)

        # 7. 构建 SignalFeature
        signal_id = self._make_signal_id(opportunity)
        signal = SignalFeature(
            id=signal_id,
            source=SIGNAL_COMMERCIAL,
            features=features,
            embedding=embedding,
            platform=opportunity.platform or (opportunity.platforms[0] if opportunity.platforms else None),
            creator_id=opportunity.brand_name,
            content_id=opportunity.opportunity_id,
            captured_at=opportunity.created_at,
            expires_at=opportunity.expires_at,
        )

        return signal

    def process_batch(
        self,
        opportunities: list[BrandOpportunity],
        generate_embedding: bool = True,
    ) -> list[SignalFeature]:
        """批量处理"""
        return [self.process(o, generate_embedding) for o in opportunities]

    # ── 品牌调性特征 ──────────────────────────────────────────────

    def _extract_brand_features(self, o: BrandOpportunity) -> dict[str, float]:
        """提取品牌调性特征"""
        features = {}

        # 品牌等级编码
        level_map = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4}
        features["brand_level_score"] = level_map.get(o.brand_level, 0.5)

        # 品牌调性 one-hot
        all_tones = list(self._tone_map.keys())
        for i, tone in enumerate(all_tones):
            features[f"tone_{i}"] = 1.0 if tone in o.brand_tone else 0.0

        # 品牌调性匹配度（与预定义调性的匹配）
        tone_match_score = 0.0
        for tone in o.brand_tone:
            if tone in self._tone_map:
                tone_match_score += 0.2
        features["brand_tone_match"] = min(1.0, tone_match_score)

        return features

    # ── 预算特征 ──────────────────────────────────────────────────

    def _extract_budget_features(self, o: BrandOpportunity) -> dict[str, float]:
        """提取预算特征"""
        features = {}

        # 预算等级
        budget_level = BUDGET_LEVELS.get(o.budget_range, 3)
        features["budget_level"] = budget_level / 5.0  # 归一化到 0-1

        # 预算金额归一化
        if o.budget_amount_max > 0:
            features["budget_amount_norm"] = min(1.0, o.budget_amount_max / 500000)
        else:
            features["budget_amount_norm"] = budget_level / 5.0

        # 预算区间宽度（越大越灵活）
        if o.budget_amount_min > 0 and o.budget_amount_max > 0:
            width = o.budget_amount_max - o.budget_amount_min
            features["budget_flexibility"] = min(1.0, width / o.budget_amount_max)
        else:
            features["budget_flexibility"] = 0.5

        return features

    # ── KPI特征 ──────────────────────────────────────────────────

    def _extract_kpi_features(self, o: BrandOpportunity) -> dict[str, float]:
        """提取KPI特征"""
        features = {}

        # KPI类型编码
        for i, kpi_type in enumerate(KPI_TYPES):
            features[f"kpi_{kpi_type}"] = 1.0 if o.kpi_type == kpi_type else 0.0

        # KPI目标归一化
        if o.kpi_target > 0:
            features["kpi_target_norm"] = min(1.0, o.kpi_target / 100000)
        else:
            features["kpi_target_norm"] = 0.5

        # KPI难度评估
        difficulty_map = {
            "exposure": 0.3,
            "engagement": 0.5,
            "conversion": 0.8,
            "brand_awareness": 0.4,
            "direct_sales": 0.9,
        }
        features["kpi_difficulty"] = difficulty_map.get(o.kpi_type, 0.5)

        return features

    # ── 目标人群特征 ──────────────────────────────────────────────

    def _extract_audience_features(self, o: BrandOpportunity) -> dict[str, float]:
        """提取目标人群特征"""
        features = {}

        # 年龄段
        age_map = {
            "18-24": [1, 0, 0, 0],
            "25-34": [0, 1, 0, 0],
            "35-44": [0, 0, 1, 0],
            "45+":   [0, 0, 0, 1],
            "all":   [0.25, 0.25, 0.25, 0.25],
        }
        age_vec = age_map.get(o.target_age_range, [0.25, 0.25, 0.25, 0.25])
        features["target_age_gen_z"] = age_vec[0]
        features["target_age_professional"] = age_vec[1]
        features["target_age_middle"] = age_vec[2]
        features["target_age_senior"] = age_vec[3]

        # 性别倾向
        gender_map = {"female": 0.8, "male": 0.2, "all": 0.5}
        features["target_female_ratio"] = gender_map.get(o.target_gender, 0.5)

        # 城市层级
        city_map = {"tier_1": 1.0, "tier_1_2": 0.8, "tier_2": 0.6, "tier_3": 0.4, "lower_tier": 0.2}
        features["target_city_tier"] = city_map.get(o.target_city_tier, 0.5)

        return features

    # ── 内容要求特征 ──────────────────────────────────────────────

    def _extract_content_features(self, o: BrandOpportunity) -> dict[str, float]:
        """提取内容要求特征"""
        features = {}

        # 内容类型
        content_types = ["short_video", "image_text", "live", "article"]
        for i, ct in enumerate(content_types):
            features[f"req_content_{ct}"] = 1.0 if o.content_type == ct else 0.0

        # 内容数量
        features["content_count_norm"] = min(1.0, o.content_count / 10)

        # 截止时间紧迫度
        if o.content_deadline_days > 0:
            features["deadline_urgency"] = min(1.0, 30 / o.content_deadline_days)
        else:
            features["deadline_urgency"] = 0.5

        # 平台数量
        features["platform_count"] = len(o.platforms) / 4.0

        return features

    # ── 嵌入向量 ──────────────────────────────────────────────────

    def _generate_embedding(
        self,
        o: BrandOpportunity,
        features: dict[str, float],
    ) -> list[float]:
        """生成特征嵌入向量"""
        vector = [0.0] * 64

        # [0-7] 预算特征
        vector[0] = features.get("budget_level", 0)
        vector[1] = features.get("budget_amount_norm", 0)
        vector[2] = features.get("budget_flexibility", 0)
        vector[3] = features.get("brand_level_score", 0)
        vector[4] = features.get("brand_tone_match", 0)

        # [8-15] KPI特征
        for i, kpi in enumerate(KPI_TYPES[:8]):
            vector[8 + i] = features.get(f"kpi_{kpi}", 0)

        # [16-23] 目标人群
        vector[16] = features.get("target_age_gen_z", 0)
        vector[17] = features.get("target_age_professional", 0)
        vector[18] = features.get("target_age_middle", 0)
        vector[19] = features.get("target_age_senior", 0)
        vector[20] = features.get("target_female_ratio", 0.5)
        vector[21] = features.get("target_city_tier", 0.5)

        # [24-31] 内容要求
        vector[24] = features.get("req_content_short_video", 0)
        vector[25] = features.get("req_content_image_text", 0)
        vector[26] = features.get("req_content_live", 0)
        vector[27] = features.get("content_count_norm", 0)
        vector[28] = features.get("deadline_urgency", 0)

        # [32-47] 品牌调性 one-hot
        all_tones = list(self._tone_map.keys())
        for i, tone in enumerate(all_tones[:16]):
            vector[32 + i] = features.get(f"tone_{i}", 0)

        # [48-63] 品牌名称哈希
        for i, char in enumerate(o.brand_name[:16]):
            hash_val = ord(char) % 16
            vector[48 + hash_val] = max(vector[48 + hash_val], 0.5)

        # L2 归一化
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    # ── 辅助方法 ──────────────────────────────────────────────────

    def _make_signal_id(self, o: BrandOpportunity) -> str:
        """生成信号ID"""
        raw = f"{o.brand_name}:{o.opportunity_id}:{o.created_at}"
        return f"commercial_{hashlib.md5(raw.encode()).hexdigest()[:12]}"
