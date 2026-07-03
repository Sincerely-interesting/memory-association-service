"""
内容特征信号处理器 (ContentSignalProcessor)

解析内容数据，提取标准化特征:
- 主题分类 (topic_classification)
- 情感分析 (sentiment)
- 视觉风格 (visual_style)
- 文案调性 (text_tone)
- 内容质量 (quality_score)
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
    from core.models import SignalFeature, SIGNAL_CONTENT
except ImportError:
    from models import SignalFeature, SIGNAL_CONTENT


# ── 主题分类定义 ──────────────────────────────────────────────────

TOPIC_CATEGORIES: dict[str, list[str]] = {
    "美妆护肤": ["护肤", "彩妆", "美妆", "面膜", "精华", "口红", "防晒"],
    "穿搭时尚": ["穿搭", "时尚", "衣服", "鞋", "包", "配饰"],
    "美食探店": ["美食", "餐厅", "探店", "小吃", "甜品", "咖啡"],
    "科技数码": ["手机", "电脑", "数码", "科技", "测评", "开箱"],
    "家居家装": ["家居", "装修", "家具", "收纳", "装饰"],
    "母婴育儿": ["母婴", "育儿", "宝宝", "婴儿", "奶粉"],
    "运动健身": ["运动", "健身", "瑜伽", "跑步", "减脂"],
    "旅行探店": ["旅行", "旅游", "酒店", "景点", "攻略"],
}

# 视觉风格定义
VISUAL_STYLES = [
    "clean_minimal",    # 简约干净
    "warm_cozy",        # 温暖舒适
    "bright_vivid",     # 明亮鲜艳
    "dark_moody",       # 暗色氛围
    "pastel_soft",      # 柔和粉彩
    "high_contrast",    # 高对比度
    "natural_organic",  # 自然有机
    "glamorous",        # 华丽精致
]

# 文案调性定义
TEXT_TONES = [
    "professional",     # 专业权威
    "casual_friendly",  # 轻松友好
    "humorous",         # 幽默搞笑
    "emotional",        # 情感共鸣
    "informative",      # 信息丰富
    "persuasive",       # 说服力强
    "storytelling",     # 故事叙述
    "urgent",           # 紧迫感
]


@dataclass
class ContentData:
    """内容数据"""
    content_id: str
    creator_id: str

    # 文本内容
    title: str = ""
    description: str = ""
    full_text: str = ""

    # 视觉信息
    has_image: bool = False
    has_video: bool = False
    image_count: int = 0
    video_duration_seconds: float = 0.0

    # 元数据
    platform: str = ""
    content_type: str = ""       # short_video / image_text / article / live
    topic: str = ""
    tags: list[str] = field(default_factory=list)

    # 互动数据（可选，用于质量评估）
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    views: int = 0

    # 时间
    published_at: float = 0.0
    collected_at: float = 0.0

    def __post_init__(self):
        if self.collected_at == 0.0:
            self.collected_at = time.time()

    @property
    def all_text(self) -> str:
        """合并所有文本"""
        return f"{self.title} {self.description} {self.full_text}"


class ContentSignalProcessor:
    """
    内容特征信号处理器

    将内容数据转换为标准化的 SignalFeature。
    """

    def __init__(self):
        self._topic_keywords = TOPIC_CATEGORIES

    # ── 主处理入口 ────────────────────────────────────────────────

    def process(
        self,
        content: ContentData,
        generate_embedding: bool = True,
    ) -> SignalFeature:
        """
        处理内容数据，生成 SignalFeature

        Args:
            content: 内容数据
            generate_embedding: 是否生成嵌入向量

        Returns:
            SignalFeature
        """
        features = {}

        # 1. 主题分类
        topic_features = self._classify_topic(content)
        features.update(topic_features)

        # 2. 情感分析
        sentiment_features = self._analyze_sentiment(content)
        features.update(sentiment_features)

        # 3. 视觉风格
        visual_features = self._detect_visual_style(content)
        features.update(visual_features)

        # 4. 文案调性
        tone_features = self._detect_text_tone(content)
        features.update(tone_features)

        # 5. 内容质量
        quality_features = self._assess_quality(content)
        features.update(quality_features)

        # 6. 生成嵌入向量
        embedding = None
        if generate_embedding:
            embedding = self._generate_embedding(content, features)

        # 7. 构建 SignalFeature
        signal_id = self._make_signal_id(content)
        signal = SignalFeature(
            id=signal_id,
            source=SIGNAL_CONTENT,
            features=features,
            embedding=embedding,
            platform=content.platform,
            creator_id=content.creator_id,
            content_id=content.content_id,
            captured_at=content.collected_at,
        )

        return signal

    def process_batch(
        self,
        contents: list[ContentData],
        generate_embedding: bool = True,
    ) -> list[SignalFeature]:
        """批量处理"""
        return [self.process(c, generate_embedding) for c in contents]

    # ── 主题分类 ──────────────────────────────────────────────────

    def _classify_topic(self, c: ContentData) -> dict[str, float]:
        """主题分类"""
        features = {}
        text = c.all_text.lower()

        # 关键词匹配
        scores = {}
        for category, keywords in self._topic_keywords.items():
            score = 0.0
            for kw in keywords:
                if kw in text:
                    score += 0.3
            scores[category] = min(1.0, score)

        # 选择最高分的主题
        if scores:
            max_category = max(scores, key=scores.get)
            features["topic_score"] = scores[max_category]
        else:
            max_category = "其他"
            features["topic_score"] = 0.0

        # 主题 one-hot
        all_categories = list(self._topic_keywords.keys()) + ["其他"]
        for i, cat in enumerate(all_categories):
            features[f"topic_{i}"] = 1.0 if cat == max_category else 0.0

        features["topic_category"] = all_categories.index(max_category) / len(all_categories)

        # 显式主题标签
        if c.topic:
            for cat, keywords in self._topic_keywords.items():
                if c.topic in keywords or c.topic == cat:
                    features["explicit_topic_match"] = 1.0
                    break
            else:
                features["explicit_topic_match"] = 0.0
        else:
            features["explicit_topic_match"] = 0.0

        return features

    # ── 情感分析 ──────────────────────────────────────────────────

    def _analyze_sentiment(self, c: ContentData) -> dict[str, float]:
        """情感分析（基于规则）"""
        text = c.all_text

        # 正面词
        positive_words = [
            "好", "棒", "喜欢", "推荐", "爱", "赞", "优秀", "完美",
            "好看", "好吃", "好用", "值得", "满意", "惊喜", "绝了",
        ]
        # 负面词
        negative_words = [
            "差", "烂", "难", "失望", "不好", "糟糕", "退货", "坑",
            "踩雷", "难用", "浪费", "后悔", "假", "骗",
        ]
        # 程度词
        intensifiers = ["非常", "特别", "超级", "真的", "太", "超", "巨"]

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        intensifier_count = sum(1 for w in intensifiers if w in text)

        # 情感得分
        total = pos_count + neg_count
        if total > 0:
            base_sentiment = (pos_count - neg_count) / total
        else:
            base_sentiment = 0.0

        # 程度词加强
        if intensifier_count > 0:
            base_sentiment *= (1 + intensifier_count * 0.2)

        features = {
            "sentiment_score": max(-1.0, min(1.0, base_sentiment)),
            "sentiment_positive": 1.0 if base_sentiment > 0.1 else 0.0,
            "sentiment_negative": 1.0 if base_sentiment < -0.1 else 0.0,
            "sentiment_neutral": 1.0 if abs(base_sentiment) <= 0.1 else 0.0,
            "positive_word_count": float(pos_count),
            "negative_word_count": float(neg_count),
        }

        return features

    # ── 视觉风格 ──────────────────────────────────────────────────

    def _detect_visual_style(self, c: ContentData) -> dict[str, float]:
        """视觉风格检测（基于元数据推断）"""
        features = {}

        # 基于内容类型推断
        style_scores = {s: 0.0 for s in VISUAL_STYLES}

        if c.has_video:
            if c.video_duration_seconds < 30:
                style_scores["bright_vivid"] += 0.3
                style_scores["glamorous"] += 0.2
            else:
                style_scores["natural_organic"] += 0.3
                style_scores["storytelling"] = 0.3

        if c.has_image:
            if c.image_count > 3:
                style_scores["high_contrast"] += 0.3
                style_scores["warm_cozy"] += 0.2
            else:
                style_scores["clean_minimal"] += 0.3
                style_scores["pastel_soft"] += 0.2

        # 基于文本推断
        text = c.all_text.lower()
        if any(w in text for w in ["简约", "极简", "clean", "minimal"]):
            style_scores["clean_minimal"] += 0.4
        if any(w in text for w in ["温暖", "温馨", "cozy", "comfort"]):
            style_scores["warm_cozy"] += 0.4
        if any(w in text for w in ["高级", "奢华", "luxury", "premium"]):
            style_scores["glamorous"] += 0.4
        if any(w in text for w in ["自然", "有机", "natural", "organic"]):
            style_scores["natural_organic"] += 0.4

        # 选择最高分
        if any(v > 0 for v in style_scores.values()):
            top_style = max(style_scores, key=style_scores.get)
            features["visual_style_score"] = style_scores[top_style]
        else:
            top_style = "clean_minimal"
            features["visual_style_score"] = 0.1

        # one-hot
        for i, style in enumerate(VISUAL_STYLES):
            features[f"style_{i}"] = 1.0 if style == top_style else 0.0

        features["has_image"] = 1.0 if c.has_image else 0.0
        features["has_video"] = 1.0 if c.has_video else 0.0
        features["image_count_norm"] = min(1.0, c.image_count / 10)

        return features

    # ── 文案调性 ──────────────────────────────────────────────────

    def _detect_text_tone(self, c: ContentData) -> dict[str, float]:
        """文案调性检测"""
        text = c.all_text
        features = {}

        tone_scores = {t: 0.0 for t in TEXT_TONES}

        # 关键词匹配
        tone_keywords = {
            "professional": ["专业", "测评", "成分", "功效", "数据"],
            "casual_friendly": ["日常", "分享", "简单", "轻松", "随手"],
            "humorous": ["哈哈", "笑死", "绝了", "离谱", "搞笑"],
            "emotional": ["感动", "治愈", "温暖", "幸福", "回忆"],
            "informative": ["教程", "攻略", "科普", "详解", "步骤"],
            "persuasive": ["必买", "推荐", "安利", "种草", "必入"],
            "storytelling": ["故事", "经历", "曾经", "那天", "回忆"],
            "urgent": ["限时", "最后", "赶紧", "快", "抢"],
        }

        for tone, keywords in tone_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            tone_scores[tone] = min(1.0, count * 0.3)

        # 选择最高分
        if any(v > 0 for v in tone_scores.values()):
            top_tone = max(tone_scores, key=tone_scores.get)
        else:
            top_tone = "casual_friendly"

        features["text_tone_score"] = tone_scores.get(top_tone, 0.0)

        # one-hot
        for i, tone in enumerate(TEXT_TONES):
            features[f"tone_{i}"] = 1.0 if tone == top_tone else 0.0

        # 文本长度特征
        features["text_length"] = min(1.0, len(text) / 1000)
        features["title_length"] = min(1.0, len(c.title) / 100)

        return features

    # ── 内容质量 ──────────────────────────────────────────────────

    def _assess_quality(self, c: ContentData) -> dict[str, float]:
        """内容质量评估"""
        features = {}

        # 互动质量
        total_engagement = c.likes + c.comments + c.shares + c.saves
        views = max(c.views, 1)

        features["engagement_rate"] = total_engagement / views
        features["like_rate"] = c.likes / views
        features["comment_rate"] = c.comments / views
        features["share_rate"] = c.shares / views
        features["save_rate"] = c.saves / views

        # 质量综合分
        quality = 0.0
        quality += min(1.0, total_engagement / 1000) * 0.3  # 互动量
        quality += features["save_rate"] * 0.3              # 收藏率（高质量指标）
        quality += features["comment_rate"] * 0.2           # 评论率
        quality += min(1.0, len(c.all_text) / 500) * 0.2   # 内容丰富度
        features["quality_score"] = min(1.0, quality)

        # 内容丰富度
        features["has_title"] = 1.0 if c.title else 0.0
        features["has_description"] = 1.0 if c.description else 0.0
        features["tag_count"] = min(1.0, len(c.tags) / 10)

        return features

    # ── 嵌入向量 ──────────────────────────────────────────────────

    def _generate_embedding(
        self,
        c: ContentData,
        features: dict[str, float],
    ) -> list[float]:
        """生成特征嵌入向量"""
        vector = [0.0] * 64

        # [0-7] 主题特征
        all_categories = list(self._topic_keywords.keys()) + ["其他"]
        for i, cat in enumerate(all_categories[:8]):
            vector[i] = features.get(f"topic_{i}", 0)

        # [8-15] 情感特征
        vector[8] = (features.get("sentiment_score", 0) + 1) / 2  # 归一化到 0-1
        vector[9] = features.get("sentiment_positive", 0)
        vector[10] = features.get("sentiment_negative", 0)
        vector[11] = features.get("positive_word_count", 0) / 5
        vector[12] = features.get("negative_word_count", 0) / 5

        # [16-23] 视觉风格
        for i, style in enumerate(VISUAL_STYLES[:8]):
            vector[16 + i] = features.get(f"style_{i}", 0)

        # [24-31] 文案调性
        for i, tone in enumerate(TEXT_TONES[:8]):
            vector[24 + i] = features.get(f"tone_{i}", 0)

        # [32-39] 内容质量
        vector[32] = features.get("quality_score", 0)
        vector[33] = features.get("engagement_rate", 0)
        vector[34] = features.get("save_rate", 0)
        vector[35] = features.get("text_length", 0)
        vector[36] = features.get("has_image", 0)
        vector[37] = features.get("has_video", 0)

        # [40-63] 内容哈希
        content_str = f"{c.title}{' '.join(c.tags)}"
        for i, char in enumerate(content_str[:24]):
            hash_val = ord(char) % 24
            vector[40 + hash_val] = max(vector[40 + hash_val], 0.5)

        # L2 归一化
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    # ── 辅助方法 ──────────────────────────────────────────────────

    def _make_signal_id(self, c: ContentData) -> str:
        """生成信号ID"""
        raw = f"{c.platform}:{c.creator_id}:{c.content_id}:{c.collected_at}"
        return f"content_{hashlib.md5(raw.encode()).hexdigest()[:12]}"
