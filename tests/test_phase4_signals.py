"""
Phase 4 单元测试: 信号处理器
验证社媒/商单/内容数据解析、特征提取、标签映射的正确性
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import SignalFeature, SIGNAL_SOCIAL, SIGNAL_COMMERCIAL, SIGNAL_CONTENT
from intelligence.signal_processors.social_signal import (
    SocialSignalProcessor, SocialMediaMetrics, PLATFORM_CONFIGS,
)
from intelligence.signal_processors.commercial_signal import (
    CommercialSignalProcessor, BrandOpportunity, BUDGET_LEVELS,
)
from intelligence.signal_processors.content_signal import (
    ContentSignalProcessor, ContentData,
    TOPIC_CATEGORIES, VISUAL_STYLES, TEXT_TONES,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def social_processor():
    return SocialSignalProcessor()


@pytest.fixture
def commercial_processor():
    return CommercialSignalProcessor()


@pytest.fixture
def content_processor():
    return ContentSignalProcessor()


@pytest.fixture
def sample_tiktok_metrics():
    return SocialMediaMetrics(
        platform="tiktok_douyin",
        content_id="video_001",
        creator_id="creator_001",
        likes=1200,
        comments=89,
        shares=45,
        saves=23,
        views=50000,
        completion_rate=0.72,
        content_type="short_video",
        topic="护肤测评",
        hashtags=["护肤", "精华", "测评"],
        audience_age_range="18-24",
        audience_gender_ratio="female_dominant",
        audience_city_tier="tier_1_2",
    )


@pytest.fixture
def sample_brand_opportunity():
    return BrandOpportunity(
        opportunity_id="opp_001",
        brand_name="某护肤品牌",
        brand_tone=["年轻时尚", "自然健康"],
        brand_category="美妆护肤",
        brand_level="tier_1",
        budget_range="medium",
        budget_amount_min=20000,
        budget_amount_max=80000,
        kpi_type="engagement",
        kpi_target=50000,
        kpi_unit="interactions",
        target_age_range="18-24",
        target_gender="female",
        target_city_tier="tier_1_2",
        content_type="short_video",
        content_count=3,
        content_deadline_days=14,
        platform="tiktok_douyin",
    )


@pytest.fixture
def sample_content_data():
    return ContentData(
        content_id="content_001",
        creator_id="creator_001",
        title="这款精华真的绝了！烟酰胺成分党必看",
        description="分享一下最近用的烟酰胺精华，效果非常好",
        full_text="今天给大家测评一款烟酰胺精华，成分很安全，效果非常好，推荐给大家",
        has_image=True,
        has_video=True,
        image_count=5,
        video_duration_seconds=45,
        platform="tiktok_douyin",
        content_type="short_video",
        topic="精华",
        tags=["护肤", "精华", "烟酰胺", "测评"],
        likes=800,
        comments=56,
        shares=23,
        saves=120,
        views=30000,
    )


# ============================================================
# Test: SocialSignalProcessor
# ============================================================

class TestSocialSignalProcessor:
    def test_process_basic(self, social_processor, sample_tiktok_metrics):
        """测试基本处理"""
        signal = social_processor.process(sample_tiktok_metrics)
        assert signal.source == SIGNAL_SOCIAL
        assert signal.platform == "tiktok_douyin"
        assert signal.creator_id == "creator_001"
        assert signal.content_id == "video_001"

    def test_engagement_features(self, social_processor, sample_tiktok_metrics):
        """测试互动率特征"""
        signal = social_processor.process(sample_tiktok_metrics)
        features = signal.features

        assert features["engagement_rate"] > 0
        assert features["like_rate"] > 0
        assert features["comment_rate"] > 0
        assert features["share_rate"] > 0
        assert features["save_rate"] > 0
        assert features["completion_rate"] == 0.72
        assert features["weighted_engagement"] > 0

    def test_content_features(self, social_processor, sample_tiktok_metrics):
        """测试内容特征"""
        signal = social_processor.process(sample_tiktok_metrics)
        features = signal.features

        assert features["content_type"] == "short_video"
        assert features["has_topic"] == 1.0
        assert features["hashtag_count"] == 3.0
        assert features["topic_heat"] > 0
        assert features.get("ct_short_video", 0) == 1.0

    def test_audience_features(self, social_processor, sample_tiktok_metrics):
        """测试受众特征"""
        signal = social_processor.process(sample_tiktok_metrics)
        features = signal.features

        assert features["audience_age_gen_z"] == 1.0
        assert features["female_ratio"] == 0.7
        assert features["city_tier_score"] == 0.8

    def test_embedding_generated(self, social_processor, sample_tiktok_metrics):
        """测试嵌入向量生成"""
        signal = social_processor.process(sample_tiktok_metrics)
        assert signal.embedding is not None
        assert len(signal.embedding) == 64
        # L2归一化后范数接近1
        norm = sum(x * x for x in signal.embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_no_embedding(self, social_processor, sample_tiktok_metrics):
        """测试不生成嵌入"""
        signal = social_processor.process(sample_tiktok_metrics, generate_embedding=False)
        assert signal.embedding is None

    def test_batch_process(self, social_processor, sample_tiktok_metrics):
        """测试批量处理"""
        signals = social_processor.process_batch([sample_tiktok_metrics] * 3)
        assert len(signals) == 3
        assert all(s.source == SIGNAL_SOCIAL for s in signals)

    def test_signal_id_unique(self, social_processor, sample_tiktok_metrics):
        """测试信号ID唯一性"""
        signal1 = social_processor.process(sample_tiktok_metrics)
        signal2 = social_processor.process(sample_tiktok_metrics)
        # 同样的数据应该生成相同的ID（因为 collected_at 相同）
        assert signal1.id == signal2.id

    def test_from_tiktok_helper(self):
        """测试便捷构造方法"""
        metrics = SocialSignalProcessor.from_tiktok(
            content_id="v1",
            creator_id="c1",
            likes=100,
            views=1000,
        )
        assert metrics.platform == "tiktok_douyin"

    def test_from_xiaohongshu_helper(self):
        """测试小红书构造方法"""
        metrics = SocialSignalProcessor.from_xiaohongshu(
            content_id="n1",
            creator_id="c1",
            likes=50,
            views=500,
        )
        assert metrics.platform == "instagram_xiaohongshu"

    def test_from_x_helper(self):
        """测试X构造方法"""
        metrics = SocialSignalProcessor.from_x(
            content_id="t1",
            creator_id="c1",
            likes=200,
            views=2000,
        )
        assert metrics.platform == "x_twitter"

    def test_zero_views(self, social_processor):
        """测试零浏览量"""
        metrics = SocialMediaMetrics(
            platform="tiktok_douyin",
            content_id="v1",
            creator_id="c1",
            likes=10,
            views=0,
        )
        signal = social_processor.process(metrics)
        # views=0 时使用 max(views,1)=1 避免除零，engagement_rate=10/1=10
        assert signal.features["engagement_rate"] == 10.0

    def test_to_dict(self, social_processor, sample_tiktok_metrics):
        """测试序列化"""
        signal = social_processor.process(sample_tiktok_metrics)
        d = signal.to_dict()
        assert d["source"] == SIGNAL_SOCIAL
        assert d["platform"] == "tiktok_douyin"
        assert "features" in d


# ============================================================
# Test: CommercialSignalProcessor
# ============================================================

class TestCommercialSignalProcessor:
    def test_process_basic(self, commercial_processor, sample_brand_opportunity):
        """测试基本处理"""
        signal = commercial_processor.process(sample_brand_opportunity)
        assert signal.source == SIGNAL_COMMERCIAL
        assert signal.creator_id == "某护肤品牌"
        assert signal.content_id == "opp_001"

    def test_brand_features(self, commercial_processor, sample_brand_opportunity):
        """测试品牌调性特征"""
        signal = commercial_processor.process(sample_brand_opportunity)
        features = signal.features

        assert features["brand_level_score"] == 1.0  # tier_1
        assert features["brand_tone_match"] > 0
        # 有两个调性匹配
        assert features["brand_tone_match"] == pytest.approx(0.4, abs=0.01)

    def test_budget_features(self, commercial_processor, sample_brand_opportunity):
        """测试预算特征"""
        signal = commercial_processor.process(sample_brand_opportunity)
        features = signal.features

        assert features["budget_level"] == 3 / 5  # medium
        assert features["budget_amount_norm"] > 0
        assert features["budget_flexibility"] > 0

    def test_kpi_features(self, commercial_processor, sample_brand_opportunity):
        """测试KPI特征"""
        signal = commercial_processor.process(sample_brand_opportunity)
        features = signal.features

        assert features["kpi_engagement"] == 1.0
        assert features["kpi_target_norm"] > 0
        assert features["kpi_difficulty"] == 0.5

    def test_audience_features(self, commercial_processor, sample_brand_opportunity):
        """测试目标人群特征"""
        signal = commercial_processor.process(sample_brand_opportunity)
        features = signal.features

        assert features["target_age_gen_z"] == 1.0
        assert features["target_female_ratio"] == 0.8
        assert features["target_city_tier"] == 0.8

    def test_content_features(self, commercial_processor, sample_brand_opportunity):
        """测试内容要求特征"""
        signal = commercial_processor.process(sample_brand_opportunity)
        features = signal.features

        assert features["req_content_short_video"] == 1.0
        assert features["content_count_norm"] == 0.3
        assert features["deadline_urgency"] > 0

    def test_embedding_generated(self, commercial_processor, sample_brand_opportunity):
        """测试嵌入向量"""
        signal = commercial_processor.process(sample_brand_opportunity)
        assert signal.embedding is not None
        assert len(signal.embedding) == 64

    def test_batch_process(self, commercial_processor, sample_brand_opportunity):
        """测试批量处理"""
        signals = commercial_processor.process_batch([sample_brand_opportunity] * 3)
        assert len(signals) == 3

    def test_low_budget(self, commercial_processor):
        """测试低预算商单"""
        opp = BrandOpportunity(
            opportunity_id="opp_002",
            brand_name="小品牌",
            brand_tone=["可爱甜美"],
            budget_range="micro",
            budget_amount_max=3000,
            kpi_type="exposure",
            target_age_range="18-24",
        )
        signal = commercial_processor.process(opp)
        assert signal.features["budget_level"] == 1 / 5
        assert signal.features["kpi_exposure"] == 1.0

    def test_high_kpi_difficulty(self, commercial_processor):
        """测试高难度KPI"""
        opp = BrandOpportunity(
            opportunity_id="opp_003",
            brand_name="品牌",
            kpi_type="direct_sales",
            kpi_target=100000,
        )
        signal = commercial_processor.process(opp)
        assert signal.features["kpi_difficulty"] == 0.9


# ============================================================
# Test: ContentSignalProcessor
# ============================================================

class TestContentSignalProcessor:
    def test_process_basic(self, content_processor, sample_content_data):
        """测试基本处理"""
        signal = content_processor.process(sample_content_data)
        assert signal.source == SIGNAL_CONTENT
        assert signal.content_id == "content_001"

    def test_topic_classification(self, content_processor, sample_content_data):
        """测试主题分类"""
        signal = content_processor.process(sample_content_data)
        features = signal.features

        assert features["topic_score"] > 0
        assert features["has_title"] == 1.0
        assert features["has_description"] == 1.0

    def test_sentiment_analysis(self, content_processor, sample_content_data):
        """测试情感分析"""
        signal = content_processor.process(sample_content_data)
        features = signal.features

        # 正面内容
        assert features["sentiment_score"] > 0
        assert features["sentiment_positive"] == 1.0

    def test_visual_style(self, content_processor, sample_content_data):
        """测试视觉风格"""
        signal = content_processor.process(sample_content_data)
        features = signal.features

        assert features["has_image"] == 1.0
        assert features["has_video"] == 1.0
        assert features["image_count_norm"] > 0
        assert features["visual_style_score"] > 0

    def test_text_tone(self, content_processor, sample_content_data):
        """测试文案调性"""
        signal = content_processor.process(sample_content_data)
        features = signal.features

        assert features["text_tone_score"] > 0
        assert features["text_length"] > 0

    def test_quality_assessment(self, content_processor, sample_content_data):
        """测试质量评估"""
        signal = content_processor.process(sample_content_data)
        features = signal.features

        assert features["quality_score"] > 0
        assert features["engagement_rate"] > 0
        assert features["save_rate"] > 0

    def test_embedding_generated(self, content_processor, sample_content_data):
        """测试嵌入向量"""
        signal = content_processor.process(sample_content_data)
        assert signal.embedding is not None
        assert len(signal.embedding) == 64

    def test_negative_sentiment(self, content_processor):
        """测试负面情感"""
        content = ContentData(
            content_id="c1",
            creator_id="cr1",
            title="踩雷！这款产品太差了",
            full_text="真的很难用，非常失望，浪费钱，后悔买了",
            likes=10,
            views=5000,
        )
        signal = content_processor.process(content)
        assert signal.features["sentiment_score"] < 0
        assert signal.features["sentiment_negative"] == 1.0

    def test_humor_detection(self, content_processor):
        """测试幽默调性检测"""
        content = ContentData(
            content_id="c1",
            creator_id="cr1",
            title="笑死！这个产品离谱了",
            full_text="哈哈哈这个东西搞笑绝了",
        )
        signal = content_processor.process(content)
        # 应该检测到幽默调性
        assert signal.features.get("text_tone_score", 0) > 0

    def test_batch_process(self, content_processor, sample_content_data):
        """测试批量处理"""
        signals = content_processor.process_batch([sample_content_data] * 3)
        assert len(signals) == 3

    def test_empty_content(self, content_processor):
        """测试空内容"""
        content = ContentData(
            content_id="c1",
            creator_id="cr1",
        )
        signal = content_processor.process(content)
        assert signal.features["quality_score"] >= 0


# ============================================================
# Test: 跨处理器集成
# ============================================================

class TestCrossProcessorIntegration:
    def test_all_processors_compatible(self):
        """测试所有处理器输出兼容"""
        social = SocialSignalProcessor()
        commercial = CommercialSignalProcessor()
        content = ContentSignalProcessor()

        s1 = social.process(SocialMediaMetrics(
            platform="tiktok_douyin", content_id="v1", creator_id="c1",
            likes=100, views=1000,
        ))
        s2 = commercial.process(BrandOpportunity(
            opportunity_id="o1", brand_name="B1",
        ))
        s3 = content.process(ContentData(
            content_id="ct1", creator_id="c1", title="test",
        ))

        # 所有信号都有相同的接口
        for s in [s1, s2, s3]:
            assert hasattr(s, "source")
            assert hasattr(s, "features")
            assert hasattr(s, "embedding")
            assert hasattr(s, "to_dict")

        # 所有嵌入都是64维
        for s in [s1, s2, s3]:
            assert s.embedding is not None
            assert len(s.embedding) == 64

    def test_signals_can_be_stored(self):
        """测试信号可以被存储"""
        from infrastructure.tag_database import TagDatabase
        import tempfile
        import os

        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        try:
            db = TagDatabase(db_path)
            db.initialize()

            processor = SocialSignalProcessor()
            signal = processor.process(SocialMediaMetrics(
                platform="tiktok_douyin", content_id="v1", creator_id="c1",
                likes=100, views=1000,
            ))

            db.upsert_signal(signal)
            retrieved = db.get_signal(signal.id)
            assert retrieved is not None
            assert retrieved.source == SIGNAL_SOCIAL

            db.close()
        finally:
            os.unlink(db_path)
