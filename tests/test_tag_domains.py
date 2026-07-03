"""
Phase 2 单元测试: 三级标签域系统
验证货品/人群/平台三个域的初始化、CRUD、层级查询
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2, LEVEL_CATEGORY_3,
    ASSOC_HIERARCHICAL,
)
from core.tag_graph import TagAssociationGraph
from intelligence.tag_system.commodity_domain import (
    CommodityDomain, CATEGORY_TREE,
    PRICE_BAND_TAGS, STYLE_TAGS, SCENARIO_TAGS, EFFICACY_TAGS,
)
from intelligence.tag_system.crowd_domain import (
    CrowdDomain,
    DEMOGRAPHICS_TAGS, INTEREST_PRIMARY_TAGS, INTEREST_SECONDARY_TAGS,
    BEHAVIOR_FREQ_TAGS, BEHAVIOR_DECISION_TAGS, BEHAVIOR_CHANNEL_TAGS,
    CONTENT_FORMAT_TAGS, CONTENT_TONE_TAGS, CONTENT_INTERACTION_TAGS,
    LIFECYCLE_TAGS,
)
from intelligence.tag_system.platform_domain import (
    PlatformDomain, PLATFORM_DATA,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def graph():
    return TagAssociationGraph()


@pytest.fixture
def commodity(graph):
    domain = CommodityDomain(graph)
    domain.initialize()
    return domain


@pytest.fixture
def crowd(graph):
    domain = CrowdDomain(graph)
    domain.initialize()
    return domain


@pytest.fixture
def platform(graph):
    domain = PlatformDomain(graph)
    domain.initialize()
    return domain


# ============================================================
# Test: 货品域 (CommodityDomain)
# ============================================================

class TestCommodityDomain:
    def test_initialize_count(self, commodity):
        """测试初始化后标签数量"""
        stats = commodity.count()
        assert stats["category"] > 0
        assert stats["price_band"] == len(PRICE_BAND_TAGS)
        assert stats["style"] == len(STYLE_TAGS)
        assert stats["scenario"] == len(SCENARIO_TAGS)
        assert stats["efficacy"] == len(EFFICACY_TAGS)
        assert stats["total"] > 50  # 品类树 + 辅助标签

    def test_l1_categories(self, commodity):
        """测试一级类目查询"""
        l1 = commodity.get_l1_categories()
        l1_names = {t.name for t in l1}
        # 品类树的一级类目
        for cat_name in CATEGORY_TREE.keys():
            assert cat_name in l1_names

    def test_l2_categories(self, commodity):
        """测试二级类目查询"""
        l2 = commodity.get_l2_categories("美妆护肤")
        l2_names = {t.name for t in l2}
        assert "护肤" in l2_names
        assert "彩妆" in l2_names
        assert "香水" in l2_names

    def test_l3_categories(self, commodity):
        """测试三级类目查询"""
        l3 = commodity.get_l3_categories("美妆护肤", "护肤")
        l3_names = {t.name for t in l3}
        assert "精华" in l3_names
        assert "面霜" in l3_names
        assert "防晒" in l3_names

    def test_category_path(self, commodity):
        """测试品类路径"""
        path = commodity.get_category_path("精华")
        assert path == ["美妆护肤", "护肤", "精华"]

    def test_category_tree_unchanged(self, commodity):
        """测试品类树未被修改"""
        tree = commodity.get_category_tree()
        assert len(tree) == len(CATEGORY_TREE)

    def test_search_categories(self, commodity):
        """测试品类搜索"""
        results = commodity.search_categories("精华")
        assert len(results) >= 1
        assert any(r.name == "精华" for r in results)

    def test_price_band_tags(self, commodity):
        """测试价格带标签"""
        tags = commodity.get_price_band_tags()
        names = {t.name for t in tags}
        assert "平价" in names
        assert "中端" in names
        assert "轻奢" in names
        assert "高端" in names

    def test_style_tags(self, commodity):
        """测试风格标签"""
        tags = commodity.get_style_tags()
        names = {t.name for t in tags}
        assert "极简" in names
        assert "国潮" in names

    def test_scenario_tags(self, commodity):
        """测试场景标签"""
        tags = commodity.get_scenario_tags()
        names = {t.name for t in tags}
        assert "日常通勤" in names
        assert "约会" in names

    def test_efficacy_tags(self, commodity):
        """测试功效标签"""
        tags = commodity.get_efficacy_tags()
        names = {t.name for t in tags}
        assert "美白" in names
        assert "抗老" in names
        assert "保湿" in names

    def test_add_custom_tag(self, commodity):
        """测试添加自定义标签"""
        tag_id = commodity.add_tag(
            name="防晒霜",
            name_en="sunscreen",
            sub_domain="category",
            parent_name="护肤",
            level=LEVEL_CATEGORY_3,
        )
        assert tag_id is not None
        tag = commodity.get_tag_by_name("防晒霜")
        assert tag is not None
        assert tag.parent_id is not None

    def test_get_tag_by_name(self, commodity):
        """测试按名称获取"""
        tag = commodity.get_tag_by_name("精华")
        assert tag is not None
        assert tag.name == "精华"
        assert tag.domain == DOMAIN_COMMODITY

    def test_remove_tag(self, commodity):
        """测试删除标签"""
        commodity.add_tag("测试标签", "test_tag", sub_domain="style")
        assert commodity.get_tag_by_name("测试标签") is not None
        commodity.remove_tag("测试标签")
        assert commodity.get_tag_by_name("测试标签") is None

    def test_hierarchical_associations(self, commodity):
        """测试层级联想关系自动建立"""
        # 检查品类树中的层级关系
        l1 = commodity.get_l1_categories()
        for cat in l1:
            assocs = commodity.graph.get_associations_for_tag(
                cat.id, association_type=ASSOC_HIERARCHICAL
            )
            children = commodity.graph.get_children(cat.id)
            # 每个子标签都应该有层级关联
            assert len(assocs) >= len(children)


# ============================================================
# Test: 人群域 (CrowdDomain)
# ============================================================

class TestCrowdDomain:
    def test_initialize_count(self, crowd):
        """测试初始化后标签数量"""
        stats = crowd.count()
        assert stats["demographics"] > 0
        assert stats["interest"] > 0
        assert stats["behavior"] > 0
        assert stats["content_preference"] > 0
        assert stats["lifecycle"] == len(LIFECYCLE_TAGS)
        assert stats["total"] > 30

    def test_demographics_tags(self, crowd):
        """测试人口统计标签"""
        tags = crowd.get_demographics_tags()
        names = {t.name for t in tags}
        # 年龄段
        assert "Z世代" in names
        assert "新锐白领" in names
        # 性别
        assert "女性主导" in names
        # 城市
        assert "一线城市" in names

    def test_interest_tags(self, crowd):
        """测试兴趣标签"""
        tags = crowd.get_interest_tags()
        names = {t.name for t in tags}
        assert "美妆护肤" in names
        assert "科技数码" in names
        assert "影视娱乐" in names

    def test_behavior_tags(self, crowd):
        """测试消费行为标签"""
        tags = crowd.get_behavior_tags()
        names = {t.name for t in tags}
        # 频次
        assert "高频消费" in names
        # 决策因子
        assert "成分党" in names
        # 渠道偏好
        assert "直播冲动型" in names

    def test_content_preference_tags(self, crowd):
        """测试内容偏好标签"""
        tags = crowd.get_content_preference_tags()
        names = {t.name for t in tags}
        # 形式
        assert "短视频" in names
        # 调性
        assert "专业硬核" in names
        # 互动
        assert "评论活跃" in names

    def test_lifecycle_tags(self, crowd):
        """测试生命周期标签"""
        tags = crowd.get_lifecycle_tags()
        names = {t.name for t in tags}
        assert "新客" in names
        assert "复购" in names
        assert "流失" in names

    def test_get_tags_by_sub_category(self, crowd):
        """测试按子分类查询"""
        tags = crowd.get_tags_by_sub_category("年龄段")
        names = {t.name for t in tags}
        assert "Z世代" in names
        assert "银发族" in names

    def test_search(self, crowd):
        """测试搜索"""
        results = crowd.search("Z世代")
        assert len(results) >= 1

    def test_add_custom_tag(self, crowd):
        """测试添加自定义标签"""
        tag_id = crowd.add_tag(
            name="二次元",
            name_en="anime",
            sub_domain="interest",
            parent_name="主兴趣簇",
        )
        assert tag_id is not None
        tag = crowd.get_tag_by_name("二次元")
        assert tag is not None

    def test_remove_tag(self, crowd):
        """测试删除标签"""
        crowd.add_tag("测试标签", "test", sub_domain="interest")
        assert crowd.get_tag_by_name("测试标签") is not None
        crowd.remove_tag("测试标签")
        assert crowd.get_tag_by_name("测试标签") is None

    def test_tag_metadata(self, crowd):
        """测试标签元数据正确性"""
        tag = crowd.get_tag_by_name("Z世代")
        assert tag is not None
        assert tag.metadata.get("age_min") == 18
        assert tag.metadata.get("age_max") == 24

    def test_lifecycle_metadata(self, crowd):
        """测试生命周期标签元数据"""
        tag = crowd.get_tag_by_name("沉睡")
        assert tag is not None
        assert tag.metadata.get("last_purchase_days") == "30-90"


# ============================================================
# Test: 平台域 (PlatformDomain)
# ============================================================

class TestPlatformDomain:
    def test_initialize_count(self, platform):
        """测试初始化后平台数量"""
        stats = platform.count()
        assert len(stats) == len(PLATFORM_DATA)
        for platform_key in PLATFORM_DATA:
            assert platform_key in stats
            assert stats[platform_key] > 0

    def test_get_all_platforms(self, platform):
        """测试获取所有平台"""
        platforms = platform.get_all_platforms()
        assert len(platforms) == len(PLATFORM_DATA)

    def test_get_platform(self, platform):
        """测试获取单个平台"""
        p = platform.get_platform("tiktok_douyin")
        assert p is not None
        assert p.name == "抖音/TikTok"

    def test_content_features(self, platform):
        """测试内容特征查询"""
        features = platform.get_content_features("tiktok_douyin")
        names = {f.name for f in features}
        assert "短视频15-60s" in names
        assert "强视觉冲击" in names
        assert "BGM驱动" in names

    def test_algorithm_weights(self, platform):
        """测试算法权重查询"""
        weights = platform.get_algorithm_weights("tiktok_douyin")
        assert "completion_rate" in weights
        assert weights["completion_rate"] == 0.4

    def test_audience_profile(self, platform):
        """测试人群画像查询"""
        profile = platform.get_audience_profile("tiktok_douyin")
        assert profile["age_range"] == "18-35"
        assert profile["gender_skew"] == "female_slight"

    def test_ecommerce_path(self, platform):
        """测试电商路径查询"""
        path = platform.get_ecommerce_path("tiktok_douyin")
        assert "短视频挂车" in path
        assert "直播转化" in path

    def test_xiaohongshu_features(self, platform):
        """测试小红书特征"""
        features = platform.get_content_features("instagram_xiaohongshu")
        names = {f.name for f in features}
        assert "图文笔记" in names
        assert "精美封面" in names

    def test_x_features(self, platform):
        """测试X/Twitter特征"""
        features = platform.get_content_features("x_twitter")
        names = {f.name for f in features}
        assert "观点输出" in names
        assert "话题讨论" in names

    def test_wechat_features(self, platform):
        """测试微信视频号特征"""
        features = platform.get_content_features("wechat_channels")
        names = {f.name for f in features}
        assert "私域沉淀" in names
        assert "社交裂变" in names

    def test_add_content_feature(self, platform):
        """测试添加内容特征"""
        tag_id = platform.add_content_feature(
            "tiktok_douyin",
            "知识付费",
            "paid_knowledge",
        )
        assert tag_id is not None
        features = platform.get_content_features("tiktok_douyin")
        names = {f.name for f in features}
        assert "知识付费" in names

    def test_search(self, platform):
        """测试搜索"""
        results = platform.search("直播")
        assert len(results) >= 1

    def test_remove_tag(self, platform):
        """测试删除标签"""
        platform.add_content_feature("tiktok_douyin", "测试", "test")
        assert platform.get_tag_by_name("测试") is not None
        platform.remove_tag("测试")
        assert platform.get_tag_by_name("测试") is None

    def test_get_tag_by_name(self, platform):
        """测试按名称获取"""
        tag = platform.get_tag_by_name("抖音/TikTok")
        assert tag is not None
        assert tag.domain == DOMAIN_PLATFORM

    def test_get_tag_by_key(self, platform):
        """测试按平台key获取"""
        tag = platform.get_tag_by_name("tiktok_douyin")
        assert tag is not None


# ============================================================
# Test: 跨域集成
# ============================================================

class TestCrossDomainIntegration:
    def test_all_domains_share_graph(self):
        """测试三个域共享同一个图"""
        graph = TagAssociationGraph()
        c = CommodityDomain(graph)
        cr = CrowdDomain(graph)
        p = PlatformDomain(graph)
        c.initialize()
        cr.initialize()
        p.initialize()

        stats = graph.stats()
        assert stats["total_tags"] > 100
        assert stats["tags_by_domain"][DOMAIN_COMMODITY] > 0
        assert stats["tags_by_domain"][DOMAIN_CROWD] > 0
        assert stats["tags_by_domain"][DOMAIN_PLATFORM] > 0

    def test_domain_tags_are_isolated(self):
        """测试域标签隔离"""
        graph = TagAssociationGraph()
        c = CommodityDomain(graph)
        cr = CrowdDomain(graph)
        c.initialize()
        cr.initialize()

        # 货品域标签不应出现在人群域查询中
        crowd_tags = cr.get_interest_tags()
        for tag in crowd_tags:
            assert tag.domain == DOMAIN_CROWD

        # 人群域标签不应出现在货品域查询中
        commodity_tags = c.get_l1_categories()
        for tag in commodity_tags:
            assert tag.domain == DOMAIN_COMMODITY

    def test_serialization_roundtrip(self):
        """测试序列化/反序列化"""
        graph = TagAssociationGraph()
        c = CommodityDomain(graph)
        cr = CrowdDomain(graph)
        p = PlatformDomain(graph)
        c.initialize()
        cr.initialize()
        p.initialize()

        # 序列化
        data = graph.to_dict()

        # 反序列化
        restored = TagAssociationGraph.from_dict(data)

        # 验证
        assert len(restored.tags) == len(graph.tags)
        assert len(restored.associations) == len(graph.associations)
