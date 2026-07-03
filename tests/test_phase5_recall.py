"""
Phase 5 单元测试: 召回引擎扩展
验证标签联想召回、跨域联想召回、与现有召回框架的集成
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    TagNode, TagAssociation, CrossDomainLink,
    DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2,
    ASSOC_SEMANTIC, ASSOC_CROSS_DOMAIN,
    RULE_BOOST,
)
from core.tag_graph import TagAssociationGraph
from infrastructure.tag_embedding import TagEmbeddingService
from intelligence.recall.tag_association_recall import (
    TagAssociationRecall, TagRecallResult,
)
from intelligence.recall.cross_domain_recall import (
    CrossDomainRecall, CrossDomainResult,
)
from intelligence.tag_system.commodity_domain import CommodityDomain
from intelligence.tag_system.crowd_domain import CrowdDomain
from intelligence.tag_system.platform_domain import PlatformDomain


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def graph():
    return TagAssociationGraph()


@pytest.fixture
def embedding_service():
    return TagEmbeddingService(mode="rule")


@pytest.fixture
def populated_graph():
    """填充完整数据的图"""
    graph = TagAssociationGraph()
    c = CommodityDomain(graph)
    cr = CrowdDomain(graph)
    p = PlatformDomain(graph)
    c.initialize()
    cr.initialize()
    p.initialize()

    # 添加跨域关联
    # 货品→人群
    comm_tags = [t for t in graph.tags.values() if t.domain == DOMAIN_COMMODITY and t.level == 1]
    crowd_tags = [t for t in graph.tags.values() if t.domain == DOMAIN_CROWD and t.level == 1]

    if comm_tags and crowd_tags:
        graph.add_association(
            comm_tags[0].id, crowd_tags[0].id,
            strength=0.8, association_type=ASSOC_CROSS_DOMAIN,
        )

    # 人群→平台
    platform_tags = [t for t in graph.tags.values() if t.domain == DOMAIN_PLATFORM and t.level == 1]
    if crowd_tags and platform_tags:
        graph.add_association(
            crowd_tags[0].id, platform_tags[0].id,
            strength=0.7, association_type=ASSOC_CROSS_DOMAIN,
        )

    return graph


@pytest.fixture
def tag_recall(populated_graph, embedding_service):
    return TagAssociationRecall(populated_graph, embedding_service)


@pytest.fixture
def cross_domain_recall(populated_graph, embedding_service):
    return CrossDomainRecall(populated_graph, embedding_service)


# ============================================================
# Test: TagAssociationRecall
# ============================================================

class TestTagAssociationRecall:
    def test_recall_basic(self, tag_recall):
        """测试基本标签联想召回"""
        results = tag_recall.recall("美妆护肤", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, TagRecallResult) for r in results)

    def test_recall_by_keyword(self, tag_recall):
        """测试关键词召回"""
        results = tag_recall.recall("精华", top_k=5)
        # 应该找到精华相关的标签
        assert len(results) > 0

    def test_recall_empty_query(self, tag_recall):
        """测试空查询"""
        results = tag_recall.recall("", top_k=5)
        assert len(results) == 0

    def test_recall_with_domain_filter(self, tag_recall):
        """测试域过滤"""
        results = tag_recall.recall("美妆护肤", top_k=5, domains=[DOMAIN_COMMODITY])
        for r in results:
            assert r.tag.domain == DOMAIN_COMMODITY

    def test_recall_score_order(self, tag_recall):
        """测试结果按得分排序"""
        results = tag_recall.recall("护肤", top_k=10)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_direct_match(self, tag_recall, populated_graph):
        """测试直接匹配"""
        # 找一个确切存在的标签
        exact_tag = None
        for tag in populated_graph.tags.values():
            if tag.level == 1 and tag.domain == DOMAIN_COMMODITY:
                exact_tag = tag
                break

        if exact_tag:
            results = tag_recall.recall(exact_tag.name, top_k=5)
            assert any(r.tag.name == exact_tag.name for r in results)

    def test_graph_walk联想(self, tag_recall, populated_graph):
        """测试图遍历联想"""
        # 找一个有邻接关系的标签
        src_tag = None
        for tag in populated_graph.tags.values():
            neighbors = populated_graph.get_neighbors(tag.id)
            if neighbors:
                src_tag = tag
                break

        if src_tag:
            results = tag_recall.recall(src_tag.name, top_k=5)
            # 应该通过图遍历找到邻接标签
            assert len(results) >= 1

    def test_recall_path(self, tag_recall, populated_graph):
        """测试联想路径"""
        # 找两个有连接的标签
        tags = list(populated_graph.tags.values())
        for tag in tags:
            neighbors = populated_graph.get_neighbors(tag.id)
            if neighbors:
                neighbor = populated_graph.tags.get(neighbors[0][0])
                if neighbor:
                    path = tag_recall.get联想路径(tag, neighbor)
                    assert path is not None
                    assert len(path) == 2
                    break

    def test_recall_no_path(self, tag_recall, populated_graph):
        """测试无路径情况"""
    def test_recall_no_path(self, tag_recall, populated_graph):
        """测试无路径情况"""
        tags = list(populated_graph.tags.values())
        if len(tags) >= 2:
            # 创建两个孤立节点
            t1 = TagNode(id="isolated_1", name="isolated1", name_en="i1", domain=DOMAIN_COMMODITY)
            t2 = TagNode(id="isolated_2", name="isolated2", name_en="i2", domain=DOMAIN_CROWD)
            path = tag_recall.get联想路径(t1, t2)
            assert path is None

    def test_max_results(self, tag_recall):
        """测试结果数量限制"""
        results = tag_recall.recall("护肤", top_k=3)
        assert len(results) <= 3


# ============================================================
# Test: CrossDomainRecall
# ============================================================

class TestCrossDomainRecall:
    def test_recall_basic(self, cross_domain_recall, populated_graph):
        """测试基本跨域召回"""
        # 找一个货品域标签
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        if comm_tags:
            results = cross_domain_recall.recall(
                DOMAIN_COMMODITY, [comm_tags[0].name],
                DOMAIN_CROWD, top_k=5,
            )
            assert isinstance(results, list)

    def test_commodity_to_crowd(self, cross_domain_recall, populated_graph):
        """测试货品→人群跨域"""
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        if comm_tags:
            results = cross_domain_recall.commodity_to_crowd(
                [comm_tags[0].name], top_k=5,
            )
            assert isinstance(results, list)
            for r in results:
                assert r.domain == DOMAIN_CROWD

    def test_crowd_to_platform(self, cross_domain_recall, populated_graph):
        """测试人群→平台跨域"""
        crowd_tags = [t for t in populated_graph.tags.values()
                      if t.domain == DOMAIN_CROWD and t.level == 1]
        if crowd_tags:
            results = cross_domain_recall.crowd_to_platform(
                [crowd_tags[0].name], top_k=5,
            )
            assert isinstance(results, list)
            for r in results:
                assert r.domain == DOMAIN_PLATFORM

    def test_commodity_to_platform(self, cross_domain_recall, populated_graph):
        """测试货品→平台跨域"""
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        if comm_tags:
            results = cross_domain_recall.commodity_to_platform(
                [comm_tags[0].name], top_k=5,
            )
            assert isinstance(results, list)

    def test_full_5c联想(self, cross_domain_recall, populated_graph):
        """测试全链路5C联想"""
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        crowd_tags = [t for t in populated_graph.tags.values()
                      if t.domain == DOMAIN_CROWD and t.level == 1]

        if comm_tags and crowd_tags:
            results = cross_domain_recall.full_5c联想(
                [comm_tags[0].name],
                [crowd_tags[0].name],
                top_k=3,
            )
            assert "commodity_to_crowd" in results
            assert "commodity_to_platform" in results

    def test_score_order(self, cross_domain_recall, populated_graph):
        """测试结果按得分排序"""
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        if comm_tags:
            results = cross_domain_recall.commodity_to_crowd(
                [comm_tags[0].name], top_k=10,
            )
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].score >= results[i + 1].score

    def test_empty_source_tags(self, cross_domain_recall):
        """测试空源标签"""
        results = cross_domain_recall.recall(
            DOMAIN_COMMODITY, [], DOMAIN_CROWD,
        )
        assert len(results) == 0


# ============================================================
# Test: 与现有召回框架集成
# ============================================================

class TestRecallIntegration:
    def test_strategy_weights_sum_to_one(self):
        """测试策略权重总和为1"""
        # 模拟 EnhancedMemoryRecall 的权重配置
        weights = {
            "semantic": 0.40,
            "keyword": 0.15,
            "associative": 0.15,
            "temporal": 0.03,
            "strength": 0.02,
            "tag_association": 0.15,
            "cross_domain": 0.10,
        }
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_tag_recall_result_format(self, tag_recall, populated_graph):
        """测试标签召回结果格式"""
        results = tag_recall.recall("护肤", top_k=3)
        for r in results:
            assert hasattr(r, "tag")
            assert hasattr(r, "score")
            assert hasattr(r, "source")
            assert hasattr(r, "path")
            assert 0 <= r.score <= 1

    def test_cross_domain_result_format(self, cross_domain_recall, populated_graph):
        """测试跨域召回结果格式"""
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        if comm_tags:
            results = cross_domain_recall.commodity_to_crowd(
                [comm_tags[0].name], top_k=3,
            )
            for r in results:
                assert hasattr(r, "tag")
                assert hasattr(r, "domain")
                assert hasattr(r, "score")
                assert hasattr(r, "source_domain")
                assert 0 <= r.score <= 1


# ============================================================
# Test: 性能验证
# ============================================================

class TestPerformance:
    def test_tag_recall_latency(self, tag_recall):
        """测试标签召回延迟"""
        import time
        start = time.time()
        results = tag_recall.recall("美妆护肤精华", top_k=10)
        latency_ms = (time.time() - start) * 1000
        # 应该在合理延迟内完成
        assert latency_ms < 100, f"标签召回延迟 {latency_ms:.1f}ms 超过 100ms"

    def test_cross_domain_latency(self, cross_domain_recall, populated_graph):
        """测试跨域召回延迟"""
        comm_tags = [t for t in populated_graph.tags.values()
                     if t.domain == DOMAIN_COMMODITY and t.level == 1]
        if comm_tags:
            import time
            start = time.time()
            results = cross_domain_recall.commodity_to_crowd(
                [comm_tags[0].name], top_k=10,
            )
            latency_ms = (time.time() - start) * 1000
            assert latency_ms < 100, f"跨域召回延迟 {latency_ms:.1f}ms 超过 100ms"
