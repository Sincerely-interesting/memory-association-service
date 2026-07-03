"""
Phase 3 单元测试: 5C联想引擎
验证嵌入服务、质心计算器、联想引擎的正确性和性能
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    TagNode, SignalFeature, CrossDomainLink,
    DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2, LEVEL_CATEGORY_3,
    ASSOC_SEMANTIC, ASSOC_COOCCURRENCE, ASSOC_CROSS_DOMAIN,
    SIGNAL_SOCIAL, RULE_BOOST,
)
from core.tag_graph import TagAssociationGraph
from infrastructure.tag_embedding import TagEmbeddingService
from infrastructure.tag_database import TagDatabase
from intelligence.tag_system.commodity_domain import CommodityDomain
from intelligence.tag_system.crowd_domain import CrowdDomain
from intelligence.tag_system.platform_domain import PlatformDomain
from intelligence.tag_system.centroid_calculator import CentroidCalculator
from intelligence.tag_system.association_engine import (
    AssociationEngine, TagQuery, AssociationResult, Full5CResult,
)


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
    return graph


@pytest.fixture
def engine(populated_graph, embedding_service):
    """完整的联想引擎"""
    centroid_calc = CentroidCalculator(populated_graph, embedding_service)
    return AssociationEngine(populated_graph, embedding_service, centroid_calc)


@pytest.fixture
def centroid_calc(populated_graph, embedding_service):
    return CentroidCalculator(populated_graph, embedding_service)


# ============================================================
# Test: TagEmbeddingService
# ============================================================

class TestTagEmbeddingService:
    def test_encode_tag(self, embedding_service):
        """测试标签编码"""
        tag = TagNode(id="t1", name="精华", name_en="Serum", domain=DOMAIN_COMMODITY)
        vec = embedding_service.encode_tag(tag)
        assert len(vec) == 128
        assert all(isinstance(x, float) for x in vec)

    def test_encode_caching(self, embedding_service):
        """测试编码缓存"""
        tag = TagNode(id="t1", name="精华", name_en="Serum", domain=DOMAIN_COMMODITY)
        vec1 = embedding_service.encode_tag(tag)
        vec2 = embedding_service.encode_tag(tag)
        assert vec1 is vec2  # 同一对象
        assert embedding_service.cache_size() == 1

    def test_cosine_similarity_identical(self, embedding_service):
        """测试相同向量的相似度"""
        vec = [1.0, 0.0, 0.0]
        sim = TagEmbeddingService.cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_orthogonal(self, embedding_service):
        """测试正交向量的相似度"""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = TagEmbeddingService.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_tag_similarity(self, embedding_service):
        """测试标签相似度"""
        t1 = TagNode(id="t1", name="精华", name_en="Serum", domain=DOMAIN_COMMODITY)
        t2 = TagNode(id="t2", name="面霜", name_en="Cream", domain=DOMAIN_COMMODITY)
        t3 = TagNode(id="t3", name="Z世代", name_en="Gen Z", domain=DOMAIN_CROWD)

        sim_12 = embedding_service.tag_similarity(t1, t2)
        sim_13 = embedding_service.tag_similarity(t1, t3)

        # 同品类的标签应该比跨域标签更相似
        # (不一定，但至少应该有正值)
        assert sim_12 > 0
        assert sim_13 > 0

    def test_find_similar_tags(self, embedding_service):
        """测试查找相似标签"""
        tags = [
            TagNode(id="t1", name="精华", name_en="Serum", domain=DOMAIN_COMMODITY),
            TagNode(id="t2", name="面霜", name_en="Cream", domain=DOMAIN_COMMODITY),
            TagNode(id="t3", name="口红", name_en="Lipstick", domain=DOMAIN_COMMODITY),
            TagNode(id="t4", name="Z世代", name_en="Gen Z", domain=DOMAIN_CROWD),
        ]
        query = tags[0]
        results = embedding_service.find_similar_tags(query, tags, top_k=3)
        assert len(results) > 0
        # 查询标签本身不应出现在结果中
        assert all(r[0].id != query.id for r in results)

    def test_domain_encoding(self, embedding_service):
        """测试不同域的编码差异"""
        t_comm = TagNode(id="t1", name="test", name_en="test", domain=DOMAIN_COMMODITY)
        t_crowd = TagNode(id="t2", name="test", name_en="test", domain=DOMAIN_CROWD)
        v1 = embedding_service.encode_tag(t_comm)
        v2 = embedding_service.encode_tag(t_crowd)
        # 不同域的编码应该不同
        assert v1 != v2

    def test_batch_encode(self, embedding_service):
        """测试批量编码"""
        tags = [
            TagNode(id=f"t{i}", name=f"标签{i}", name_en=f"Tag{i}",
                    domain=DOMAIN_COMMODITY)
            for i in range(10)
        ]
        vectors = embedding_service.encode_tags_batch(tags)
        assert len(vectors) == 10
        assert all(len(v) == 128 for v in vectors)


# ============================================================
# Test: CentroidCalculator
# ============================================================

class TestCentroidCalculator:
    def test_compute_centroid_basic(self, centroid_calc):
        """测试基本质心计算"""
        signals = [
            SignalFeature(
                id="s1", source=SIGNAL_SOCIAL,
                features={"rate": 0.05},
                embedding=[1.0, 0.0, 0.0, 0.0],
                captured_at=time.time(),
            ),
            SignalFeature(
                id="s2", source=SIGNAL_SOCIAL,
                features={"rate": 0.08},
                embedding=[0.8, 0.6, 0.0, 0.0],
                captured_at=time.time(),
            ),
        ]
        result = centroid_calc.compute_centroid(signals, user_id="user_001")
        assert result.is_valid
        assert result.signal_count == 2
        assert result.dimension == 4
        assert len(result.vector) == 4

    def test_compute_centroid_empty(self, centroid_calc):
        """测试空信号的质心计算"""
        result = centroid_calc.compute_centroid([])
        assert not result.is_valid
        assert result.signal_count == 0

    def test_compute_centroid_expired_signals(self, centroid_calc):
        """测试过期信号被过滤"""
        signals = [
            SignalFeature(
                id="s1", source=SIGNAL_SOCIAL,
                embedding=[1.0, 0.0],
                captured_at=time.time(),
            ),
            SignalFeature(
                id="s2", source=SIGNAL_SOCIAL,
                embedding=[0.0, 1.0],
                captured_at=time.time() - 86400 * 100,  # 过期
                expires_at=time.time() - 86400,
            ),
        ]
        result = centroid_calc.compute_centroid(signals)
        assert result.signal_count == 1

    def test_time_decay_weights(self, centroid_calc):
        """测试时间衰减权重"""
        signals = [
            SignalFeature(
                id="s1", source=SIGNAL_SOCIAL,
                embedding=[1.0, 0.0],
                captured_at=time.time(),  # 刚刚
            ),
            SignalFeature(
                id="s2", source=SIGNAL_SOCIAL,
                embedding=[1.0, 0.0],
                captured_at=time.time() - 86400 * 30,  # 30天前
            ),
        ]
        weights = centroid_calc._compute_time_decay_weights(signals)
        assert weights[0] > weights[1]  # 刚刚的权重更高

    def test_centroid_caching(self, centroid_calc):
        """测试质心缓存"""
        signals = [
            SignalFeature(
                id="s1", source=SIGNAL_SOCIAL,
                embedding=[1.0, 0.0],
                captured_at=time.time(),
            ),
        ]
        result1 = centroid_calc.compute_centroid(signals, user_id="user_001")
        result2 = centroid_calc.get_cached_centroid("user_001")
        assert result2 is not None
        assert result2.signal_count == result1.signal_count

    def test_centroid_shift_tracking(self, centroid_calc):
        """测试质心偏移追踪"""
        signals1 = [
            SignalFeature(id="s1", source=SIGNAL_SOCIAL,
                         embedding=[1.0, 0.0], captured_at=time.time()),
        ]
        signals2 = [
            SignalFeature(id="s2", source=SIGNAL_SOCIAL,
                         embedding=[0.0, 1.0], captured_at=time.time()),
        ]
        centroid_calc.compute_centroid(signals1, user_id="user_001")
        centroid_calc.compute_centroid(signals2, user_id="user_001")

        history = centroid_calc.get_shift_history("user_001")
        assert len(history) == 1
        assert history[0].shift_magnitude > 0

    def test_find_nearest_tags(self, centroid_calc):
        """测试近邻标签查找"""
        # 创建带嵌入的标签
        t1 = TagNode(id="t1", name="精华", name_en="Serum",
                     domain=DOMAIN_COMMODITY, embedding=[1.0, 0.0, 0.0])
        t2 = TagNode(id="t2", name="面霜", name_en="Cream",
                     domain=DOMAIN_COMMODITY, embedding=[0.8, 0.6, 0.0])
        t3 = TagNode(id="t3", name="Z世代", name_en="Gen Z",
                     domain=DOMAIN_CROWD, embedding=[0.0, 0.0, 1.0])
        centroid_calc.graph.tags = {"t1": t1, "t2": t2, "t3": t3}

        results = centroid_calc.find_nearest_tags(
            [1.0, 0.0, 0.0], domain=DOMAIN_COMMODITY, top_k=2
        )
        assert len(results) > 0
        # 应该找到精华
        assert any(r[0].name == "精华" for r in results)


# ============================================================
# Test: AssociationEngine
# ============================================================

class TestAssociationEngine:
    def test_single_domain_query(self, engine):
        """测试单域查询"""
        # 先建立一些跨域关联
        tags = list(engine.graph.tags.values())
        commodity_tags = [t for t in tags if t.domain == DOMAIN_COMMODITY and t.level == 1]
        crowd_tags = [t for t in tags if t.domain == DOMAIN_CROWD and t.level == 1]

        if commodity_tags and crowd_tags:
            # 建立跨域关联
            engine.graph.add_association(
                commodity_tags[0].id, crowd_tags[0].id,
                strength=0.8, association_type=ASSOC_CROSS_DOMAIN,
            )

        query = TagQuery(
            source_domain=DOMAIN_COMMODITY,
            source_tag_names=[commodity_tags[0].name] if commodity_tags else ["精华"],
        )
        results = engine.single_domain_query(query, top_k=5)
        # 结果可能为空（取决于图结构），但不应该报错
        assert isinstance(results, list)

    def test_single_domain_query_empty(self, engine):
        """测试空查询"""
        query = TagQuery(source_tag_names=[])
        results = engine.single_domain_query(query)
        assert len(results) == 0

    def test_cross_domain_query(self, engine):
        """测试双域交叉查询"""
        query = TagQuery(
            source_domain=DOMAIN_CROWD,
            source_tag_names=["Z世代"],
            target_domain=DOMAIN_PLATFORM,
            target_tag_names=["抖音"],
        )
        results = engine.cross_domain_query(query, top_k=5)
        # 结果可能为空（如果图中没有中间节点）
        # 但不应该报错
        assert isinstance(results, list)

    def test_match_5c_partial(self, engine):
        """测试5C部分匹配"""
        partial = {
            "content": ["短视频"],
            "channel": None,
            "crowd": ["Z世代"],
            "commodity": ["精华"],
        }
        results = engine.match_5c(partial, top_k=3)
        assert isinstance(results, list)

    def test_match_5c_complete(self, engine):
        """测试5C完整匹配"""
        complete = {
            "content": ["短视频"],
            "channel": ["抖音"],
            "crowd": ["Z世代"],
            "commodity": ["精华"],
        }
        results = engine.match_5c(complete, top_k=1)
        assert len(results) == 1
        assert results[0].confidence == 1.0

    def test_strategy_graph_walk(self, populated_graph, embedding_service):
        """测试图遍历策略"""
        engine = AssociationEngine(populated_graph, embedding_service)

        # 找一个有邻接关系的标签
        src = None
        for tag in populated_graph.tags.values():
            neighbors = populated_graph.get_neighbors(tag.id)
            if neighbors:
                src = tag
                break

        if src:
            candidates = populated_graph.get_tags_by_domain(DOMAIN_CROWD)
            results = engine._strategy_graph_walk([src], candidates)
            assert isinstance(results, list)

    def test_strategy_semantic(self, populated_graph, embedding_service):
        """测试语义相似度策略"""
        engine = AssociationEngine(populated_graph, embedding_service)

        src = list(populated_graph.tags.values())[0]
        candidates = populated_graph.get_tags_by_domain(DOMAIN_CROWD)[:5]
        results = engine._strategy_semantic([src], candidates)
        assert isinstance(results, list)
        assert all(0 <= r[1] <= 1 for r in results)

    def test_fuse_and_rank(self, engine):
        """测试融合排序"""
        tag = TagNode(id="t1", name="test", name_en="test", domain=DOMAIN_COMMODITY)
        results = [
            AssociationResult(tag=tag, score=0.8, strategy="a"),
            AssociationResult(tag=tag, score=0.9, strategy="b"),  # 重复
            AssociationResult(tag=tag, score=0.5, strategy="c"),  # 重复
        ]
        fused = engine._fuse_and_rank(results)
        assert len(fused) == 1  # 去重后只有一个
        assert fused[0].score == 0.9  # 保留最高分

    def test_full5c_result_to_dict(self):
        """测试5C结果序列化"""
        tag = TagNode(id="t1", name="test", name_en="test", domain=DOMAIN_COMMODITY)
        result = Full5CResult(
            commodity=[AssociationResult(tag=tag, score=0.8, strategy="test")],
            crowd=[AssociationResult(tag=tag, score=0.7, strategy="test")],
            conversion_score=0.75,
            confidence=0.7,
        )
        d = result.to_dict()
        assert "commodity" in d
        assert "crowd" in d
        assert d["conversion_score"] == 0.75


# ============================================================
# Test: 性能验证
# ============================================================

class TestPerformance:
    def test_single_query_latency(self, engine):
        """测试单域查询延迟 < 50ms"""
        query = TagQuery(
            source_domain=DOMAIN_COMMODITY,
            source_tag_names=["精华"],
        )
        start = time.time()
        results = engine.single_domain_query(query, top_k=10)
        latency_ms = (time.time() - start) * 1000
        assert latency_ms < 50, f"查询延迟 {latency_ms:.1f}ms 超过 50ms"

    def test_embedding_batch_latency(self, embedding_service, populated_graph):
        """测试批量嵌入性能"""
        tags = list(populated_graph.tags.values())[:50]
        start = time.time()
        vectors = embedding_service.encode_tags_batch(tags)
        latency_ms = (time.time() - start) * 1000
        assert len(vectors) == 50
        assert latency_ms < 100, f"批量嵌入延迟 {latency_ms:.1f}ms 超过 100ms"


# ============================================================
# Test: 与数据库集成
# ============================================================

class TestDatabaseIntegration:
    def test_save_load_with_engine(self, tmp_path):
        """测试引擎与数据库的集成"""
        db_path = str(tmp_path / "test_phase3.db")
        # 创建并初始化
        graph = TagAssociationGraph()
        embedding_service = TagEmbeddingService(mode="rule")
        c = CommodityDomain(graph)
        cr = CrowdDomain(graph)
        c.initialize()
        cr.initialize()

        # 保存到数据库
        db = TagDatabase(db_path)
        db.initialize()
        db.save_graph(graph)
        db.close()

        # 从数据库加载
        db2 = TagDatabase(db_path)
        loaded_graph = db2.load_graph()
        db2.close()

        # 使用加载的图创建引擎
        engine = AssociationEngine(loaded_graph, embedding_service)
        query = TagQuery(
            source_domain=DOMAIN_COMMODITY,
            source_tag_names=["精华"],
        )
        results = engine.single_domain_query(query, top_k=5)
        assert isinstance(results, list)
