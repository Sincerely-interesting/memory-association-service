"""
标签系统单元测试
验证 Phase 1 的核心数据模型和图操作的正确性
"""

import os
import sys
import time
import tempfile

import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    TagNode, TagAssociation, CrossDomainLink, SignalFeature,
    DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2, LEVEL_CATEGORY_3,
    ASSOC_SEMANTIC, ASSOC_COOCCURRENCE, ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
    RULE_CONSTRAINT, RULE_BOOST, RULE_FILTER,
    SIGNAL_SOCIAL, SIGNAL_COMMERCIAL, SIGNAL_CONTENT,
)
from core.tag_graph import TagAssociationGraph
from infrastructure.tag_database import TagDatabase


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def graph():
    """创建空标签图"""
    return TagAssociationGraph()


@pytest.fixture
def db_path():
    """创建临时数据库路径"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def db(db_path):
    """创建已初始化的标签数据库"""
    database = TagDatabase(db_path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def populated_graph(graph):
    """填充示例数据的标签图"""
    # 货品域
    g1 = graph.add_tag("美妆护肤", "Beauty & Skincare", DOMAIN_COMMODITY, LEVEL_CATEGORY_1)
    g2 = graph.add_tag("护肤", "Skincare", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=g1)
    g3 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY, LEVEL_CATEGORY_3, parent_id=g2)
    g4 = graph.add_tag("烟酰胺精华", "Niacinamide Serum", DOMAIN_COMMODITY, LEVEL_CATEGORY_3, parent_id=g2)
    g5 = graph.add_tag("服饰鞋包", "Fashion", DOMAIN_COMMODITY, LEVEL_CATEGORY_1)

    # 人群域
    c1 = graph.add_tag("Z世代", "Gen Z", DOMAIN_CROWD, LEVEL_CATEGORY_1)
    c2 = graph.add_tag("新锐白领", "Young Professional", DOMAIN_CROWD, LEVEL_CATEGORY_1)
    c3 = graph.add_tag("成分党", "Ingredient Focused", DOMAIN_CROWD, LEVEL_CATEGORY_2, parent_id=c1)

    # 平台域
    p1 = graph.add_tag("抖音", "TikTok/Douyin", DOMAIN_PLATFORM, LEVEL_CATEGORY_1)
    p2 = graph.add_tag("小红书", "Xiaohongshu/RED", DOMAIN_PLATFORM, LEVEL_CATEGORY_1)
    p3 = graph.add_tag("Instagram", "Instagram", DOMAIN_PLATFORM, LEVEL_CATEGORY_1)

    # 联想关系
    graph.add_association(g4, c3, strength=0.85, association_type=ASSOC_COOCCURRENCE)
    graph.add_association(g3, c1, strength=0.7, association_type=ASSOC_CROSS_DOMAIN)
    graph.add_association(g3, p2, strength=0.75, association_type=ASSOC_CROSS_DOMAIN)
    graph.add_association(c1, p1, strength=0.8, association_type=ASSOC_CROSS_DOMAIN)
    graph.add_association(p2, c2, strength=0.65, association_type=ASSOC_CROSS_DOMAIN)

    # 跨域规则
    graph.add_cross_domain_link(
        DOMAIN_CROWD, DOMAIN_PLATFORM,
        RULE_BOOST, "IF gen_z THEN tiktok BOOST 0.3"
    )

    return graph


# ============================================================
# Test: TagNode 数据模型
# ============================================================

class TestTagNode:
    def test_create_tag_basic(self):
        """测试基本标签创建"""
        tag = TagNode(
            id="tag_001",
            name="美妆护肤",
            name_en="Beauty & Skincare",
            level=LEVEL_CATEGORY_1,
            domain=DOMAIN_COMMODITY,
        )
        assert tag.id == "tag_001"
        assert tag.name == "美妆护肤"
        assert tag.name_en == "Beauty & Skincare"
        assert tag.level == 1
        assert tag.domain == DOMAIN_COMMODITY
        assert tag.parent_id is None
        assert tag.is_root is True
        assert tag.is_leaf is False
        assert tag.created_at > 0

    def test_create_tag_with_parent(self):
        """测试带父标签创建"""
        tag = TagNode(
            id="tag_002",
            name="精华",
            name_en="Serum",
            level=LEVEL_CATEGORY_3,
            domain=DOMAIN_COMMODITY,
            parent_id="tag_001",
        )
        assert tag.parent_id == "tag_001"
        assert tag.is_root is False
        assert tag.is_leaf is True

    def test_tag_synonyms_from_string(self):
        """测试同义词字符串自动拆分"""
        tag = TagNode(
            id="tag_003",
            name="精华",
            name_en="Serum",
            synonyms="面部精华,脸部精华液",
        )
        assert tag.synonyms == ["面部精华", "脸部精华液"]

    def test_tag_to_dict(self):
        """测试序列化"""
        tag = TagNode(
            id="tag_001",
            name="测试",
            name_en="Test",
            synonyms=["a", "b"],
        )
        d = tag.to_dict()
        assert d["id"] == "tag_001"
        assert d["synonyms"] == ["a", "b"]
        assert "embedding" not in d  # embedding不在to_dict中序列化

    def test_tag_with_embedding(self):
        """测试带嵌入向量的标签"""
        tag = TagNode(
            id="tag_004",
            name="test",
            name_en="test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        assert tag.embedding == [0.1, 0.2, 0.3, 0.4]


# ============================================================
# Test: TagAssociation 数据模型
# ============================================================

class TestTagAssociation:
    def test_create_association(self):
        """测试基本联想创建"""
        assoc = TagAssociation(
            id="assoc_001",
            source_tag_id="tag_001",
            target_tag_id="tag_002",
            strength=0.8,
            confidence=0.9,
            association_type=ASSOC_SEMANTIC,
        )
        assert assoc.strength == 0.8
        assert assoc.confidence == 0.9
        assert assoc.association_type == ASSOC_SEMANTIC
        assert assoc.created_at > 0

    def test_strengthen(self):
        """测试强化联想"""
        assoc = TagAssociation(
            id="assoc_001",
            source_tag_id="tag_001",
            target_tag_id="tag_002",
            strength=0.5,
        )
        old_time = assoc.last_strengthened
        time.sleep(0.01)
        assoc.strengthen(0.2)
        assert assoc.strength == pytest.approx(0.7, abs=0.01)
        assert assoc.hit_count == 1
        assert assoc.last_strengthened > old_time

    def test_strengthen_clamp(self):
        """测试强化不超过上限"""
        assoc = TagAssociation(
            id="assoc_001",
            source_tag_id="tag_001",
            target_tag_id="tag_002",
            strength=0.95,
        )
        assoc.strengthen(0.2)
        assert assoc.strength == 1.0

    def test_weaken(self):
        """测试弱化联想"""
        assoc = TagAssociation(
            id="assoc_001",
            source_tag_id="tag_001",
            target_tag_id="tag_002",
            strength=0.8,
        )
        assoc.weaken(0.3)
        assert assoc.strength == pytest.approx(0.5, abs=0.01)

    def test_weaken_clamp(self):
        """测试弱化不低于下限"""
        assoc = TagAssociation(
            id="assoc_001",
            source_tag_id="tag_001",
            target_tag_id="tag_002",
            strength=0.05,
        )
        assoc.weaken(0.3)
        assert assoc.strength == 0.0

    def test_is_cross_domain(self):
        """测试跨域判断"""
        assoc1 = TagAssociation(
            id="a1", source_tag_id="s", target_tag_id="t",
            association_type=ASSOC_CROSS_DOMAIN,
        )
        assoc2 = TagAssociation(
            id="a2", source_tag_id="s", target_tag_id="t",
            association_type=ASSOC_SEMANTIC,
        )
        assert assoc1.is_cross_domain is True
        assert assoc2.is_cross_domain is False


# ============================================================
# Test: CrossDomainLink 数据模型
# ============================================================

class TestCrossDomainLink:
    def test_create_link(self):
        """测试跨域规则创建"""
        link = CrossDomainLink(
            id="link_001",
            source_domain=DOMAIN_CROWD,
            target_domain=DOMAIN_PLATFORM,
            rule_type=RULE_BOOST,
            rule_expression="IF gen_z THEN tiktok",
            weight=0.8,
        )
        assert link.source_domain == DOMAIN_CROWD
        assert link.target_domain == DOMAIN_PLATFORM
        assert link.enabled is True
        assert link.trigger_count == 0

    def test_trigger(self):
        """测试规则触发记录"""
        link = CrossDomainLink(
            id="link_001",
            source_domain=DOMAIN_CROWD,
            target_domain=DOMAIN_PLATFORM,
            rule_type=RULE_BOOST,
        )
        link.trigger()
        link.trigger()
        assert link.trigger_count == 2
        assert link.last_triggered > 0


# ============================================================
# Test: SignalFeature 数据模型
# ============================================================

class TestSignalFeature:
    def test_create_signal(self):
        """测试信号特征创建"""
        signal = SignalFeature(
            id="sig_001",
            source=SIGNAL_SOCIAL,
            features={"engagement_rate": 0.05, "video_completion": 0.72},
            platform="tiktok_douyin",
            creator_id="creator_123",
        )
        assert signal.source == SIGNAL_SOCIAL
        assert signal.platform == "tiktok_douyin"
        assert signal.get_feature("engagement_rate") == 0.05
        assert signal.get_feature("nonexistent", default=-1) == -1

    def test_not_expired(self):
        """测试未过期信号"""
        signal = SignalFeature(
            id="sig_001", source=SIGNAL_SOCIAL, expires_at=None,
        )
        assert signal.is_expired is False

    def test_expired(self):
        """测试已过期信号"""
        signal = SignalFeature(
            id="sig_001", source=SIGNAL_SOCIAL,
            expires_at=time.time() - 10,  # 10秒前过期
        )
        assert signal.is_expired is True


# ============================================================
# Test: TagAssociationGraph 图操作
# ============================================================

class TestTagAssociationGraph:
    def test_add_tag(self, graph):
        """测试添加标签"""
        tag_id = graph.add_tag("测试", "Test", DOMAIN_COMMODITY)
        assert tag_id in graph.tags
        assert graph.tags[tag_id].name == "测试"
        assert len(graph.get_tags_by_domain(DOMAIN_COMMODITY)) == 1

    def test_add_tag_hierarchy(self, graph):
        """测试标签层级"""
        parent = graph.add_tag("护肤", "Skincare", DOMAIN_COMMODITY, LEVEL_CATEGORY_1)
        child1 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=parent)
        child2 = graph.add_tag("面霜", "Cream", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=parent)

        children = graph.get_children(parent)
        assert len(children) == 2
        assert {c.id for c in children} == {child1, child2}

    def test_get_descendants(self, graph):
        """测试获取所有后代"""
        g1 = graph.add_tag("美妆", "Beauty", DOMAIN_COMMODITY, LEVEL_CATEGORY_1)
        g2 = graph.add_tag("护肤", "Skincare", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=g1)
        g3 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY, LEVEL_CATEGORY_3, parent_id=g2)
        g4 = graph.add_tag("面霜", "Cream", DOMAIN_COMMODITY, LEVEL_CATEGORY_3, parent_id=g2)

        descendants = graph.get_descendants(g1)
        assert len(descendants) == 3

    def test_get_ancestors(self, graph):
        """测试获取祖先"""
        g1 = graph.add_tag("美妆", "Beauty", DOMAIN_COMMODITY, LEVEL_CATEGORY_1)
        g2 = graph.add_tag("护肤", "Skincare", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=g1)
        g3 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY, LEVEL_CATEGORY_3, parent_id=g2)

        ancestors = graph.get_ancestors(g3)
        assert len(ancestors) == 2
        assert ancestors[0].id == g2  # 最近的祖先
        assert ancestors[1].id == g1

    def test_get_siblings(self, graph):
        """测试获取兄弟"""
        g1 = graph.add_tag("美妆", "Beauty", DOMAIN_COMMODITY, LEVEL_CATEGORY_1)
        g2 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=g1)
        g3 = graph.add_tag("面霜", "Cream", DOMAIN_COMMODITY, LEVEL_CATEGORY_2, parent_id=g1)

        siblings = graph.get_siblings(g2)
        assert len(siblings) == 1
        assert siblings[0].id == g3

    def test_add_association(self, graph):
        """测试添加联想关系"""
        t1 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY)
        t2 = graph.add_tag("成分党", "Ingredient Focused", DOMAIN_CROWD)

        assoc_id = graph.add_association(t1, t2, strength=0.8)
        assert assoc_id is not None
        assert len(graph.associations) == 1

        # 邻接表
        neighbors = graph.get_neighbors(t1)
        assert len(neighbors) == 1
        assert neighbors[0][0] == t2

    def test_association_strengthen_on_duplicate(self, graph):
        """测试重复添加联想时自动强化"""
        t1 = graph.add_tag("A", "A", DOMAIN_COMMODITY)
        t2 = graph.add_tag("B", "B", DOMAIN_CROWD)

        graph.add_association(t1, t2, strength=0.5)
        graph.add_association(t1, t2, strength=0.5)

        assert len(graph.associations) == 1
        assert graph.associations[0].strength > 0.5

    def test_get_associations_for_tag(self, graph):
        """测试获取标签的所有联想"""
        t1 = graph.add_tag("A", "A", DOMAIN_COMMODITY)
        t2 = graph.add_tag("B", "B", DOMAIN_CROWD)
        t3 = graph.add_tag("C", "C", DOMAIN_PLATFORM)

        graph.add_association(t1, t2, strength=0.8)
        graph.add_association(t1, t3, strength=0.6)
        graph.add_association(t2, t3, strength=0.5)

        assocs = graph.get_associations_for_tag(t1)
        assert len(assocs) == 2

    def test_cross_domain_neighbors(self, graph):
        """测试跨域邻居查询"""
        t1 = graph.add_tag("精华", "Serum", DOMAIN_COMMODITY)
        t2 = graph.add_tag("Z世代", "Gen Z", DOMAIN_CROWD)
        t3 = graph.add_tag("抖音", "TikTok", DOMAIN_PLATFORM)
        t4 = graph.add_tag("面霜", "Cream", DOMAIN_COMMODITY)

        graph.add_association(t1, t2, strength=0.8)
        graph.add_association(t1, t3, strength=0.7)
        graph.add_association(t1, t4, strength=0.9)

        # 只返回人群域的跨域邻居
        crowd_neighbors = graph.get_cross_domain_neighbors(t1, DOMAIN_CROWD)
        assert len(crowd_neighbors) == 1
        assert crowd_neighbors[0][0] == t2

    def test_remove_tag(self, graph):
        """测试删除标签"""
        t1 = graph.add_tag("A", "A", DOMAIN_COMMODITY)
        t2 = graph.add_tag("B", "B", DOMAIN_CROWD)
        graph.add_association(t1, t2, strength=0.8)

        graph.remove_tag(t1)
        assert t1 not in graph.tags
        assert len(graph.associations) == 0  # 关联也被删除

    def test_remove_association(self, graph):
        """测试删除联想"""
        t1 = graph.add_tag("A", "A", DOMAIN_COMMODITY)
        t2 = graph.add_tag("B", "B", DOMAIN_CROWD)
        assoc_id = graph.add_association(t1, t2, strength=0.8)

        graph.remove_association(assoc_id)
        assert len(graph.associations) == 0
        assert len(graph.get_neighbors(t1)) == 0

    def test_cross_domain_links(self, graph):
        """测试跨域规则"""
        link_id = graph.add_cross_domain_link(
            DOMAIN_CROWD, DOMAIN_PLATFORM,
            RULE_BOOST, "IF gen_z THEN tiktok", weight=0.8,
        )
        links = graph.get_cross_domain_links(source_domain=DOMAIN_CROWD)
        assert len(links) == 1
        assert links[0].weight == 0.8

    def test_signal_management(self, graph):
        """测试信号管理"""
        sig = SignalFeature(
            id="sig_001",
            source=SIGNAL_SOCIAL,
            features={"rate": 0.05},
            platform="tiktok",
        )
        graph.add_signal(sig)

        retrieved = graph.get_signal("sig_001")
        assert retrieved is not None
        assert retrieved.features["rate"] == 0.05

        signals = graph.get_signals_by_source(source=SIGNAL_SOCIAL, platform="tiktok")
        assert len(signals) == 1

    def test_stats(self, populated_graph):
        """测试统计"""
        stats = populated_graph.stats()
        assert stats["total_tags"] == 11
        assert stats["total_associations"] == 5
        assert stats["total_cross_domain_links"] == 1
        assert stats["tags_by_domain"][DOMAIN_COMMODITY] == 5
        assert stats["tags_by_domain"][DOMAIN_CROWD] == 3
        assert stats["tags_by_domain"][DOMAIN_PLATFORM] == 3

    def test_serialization_roundtrip(self, populated_graph):
        """测试序列化/反序列化往返"""
        data = populated_graph.to_dict()
        restored = TagAssociationGraph.from_dict(data)

        assert len(restored.tags) == len(populated_graph.tags)
        assert len(restored.associations) == len(populated_graph.associations)
        assert len(restored.cross_domain_links) == len(populated_graph.cross_domain_links)

        # 验证图结构一致性
        for tag_id, tag in restored.tags.items():
            assert tag.name == populated_graph.tags[tag_id].name
            assert tag.domain == populated_graph.tags[tag_id].domain


# ============================================================
# Test: TagDatabase 持久化
# ============================================================

class TestTagDatabase:
    def test_initialize(self, db):
        """测试数据库初始化"""
        assert db is not None
        stats = db.get_stats()
        assert stats["total_tags"] == 0

    def test_upsert_and_get_tag(self, db):
        """测试标签CRUD"""
        tag = TagNode(
            id="tag_001",
            name="美妆护肤",
            name_en="Beauty",
            domain=DOMAIN_COMMODITY,
            level=LEVEL_CATEGORY_1,
        )
        db.upsert_tag(tag)
        retrieved = db.get_tag("tag_001")
        assert retrieved is not None
        assert retrieved.name == "美妆护肤"
        assert retrieved.domain == DOMAIN_COMMODITY

    def test_upsert_tag_update(self, db):
        """测试标签更新（UPSERT）"""
        tag = TagNode(
            id="tag_001", name="旧名称", name_en="Old",
            domain=DOMAIN_COMMODITY, level=LEVEL_CATEGORY_1,
        )
        db.upsert_tag(tag)

        tag.name = "新名称"
        db.upsert_tag(tag)

        retrieved = db.get_tag("tag_001")
        assert retrieved.name == "新名称"

    def test_bulk_upsert(self, db):
        """测试批量插入"""
        tags = [
            TagNode(id=f"t{i}", name=f"标签{i}", name_en=f"Tag{i}",
                    domain=DOMAIN_COMMODITY, level=LEVEL_CATEGORY_1)
            for i in range(100)
        ]
        db.upsert_tags_bulk(tags)
        assert db.count_tags() == 100

    def test_get_tags_filter(self, db):
        """测试标签过滤查询"""
        db.upsert_tag(TagNode(id="t1", name="美妆", name_en="Beauty",
                              domain=DOMAIN_COMMODITY, level=LEVEL_CATEGORY_1))
        db.upsert_tag(TagNode(id="t2", name="Z世代", name_en="Gen Z",
                              domain=DOMAIN_CROWD, level=LEVEL_CATEGORY_1))
        db.upsert_tag(TagNode(id="t3", name="护肤", name_en="Skincare",
                              domain=DOMAIN_COMMODITY, level=LEVEL_CATEGORY_2))

        # 按域过滤
        comm_tags = db.get_tags(domain=DOMAIN_COMMODITY)
        assert len(comm_tags) == 2

        # 按层级过滤
        l1_tags = db.get_tags(level=LEVEL_CATEGORY_1)
        assert len(l1_tags) == 2

    def test_delete_tag(self, db):
        """测试删除标签"""
        db.upsert_tag(TagNode(id="t1", name="A", name_en="A",
                              domain=DOMAIN_COMMODITY, level=LEVEL_CATEGORY_1))
        assert db.delete_tag("t1") is True
        assert db.get_tag("t1") is None
        assert db.delete_tag("t1") is False  # 不存在的标签

    def test_association_crud(self, db):
        """测试联想关系CRUD"""
        assoc = TagAssociation(
            id="a1",
            source_tag_id="t1",
            target_tag_id="t2",
            strength=0.8,
            confidence=0.9,
            association_type=ASSOC_SEMANTIC,
            context={"platform": "tiktok"},
        )
        db.upsert_association(assoc)

        retrieved = db.get_association("t1", "t2")
        assert retrieved is not None
        assert retrieved.strength == 0.8
        assert retrieved.context["platform"] == "tiktok"

        # 查询标签的联想
        assocs = db.get_associations_for_tag("t1")
        assert len(assocs) == 1

        # 删除
        assert db.delete_association("a1") is True

    def test_link_crud(self, db):
        """测试跨域规则CRUD"""
        link = CrossDomainLink(
            id="l1",
            source_domain=DOMAIN_CROWD,
            target_domain=DOMAIN_PLATFORM,
            rule_type=RULE_BOOST,
            weight=0.8,
        )
        db.upsert_link(link)

        links = db.get_links(source_domain=DOMAIN_CROWD)
        assert len(links) == 1
        assert links[0].weight == 0.8

    def test_signal_crud(self, db):
        """测试信号CRUD"""
        sig = SignalFeature(
            id="s1",
            source=SIGNAL_SOCIAL,
            features={"rate": 0.05},
            platform="tiktok",
        )
        db.upsert_signal(sig)

        retrieved = db.get_signal("s1")
        assert retrieved is not None
        assert retrieved.features["rate"] == 0.05

        signals = db.get_signals(platform="tiktok")
        assert len(signals) == 1

        assert db.delete_signal("s1") is True

    def test_save_load_graph(self, db, populated_graph):
        """测试图保存/加载往返"""
        # 保存
        db.save_graph(populated_graph)

        # 加载
        loaded = db.load_graph()

        # 验证
        assert len(loaded.tags) == len(populated_graph.tags)
        assert len(loaded.associations) == len(populated_graph.associations)
        assert len(loaded.cross_domain_links) == len(populated_graph.cross_domain_links)

        # 验证图结构
        for tag_id in populated_graph.tags:
            assert tag_id in loaded.tags
            assert loaded.tags[tag_id].name == populated_graph.tags[tag_id].name

        # 验证邻接表
        for tag_id in populated_graph.tags:
            orig_neighbors = set(n for n, _ in populated_graph.get_neighbors(tag_id))
            loaded_neighbors = set(n for n, _ in loaded.get_neighbors(tag_id))
            assert orig_neighbors == loaded_neighbors

    def test_persistence_across_instances(self, db_path, populated_graph):
        """测试数据在不同实例间持久化"""
        # 保存到数据库
        db1 = TagDatabase(db_path)
        db1.initialize()
        db1.save_graph(populated_graph)
        db1.close()

        # 重新加载
        db2 = TagDatabase(db_path)
        loaded = db2.load_graph()
        db2.close()

        assert len(loaded.tags) == len(populated_graph.tags)
        stats = loaded.stats()
        assert stats["total_associations"] == 5


# ============================================================
# Test: 兼容性验证
# ============================================================

class TestBackwardCompatibility:
    def test_legacy_concept_still_works(self):
        """测试旧 Concept 模型仍然可用"""
        from core.models import Concept
        c = Concept(id="c1", name="test")
        assert c.id == "c1"
        assert c.name == "test"

    def test_legacy_memory_still_works(self):
        """测试旧 Memory 模型仍然可用"""
        from core.models import Memory
        m = Memory(id="m1", concept_id="c1", content="test")
        assert m.id == "m1"
        assert m.content == "test"

    def test_legacy_connection_still_works(self):
        """测试旧 Connection 模型仍然可用"""
        from core.models import Connection
        conn = Connection(id="conn1", from_concept="c1", to_concept="c2")
        assert conn.id == "conn1"
        assert conn.from_concept == "c1"

    def test_legacy_memory_graph_still_works(self):
        """测试旧 MemoryGraph 仍然可用"""
        from core.memory_graph import MemoryGraph
        mg = MemoryGraph()
        cid = mg.add_concept("test")
        assert cid in mg.concepts
        mid = mg.add_memory("content", cid)
        assert mid in mg.memories
