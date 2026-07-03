"""
Phase 6 单元测试: API与配置
验证API端点、配置管理、命令注册的正确性
"""
from __future__ import annotations

import os
import sys
import tempfile
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM,
)
from api.tag_gateway import TagAPIGateway


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test_tag_api.db")


@pytest.fixture
def gateway(tmp_db_path):
    """创建已初始化的API网关"""
    gw = TagAPIGateway(db_path=tmp_db_path, auto_initialize=True)
    return gw


# ============================================================
# Test: TagAPIGateway 初始化
# ============================================================

class TestTagAPIGatewayInit:
    def test_initialize(self, gateway):
        """测试初始化"""
        assert gateway.graph is not None
        assert gateway.db is not None
        stats = gateway.get_stats()
        assert stats["ok"] is True
        assert stats["data"]["graph"]["total_tags"] > 0

    def test_domain_managers(self, gateway):
        """测试域管理器"""
        assert gateway.commodity is not None
        assert gateway.crowd is not None
        assert gateway.platform is not None

    def test_engines(self, gateway):
        """测试引擎"""
        assert gateway.association_engine is not None
        assert gateway.centroid_calc is not None

    def test_persistence(self, tmp_db_path):
        """测试持久化"""
        gw1 = TagAPIGateway(db_path=tmp_db_path, auto_initialize=True)
        initial_count = gw1.graph.stats()["total_tags"]
        gw1.close()

        gw2 = TagAPIGateway(db_path=tmp_db_path, auto_initialize=True)
        assert gw2.graph.stats()["total_tags"] == initial_count
        gw2.close()


# ============================================================
# Test: 标签CRUD API
# ============================================================

class TestTagCRUDAPI:
    def test_create_tag(self, gateway):
        """测试创建标签"""
        result = gateway.create_tag({
            "name": "测试标签",
            "name_en": "test_tag",
            "domain": DOMAIN_COMMODITY,
            "level": 1,
        })
        assert result["ok"] is True
        assert "id" in result["data"]

    def test_get_tag(self, gateway):
        """测试获取标签"""
        # 先创建
        create_result = gateway.create_tag({
            "name": "获取测试",
            "name_en": "get_test",
            "domain": DOMAIN_COMMODITY,
        })
        tag_id = create_result["data"]["id"]

        # 获取
        result = gateway.get_tag(tag_id)
        assert result["ok"] is True
        assert result["data"]["name"] == "获取测试"

    def test_get_tag_not_found(self, gateway):
        """测试获取不存在的标签"""
        result = gateway.get_tag("nonexistent")
        assert result["ok"] is False

    def test_update_tag(self, gateway):
        """测试更新标签"""
        create_result = gateway.create_tag({
            "name": "更新前",
            "name_en": "update_test",
            "domain": DOMAIN_COMMODITY,
        })
        tag_id = create_result["data"]["id"]

        result = gateway.update_tag(tag_id, {"name": "更新后"})
        assert result["ok"] is True

    def test_delete_tag(self, gateway):
        """测试删除标签"""
        create_result = gateway.create_tag({
            "name": "删除测试",
            "name_en": "delete_test",
            "domain": DOMAIN_COMMODITY,
        })
        tag_id = create_result["data"]["id"]

        result = gateway.delete_tag(tag_id)
        assert result["ok"] is True

        # 验证已删除
        get_result = gateway.get_tag(tag_id)
        assert get_result["ok"] is False

    def test_list_tags(self, gateway):
        """测试标签列表"""
        result = gateway.list_tags(domain=DOMAIN_COMMODITY, limit=10)
        assert result["ok"] is True
        assert "tags" in result["data"]
        assert "total" in result["data"]

    def test_get_children(self, gateway):
        """测试获取子标签"""
        # 找一个有子节点的标签
        tags = gateway.graph.get_tags_by_domain(DOMAIN_COMMODITY)
        parent = None
        for tag in tags:
            children = gateway.graph.get_children(tag.id)
            if children:
                parent = tag
                break

        if parent:
            result = gateway.get_children(parent.id)
            assert result["ok"] is True
            assert len(result["data"]) > 0

    def test_get_path(self, gateway):
        """测试获取标签路径"""
    def test_get_path(self, gateway):
        """测试获取标签路径"""
        tags = gateway.graph.get_tags_by_domain(DOMAIN_COMMODITY)
        # 找一个有父节点的标签
        child = None
        for tag in tags:
            if tag.parent_id:
                child = tag
                break

        if child:
            result = gateway.get_path(child.id)
            assert result["ok"] is True
            assert len(result["data"]["path"]) > 1

    def test_batch_create(self, gateway):
        """测试批量创建"""
        result = gateway.batch_create_tags([
            {"name": "批量1", "name_en": "batch1", "domain": DOMAIN_COMMODITY},
            {"name": "批量2", "name_en": "batch2", "domain": DOMAIN_COMMODITY},
        ])
        assert result["ok"] is True
        assert result["data"]["created"] == 2


# ============================================================
# Test: 联想查询 API
# ============================================================

class TestAssociationAPI:
    def test_associate_single(self, gateway):
        """测试单域联想"""
        result = gateway.associate_single({
            "domain": DOMAIN_COMMODITY,
            "tags": ["精华"],
            "top_k": 5,
        })
        assert result["ok"] is True
        assert "results" in result["data"]

    def test_associate_cross(self, gateway):
        """测试双域交叉"""
        result = gateway.associate_cross({
            "source_domain": DOMAIN_CROWD,
            "source_tags": ["Z世代"],
            "target_domain": DOMAIN_PLATFORM,
            "top_k": 5,
        })
        assert result["ok"] is True

    def test_associate_5c(self, gateway):
        """测试5C匹配"""
        result = gateway.associate_5c({
            "content": ["短视频"],
            "crowd": ["Z世代"],
            "commodity": ["精华"],
            "top_k": 3,
        })
        assert result["ok"] is True


# ============================================================
# Test: 质心计算 API
# ============================================================

class TestCentroidAPI:
    def test_compute_centroid(self, gateway):
        """测试质心计算"""
        # 先创建信号
        from core.models import SignalFeature, SIGNAL_SOCIAL
        sig = SignalFeature(
            id="test_sig_1",
            source=SIGNAL_SOCIAL,
            embedding=[1.0, 0.0, 0.0],
            features={"rate": 0.05},
        )
        gateway.graph.add_signal(sig)

        result = gateway.compute_centroid({
            "user_id": "test_user",
            "signal_ids": ["test_sig_1"],
        })
        assert result["ok"] is True
        assert "vector" in result["data"]

    def test_find_tags_by_centroid(self, gateway):
        """测试根据质心查找标签"""
        result = gateway.find_tags_by_centroid({
            "centroid": [1.0, 0.0, 0.0],
            "top_k": 5,
        })
        assert result["ok"] is True


# ============================================================
# Test: 信号导入 API
# ============================================================

class TestSignalImportAPI:
    def test_import_social_signal(self, gateway):
        """测试导入社媒数据"""
        result = gateway.import_social_signal({
            "platform": "tiktok_douyin",
            "content_id": "video_test",
            "creator_id": "creator_test",
            "likes": 100,
            "comments": 10,
            "shares": 5,
            "views": 5000,
            "content_type": "short_video",
            "topic": "护肤",
        })
        assert result["ok"] is True
        assert "id" in result["data"]

    def test_import_commercial_signal(self, gateway):
        """测试导入商单数据"""
        result = gateway.import_commercial_signal({
            "brand_name": "测试品牌",
            "brand_tone": ["年轻时尚"],
            "budget_range": "medium",
            "kpi_type": "engagement",
        })
        assert result["ok"] is True

    def test_import_content_signal(self, gateway):
        """测试导入内容数据"""
        result = gateway.import_content_signal({
            "content_id": "content_test",
            "creator_id": "creator_test",
            "title": "测试标题",
            "full_text": "测试内容",
            "topic": "护肤",
        })
        assert result["ok"] is True


# ============================================================
# Test: 统计与配置 API
# ============================================================

class TestStatsConfigAPI:
    def test_get_stats(self, gateway):
        """测试获取统计"""
        result = gateway.get_stats()
        assert result["ok"] is True
        assert "graph" in result["data"]
        assert "database" in result["data"]

    def test_get_domain_stats(self, gateway):
        """测试获取域统计"""
        result = gateway.get_domain_stats()
        assert result["ok"] is True
        assert "commodity" in result["data"]
        assert "crowd" in result["data"]
        assert "platform" in result["data"]

    def test_get_config(self, gateway):
        """测试获取配置"""
        result = gateway.get_config()
        assert result["ok"] is True
        assert "tag_system_enabled" in result["data"]

    def test_update_config(self, gateway):
        """测试更新配置"""
        result = gateway.update_config({"max_results": 50})
        assert result["ok"] is True
        assert result["data"]["max_results"] == 50


# ============================================================
# Test: 响应格式
# ============================================================

class TestResponseFormat:
    def test_success_format(self):
        """测试成功响应格式"""
        result = TagAPIGateway._success({"key": "value"})
        assert result["ok"] is True
        assert result["data"] == {"key": "value"}

    def test_error_format(self):
        """测试错误响应格式"""
        result = TagAPIGateway._error("错误信息", 404)
        assert result["ok"] is False
        assert result["error"] == "错误信息"
        assert result["code"] == 404


# ============================================================
# Test: 配置文件验证
# ============================================================

class TestConfigSchema:
    def test_schema_loadable(self):
        """测试配置模式可加载"""
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "_conf_schema.json"
        )
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # 验证新增的配置项
        assert "tag_system_enabled" in schema
        assert "tag_domains" in schema
        assert "association_weights" in schema
        assert "tag_embedding_mode" in schema
        assert "recall_weights" in schema
        assert "tag_system_api" in schema

    def test_tag_system_config_defaults(self):
        """测试标签系统配置默认值"""
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "_conf_schema.json"
        )
        with open(schema_path, "r") as f:
            schema = json.load(f)

        assert schema["tag_system_enabled"]["default"] is True
        assert schema["tag_embedding_mode"]["default"] == "rule"
        assert schema["tag_domains"]["items"]["commodity"]["default"] is True
