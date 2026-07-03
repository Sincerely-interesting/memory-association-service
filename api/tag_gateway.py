"""
标签系统API网关 (TagAPIGateway)

提供标签系统的REST API接口:
- 标签CRUD
- 联想查询（单域/双域/5C）
- 信号导入
- 质心计算
- 统计信息

可独立运行（aiohttp）或集成到现有Web服务。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.models import (
        TagNode, TagAssociation, SignalFeature,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM, ALL_DOMAINS,
    )
    from core.tag_graph import TagAssociationGraph
    from infrastructure.tag_database import TagDatabase
    from infrastructure.tag_embedding import TagEmbeddingService
    from intelligence.tag_system import (
        CommodityDomain, CrowdDomain, PlatformDomain,
        CentroidCalculator, AssociationEngine, TagQuery,
    )
    from intelligence.signal_processors import (
        SocialSignalProcessor, CommercialSignalProcessor, ContentSignalProcessor,
    )
    from intelligence.signal_processors.social_signal import SocialMediaMetrics
    from intelligence.signal_processors.commercial_signal import BrandOpportunity
    from intelligence.signal_processors.content_signal import ContentData
except ImportError:
    from models import (
        TagNode, TagAssociation, SignalFeature,
        DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM, ALL_DOMAINS,
    )
    from tag_graph import TagAssociationGraph
    from tag_database import TagDatabase
    from tag_embedding import TagEmbeddingService
    from commodity_domain import CommodityDomain
    from crowd_domain import CrowdDomain
    from platform_domain import PlatformDomain
    from centroid_calculator import CentroidCalculator
    from association_engine import AssociationEngine, TagQuery
    from social_signal import SocialSignalProcessor, SocialMediaMetrics
    from commercial_signal import CommercialSignalProcessor, BrandOpportunity
    from content_signal import ContentSignalProcessor, ContentData


class TagAPIGateway:
    """
    标签系统API网关

    封装所有标签系统功能为统一的API接口。
    每个方法对应一个HTTP端点，返回标准化的响应格式。
    """

    def __init__(
        self,
        db_path: str = "data/tag_system.db",
        auto_initialize: bool = True,
    ):
        self.db_path = db_path
        self.db = TagDatabase(db_path)
        self.graph = TagAssociationGraph()
        self.embedding_service = TagEmbeddingService(mode="rule")

        # 域管理器
        self.commodity = CommodityDomain(self.graph)
        self.crowd = CrowdDomain(self.graph)
        self.platform = PlatformDomain(self.graph)

        # 引擎
        self.centroid_calc = CentroidCalculator(self.graph, self.embedding_service)
        self.association_engine = AssociationEngine(
            self.graph, self.embedding_service, self.centroid_calc
        )

        # 信号处理器
        self.social_processor = SocialSignalProcessor()
        self.commercial_processor = CommercialSignalProcessor()
        self.content_processor = ContentSignalProcessor()

        # 配置
        self.config = {
            "tag_system_enabled": True,
            "auto_persist": True,
            "max_results": 20,
        }

        # 初始化数据库表
        self.db.initialize()

        if auto_initialize:
            self.initialize()

    def initialize(self):
        """初始化系统"""
        # 尝试从数据库加载
        try:
            loaded = self.db.load_graph()
            if loaded.tags:
                self.graph = loaded
                # 重新绑定域管理器到加载的图
                self.commodity = CommodityDomain(self.graph)
                self.crowd = CrowdDomain(self.graph)
                self.platform = PlatformDomain(self.graph)
                self.centroid_calc = CentroidCalculator(self.graph, self.embedding_service)
                self.association_engine = AssociationEngine(
                    self.graph, self.embedding_service, self.centroid_calc
                )
                logger.info(f"从数据库加载标签图: {len(self.graph.tags)} 个标签")
                return
        except Exception:
            pass

        # 初始化预置数据
        self.commodity.initialize()
        self.crowd.initialize()
        self.platform.initialize()

        # 保存到数据库
        self._persist()
        logger.info("标签系统初始化完成")

    def _persist(self):
        """持久化到数据库"""
        if self.config["auto_persist"]:
            try:
                self.db.save_graph(self.graph)
            except Exception as e:
                logger.warning(f"持久化失败: {e}")

    # ── 标签CRUD ──────────────────────────────────────────────────

    def create_tag(self, data: dict) -> dict:
        """创建标签"""
        try:
            tag_id = self.graph.add_tag(
                name=data["name"],
                name_en=data.get("name_en", data["name"]),
                domain=data.get("domain", DOMAIN_COMMODITY),
                level=data.get("level", 1),
                parent_id=data.get("parent_id"),
                description=data.get("description", ""),
                synonyms=data.get("synonyms", []),
                metadata=data.get("metadata", {}),
            )
            self._persist()
            tag = self.graph.tags.get(tag_id)
            return self._success(tag.to_dict() if tag else {"id": tag_id})
        except Exception as e:
            return self._error(str(e))

    def get_tag(self, tag_id: str) -> dict:
        """获取标签"""
        tag = self.graph.tags.get(tag_id)
        if tag:
            return self._success(tag.to_dict())
        return self._error("标签不存在", 404)

    def update_tag(self, tag_id: str, data: dict) -> dict:
        """更新标签"""
        if tag_id not in self.graph.tags:
            return self._error("标签不存在", 404)
        try:
            self.graph.update_tag(tag_id, **data)
            self._persist()
            tag = self.graph.tags.get(tag_id)
            return self._success(tag.to_dict() if tag else {"id": tag_id})
        except Exception as e:
            return self._error(str(e))

    def delete_tag(self, tag_id: str) -> dict:
        """删除标签"""
        if self.graph.remove_tag(tag_id):
            self._persist()
            return self._success({"deleted": True})
        return self._error("标签不存在", 404)

    def list_tags(
        self,
        domain: str | None = None,
        level: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """查询标签列表"""
        try:
            tags = self.db.get_tags(domain=domain, level=level, limit=limit, offset=offset)
            total = self.db.count_tags(domain=domain)
            return self._success({
                "tags": [t.to_dict() for t in tags],
                "total": total,
                "limit": limit,
                "offset": offset,
            })
        except Exception as e:
            return self._error(str(e))

    def get_children(self, tag_id: str) -> dict:
        """获取子标签"""
        children = self.graph.get_children(tag_id)
        return self._success([c.to_dict() for c in children])

    def get_path(self, tag_id: str) -> dict:
        """获取标签路径"""
        ancestors = self.graph.get_ancestors(tag_id)
        tag = self.graph.tags.get(tag_id)
        path = [a.name for a in reversed(ancestors)]
        if tag:
            path.append(tag.name)
        return self._success({"path": path})

    def batch_create_tags(self, tags_data: list[dict]) -> dict:
        """批量创建标签"""
        created = []
        for data in tags_data:
            try:
                tag_id = self.graph.add_tag(
                    name=data["name"],
                    name_en=data.get("name_en", data["name"]),
                    domain=data.get("domain", DOMAIN_COMMODITY),
                    level=data.get("level", 1),
                    parent_id=data.get("parent_id"),
                    metadata=data.get("metadata", {}),
                )
                created.append(tag_id)
            except Exception as e:
                logger.warning(f"批量创建标签失败: {e}")
        self._persist()
        return self._success({"created": len(created), "ids": created})

    # ── 联想查询 ──────────────────────────────────────────────────

    def associate_single(self, data: dict) -> dict:
        """单域联想查询"""
        try:
            query = TagQuery(
                source_domain=data.get("domain"),
                source_tag_names=data.get("tags", []),
                filters=data.get("filters", {}),
            )
            results = self.association_engine.single_domain_query(
                query, top_k=data.get("top_k", 10)
            )
            return self._success({
                "results": [r.to_dict() for r in results],
                "count": len(results),
            })
        except Exception as e:
            return self._error(str(e))

    def associate_cross(self, data: dict) -> dict:
        """双域交叉查询"""
        try:
            query = TagQuery(
                source_domain=data.get("source_domain"),
                source_tag_names=data.get("source_tags", []),
                target_domain=data.get("target_domain"),
                target_tag_names=data.get("target_tags", []),
            )
            results = self.association_engine.cross_domain_query(
                query, top_k=data.get("top_k", 10)
            )
            return self._success({
                "results": [r.to_dict() for r in results],
                "count": len(results),
            })
        except Exception as e:
            return self._error(str(e))

    def associate_5c(self, data: dict) -> dict:
        """全链路5C匹配"""
        try:
            partial_5c = {
                "content": data.get("content"),
                "channel": data.get("channel"),
                "crowd": data.get("crowd"),
                "commodity": data.get("commodity"),
            }
            results = self.association_engine.match_5c(
                partial_5c, top_k=data.get("top_k", 5)
            )
            return self._success({
                "results": [r.to_dict() for r in results],
                "count": len(results),
            })
        except Exception as e:
            return self._error(str(e))

    # ── 质心计算 ──────────────────────────────────────────────────

    def compute_centroid(self, data: dict) -> dict:
        """计算用户兴趣质心"""
        try:
            signal_ids = data.get("signal_ids", [])
            user_id = data.get("user_id")

            # 获取信号
            signals = []
            for sig_id in signal_ids:
                sig = self.graph.signals.get(sig_id) or self.db.get_signal(sig_id)
                if sig:
                    signals.append(sig)

            if not signals:
                return self._error("未找到有效信号")

            result = self.centroid_calc.compute_centroid(signals, user_id=user_id)
            return self._success({
                "vector": result.vector,
                "dimension": result.dimension,
                "signal_count": result.signal_count,
                "total_weight": result.total_weight,
            })
        except Exception as e:
            return self._error(str(e))

    def find_tags_by_centroid(self, data: dict) -> dict:
        """根据质心查找标签"""
        try:
            centroid = data.get("centroid", [])
            domain = data.get("domain")
            top_k = data.get("top_k", 10)

            results = self.centroid_calc.find_nearest_tags(
                centroid, domain=domain, top_k=top_k
            )
            return self._success({
                "results": [{"tag": t.to_dict(), "score": s} for t, s in results],
                "count": len(results),
            })
        except Exception as e:
            return self._error(str(e))

    # ── 信号导入 ──────────────────────────────────────────────────

    def import_social_signal(self, data: dict) -> dict:
        """导入社媒数据"""
        try:
            metrics = SocialMediaMetrics(
                platform=data.get("platform", ""),
                content_id=data.get("content_id", ""),
                creator_id=data.get("creator_id", ""),
                likes=data.get("likes", 0),
                comments=data.get("comments", 0),
                shares=data.get("shares", 0),
                saves=data.get("saves", 0),
                views=data.get("views", 0),
                completion_rate=data.get("completion_rate", 0.0),
                content_type=data.get("content_type", "short_video"),
                topic=data.get("topic", ""),
                hashtags=data.get("hashtags", []),
                audience_age_range=data.get("audience_age_range", ""),
                audience_gender_ratio=data.get("audience_gender_ratio", ""),
                audience_city_tier=data.get("audience_city_tier", ""),
            )
            signal = self.social_processor.process(metrics)
            self.graph.add_signal(signal)
            self._persist()
            return self._success(signal.to_dict())
        except Exception as e:
            return self._error(str(e))

    def import_commercial_signal(self, data: dict) -> dict:
        """导入商单数据"""
        try:
            opportunity = BrandOpportunity(
                opportunity_id=data.get("opportunity_id", f"opp_{int(time.time())}"),
                brand_name=data.get("brand_name", ""),
                brand_tone=data.get("brand_tone", []),
                brand_level=data.get("brand_level", ""),
                budget_range=data.get("budget_range", ""),
                budget_amount_min=data.get("budget_amount_min", 0),
                budget_amount_max=data.get("budget_amount_max", 0),
                kpi_type=data.get("kpi_type", "engagement"),
                kpi_target=data.get("kpi_target", 0),
                target_age_range=data.get("target_age_range", ""),
                target_gender=data.get("target_gender", ""),
                content_type=data.get("content_type", ""),
                platform=data.get("platform", ""),
            )
            signal = self.commercial_processor.process(opportunity)
            self.graph.add_signal(signal)
            self._persist()
            return self._success(signal.to_dict())
        except Exception as e:
            return self._error(str(e))

    def import_content_signal(self, data: dict) -> dict:
        """导入内容数据"""
        try:
            content = ContentData(
                content_id=data.get("content_id", f"ct_{int(time.time())}"),
                creator_id=data.get("creator_id", ""),
                title=data.get("title", ""),
                description=data.get("description", ""),
                full_text=data.get("full_text", ""),
                has_image=data.get("has_image", False),
                has_video=data.get("has_video", False),
                image_count=data.get("image_count", 0),
                platform=data.get("platform", ""),
                content_type=data.get("content_type", ""),
                topic=data.get("topic", ""),
                tags=data.get("tags", []),
                likes=data.get("likes", 0),
                comments=data.get("comments", 0),
                shares=data.get("shares", 0),
                saves=data.get("saves", 0),
                views=data.get("views", 0),
            )
            signal = self.content_processor.process(content)
            self.graph.add_signal(signal)
            self._persist()
            return self._success(signal.to_dict())
        except Exception as e:
            return self._error(str(e))

    # ── 统计信息 ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """获取系统统计"""
        try:
            graph_stats = self.graph.stats()
            db_stats = self.db.get_stats()
            return self._success({
                "graph": graph_stats,
                "database": db_stats,
                "config": self.config,
            })
        except Exception as e:
            return self._error(str(e))

    def get_domain_stats(self) -> dict:
        """获取各域统计"""
        return self._success({
            "commodity": self.commodity.count(),
            "crowd": self.crowd.count(),
            "platform": self.platform.count(),
        })

    # ── 配置管理 ──────────────────────────────────────────────────

    def get_config(self) -> dict:
        """获取配置"""
        return self._success(self.config)

    def update_config(self, data: dict) -> dict:
        """更新配置"""
        for key, value in data.items():
            if key in self.config:
                self.config[key] = value
        return self._success(self.config)

    # ── 资源管理 ──────────────────────────────────────────────────

    def close(self):
        """关闭资源"""
        self._persist()
        self.db.close()

    # ── 响应格式 ──────────────────────────────────────────────────

    @staticmethod
    def _success(data: Any) -> dict:
        """成功响应"""
        return {"ok": True, "data": data}

    @staticmethod
    def _error(message: str, code: int = 500) -> dict:
        """错误响应"""
        return {"ok": False, "error": message, "code": code}


# ── aiohttp 路由（可选） ─────────────────────────────────────────

def create_tag_routes(gateway: TagAPIGateway) -> list:
    """
    创建 aiohttp 路由

    使用方法:
        from aiohttp import web
        routes = create_tag_routes(gateway)
        app = web.Application()
        app.router.add_routes(routes)
    """
    try:
        from aiohttp import web
    except ImportError:
        logger.warning("aiohttp 未安装，无法创建HTTP路由")
        return []

    routes = web.RouteTableDef()

    # 标签CRUD
    @routes.post("/api/tags")
    async def create_tag(request):
        data = await request.json()
        return web.json_response(gateway.create_tag(data))

    @routes.get("/api/tags/{tag_id}")
    async def get_tag(request):
        tag_id = request.match_info["tag_id"]
        return web.json_response(gateway.get_tag(tag_id))

    @routes.put("/api/tags/{tag_id}")
    async def update_tag(request):
        tag_id = request.match_info["tag_id"]
        data = await request.json()
        return web.json_response(gateway.update_tag(tag_id, data))

    @routes.delete("/api/tags/{tag_id}")
    async def delete_tag(request):
        tag_id = request.match_info["tag_id"]
        return web.json_response(gateway.delete_tag(tag_id))

    @routes.get("/api/tags")
    async def list_tags(request):
        domain = request.query.get("domain")
        limit = int(request.query.get("limit", 100))
        offset = int(request.query.get("offset", 0))
        return web.json_response(gateway.list_tags(domain=domain, limit=limit, offset=offset))

    @routes.get("/api/tags/{tag_id}/children")
    async def get_children(request):
        tag_id = request.match_info["tag_id"]
        return web.json_response(gateway.get_children(tag_id))

    @routes.get("/api/tags/{tag_id}/path")
    async def get_path(request):
        tag_id = request.match_info["tag_id"]
        return web.json_response(gateway.get_path(tag_id))

    @routes.post("/api/tags/batch")
    async def batch_create(request):
        data = await request.json()
        return web.json_response(gateway.batch_create_tags(data))

    # 联想查询
    @routes.post("/api/associate/single")
    async def associate_single(request):
        data = await request.json()
        return web.json_response(gateway.associate_single(data))

    @routes.post("/api/associate/cross")
    async def associate_cross(request):
        data = await request.json()
        return web.json_response(gateway.associate_cross(data))

    @routes.post("/api/associate/full-5c")
    async def associate_5c(request):
        data = await request.json()
        return web.json_response(gateway.associate_5c(data))

    # 质心计算
    @routes.post("/api/centroid/compute")
    async def compute_centroid(request):
        data = await request.json()
        return web.json_response(gateway.compute_centroid(data))

    @routes.post("/api/centroid/find-tags")
    async def find_tags(request):
        data = await request.json()
        return web.json_response(gateway.find_tags_by_centroid(data))

    # 信号导入
    @routes.post("/api/signals/social")
    async def import_social(request):
        data = await request.json()
        return web.json_response(gateway.import_social_signal(data))

    @routes.post("/api/signals/commercial")
    async def import_commercial(request):
        data = await request.json()
        return web.json_response(gateway.import_commercial_signal(data))

    @routes.post("/api/signals/content")
    async def import_content(request):
        data = await request.json()
        return web.json_response(gateway.import_content_signal(data))

    # 统计
    @routes.get("/api/stats")
    async def get_stats(request):
        return web.json_response(gateway.get_stats())

    @routes.get("/api/stats/domains")
    async def get_domain_stats(request):
        return web.json_response(gateway.get_domain_stats())

    # 配置
    @routes.get("/api/config")
    async def get_config(request):
        return web.json_response(gateway.get_config())

    @routes.put("/api/config")
    async def update_config(request):
        data = await request.json()
        return web.json_response(gateway.update_config(data))

    return routes
