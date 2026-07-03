"""
货品域 (Commodity Domain)

管理商品相关的三级标签体系:
- 品类标签 (Category): 一级类目 → 二级类目 → 三级类目
- 价格带标签 (PriceBand): 平价/中端/轻奢/高端
- 风格标签 (Style): 极简/国潮/快时尚/...
- 场景标签 (Scenario): 日常通勤/约会/运动/...
- 功效标签 (Efficacy): 美白/抗老/保湿/...
"""
from __future__ import annotations

import time

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from core.models import (
    TagNode, TagAssociation,
    DOMAIN_COMMODITY,
    LEVEL_CATEGORY_1, LEVEL_CATEGORY_2, LEVEL_CATEGORY_3,
    ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
)
from core.tag_graph import TagAssociationGraph


# ── 预置品类树 ──────────────────────────────────────────────────

# 品类树定义: {一级类目: {二级类目: [三级类目]}}
CATEGORY_TREE: dict[str, dict[str, list[str]]] = {
    "美妆护肤": {
        "护肤": ["精华", "面霜", "防晒", "面膜", "洁面", "爽肤水"],
        "彩妆": ["口红", "粉底", "眼影", "腮红", "睫毛膏"],
        "香水": ["女士香水", "男士香水", "中性香水"],
    },
    "服饰鞋包": {
        "女装": ["连衣裙", "衬衫", "T恤", "外套", "裤装"],
        "男装": ["衬衫", "T恤", "外套", "裤装", "西装"],
        "鞋靴": ["运动鞋", "高跟鞋", "靴子", "凉鞋"],
        "箱包": ["双肩包", "手提包", "斜挎包", "钱包"],
    },
    "食品饮料": {
        "零食": ["坚果", "饼干", "糖果", "膨化食品"],
        "饮品": ["咖啡", "茶饮", "果汁", "乳制品"],
        "生鲜": ["水果", "蔬菜", "肉类", "海鲜"],
    },
    "3C数码": {
        "手机": ["旗舰手机", "中端手机", "折叠屏"],
        "电脑": ["笔记本", "台式机", "平板"],
        "配件": ["耳机", "充电器", "手机壳", "数据线"],
    },
    "家居生活": {
        "家具": ["沙发", "床", "桌子", "椅子"],
        "家纺": ["床品", "毛巾", "窗帘"],
        "厨具": ["锅具", "餐具", "厨房小工具"],
    },
    "母婴亲子": {
        "奶粉": ["婴儿奶粉", "儿童奶粉"],
        "纸尿裤": ["新生儿", "L码", "XL码"],
        "童装": ["婴儿服", "儿童服"],
    },
    "运动户外": {
        "运动装备": ["瑜伽", "跑步", "健身"],
        "户外装备": ["登山", "露营", "骑行"],
    },
    "宠物用品": {
        "猫粮": ["幼猫粮", "成猫粮", "处方粮"],
        "狗粮": ["幼犬粮", "成犬粮"],
        "宠物玩具": ["猫玩具", "狗玩具"],
    },
}

# ── 预置辅助标签 ────────────────────────────────────────────────

PRICE_BAND_TAGS: list[tuple[str, str, dict]] = [
    ("平价", "budget", {"range_min": 0, "range_max": 99, "currency": "CNY"}),
    ("中端", "mid_range", {"range_min": 100, "range_max": 499, "currency": "CNY"}),
    ("轻奢", "affordable_luxury", {"range_min": 500, "range_max": 1999, "currency": "CNY"}),
    ("高端", "premium", {"range_min": 2000, "range_max": 99999, "currency": "CNY"}),
]

STYLE_TAGS: list[tuple[str, str]] = [
    ("极简", "minimalist"),
    ("国潮", "guochao"),
    ("小众设计师", "indie_designer"),
    ("快时尚", "fast_fashion"),
    ("复古", "vintage"),
    ("甜美", "sweet"),
    ("酷飒", "cool"),
    ("运动休闲", "athleisure"),
]

SCENARIO_TAGS: list[tuple[str, str]] = [
    ("日常通勤", "daily_commute"),
    ("约会", "date_night"),
    ("运动健身", "workout"),
    ("居家办公", "home_office"),
    ("旅行", "travel"),
    ("聚会", "party"),
    ("送礼", "gift"),
    ("开学季", "back_to_school"),
]

EFFICACY_TAGS: list[tuple[str, str]] = [
    ("美白", "brightening"),
    ("抗老", "anti_aging"),
    ("保湿", "hydrating"),
    ("修护", "repairing"),
    ("控油", "oil_control"),
    ("祛痘", "acne_treatment"),
    ("舒缓", "soothing"),
    ("紧致", "firming"),
    ("抗氧化", "antioxidant"),
]


class CommodityDomain:
    """
    货品域标签管理器

    提供品类树管理和辅助标签（价格带/风格/场景/功效）的CRUD。
    所有标签通过 TagAssociationGraph 存储，支持层级查询。
    """

    def __init__(self, graph: TagAssociationGraph):
        self.graph = graph

        # 内部索引: tag_name -> tag_id (加速查询)
        self._name_index: dict[str, str] = {}

        # 子域索引: sub_domain -> {tag_name: tag_id}
        self._sub_domains: dict[str, dict[str, str]] = {
            "price_band": {},
            "style": {},
            "scenario": {},
            "efficacy": {},
        }

    # ── 初始化 ────────────────────────────────────────────────────

    def initialize(self):
        """初始化预置标签数据"""
        self._init_category_tree()
        self._init_price_band_tags()
        self._init_style_tags()
        self._init_scenario_tags()
        self._init_efficacy_tags()
        logger.info(
            f"货品域初始化完成: 品类树{len(CATEGORY_TREE)}个一级类目, "
            f"价格带{len(PRICE_BAND_TAGS)}个, 风格{len(STYLE_TAGS)}个, "
            f"场景{len(SCENARIO_TAGS)}个, 功效{len(EFFICACY_TAGS)}个"
        )

    def _init_category_tree(self):
        """初始化品类树"""
        for l1_name, l2_dict in CATEGORY_TREE.items():
            l1_id = self.graph.add_tag(
                name=l1_name,
                name_en=l1_name,
                domain=DOMAIN_COMMODITY,
                level=LEVEL_CATEGORY_1,
                description=f"一级类目: {l1_name}",
                metadata={"sub_domain": "category"},
            )
            self._name_index[l1_name] = l1_id

            for l2_name, l3_list in l2_dict.items():
                l2_id = self.graph.add_tag(
                    name=l2_name,
                    name_en=l2_name,
                    domain=DOMAIN_COMMODITY,
                    level=LEVEL_CATEGORY_2,
                    parent_id=l1_id,
                    description=f"二级类目: {l2_name}",
                    metadata={"sub_domain": "category"},
                )
                self._name_index[l2_name] = l2_id

                # 自动建立层级关联
                self.graph.add_association(
                    l1_id, l2_id,
                    strength=0.9,
                    association_type=ASSOC_HIERARCHICAL,
                )

                for l3_name in l3_list:
                    l3_id = self.graph.add_tag(
                        name=l3_name,
                        name_en=l3_name,
                        domain=DOMAIN_COMMODITY,
                        level=LEVEL_CATEGORY_3,
                        parent_id=l2_id,
                        description=f"三级类目: {l3_name}",
                        metadata={"sub_domain": "category"},
                    )
                    self._name_index[l3_name] = l3_id

                    self.graph.add_association(
                        l2_id, l3_id,
                        strength=0.9,
                        association_type=ASSOC_HIERARCHICAL,
                    )

    def _init_price_band_tags(self):
        """初始化价格带标签"""
        for name, name_en, meta in PRICE_BAND_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_COMMODITY,
                level=LEVEL_CATEGORY_1,
                description=f"价格带: {name}",
                metadata={"sub_domain": "price_band", **meta},
            )
            self._name_index[name] = tag_id
            self._sub_domains["price_band"][name] = tag_id

    def _init_style_tags(self):
        """初始化风格标签"""
        for name, name_en in STYLE_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_COMMODITY,
                level=LEVEL_CATEGORY_1,
                description=f"风格: {name}",
                metadata={"sub_domain": "style"},
            )
            self._name_index[name] = tag_id
            self._sub_domains["style"][name] = tag_id

    def _init_scenario_tags(self):
        """初始化场景标签"""
        for name, name_en in SCENARIO_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_COMMODITY,
                level=LEVEL_CATEGORY_1,
                description=f"场景: {name}",
                metadata={"sub_domain": "scenario"},
            )
            self._name_index[name] = tag_id
            self._sub_domains["scenario"][name] = tag_id

    def _init_efficacy_tags(self):
        """初始化功效标签"""
        for name, name_en in EFFICACY_TAGS:
            tag_id = self.graph.add_tag(
                name=name,
                name_en=name_en,
                domain=DOMAIN_COMMODITY,
                level=LEVEL_CATEGORY_1,
                description=f"功效: {name}",
                metadata={"sub_domain": "efficacy"},
            )
            self._name_index[name] = tag_id
            self._sub_domains["efficacy"][name] = tag_id

    # ── 品类树查询 ────────────────────────────────────────────────

    def get_category_tree(self) -> dict[str, dict[str, list[str]]]:
        """获取完整品类树结构"""
        return CATEGORY_TREE.copy()

    def get_l1_categories(self) -> list[TagNode]:
        """获取所有一级类目"""
        return [
            self.graph.tags[tid]
            for name, tid in self._name_index.items()
            if tid in self.graph.tags
            and self.graph.tags[tid].metadata.get("sub_domain") == "category"
            and self.graph.tags[tid].level == LEVEL_CATEGORY_1
        ]

    def get_l2_categories(self, l1_name: str) -> list[TagNode]:
        """获取某个一级类目下的所有二级类目"""
        l1_id = self._name_index.get(l1_name)
        if not l1_id:
            return []
        children = self.graph.get_children(l1_id)
        return [c for c in children if c.level == LEVEL_CATEGORY_2]

    def get_l3_categories(self, l1_name: str, l2_name: str) -> list[TagNode]:
        """获取某个二级类目下的所有三级类目"""
        l2_id = self._name_index.get(l2_name)
        if not l2_id:
            return []
        children = self.graph.get_children(l2_id)
        return [c for c in children if c.level == LEVEL_CATEGORY_3]

    def get_category_path(self, tag_name: str) -> list[str]:
        """获取标签的完整路径: ['一级', '二级', '三级']"""
        tag_id = self._name_index.get(tag_name)
        if not tag_id:
            return []
        ancestors = self.graph.get_ancestors(tag_id)
        # ancestors 是从近到远排列，需要反转
        path = [a.name for a in reversed(ancestors)]
        path.append(tag_name)
        return path

    def search_categories(self, keyword: str) -> list[TagNode]:
        """按关键词搜索品类标签"""
        results = []
        for name, tag_id in self._name_index.items():
            if keyword in name and tag_id in self.graph.tags:
                node = self.graph.tags[tag_id]
                if node.metadata.get("sub_domain") == "category":
                    results.append(node)
        return results

    # ── 辅助标签查询 ──────────────────────────────────────────────

    def get_price_band_tags(self) -> list[TagNode]:
        """获取所有价格带标签"""
        return self._get_sub_domain_tags("price_band")

    def get_style_tags(self) -> list[TagNode]:
        """获取所有风格标签"""
        return self._get_sub_domain_tags("style")

    def get_scenario_tags(self) -> list[TagNode]:
        """获取所有场景标签"""
        return self._get_sub_domain_tags("scenario")

    def get_efficacy_tags(self) -> list[TagNode]:
        """获取所有功效标签"""
        return self._get_sub_domain_tags("efficacy")

    def _get_sub_domain_tags(self, sub_domain: str) -> list[TagNode]:
        """获取某个子域的所有标签"""
        tag_ids = self._sub_domains.get(sub_domain, {})
        return [
            self.graph.tags[tid]
            for tid in tag_ids.values()
            if tid in self.graph.tags
        ]

    # ── CRUD ──────────────────────────────────────────────────────

    def add_tag(
        self,
        name: str,
        name_en: str,
        sub_domain: str = "category",
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
            domain=DOMAIN_COMMODITY,
            level=level,
            parent_id=parent_id,
            description=description or f"{sub_domain}: {name}",
            metadata=meta,
        )
        self._name_index[name] = tag_id
        if sub_domain in self._sub_domains:
            self._sub_domains[sub_domain][name] = tag_id

        return tag_id

    def get_tag_by_name(self, name: str) -> TagNode | None:
        """按名称获取标签"""
        tag_id = self._name_index.get(name)
        if tag_id:
            return self.graph.tags.get(tag_id)
        return None

    def remove_tag(self, name: str) -> bool:
        """按名称删除标签"""
        tag_id = self._name_index.pop(name, None)
        if not tag_id:
            return False

        # 从子域索引移除
        for sub_domain in self._sub_domains.values():
            sub_domain.pop(name, None)

        return self.graph.remove_tag(tag_id)

    def count(self) -> dict:
        """统计各子域标签数量"""
        return {
            "category": len([
                n for n, tid in self._name_index.items()
                if tid in self.graph.tags
                and self.graph.tags[tid].metadata.get("sub_domain") == "category"
            ]),
            "price_band": len(self._sub_domains.get("price_band", {})),
            "style": len(self._sub_domains.get("style", {})),
            "scenario": len(self._sub_domains.get("scenario", {})),
            "efficacy": len(self._sub_domains.get("efficacy", {})),
            "total": len(self._name_index),
        }
