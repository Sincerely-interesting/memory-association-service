"""
标签联想图数据结构
管理三级标签节点、联想关系和跨域关联
复用 MemoryGraph 的邻接表和连接管理架构
"""
from __future__ import annotations

import time

try:
    from .models import (
        TagNode,
        TagAssociation,
        CrossDomainLink,
        SignalFeature,
        DOMAIN_COMMODITY,
        DOMAIN_CROWD,
        DOMAIN_PLATFORM,
        ALL_DOMAINS,
        LEVEL_CATEGORY_1,
        LEVEL_CATEGORY_2,
        LEVEL_CATEGORY_3,
        ASSOC_SEMANTIC,
        ASSOC_COOCCURRENCE,
        ASSOC_HIERARCHICAL,
        ASSOC_CROSS_DOMAIN,
        ALL_ASSOC_TYPES,
    )
except ImportError:
    from models import (
        TagNode,
        TagAssociation,
        CrossDomainLink,
        SignalFeature,
        DOMAIN_COMMODITY,
        DOMAIN_CROWD,
        DOMAIN_PLATFORM,
        ALL_DOMAINS,
        LEVEL_CATEGORY_1,
        LEVEL_CATEGORY_2,
        LEVEL_CATEGORY_3,
        ASSOC_SEMANTIC,
        ASSOC_COOCCURRENCE,
        ASSOC_HIERARCHICAL,
        ASSOC_CROSS_DOMAIN,
        ALL_ASSOC_TYPES,
    )


class TagAssociationGraph:
    """
    标签联想图

    结构:
        TagNode (标签节点)
            ├── 同域层级连接 (parent_id / children)
            ├── 跨域联想连接 (TagAssociation)
            └── 跨域规则连接 (CrossDomainLink)

    复用 MemoryGraph 的邻接表设计，支持:
        - 三级标签的层级查询（祖先/后代/兄弟）
        - 跨域联想遍历
        - 标签嵌入向量管理
    """

    def __init__(self):
        # 核心存储
        self.tags: dict[str, TagNode] = {}
        self.associations: list[TagAssociation] = []
        self.cross_domain_links: list[CrossDomainLink] = []
        self.signals: dict[str, SignalFeature] = {}

        # 全局计数器（用于生成唯一ID）
        self._tag_counter: int = 0

        # 索引结构（加速查询）
        self._domain_index: dict[str, set[str]] = {d: set() for d in ALL_DOMAINS}
        self._level_index: dict[int, set[str]] = {
            LEVEL_CATEGORY_1: set(),
            LEVEL_CATEGORY_2: set(),
            LEVEL_CATEGORY_3: set(),
        }
        self._parent_index: dict[str, set[str]] = {}  # parent_id -> {child_ids}
        self._adjacency_list: dict[str, list[tuple[str, float]]] = {}
        self._association_index: dict[tuple[str, str], TagAssociation] = {}

    # ── TagNode CRUD ──────────────────────────────────────────────

    def add_tag(
        self,
        name: str,
        name_en: str,
        domain: str,
        level: int = LEVEL_CATEGORY_1,
        parent_id: str | None = None,
        tag_id: str | None = None,
        description: str = "",
        synonyms: list[str] | None = None,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """添加标签节点"""
        if tag_id is None:
            # 使用全局计数器避免同一毫秒内的ID碰撞
            self._tag_counter += 1
            tag_id = f"tag_{domain}_{self._tag_counter}"

        if tag_id in self.tags:
            return tag_id

        node = TagNode(
            id=tag_id,
            name=name,
            name_en=name_en,
            level=level,
            domain=domain,
            parent_id=parent_id,
            description=description,
            synonyms=synonyms or [],
            embedding=embedding,
            metadata=metadata or {},
        )
        self.tags[tag_id] = node

        # 更新索引
        self._domain_index.setdefault(domain, set()).add(tag_id)
        self._level_index.setdefault(level, set()).add(tag_id)
        if parent_id:
            self._parent_index.setdefault(parent_id, set()).add(tag_id)

        # 初始化邻接表
        if tag_id not in self._adjacency_list:
            self._adjacency_list[tag_id] = []

        return tag_id

    def get_tag(self, tag_id: str) -> TagNode | None:
        """获取标签节点"""
        return self.tags.get(tag_id)

    def update_tag(self, tag_id: str, **fields) -> bool:
        """更新标签字段"""
        node = self.tags.get(tag_id)
        if not node:
            return False
        allowed = {
            "name", "name_en", "level", "domain", "parent_id",
            "embedding", "description", "synonyms", "usage_count",
            "last_used", "metadata",
        }
        for k, v in fields.items():
            if k in allowed and v is not None:
                setattr(node, k, v)
        return True

    def remove_tag(self, tag_id: str) -> bool:
        """删除标签及其所有关联"""
        node = self.tags.get(tag_id)
        if not node:
            return False

        # 1. 删除所有关联到此标签的联想关系
        self.associations = [
            a for a in self.associations
            if a.source_tag_id != tag_id and a.target_tag_id != tag_id
        ]

        # 2. 删除子标签的 parent_id 引用
        children = self._parent_index.pop(tag_id, set())
        for child_id in children:
            child = self.tags.get(child_id)
            if child:
                child.parent_id = None

        # 3. 更新索引
        self._domain_index.get(node.domain, set()).discard(tag_id)
        self._level_index.get(node.level, set()).discard(tag_id)
        if node.parent_id and node.parent_id in self._parent_index:
            self._parent_index[node.parent_id].discard(tag_id)

        # 4. 清理邻接表
        if tag_id in self._adjacency_list:
            del self._adjacency_list[tag_id]
        for neighbor_id in list(self._adjacency_list.keys()):
            self._adjacency_list[neighbor_id] = [
                (n, s) for n, s in self._adjacency_list[neighbor_id] if n != tag_id
            ]

        # 5. 删除标签
        del self.tags[tag_id]
        return True

    # ── TagAssociation CRUD ────────────────────────────────────────

    def add_association(
        self,
        source_tag_id: str,
        target_tag_id: str,
        strength: float = 0.5,
        confidence: float = 0.5,
        association_type: str = ASSOC_SEMANTIC,
        context: dict | None = None,
        association_id: str | None = None,
    ) -> str:
        """添加标签联想关系"""
        # 检查是否已存在
        key = (source_tag_id, target_tag_id)
        reverse_key = (target_tag_id, source_tag_id)

        if key in self._association_index:
            existing = self._association_index[key]
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.last_strengthened = time.time()
            return existing.id
        if reverse_key in self._association_index:
            existing = self._association_index[reverse_key]
            existing.strength = min(1.0, existing.strength + 0.1)
            existing.last_strengthened = time.time()
            return existing.id

        if association_id is None:
            association_id = f"assoc_{source_tag_id}_{target_tag_id}"

        assoc = TagAssociation(
            id=association_id,
            source_tag_id=source_tag_id,
            target_tag_id=target_tag_id,
            strength=strength,
            confidence=confidence,
            association_type=association_type,
            context=context or {},
        )
        self.associations.append(assoc)
        self._association_index[key] = assoc

        # 更新邻接表（双向）
        self._adjacency_list.setdefault(source_tag_id, [])
        self._adjacency_list.setdefault(target_tag_id, [])
        self._adjacency_list[source_tag_id].append((target_tag_id, strength))
        self._adjacency_list[target_tag_id].append((source_tag_id, strength))

        return assoc.id

    def get_association(self, source_tag_id: str, target_tag_id: str) -> TagAssociation | None:
        """获取指定两个标签间的联想关系"""
        return self._association_index.get((source_tag_id, target_tag_id))

    def remove_association(self, association_id: str) -> bool:
        """删除联想关系"""
        target = None
        for assoc in self.associations:
            if assoc.id == association_id:
                target = assoc
                break

        if not target:
            return False

        # 从列表移除
        self.associations = [a for a in self.associations if a.id != association_id]
        # 从索引移除
        key = (target.source_tag_id, target.target_tag_id)
        self._association_index.pop(key, None)

        # 更新邻接表
        src = target.source_tag_id
        tgt = target.target_tag_id
        if src in self._adjacency_list:
            self._adjacency_list[src] = [
                (n, s) for n, s in self._adjacency_list[src] if n != tgt
            ]
        if tgt in self._adjacency_list:
            self._adjacency_list[tgt] = [
                (n, s) for n, s in self._adjacency_list[tgt] if n != src
            ]

        return True

    def get_associations_for_tag(
        self,
        tag_id: str,
        association_type: str | None = None,
        min_strength: float = 0.0,
    ) -> list[TagAssociation]:
        """获取某个标签的所有联想关系"""
        results = []
        for assoc in self.associations:
            if assoc.source_tag_id == tag_id or assoc.target_tag_id == tag_id:
                if association_type and assoc.association_type != association_type:
                    continue
                if assoc.strength < min_strength:
                    continue
                results.append(assoc)
        return results

    # ── CrossDomainLink CRUD ──────────────────────────────────────

    def add_cross_domain_link(
        self,
        source_domain: str,
        target_domain: str,
        rule_type: str,
        rule_expression: str = "",
        weight: float = 1.0,
        link_id: str | None = None,
    ) -> str:
        """添加跨域关联规则"""
        if link_id is None:
            link_id = f"link_{source_domain}_{target_domain}_{int(time.time() * 1000)}"

        link = CrossDomainLink(
            id=link_id,
            source_domain=source_domain,
            target_domain=target_domain,
            rule_type=rule_type,
            rule_expression=rule_expression,
            weight=weight,
        )
        self.cross_domain_links.append(link)
        return link.id

    def remove_cross_domain_link(self, link_id: str) -> bool:
        """删除跨域关联规则"""
        before = len(self.cross_domain_links)
        self.cross_domain_links = [l for l in self.cross_domain_links if l.id != link_id]
        return len(self.cross_domain_links) < before

    def get_cross_domain_links(
        self,
        source_domain: str | None = None,
        target_domain: str | None = None,
        rule_type: str | None = None,
    ) -> list[CrossDomainLink]:
        """查询跨域关联规则"""
        results = self.cross_domain_links
        if source_domain:
            results = [l for l in results if l.source_domain == source_domain]
        if target_domain:
            results = [l for l in results if l.target_domain == target_domain]
        if rule_type:
            results = [l for l in results if l.rule_type == rule_type]
        return [l for l in results if l.enabled]

    # ── SignalFeature CRUD ────────────────────────────────────────

    def add_signal(self, signal: SignalFeature) -> str:
        """添加信号特征"""
        self.signals[signal.id] = signal
        return signal.id

    def get_signal(self, signal_id: str) -> SignalFeature | None:
        """获取信号特征"""
        return self.signals.get(signal_id)

    def remove_signal(self, signal_id: str) -> bool:
        """删除信号特征"""
        if signal_id in self.signals:
            del self.signals[signal_id]
            return True
        return False

    def get_signals_by_source(
        self,
        source: str | None = None,
        platform: str | None = None,
        creator_id: str | None = None,
    ) -> list[SignalFeature]:
        """按来源/平台/创作者查询信号"""
        results = list(self.signals.values())
        if source:
            results = [s for s in results if s.source == source]
        if platform:
            results = [s for s in results if s.platform == platform]
        if creator_id:
            results = [s for s in results if s.creator_id == creator_id]
        return results

    # ── 层级查询 ──────────────────────────────────────────────────

    def get_children(self, tag_id: str) -> list[TagNode]:
        """获取直接子标签"""
        child_ids = self._parent_index.get(tag_id, set())
        return [self.tags[cid] for cid in child_ids if cid in self.tags]

    def get_descendants(self, tag_id: str) -> list[TagNode]:
        """获取所有后代标签（递归）"""
        result = []
        stack = list(self._parent_index.get(tag_id, set()))
        while stack:
            cid = stack.pop()
            if cid in self.tags:
                result.append(self.tags[cid])
                stack.extend(self._parent_index.get(cid, set()))
        return result

    def get_ancestors(self, tag_id: str) -> list[TagNode]:
        """获取所有祖先标签（从近到远）"""
        result = []
        node = self.tags.get(tag_id)
        while node and node.parent_id:
            parent = self.tags.get(node.parent_id)
            if parent:
                result.append(parent)
            node = parent
        return result

    def get_siblings(self, tag_id: str) -> list[TagNode]:
        """获取兄弟标签（同父）"""
        node = self.tags.get(tag_id)
        if not node or not node.parent_id:
            return []
        child_ids = self._parent_index.get(node.parent_id, set())
        return [
            self.tags[cid] for cid in child_ids
            if cid != tag_id and cid in self.tags
        ]

    def get_root_tags(self, domain: str | None = None) -> list[TagNode]:
        """获取顶级标签"""
        results = [n for n in self.tags.values() if n.parent_id is None]
        if domain:
            results = [n for n in results if n.domain == domain]
        return results

    def get_tags_by_domain(self, domain: str) -> list[TagNode]:
        """获取某个域的所有标签"""
        ids = self._domain_index.get(domain, set())
        return [self.tags[tid] for tid in ids if tid in self.tags]

    def get_tags_by_level(self, level: int) -> list[TagNode]:
        """获取某个层级的所有标签"""
        ids = self._level_index.get(level, set())
        return [self.tags[tid] for tid in ids if tid in self.tags]

    def get_neighbors(self, tag_id: str) -> list[tuple[str, float]]:
        """获取标签的联想邻居及其强度"""
        return self._adjacency_list.get(tag_id, [])

    def get_cross_domain_neighbors(
        self,
        tag_id: str,
        target_domain: str,
    ) -> list[tuple[str, float]]:
        """获取跨域联想邻居"""
        node = self.tags.get(tag_id)
        if not node:
            return []

        neighbors = []
        for neighbor_id, strength in self._adjacency_list.get(tag_id, []):
            neighbor = self.tags.get(neighbor_id)
            if neighbor and neighbor.domain == target_domain:
                neighbors.append((neighbor_id, strength))
        return neighbors

    # ── 统计 ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回图统计信息"""
        return {
            "total_tags": len(self.tags),
            "total_associations": len(self.associations),
            "total_cross_domain_links": len(self.cross_domain_links),
            "total_signals": len(self.signals),
            "tags_by_domain": {
                d: len(self._domain_index.get(d, set())) for d in ALL_DOMAINS
            },
            "tags_by_level": {
                f"L{l}": len(self._level_index.get(l, set()))
                for l in [LEVEL_CATEGORY_1, LEVEL_CATEGORY_2, LEVEL_CATEGORY_3]
            },
            "avg_association_strength": (
                sum(a.strength for a in self.associations) / len(self.associations)
                if self.associations else 0.0
            ),
        }

    # ── 序列化 ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """序列化整个图（用于持久化）"""
        return {
            "tags": {k: v.to_dict() for k, v in self.tags.items()},
            "associations": [a.to_dict() for a in self.associations],
            "cross_domain_links": [l.to_dict() for l in self.cross_domain_links],
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TagAssociationGraph":
        """从字典反序列化"""
        graph = cls()

        # 恢复标签
        for tag_data in data.get("tags", {}).values():
            graph.tags[tag_data["id"]] = TagNode(**tag_data)

        # 重建索引
        for tag_id, node in graph.tags.items():
            graph._domain_index.setdefault(node.domain, set()).add(tag_id)
            graph._level_index.setdefault(node.level, set()).add(tag_id)
            if node.parent_id:
                graph._parent_index.setdefault(node.parent_id, set()).add(tag_id)
            graph._adjacency_list.setdefault(tag_id, [])

        # 恢复联想关系
        for assoc_data in data.get("associations", []):
            assoc = TagAssociation(**assoc_data)
            graph.associations.append(assoc)
            key = (assoc.source_tag_id, assoc.target_tag_id)
            graph._association_index[key] = assoc
            # 更新邻接表
            graph._adjacency_list.setdefault(assoc.source_tag_id, [])
            graph._adjacency_list.setdefault(assoc.target_tag_id, [])
            graph._adjacency_list[assoc.source_tag_id].append(
                (assoc.target_tag_id, assoc.strength)
            )
            graph._adjacency_list[assoc.target_tag_id].append(
                (assoc.source_tag_id, assoc.strength)
            )

        # 恢复跨域规则
        for link_data in data.get("cross_domain_links", []):
            graph.cross_domain_links.append(CrossDomainLink(**link_data))

        # 恢复信号
        for sig_data in data.get("signals", {}).values():
            graph.signals[sig_data["id"]] = SignalFeature(**sig_data)

        return graph
