"""
标签系统数据持久化
基于 SQLite 实现标签节点、联想关系、跨域规则、信号特征的 CRUD
"""
from __future__ import annotations

import json
import sqlite3
import time

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from ..core.models import (
        TagNode, TagAssociation, CrossDomainLink, SignalFeature,
        ALL_DOMAINS, ALL_ASSOC_TYPES, ALL_RULE_TYPES, ALL_SIGNAL_SOURCES,
    )
    from ..core.tag_graph import TagAssociationGraph
except ImportError:
    from core.models import (
        TagNode, TagAssociation, CrossDomainLink, SignalFeature,
        ALL_DOMAINS, ALL_ASSOC_TYPES, ALL_RULE_TYPES, ALL_SIGNAL_SOURCES,
    )
    from core.tag_graph import TagAssociationGraph


class TagDatabase:
    """
    标签系统持久化层

    表结构:
        tag_nodes         — 标签节点
        tag_associations  — 标签联想关系
        cross_domain_links — 跨域规则
        signal_features   — 信号特征
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def initialize(self):
        """初始化表结构"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 标签节点表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_en TEXT NOT NULL DEFAULT '',
                level INTEGER NOT NULL DEFAULT 1,
                domain TEXT NOT NULL,
                parent_id TEXT,
                embedding TEXT,
                description TEXT DEFAULT '',
                synonyms TEXT DEFAULT '[]',
                usage_count INTEGER DEFAULT 0,
                last_used REAL DEFAULT 0,
                created_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)

        # 标签联想关系表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tag_associations (
                id TEXT PRIMARY KEY,
                source_tag_id TEXT NOT NULL,
                target_tag_id TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.5,
                association_type TEXT NOT NULL DEFAULT 'semantic',
                context TEXT DEFAULT '{}',
                hit_count INTEGER DEFAULT 0,
                last_hit REAL DEFAULT 0,
                created_at REAL NOT NULL,
                last_strengthened REAL NOT NULL DEFAULT 0,
                UNIQUE(source_tag_id, target_tag_id)
            )
        """)

        # 跨域规则表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_domain_links (
                id TEXT PRIMARY KEY,
                source_domain TEXT NOT NULL,
                target_domain TEXT NOT NULL,
                rule_type TEXT NOT NULL DEFAULT 'boost',
                rule_expression TEXT DEFAULT '',
                weight REAL NOT NULL DEFAULT 1.0,
                enabled INTEGER NOT NULL DEFAULT 1,
                trigger_count INTEGER DEFAULT 0,
                last_triggered REAL DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)

        # 信号特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_features (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                features TEXT NOT NULL DEFAULT '{}',
                embedding TEXT,
                platform TEXT,
                creator_id TEXT,
                content_id TEXT,
                captured_at REAL NOT NULL,
                expires_at REAL
            )
        """)

        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_domain ON tag_nodes(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_level ON tag_nodes(level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag_parent ON tag_nodes(parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assoc_source ON tag_associations(source_tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assoc_target ON tag_associations(target_tag_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assoc_type ON tag_associations(association_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_link_domain ON cross_domain_links(source_domain, target_domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_source ON signal_features(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_platform ON signal_features(platform)")

        conn.commit()
        logger.info(f"标签数据库初始化完成: {self.db_path}")

    def close(self):
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── TagNode CRUD ──────────────────────────────────────────────

    def _row_to_tag(self, row: sqlite3.Row) -> TagNode:
        """将数据库行转换为 TagNode"""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])
        synonyms = json.loads(row["synonyms"]) if row["synonyms"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return TagNode(
            id=row["id"],
            name=row["name"],
            name_en=row["name_en"],
            level=row["level"],
            domain=row["domain"],
            parent_id=row["parent_id"],
            embedding=embedding,
            description=row["description"],
            synonyms=synonyms,
            usage_count=row["usage_count"],
            last_used=row["last_used"],
            created_at=row["created_at"],
            metadata=metadata,
        )

    def upsert_tag(self, tag: TagNode):
        """插入或更新标签"""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO tag_nodes
            (id, name, name_en, level, domain, parent_id, embedding,
             description, synonyms, usage_count, last_used, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tag.id, tag.name, tag.name_en, tag.level, tag.domain,
            tag.parent_id,
            json.dumps(tag.embedding) if tag.embedding else None,
            tag.description,
            json.dumps(tag.synonyms, ensure_ascii=False),
            tag.usage_count, tag.last_used, tag.created_at,
            json.dumps(tag.metadata, ensure_ascii=False),
        ))
        conn.commit()

    def upsert_tags_bulk(self, tags: list[TagNode]):
        """批量插入或更新标签"""
        conn = self._get_conn()
        conn.executemany("""
            INSERT OR REPLACE INTO tag_nodes
            (id, name, name_en, level, domain, parent_id, embedding,
             description, synonyms, usage_count, last_used, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                t.id, t.name, t.name_en, t.level, t.domain,
                t.parent_id,
                json.dumps(t.embedding) if t.embedding else None,
                t.description,
                json.dumps(t.synonyms, ensure_ascii=False),
                t.usage_count, t.last_used, t.created_at,
                json.dumps(t.metadata, ensure_ascii=False),
            )
            for t in tags
        ])
        conn.commit()

    def get_tag(self, tag_id: str) -> TagNode | None:
        """获取单个标签"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tag_nodes WHERE id = ?", (tag_id,)
        ).fetchone()
        if row:
            return self._row_to_tag(row)
        return None

    def get_tags(
        self,
        domain: str | None = None,
        level: int | None = None,
        parent_id: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[TagNode]:
        """查询标签列表"""
        conn = self._get_conn()
        query = "SELECT * FROM tag_nodes WHERE 1=1"
        params: list = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)
        if level is not None:
            query += " AND level = ?"
            params.append(level)
        if parent_id is not None:
            if parent_id == "":
                query += " AND parent_id IS NULL"
            else:
                query += " AND parent_id = ?"
                params.append(parent_id)

        query += " ORDER BY usage_count DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_tag(row) for row in rows]

    def delete_tag(self, tag_id: str) -> bool:
        """删除标签"""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM tag_nodes WHERE id = ?", (tag_id,))
        conn.commit()
        return cursor.rowcount > 0

    def count_tags(self, domain: str | None = None) -> int:
        """统计标签数量"""
        conn = self._get_conn()
        if domain:
            row = conn.execute(
                "SELECT COUNT(*) FROM tag_nodes WHERE domain = ?", (domain,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM tag_nodes").fetchone()
        return row[0] if row else 0

    # ── TagAssociation CRUD ────────────────────────────────────────

    def _row_to_association(self, row: sqlite3.Row) -> TagAssociation:
        """将数据库行转换为 TagAssociation"""
        context = json.loads(row["context"]) if row["context"] else {}
        return TagAssociation(
            id=row["id"],
            source_tag_id=row["source_tag_id"],
            target_tag_id=row["target_tag_id"],
            strength=row["strength"],
            confidence=row["confidence"],
            association_type=row["association_type"],
            context=context,
            hit_count=row["hit_count"],
            last_hit=row["last_hit"],
            created_at=row["created_at"],
            last_strengthened=row["last_strengthened"],
        )

    def upsert_association(self, assoc: TagAssociation):
        """插入或更新联想关系"""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO tag_associations
            (id, source_tag_id, target_tag_id, strength, confidence,
             association_type, context, hit_count, last_hit, created_at, last_strengthened)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assoc.id, assoc.source_tag_id, assoc.target_tag_id,
            assoc.strength, assoc.confidence, assoc.association_type,
            json.dumps(assoc.context, ensure_ascii=False),
            assoc.hit_count, assoc.last_hit, assoc.created_at,
            assoc.last_strengthened,
        ))
        conn.commit()

    def get_association(self, source_tag_id: str, target_tag_id: str) -> TagAssociation | None:
        """获取指定标签间的联想关系"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM tag_associations WHERE source_tag_id = ? AND target_tag_id = ?",
            (source_tag_id, target_tag_id)
        ).fetchone()
        if row:
            return self._row_to_association(row)
        return None

    def get_associations_for_tag(
        self,
        tag_id: str,
        association_type: str | None = None,
        min_strength: float = 0.0,
        limit: int = 100,
    ) -> list[TagAssociation]:
        """获取某个标签的所有联想关系"""
        conn = self._get_conn()
        query = """
            SELECT * FROM tag_associations
            WHERE (source_tag_id = ? OR target_tag_id = ?)
              AND strength >= ?
        """
        params: list = [tag_id, tag_id, min_strength]

        if association_type:
            query += " AND association_type = ?"
            params.append(association_type)

        query += " ORDER BY strength DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_association(row) for row in rows]

    def delete_association(self, association_id: str) -> bool:
        """删除联想关系"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM tag_associations WHERE id = ?", (association_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def count_associations(self) -> int:
        """统计联想关系数量"""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM tag_associations").fetchone()
        return row[0] if row else 0

    # ── CrossDomainLink CRUD ──────────────────────────────────────

    def _row_to_link(self, row: sqlite3.Row) -> CrossDomainLink:
        """将数据库行转换为 CrossDomainLink"""
        return CrossDomainLink(
            id=row["id"],
            source_domain=row["source_domain"],
            target_domain=row["target_domain"],
            rule_type=row["rule_type"],
            rule_expression=row["rule_expression"],
            weight=row["weight"],
            enabled=bool(row["enabled"]),
            trigger_count=row["trigger_count"],
            last_triggered=row["last_triggered"],
            created_at=row["created_at"],
        )

    def upsert_link(self, link: CrossDomainLink):
        """插入或更新跨域规则"""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO cross_domain_links
            (id, source_domain, target_domain, rule_type, rule_expression,
             weight, enabled, trigger_count, last_triggered, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            link.id, link.source_domain, link.target_domain,
            link.rule_type, link.rule_expression, link.weight,
            int(link.enabled), link.trigger_count, link.last_triggered,
            link.created_at,
        ))
        conn.commit()

    def get_links(
        self,
        source_domain: str | None = None,
        target_domain: str | None = None,
        rule_type: str | None = None,
    ) -> list[CrossDomainLink]:
        """查询跨域规则"""
        conn = self._get_conn()
        query = "SELECT * FROM cross_domain_links WHERE enabled = 1"
        params: list = []

        if source_domain:
            query += " AND source_domain = ?"
            params.append(source_domain)
        if target_domain:
            query += " AND target_domain = ?"
            params.append(target_domain)
        if rule_type:
            query += " AND rule_type = ?"
            params.append(rule_type)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_link(row) for row in rows]

    def delete_link(self, link_id: str) -> bool:
        """删除跨域规则"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM cross_domain_links WHERE id = ?", (link_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    # ── SignalFeature CRUD ────────────────────────────────────────

    def _row_to_signal(self, row: sqlite3.Row) -> SignalFeature:
        """将数据库行转换为 SignalFeature"""
        features = json.loads(row["features"]) if row["features"] else {}
        embedding = json.loads(row["embedding"]) if row["embedding"] else None

        return SignalFeature(
            id=row["id"],
            source=row["source"],
            features=features,
            embedding=embedding,
            platform=row["platform"],
            creator_id=row["creator_id"],
            content_id=row["content_id"],
            captured_at=row["captured_at"],
            expires_at=row["expires_at"],
        )

    def upsert_signal(self, signal: SignalFeature):
        """插入或更新信号特征"""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO signal_features
            (id, source, features, embedding, platform, creator_id,
             content_id, captured_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.id, signal.source,
            json.dumps(signal.features, ensure_ascii=False),
            json.dumps(signal.embedding) if signal.embedding else None,
            signal.platform, signal.creator_id, signal.content_id,
            signal.captured_at, signal.expires_at,
        ))
        conn.commit()

    def get_signal(self, signal_id: str) -> SignalFeature | None:
        """获取单个信号"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM signal_features WHERE id = ?", (signal_id,)
        ).fetchone()
        if row:
            return self._row_to_signal(row)
        return None

    def get_signals(
        self,
        source: str | None = None,
        platform: str | None = None,
        creator_id: str | None = None,
        limit: int = 100,
    ) -> list[SignalFeature]:
        """查询信号列表"""
        conn = self._get_conn()
        query = "SELECT * FROM signal_features WHERE 1=1"
        params: list = []

        if source:
            query += " AND source = ?"
            params.append(source)
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        if creator_id:
            query += " AND creator_id = ?"
            params.append(creator_id)

        query += " ORDER BY captured_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_signal(row) for row in rows]

    def delete_signal(self, signal_id: str) -> bool:
        """删除信号"""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM signal_features WHERE id = ?", (signal_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def count_signals(self) -> int:
        """统计信号数量"""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM signal_features").fetchone()
        return row[0] if row else 0

    # ── 图加载/保存 ───────────────────────────────────────────────

    def load_graph(self) -> TagAssociationGraph:
        """从数据库加载完整标签图"""
        graph = TagAssociationGraph()

        # 加载标签
        conn = self._get_conn()
        for row in conn.execute("SELECT * FROM tag_nodes").fetchall():
            tag = self._row_to_tag(row)
            graph.tags[tag.id] = tag
            graph._domain_index.setdefault(tag.domain, set()).add(tag.id)
            graph._level_index.setdefault(tag.level, set()).add(tag.id)
            if tag.parent_id:
                graph._parent_index.setdefault(tag.parent_id, set()).add(tag.id)
            graph._adjacency_list.setdefault(tag.id, [])

        # 加载联想关系
        for row in conn.execute("SELECT * FROM tag_associations").fetchall():
            assoc = self._row_to_association(row)
            graph.associations.append(assoc)
            key = (assoc.source_tag_id, assoc.target_tag_id)
            graph._association_index[key] = assoc
            graph._adjacency_list.setdefault(assoc.source_tag_id, [])
            graph._adjacency_list.setdefault(assoc.target_tag_id, [])
            graph._adjacency_list[assoc.source_tag_id].append(
                (assoc.target_tag_id, assoc.strength)
            )
            graph._adjacency_list[assoc.target_tag_id].append(
                (assoc.source_tag_id, assoc.strength)
            )

        # 加载跨域规则
        for row in conn.execute("SELECT * FROM cross_domain_links").fetchall():
            graph.cross_domain_links.append(self._row_to_link(row))

        # 加载信号
        for row in conn.execute("SELECT * FROM signal_features").fetchall():
            sig = self._row_to_signal(row)
            graph.signals[sig.id] = sig

        return graph

    def save_graph(self, graph: TagAssociationGraph):
        """将完整标签图保存到数据库"""
        conn = self._get_conn()
        try:
            conn.execute("BEGIN TRANSACTION")

            # 清空现有数据
            conn.execute("DELETE FROM tag_nodes")
            conn.execute("DELETE FROM tag_associations")
            conn.execute("DELETE FROM cross_domain_links")
            conn.execute("DELETE FROM signal_features")

            # 保存标签
            for tag in graph.tags.values():
                conn.execute("""
                    INSERT INTO tag_nodes
                    (id, name, name_en, level, domain, parent_id, embedding,
                     description, synonyms, usage_count, last_used, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tag.id, tag.name, tag.name_en, tag.level, tag.domain,
                    tag.parent_id,
                    json.dumps(tag.embedding) if tag.embedding else None,
                    tag.description,
                    json.dumps(tag.synonyms, ensure_ascii=False),
                    tag.usage_count, tag.last_used, tag.created_at,
                    json.dumps(tag.metadata, ensure_ascii=False),
                ))

            # 保存联想关系
            for assoc in graph.associations:
                conn.execute("""
                    INSERT INTO tag_associations
                    (id, source_tag_id, target_tag_id, strength, confidence,
                     association_type, context, hit_count, last_hit, created_at, last_strengthened)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    assoc.id, assoc.source_tag_id, assoc.target_tag_id,
                    assoc.strength, assoc.confidence, assoc.association_type,
                    json.dumps(assoc.context, ensure_ascii=False),
                    assoc.hit_count, assoc.last_hit, assoc.created_at,
                    assoc.last_strengthened,
                ))

            # 保存跨域规则
            for link in graph.cross_domain_links:
                conn.execute("""
                    INSERT INTO cross_domain_links
                    (id, source_domain, target_domain, rule_type, rule_expression,
                     weight, enabled, trigger_count, last_triggered, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    link.id, link.source_domain, link.target_domain,
                    link.rule_type, link.rule_expression, link.weight,
                    int(link.enabled), link.trigger_count, link.last_triggered,
                    link.created_at,
                ))

            # 保存信号
            for sig in graph.signals.values():
                conn.execute("""
                    INSERT INTO signal_features
                    (id, source, features, embedding, platform, creator_id,
                     content_id, captured_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sig.id, sig.source,
                    json.dumps(sig.features, ensure_ascii=False),
                    json.dumps(sig.embedding) if sig.embedding else None,
                    sig.platform, sig.creator_id, sig.content_id,
                    sig.captured_at, sig.expires_at,
                ))

            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            raise e

    # ── 统计 ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        return {
            "total_tags": self.count_tags(),
            "tags_by_domain": {
                d: self.count_tags(d) for d in ALL_DOMAINS
            },
            "total_associations": self.count_associations(),
            "total_signals": self.count_signals(),
        }
