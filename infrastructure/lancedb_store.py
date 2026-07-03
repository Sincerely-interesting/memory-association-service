"""
LanceDB 向量存储
基于 LanceDB 的多模态向量存储，支持向量索引和标量过滤

兼容 lancedb 0.11.0 (async API)
"""

import os
import time
from typing import Optional

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pyarrow as pa

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from .multimodal_encoder import CLIP_VECTOR_DIM
except (ImportError, SystemError):
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from multimodal_encoder import CLIP_VECTOR_DIM


def compute_combined_vector(
    text_vector: list,
    image_vector: Optional[list],
    modality: str,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> list:
    """计算图文融合向量"""
    try:
        from .multimodal_encoder import compute_combined_vector as _compute
    except (ImportError, SystemError):
        from multimodal_encoder import compute_combined_vector as _compute

    return _compute(text_vector, image_vector, modality, text_weight, image_weight)


def _build_schema(table_name: str) -> "pa.Schema":
    """构建 LanceDB 表的 PyArrow Schema

    关键：向量字段必须用 pa.list_(pa.float32(), list_size=DIM)
    而非 Python 的 list[float]（会被推断为 List(Float64)，不支持向量检索）
    """
    return pa.schema(
        [
            pa.field("memory_id", pa.utf8()),
            pa.field("group_id", pa.utf8()),
            pa.field(
                "text_vector", pa.list_(pa.float32(), list_size=CLIP_VECTOR_DIM)
            ),
            pa.field(
                "image_vector", pa.list_(pa.float32(), list_size=CLIP_VECTOR_DIM)
            ),
            pa.field(
                "combined_vector", pa.list_(pa.float32(), list_size=CLIP_VECTOR_DIM)
            ),
            pa.field("modality", pa.utf8()),
            pa.field("concept_id", pa.utf8()),
            pa.field("concept_name", pa.utf8()),
            pa.field("has_image", pa.bool_()),
            pa.field("emotion", pa.utf8()),
            pa.field("content_preview", pa.utf8()),
            pa.field("image_url", pa.utf8()),
            pa.field("created_at", pa.int64()),
            pa.field("updated_at", pa.int64()),
            pa.field("strength", pa.float32()),
        ]
    )


async def _table_to_list(table_query) -> list:
    """将 LanceDB 查询结果转为 list[dict]（兼容不同版本 API）"""
    try:
        arrow_table = await table_query.to_arrow()
        return arrow_table.to_pylist()
    except AttributeError:
        return await table_query.to_list()


class LanceDBVectorStore:
    """LanceDB 向量存储 — 管理多模态记忆向量

    存储结构:
        - text_vector: 文本向量 [512d, float32, fixed_size_list]
        - image_vector: 图片向量 [512d, float32, fixed_size_list]
        - combined_vector: 融合向量 [512d, float32, fixed_size_list]

    索引策略:
        - combined_vector: IVF_FLAT (主检索, cosine)
        - text_vector: IVF_FLAT (纯文本检索, cosine)
        - group_id/modality/has_image: BITMAP (标量过滤)
    """

    def __init__(self, db_path: str, table_name: str = "memory_vectors"):
        self.db_path = db_path
        self.table_name = table_name
        self._db = None
        self._table = None
        self._initialized = False
        self._schema = _build_schema(table_name)

    async def initialize(self) -> bool:
        """初始化 LanceDB 连接"""
        try:
            import lancedb

            os.makedirs(self.db_path, exist_ok=True)
            self._db = await lancedb.connect_async(self.db_path)

            try:
                self._table = await self._db.open_table(self.table_name)
                logger.info(f"LanceDB 表 '{self.table_name}' 已打开")
            except Exception:
                logger.info(
                    f"LanceDB 表 '{self.table_name}' 不存在，将在首次写入时创建"
                )
                self._table = None

            self._initialized = True
            return True

        except ImportError:
            logger.warning("lancedb 包未安装，向量存储不可用")
            return False
        except Exception as e:
            logger.warning(f"LanceDB 初始化失败: {e}")
            return False

    async def upsert_memory_vector(
        self,
        memory_id: str,
        group_id: str,
        text_vector: list,
        image_vector: Optional[list],
        modality: str,
        concept_id: str,
        concept_name: str,
        content_preview: str,
        emotion: Optional[str] = None,
        image_url: Optional[str] = None,
        strength: float = 1.0,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ) -> bool:
        """写入或更新记忆向量"""
        if not self._initialized:
            return False

        try:
            combined = compute_combined_vector(
                text_vector, image_vector, modality, text_weight, image_weight
            )

            # 确保向量是 float32 列表（LanceDB schema 要求）
            def to_float32_list(v):
                if v is None:
                    return [0.0] * CLIP_VECTOR_DIM
                return [float(x) for x in v]

            record = {
                "memory_id": memory_id,
                "group_id": group_id,
                "text_vector": to_float32_list(text_vector),
                "image_vector": to_float32_list(image_vector),
                "combined_vector": to_float32_list(combined),
                "modality": modality,
                "concept_id": concept_id,
                "concept_name": concept_name or "",
                "has_image": image_vector is not None,
                "emotion": emotion or "",
                "content_preview": (content_preview or "")[:200],
                "image_url": image_url or "",
                "created_at": int(time.time() * 1000),
                "updated_at": int(time.time() * 1000),
                "strength": float(strength),
            }

            if self._table is None:
                # 首次写入，创建表（使用 schema 确保 fixed_size_list<float32>）
                self._table = await self._db.create_table(
                    self.table_name, data=[record], schema=self._schema
                )
                logger.info(f"LanceDB 表 '{self.table_name}' 已创建")
                await self._create_indexes()
            else:
                # 检查是否存在
                existing = await _table_to_list(
                    self._table.query()
                    .where(f"memory_id = '{memory_id}'")
                    .limit(1)
                )
                if existing:
                    # lancedb 0.11: update(values_dict, where=...)
                    await self._table.update(
                        record,
                        where=f"memory_id = '{memory_id}'",
                    )
                else:
                    await self._table.add([record])

            return True

        except Exception as e:
            logger.error(f"LanceDB 写入失败 memory_id={memory_id}: {e}")
            return False

    async def search(
        self,
        query_vector: list,
        vector_column: str = "combined_vector",
        group_id: Optional[str] = None,
        modality_filter: Optional[str] = None,
        has_image: Optional[bool] = None,
        limit: int = 10,
        nprobes: int = 10,
    ) -> list:
        """向量检索"""
        if not self._initialized or self._table is None:
            return []

        try:
            # 构建过滤条件
            conditions = []
            if group_id:
                conditions.append(f"group_id = '{group_id}'")
            if modality_filter:
                conditions.append(f"modality = '{modality_filter}'")
            if has_image is not None:
                conditions.append(f"has_image = {str(has_image).lower()}")

            filter_expr = " AND ".join(conditions) if conditions else None

            # lancedb 0.11: table.vector_search(vec).column(col) → AsyncVectorQuery
            # 多个向量列时必须用 .column() 指定检索哪一列
            query = self._table.vector_search([float(x) for x in query_vector]).column(vector_column)

            if filter_expr:
                query = query.where(filter_expr)

            results = await _table_to_list(query.limit(limit))

            return [
                {
                    "memory_id": r["memory_id"],
                    "score": 1 - r.get("_distance", 0),
                    "modality": r.get("modality", "text"),
                    "concept_name": r.get("concept_name", ""),
                    "content_preview": r.get("content_preview", ""),
                    "image_url": r.get("image_url", ""),
                    "strength": r.get("strength", 1.0),
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"LanceDB 检索失败: {e}")
            return []

    async def delete_memory_vector(self, memory_id: str) -> bool:
        """删除记忆向量"""
        if not self._initialized or self._table is None:
            return False

        try:
            await self._table.delete(f"memory_id = '{memory_id}'")
            return True
        except Exception as e:
            logger.error(f"LanceDB 删除失败 memory_id={memory_id}: {e}")
            return False

    async def batch_delete_by_group(self, group_id: str) -> bool:
        """按群组批量删除"""
        if not self._initialized or self._table is None:
            return False

        try:
            await self._table.delete(f"group_id = '{group_id}'")
            return True
        except Exception as e:
            logger.error(f"LanceDB 群组删除失败 group_id={group_id}: {e}")
            return False

    async def count(self, group_id: Optional[str] = None) -> int:
        """获取记录数"""
        if not self._initialized or self._table is None:
            return 0

        try:
            if group_id:
                # query().where().count_rows() 不可用时降级
                rows = await _table_to_list(
                    self._table.query()
                    .where(f"group_id = '{group_id}'")
                    .select(["memory_id"])
                )
                return len(rows)
            else:
                return await self._table.count_rows()
        except Exception:
            return 0

    async def _create_indexes(self):
        """创建向量索引

        lancedb 0.11: create_index(column_name) 仅接受列名
        索引在数据量足够时自动选择 IVF_PQ/IVF_FLAT
        """
        if self._table is None:
            return

        for col in ["combined_vector", "text_vector"]:
            try:
                await self._table.create_index(col, replace=True)
                logger.info(f"向量索引创建成功: {col}")
            except Exception as e:
                # 小数据量时 KMeans 训练会失败，这是正常的
                logger.info(f"向量索引创建跳过 {col}: {e}")

    def get_status(self) -> dict:
        """获取存储状态"""
        return {
            "initialized": self._initialized,
            "db_path": self.db_path,
            "table_name": self.table_name,
            "has_table": self._table is not None,
        }
