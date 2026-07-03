#!/usr/bin/env python3
"""
多模态记忆迁移脚本
扫描 SQLite 中所有现有记忆，为每条生成文本向量，写入 LanceDB。

用法:
    python -m plugin_memora_connect.scripts.migrate_to_multimodal [--db-path PATH] [--lancedb-path PATH]

    或直接:
        python scripts/migrate_to_multimodal.py --db-path /path/to/memory.db
"""

import argparse
import asyncio
import os
import sqlite3
import sys
import time


async def migrate(db_path: str, lancedb_path: str, clip_model: str):
    """执行迁移"""
    print(f"数据库: {db_path}")
    print(f"LanceDB: {lancedb_path}")
    print(f"CLIP 模型: {clip_model}")
    print()

    # 1. 检查 SQLite 数据库
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return False

    # 2. 读取所有记忆
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT id, concept_id, content, details, emotion, strength, group_id "
            "FROM memories ORDER BY created_at DESC"
        )
        memories = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"读取记忆失败: {e}")
        conn.close()
        return False

    if not memories:
        print("没有记忆需要迁移")
        conn.close()
        return True

    print(f"找到 {len(memories)} 条记忆")

    # 3. 读取概念名称映射
    try:
        cursor.execute("SELECT id, name FROM concepts")
        concepts = {row[0]: row[1] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        concepts = {}

    conn.close()

    # 4. 初始化编码器
    print(f"正在加载 CLIP 模型: {clip_model} ...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from infrastructure.multimodal_encoder import MultimodalEncoder

        encoder = MultimodalEncoder(model_name=clip_model)
        if not encoder.is_available:
            print("错误: CLIP 模型加载失败")
            return False
        print(f"CLIP 模型加载成功，向量维度: {encoder.dimension}")
    except Exception as e:
        print(f"错误: 初始化编码器失败: {e}")
        return False

    # 5. 初始化 LanceDB
    print(f"正在连接 LanceDB: {lancedb_path} ...")
    try:
        from infrastructure.lancedb_store import LanceDBVectorStore

        store = LanceDBVectorStore(db_path=lancedb_path)
        success = await store.initialize()
        if not success:
            print("错误: LanceDB 初始化失败")
            return False
        print("LanceDB 连接成功")
    except Exception as e:
        print(f"错误: 初始化 LanceDB 失败: {e}")
        return False

    # 6. 批量迁移
    print()
    print("开始迁移...")
    start_time = time.time()

    synced = 0
    errors = 0
    total = len(memories)

    for i, (memory_id, concept_id, content, details, emotion, strength, group_id) in enumerate(
        memories
    ):
        try:
            # 编码文本
            text_content = f"{content} {details or ''}"
            text_vector = encoder.encode_text(text_content)

            # 检查是否为零向量（编码失败）
            if all(v == 0.0 for v in text_vector):
                errors += 1
                continue

            # 写入 LanceDB
            concept_name = concepts.get(concept_id, "")
            result = await store.upsert_memory_vector(
                memory_id=memory_id,
                group_id=group_id or "",
                text_vector=text_vector,
                image_vector=None,
                modality="text",
                concept_id=concept_id or "",
                concept_name=concept_name,
                content_preview=content,
                emotion=emotion or "",
                strength=strength or 1.0,
            )

            if result:
                synced += 1
            else:
                errors += 1

            # 进度输出
            if (i + 1) % 10 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"\r  进度: {i + 1}/{total} "
                    f"({(i + 1) / total * 100:.1f}%) "
                    f"成功: {synced} 失败: {errors} "
                    f"速率: {rate:.1f} 条/秒",
                    end="",
                    flush=True,
                )

        except Exception as e:
            errors += 1
            print(f"\n  迁移记忆 {memory_id} 失败: {e}")

    elapsed = time.time() - start_time
    print()
    print()
    print("=" * 50)
    print(f"迁移完成!")
    print(f"  总计: {total} 条")
    print(f"  成功: {synced} 条")
    print(f"  失败: {errors} 条")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  速率: {total / elapsed:.1f} 条/秒" if elapsed > 0 else "")
    print("=" * 50)

    return errors == 0


def main():
    parser = argparse.ArgumentParser(description="多模态记忆迁移脚本")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite 数据库路径 (默认: 自动查找)",
    )
    parser.add_argument(
        "--lancedb-path",
        type=str,
        default="data/lancedb",
        help="LanceDB 数据目录 (默认: data/lancedb)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP 模型名称 (默认: openai/clip-vit-base-patch32)",
    )

    args = parser.parse_args()

    # 自动查找数据库路径
    db_path = args.db_path
    if db_path is None:
        # 尝试常见路径
        candidates = [
            "data/memora_connect/memory.db",
            "data/memory.db",
            "memory.db",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                db_path = candidate
                break

    if db_path is None:
        print("错误: 无法找到数据库文件，请使用 --db-path 指定")
        sys.exit(1)

    success = asyncio.run(migrate(db_path, args.lancedb_path, args.clip_model))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
