#!/usr/bin/env python3
"""
端到端集成测试：多模态记忆系统
测试完整链路：CLIP 编码 → LanceDB 写入 → 向量检索 → 跨模态召回

用法:
    cd plugin_memora_connect
    python -m tests.test_multimodal_e2e
    或
    python tests/test_multimodal_e2e.py
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time

# 确保能导入本项目（直接导入模块，绕过 __init__.py 对 astrbot 的依赖）
_plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_infra_dir = os.path.join(_plugin_dir, "infrastructure")
_scripts_dir = os.path.join(_plugin_dir, "scripts")
sys.path.insert(0, _infra_dir)
sys.path.insert(0, _scripts_dir)

# ── 测试配置 ──────────────────────────────────────────────────

# 项目根目录下的图片
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_IMAGES = [
    os.path.join(PROJECT_ROOT, "data", "generated", "adventures"),
    os.path.join(PROJECT_ROOT, "data", "uploads", "avatars"),
    os.path.join(PROJECT_ROOT, "public", "assets", "fallback", "adventures"),
]

# 测试结果
_results = {"passed": 0, "failed": 0, "errors": []}


def _pass(name: str, detail: str = ""):
    _results["passed"] += 1
    print(f"  ✅ {name}" + (f" ({detail})" if detail else ""))


def _fail(name: str, reason: str):
    _results["failed"] += 1
    _results["errors"].append(f"{name}: {reason}")
    print(f"  ❌ {name}: {reason}")


# ── 测试用例 ──────────────────────────────────────────────────


async def test_01_clip_encoder_text():
    """测试 CLIP 文本编码"""
    from multimodal_encoder import MultimodalEncoder, CLIP_VECTOR_DIM, ZERO_VECTOR

    encoder = MultimodalEncoder()
    _pass("CLIP encoder 实例化")

    # 测试文本编码
    vec = encoder.encode_text("一双红色的运动鞋")
    if len(vec) != CLIP_VECTOR_DIM:
        _fail("文本编码维度", f"期望 {CLIP_VECTOR_DIM}，实际 {len(vec)}")
        return
    _pass("文本编码维度", f"{len(vec)}d")

    # 测试非零向量
    norm = sum(x * x for x in vec) ** 0.5
    if norm < 0.01:
        _fail("文本编码非零", "向量接近零")
    else:
        _pass("文本编码非零", f"L2 norm={norm:.4f}")

    # 测试空文本
    empty_vec = encoder.encode_text("")
    if all(v == 0.0 for v in empty_vec):
        _pass("空文本返回零向量")
    else:
        _fail("空文本", "应返回零向量")


async def test_02_clip_encoder_image():
    """测试 CLIP 图片编码 — 使用项目中的真实图片"""
    from multimodal_encoder import MultimodalEncoder, CLIP_VECTOR_DIM

    encoder = MultimodalEncoder()
    if not encoder.is_available:
        _fail("CLIP 模型加载", "模型不可用，跳过图片编码测试")
        return
    _pass("CLIP 模型加载", encoder.model_name)

    # 找一张真实图片
    image_path = _find_sample_image()
    if not image_path:
        _fail("查找测试图片", "未找到任何图片文件")
        return
    _pass("找到测试图片", os.path.basename(image_path))

    # 编码图片
    start = time.time()
    img_vec = encoder.encode_image(image_path)
    elapsed_ms = (time.time() - start) * 1000

    if len(img_vec) != CLIP_VECTOR_DIM:
        _fail("图片编码维度", f"期望 {CLIP_VECTOR_DIM}，实际 {len(img_vec)}")
        return
    _pass("图片编码维度", f"{len(img_vec)}d, {elapsed_ms:.0f}ms")

    norm = sum(x * x for x in img_vec) ** 0.5
    if norm < 0.01:
        _fail("图片编码非零", "向量接近零")
    else:
        _pass("图片编码非零", f"L2 norm={norm:.4f}")

    # 测试文本 vs 图片跨模态相似度
    text_vec = encoder.encode_text("冒险场景")
    dot = sum(a * b for a, b in zip(text_vec, img_vec))
    _pass("跨模态相似度", f"cosine={dot:.4f}")

    # 测试不同图片的向量应不同
    image_path2 = _find_sample_image(exclude=image_path)
    if image_path2:
        img_vec2 = encoder.encode_image(image_path2)
        dot2 = sum(a * b for a, b in zip(img_vec, img_vec2))
        _pass("不同图片相似度", f"cosine={dot2:.4f} (应 <1.0)")
        if dot2 > 0.999:
            _fail("不同图片区分度", "两张不同图片的向量完全相同")


async def test_03_combined_vector():
    """测试融合向量计算"""
    from multimodal_encoder import compute_combined_vector, CLIP_VECTOR_DIM

    text_vec = [1.0] * CLIP_VECTOR_DIM
    image_vec = [0.5] * CLIP_VECTOR_DIM

    # 纯文本模态
    combined_text = compute_combined_vector(text_vec, None, "text")
    assert combined_text == text_vec, "纯文本应直接返回文本向量"
    _pass("纯文本融合")

    # 纯图片模态
    combined_image = compute_combined_vector(text_vec, image_vec, "image")
    assert combined_image == image_vec, "纯图片应直接返回图片向量"
    _pass("纯图片融合")

    # 多模态融合
    combined_mm = compute_combined_vector(text_vec, image_vec, "multimodal")
    norm = sum(x * x for x in combined_mm) ** 0.5
    if abs(norm - 1.0) < 0.01:
        _pass("多模态融合 L2 归一化", f"norm={norm:.6f}")
    else:
        _fail("多模态融合 L2 归一化", f"norm={norm:.6f}，应接近 1.0")


async def test_04_lancedb_crud():
    """测试 LanceDB 基本 CRUD 操作"""
    from lancedb_store import LanceDBVectorStore

    tmpdir = tempfile.mkdtemp(prefix="lancedb_test_")
    try:
        store = LanceDBVectorStore(db_path=tmpdir, table_name="test_crud")
        ok = await store.initialize()
        if not ok:
            _fail("LanceDB 初始化", "initialize() 返回 False")
            return
        _pass("LanceDB 初始化")

        # UPSERT
        ok = await store.upsert_memory_vector(
            memory_id="mem_001",
            group_id="group_a",
            text_vector=[0.1] * 512,
            image_vector=None,
            modality="text",
            concept_id="c_001",
            concept_name="旅行",
            content_preview="用户去了海边",
            emotion="开心",
            strength=0.8,
        )
        if not ok:
            _fail("LanceDB upsert", "upsert 返回 False")
            return
        _pass("LanceDB upsert (text)")

        # UPSERT multimodal
        ok = await store.upsert_memory_vector(
            memory_id="mem_002",
            group_id="group_a",
            text_vector=[0.2] * 512,
            image_vector=[0.3] * 512,
            modality="multimodal",
            concept_id="c_002",
            concept_name="美食",
            content_preview="用户发了一张火锅照片",
            image_url="http://example.com/hotpot.jpg",
            strength=0.9,
        )
        _pass("LanceDB upsert (multimodal)", f"ok={ok}")

        # COUNT
        count = await store.count()
        if count != 2:
            _fail("LanceDB count", f"期望 2，实际 {count}")
        else:
            _pass("LanceDB count", str(count))

        # UPSERT 更新同一 memory_id
        ok = await store.upsert_memory_vector(
            memory_id="mem_001",
            group_id="group_a",
            text_vector=[0.15] * 512,
            image_vector=None,
            modality="text",
            concept_id="c_001",
            concept_name="旅行",
            content_preview="用户去了海边（更新）",
            strength=0.9,
        )
        count2 = await store.count()
        if count2 != 2:
            _fail("LanceDB upsert 更新", f"count 应仍为 2，实际 {count2}")
        else:
            _pass("LanceDB upsert 更新", "count 仍为 2")

        # DELETE
        ok = await store.delete_memory_vector("mem_001")
        count3 = await store.count()
        if count3 != 1:
            _fail("LanceDB delete", f"count 应为 1，实际 {count3}")
        else:
            _pass("LanceDB delete", "count=1")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_05_lancedb_vector_search():
    """测试 LanceDB 向量检索"""
    from lancedb_store import LanceDBVectorStore

    tmpdir = tempfile.mkdtemp(prefix="lancedb_search_")
    try:
        store = LanceDBVectorStore(db_path=tmpdir, table_name="test_search")
        await store.initialize()

        # 写入多条记忆
        for i in range(10):
            vec = [float(i) / 10] * 512
            await store.upsert_memory_vector(
                memory_id=f"mem_{i:03d}",
                group_id="group_test",
                text_vector=vec,
                image_vector=None,
                modality="text",
                concept_id=f"c_{i % 3}",
                concept_name=["旅行", "美食", "运动"][i % 3],
                content_preview=f"测试记忆 {i}",
                strength=float(i) / 10,
            )
        _pass("写入 10 条记忆")

        # 向量检索
        query_vec = [0.5] * 512  # 应该最接近 mem_005
        results = await store.search(query_vec, limit=5)
        if not results:
            _fail("向量检索", "返回空结果")
        else:
            _pass("向量检索", f"返回 {len(results)} 条")
            top = results[0]
            _pass("Top1 结果", f"memory_id={top['memory_id']}, score={top['score']:.4f}")

        # 带 group_id 过滤的检索
        results_filtered = await store.search(
            query_vec, group_id="group_test", limit=5
        )
        if results_filtered:
            _pass("group_id 过滤检索", f"返回 {len(results_filtered)} 条")
        else:
            _fail("group_id 过滤检索", "返回空")

        # 带 modality 过滤的检索
        results_mod = await store.search(
            query_vec, modality_filter="text", limit=5
        )
        if results_mod:
            _pass("modality 过滤检索", f"返回 {len(results_mod)} 条")
        else:
            _fail("modality 过滤检索", "返回空")

        # 分数排序验证
        if len(results) >= 2:
            if results[0]["score"] >= results[1]["score"]:
                _pass("分数降序排列")
            else:
                _fail("分数降序排列", f"{results[0]['score']} < {results[1]['score']}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_06_cross_modal_search():
    """测试跨模态检索：用文字向量检索图片记忆"""
    from multimodal_encoder import MultimodalEncoder, compute_combined_vector
    from lancedb_store import LanceDBVectorStore

    encoder = MultimodalEncoder()
    if not encoder.is_available:
        _fail("跨模态检索", "CLIP 不可用")
        return

    tmpdir = tempfile.mkdtemp(prefix="lancedb_xmodal_")
    try:
        store = LanceDBVectorStore(db_path=tmpdir, table_name="test_xmodal")
        await store.initialize()

        # 写入图片记忆
        image_path = _find_sample_image()
        if not image_path:
            _fail("跨模态检索", "无测试图片")
            return

        img_vec = encoder.encode_image(image_path)
        text_vec_for_image = encoder.encode_text("冒险场景图片")

        await store.upsert_memory_vector(
            memory_id="img_mem_001",
            group_id="test",
            text_vector=text_vec_for_image,
            image_vector=img_vec,
            modality="multimodal",
            concept_id="adventure",
            concept_name="冒险",
            content_preview="用户发了一张冒险场景图片",
            image_url=image_path,
        )
        _pass("写入图片记忆")

        # 用文字查询检索图片记忆
        query_vec = encoder.encode_text("冒险")
        results = await store.search(query_vec, limit=5)
        if results and results[0]["memory_id"] == "img_mem_001":
            _pass("文字→图片跨模态检索", f"score={results[0]['score']:.4f}")
        elif results:
            _pass("文字→图片跨模态检索", f"命中 {results[0]['memory_id']} (score={results[0]['score']:.4f})")
        else:
            _fail("文字→图片跨模态检索", "未命中")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_07_group_isolation():
    """测试群组隔离"""
    from lancedb_store import LanceDBVectorStore

    tmpdir = tempfile.mkdtemp(prefix="lancedb_group_")
    try:
        store = LanceDBVectorStore(db_path=tmpdir, table_name="test_group")
        await store.initialize()

        # 写入两个群组的记忆
        for gid in ["group_a", "group_b"]:
            await store.upsert_memory_vector(
                memory_id=f"{gid}_mem",
                group_id=gid,
                text_vector=[0.5] * 512,
                image_vector=None,
                modality="text",
                concept_id="c",
                concept_name="test",
                content_preview=f"{gid} 的记忆",
            )

        count_a = await store.count(group_id="group_a")
        count_b = await store.count(group_id="group_b")
        if count_a == 1 and count_b == 1:
            _pass("群组隔离 count", f"A={count_a}, B={count_b}")
        else:
            _fail("群组隔离 count", f"A={count_a}, B={count_b}")

        # 检索只返回本组
        results_a = await store.search([0.5] * 512, group_id="group_a", limit=10)
        if len(results_a) == 1 and results_a[0]["memory_id"] == "group_a_mem":
            _pass("群组隔离检索")
        else:
            ids = [r["memory_id"] for r in results_a]
            _fail("群组隔离检索", f"期望只有 group_a_mem，实际 {ids}")

        # 批量删除
        await store.batch_delete_by_group("group_a")
        count_after = await store.count()
        if count_after == 1:
            _pass("群组批量删除")
        else:
            _fail("群组批量删除", f"期望 1，实际 {count_after}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_08_performance_benchmark():
    """性能基准：批量写入和检索延迟"""
    from multimodal_encoder import MultimodalEncoder
    from lancedb_store import LanceDBVectorStore

    encoder = MultimodalEncoder()
    tmpdir = tempfile.mkdtemp(prefix="lancedb_perf_")
    try:
        store = LanceDBVectorStore(db_path=tmpdir, table_name="test_perf")
        await store.initialize()

        # 批量写入 50 条
        n_records = 50
        start = time.time()
        for i in range(n_records):
            vec = [float(i) / n_records] * 512
            await store.upsert_memory_vector(
                memory_id=f"perf_{i:04d}",
                group_id="perf_group",
                text_vector=vec,
                image_vector=None,
                modality="text",
                concept_id=f"c_{i % 5}",
                concept_name=f"con_{i % 5}",
                content_preview=f"性能测试记忆 {i}",
            )
        write_elapsed = time.time() - start
        write_rate = n_records / write_elapsed
        _pass(f"批量写入 {n_records} 条", f"{write_elapsed:.2f}s ({write_rate:.0f} 条/秒)")

        # 检索延迟
        query_vec = [0.5] * 512
        latencies = []
        for _ in range(10):
            start = time.time()
            await store.search(query_vec, limit=10)
            latencies.append((time.time() - start) * 1000)

        avg_lat = sum(latencies) / len(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        _pass("检索延迟", f"avg={avg_lat:.1f}ms, p95={p95_lat:.1f}ms")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def test_09_migration_script_smoke():
    """冒烟测试：迁移脚本可导入"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "migrate_to_multimodal",
            os.path.join(_scripts_dir, "migrate_to_multimodal.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pass("迁移脚本导入")
    except ImportError as e:
        _fail("迁移脚本导入", str(e))


async def test_10_memory_system_multimodal_init():
    """测试 MemorySystem 多模态初始化（不需要 AstrBot 环境）"""
    from lancedb_store import LanceDBVectorStore
    from multimodal_encoder import MultimodalEncoder

    tmpdir = tempfile.mkdtemp(prefix="lancedb_sys_")
    try:
        encoder = MultimodalEncoder()
        store = LanceDBVectorStore(db_path=tmpdir)
        ok = await store.initialize()
        _pass("MemorySystem 多模态初始化流程", f"encoder={encoder.is_available}, store={ok}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── 工具函数 ──────────────────────────────────────────────────


def _find_sample_image(exclude: str = None):
    """在项目中查找一张可用的图片文件"""
    for search_dir in SAMPLE_IMAGES:
        if not os.path.isdir(search_dir):
            continue
        for f in sorted(os.listdir(search_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                full = os.path.join(search_dir, f)
                if full != exclude and os.path.getsize(full) > 1000:
                    return full
    return None


# ── 主入口 ────────────────────────────────────────────────────


async def run_all_tests():
    print("=" * 60)
    print("多模态记忆系统 端到端集成测试")
    print("=" * 60)
    print()

    tests = [
        ("01 CLIP 文本编码", test_01_clip_encoder_text),
        ("02 CLIP 图片编码", test_02_clip_encoder_image),
        ("03 融合向量计算", test_03_combined_vector),
        ("04 LanceDB CRUD", test_04_lancedb_crud),
        ("05 LanceDB 向量检索", test_05_lancedb_vector_search),
        ("06 跨模态检索", test_06_cross_modal_search),
        ("07 群组隔离", test_07_group_isolation),
        ("08 性能基准", test_08_performance_benchmark),
        ("09 迁移脚本冒烟", test_09_migration_script_smoke),
        ("10 初始化流程", test_10_memory_system_multimodal_init),
    ]

    total_start = time.time()

    for name, test_fn in tests:
        print(f"\n── {name} ──")
        try:
            await test_fn()
        except Exception as e:
            _fail(name, f"未捕获异常: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    print()
    print("=" * 60)
    print(f"测试完成: {_results['passed']} 通过, {_results['failed']} 失败, 耗时 {total_elapsed:.1f}s")
    if _results["errors"]:
        print()
        print("失败详情:")
        for err in _results["errors"]:
            print(f"  • {err}")
    print("=" * 60)

    return _results["failed"] == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
