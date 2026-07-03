# Memory Association Service

<div align="center">

![Version](https://img.shields.io/badge/version-v1.0.0-green?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-205%20passed-brightgreen?style=for-the-badge)

**货品-人群-平台 三级标签泛化联想服务**

基于5C法则（内容×渠道×人群×货品×转化）的智能标签联想引擎

[核心架构](#-核心架构) • [快速开始](#-快速开始) • [API参考](#-api参考) • [配置说明](#-配置说明)

</div>

---

## 📖 项目简介

Memory Association Service 是一个面向**内容电商+货架电商**的三级标签泛化联想服务。它基于生物海马体的记忆模型，结合货品-人群-平台的三级标签体系，通过5C法则实现智能的跨域联想匹配。

### 🎯 核心价值

> **用什么内容** × **在什么渠道** × **向什么人群** × **推什么商品** × **得到什么回报**

### ✨ 核心特性

- **🏷️ 三级标签体系**：货品域(品类/价格/风格/场景/功效) × 人群域(人口/兴趣/行为/内容/生命周期) × 平台域(TikTok/Instagram/X/微信)
- **🧠 5C联想引擎**：支持单域查询、双域交叉、全链路5C匹配
- **📊 多策略融合**：图遍历(0.35) + 语义相似度(0.30) + 共现关系(0.20) + 规则驱动(0.15)
- **🔌 信号处理器**：社媒数据/商单机会/内容特征的标准化处理
- **🔄 7种召回策略**：语义/关键词/联想/时间/强度 + 标签联想/跨域联想
- **⚡ 高性能**：单域查询 <50ms，支持10万级标签

---

## 🏗️ 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        5C 联想引擎                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Content  │  │ Channel  │  │  Crowd   │  │Commodity │           │
│  │ (内容)   │  │ (渠道)   │  │ (人群)   │  │ (货品)   │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │             │             │             │                   │
│       └─────────────┴──────┬──────┴─────────────┘                   │
│                            ↓                                        │
│                   ┌──────────────┐                                  │
│                   │  联想引擎     │──→ Conversion (转化回报)         │
│                   │  Association │                                  │
│                   │    Engine    │                                  │
│                   └──────┬───────┘                                  │
│                          │                                          │
│              ┌───────────┼───────────┐                              │
│              ↓           ↓           ↓                              │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│     │ TagGraph │  │ Embedding│  │ Signal   │                       │
│     │ (标签图)  │  │ (嵌入)   │  │ (信号)   │                       │
│     └──────────┘  └──────────┘  └──────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 模块结构

```
memory-association-service/
├── core/                           # 核心层
│   ├── models.py                   # 数据模型 (TagNode, TagAssociation, SignalFeature...)
│   ├── tag_graph.py                # 标签联想图 (层级查询/跨域遍历)
│   ├── memory_graph.py             # Legacy: 记忆图
│   └── memory_system.py            # Legacy: 记忆系统
│
├── intelligence/                   # 智能层
│   ├── tag_system/                 # 三级标签域
│   │   ├── commodity_domain.py     # 货品域 (品类树/价格/风格/场景/功效)
│   │   ├── crowd_domain.py         # 人群域 (人口/兴趣/行为/内容/生命周期)
│   │   ├── platform_domain.py      # 平台域 (特征/算法/人群/路径)
│   │   ├── association_engine.py   # 5C联想引擎
│   │   └── centroid_calculator.py  # 质心计算器
│   │
│   ├── signal_processors/          # 信号处理器
│   │   ├── social_signal.py        # 社媒数据 (TikTok/Instagram/X)
│   │   ├── commercial_signal.py    # 商单机会
│   │   └── content_signal.py       # 内容特征
│   │
│   └── recall/                     # 联想召回
│       ├── tag_association_recall.py  # 标签联想召回
│       └── cross_domain_recall.py     # 跨域联想召回
│
├── infrastructure/                 # 基础设施
│   ├── tag_database.py             # 标签持久化 (SQLite)
│   ├── tag_embedding.py            # 标签嵌入服务 (128维规则编码)
│   └── embedding.py                # Legacy: 向量缓存
│
├── api/                            # API层
│   └── tag_gateway.py              # 标签系统API网关 (20+端点)
│
├── memory/                         # Legacy: 记忆召回
│   └── memory_recall.py            # 7种召回策略
│
├── tests/                          # 测试 (205个用例)
│   ├── test_tag_system.py          # Phase 1: 数据模型
│   ├── test_tag_domains.py         # Phase 2: 三级标签域
│   ├── test_phase3_engine.py       # Phase 3: 联想引擎
│   ├── test_phase4_signals.py      # Phase 4: 信号处理器
│   ├── test_phase5_recall.py       # Phase 5: 召回扩展
│   └── test_phase6_api.py          # Phase 6: API与配置
│
├── docs/
│   └── TAG-ASSOCIATION-SERVICE-DESIGN.md  # 完整设计文档
│
├── main.py                         # AstrBot插件入口
└── _conf_schema.json               # 配置模式
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### Python API 使用

```python
from api.tag_gateway import TagAPIGateway

# 初始化（自动加载预置标签数据）
gateway = TagAPIGateway(db_path="data/tag_system.db")

# 查看统计
stats = gateway.get_stats()
print(f"总标签数: {stats['data']['graph']['total_tags']}")

# 标签联想查询
results = gateway.associate_single({
    "domain": "commodity",
    "tags": ["精华"],
    "top_k": 5,
})
for r in results["data"]["results"]:
    print(f"{r['tag_name']} ({r['tag_domain']}) - 相关度: {r['score']:.3f}")

# 5C法则匹配
results = gateway.associate_5c({
    "content": ["短视频"],
    "channel": ["抖音"],
    "crowd": ["Z世代"],
    "commodity": ["精华"],
    "top_k": 3,
})

# 导入社媒数据
gateway.import_social_signal({
    "platform": "tiktok_douyin",
    "content_id": "video_001",
    "creator_id": "creator_001",
    "likes": 1200,
    "comments": 89,
    "shares": 45,
    "views": 50000,
})
```

### 启动HTTP服务

```python
from aiohttp import web
from api.tag_gateway import TagAPIGateway, create_tag_routes

gateway = TagAPIGateway(db_path="data/tag_system.db")
routes = create_tag_routes(gateway)

app = web.Application()
app.router.add_routes(routes)
web.run_app(app, host="127.0.0.1", port=8351)
```

### AstrBot 插件命令

```
/记忆 标签查询           # 查看标签统计
/记忆 标签查询 精华      # 搜索标签
/记忆 联想 精华          # 标签联想查询
/记忆 5C匹配 货品:精华   # 5C法则匹配
```

---

## 📡 API 参考

### 标签CRUD

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/tags` | POST | 创建标签 |
| `/api/tags/{id}` | GET | 获取标签 |
| `/api/tags/{id}` | PUT | 更新标签 |
| `/api/tags/{id}` | DELETE | 删除标签 |
| `/api/tags?domain=commodity` | GET | 列表查询 |
| `/api/tags/{id}/children` | GET | 获取子标签 |
| `/api/tags/{id}/path` | GET | 获取标签路径 |
| `/api/tags/batch` | POST | 批量创建 |

### 联想查询

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/associate/single` | POST | 单域联想 |
| `/api/associate/cross` | POST | 双域交叉 |
| `/api/associate/full-5c` | POST | 全链路5C匹配 |

### 质心计算

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/centroid/compute` | POST | 计算用户兴趣质心 |
| `/api/centroid/find-tags` | POST | 根据质心查找标签 |

### 信号导入

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/signals/social` | POST | 导入社媒数据 |
| `/api/signals/commercial` | POST | 导入商单数据 |
| `/api/signals/content` | POST | 导入内容数据 |

### 统计配置

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/stats` | GET | 系统统计 |
| `/api/stats/domains` | GET | 域统计 |
| `/api/config` | GET | 获取配置 |
| `/api/config` | PUT | 更新配置 |

---

## 🏷️ 三级标签体系

### 货品域 (Commodity)

```
品类树:
  美妆护肤 → 护肤 → 精华 → 烟酰胺精华
                        → 玻尿酸精华
              → 彩妆 → 口红 / 粉底 / 眼影
  服饰鞋包 → 女装 / 男装 / 鞋靴 / 箱包
  食品饮料 → 零食 / 饮品 / 生鲜
  3C数码   → 手机 / 电脑 / 配件
  ...

辅助标签:
  价格带: 平价(0-99) | 中端(100-499) | 轻奢(500-1999) | 高端(2000+)
  风格: 极简 | 国潮 | 小众设计师 | 快时尚 | 复古 | 甜美 | 酷飒
  场景: 日常通勤 | 约会 | 运动健身 | 居家办公 | 旅行 | 送礼
  功效: 美白 | 抗老 | 保湿 | 修护 | 控油 | 祛痘 | 舒缓
```

### 人群域 (Crowd)

```
人口统计:
  年龄: Z世代(18-24) | 新锐白领(25-34) | 精致妈妈(30-40) | 银发族(50+)
  性别: 女性主导 | 男性主导 | 中性
  城市: 一线 | 新一线 | 二线 | 下沉市场

兴趣质心:
  主兴趣: 美妆护肤 | 穿搭时尚 | 美食探店 | 科技数码 | 家居家装 | 母婴育儿
  次兴趣: 宠物 | 影视娱乐 | 学习成长 | 理财投资

消费行为:
  频次: 高频(月3+) | 中频(月1-2) | 低频(季1)
  决策: 成分党 | 颜值党 | 性价比 | 品牌忠诚 | KOL驱动
  渠道: 直播冲动型 | 搜索目的型 | 种草拔草型

内容偏好:
  形式: 短视频 | 图文笔记 | 直播 | 测评长文
  调性: 专业硬核 | 轻松日常 | 种草安利 | 避坑指南
  互动: 评论活跃 | 点赞收藏 | 分享传播

生命周期: 新客 → 首购 → 复购 → 活跃 → 沉睡 → 流失
```

### 平台域 (Platform)

```
TikTok/抖音:
  内容: 短视频15-60s | 强视觉冲击 | BGM驱动 | 剧情化
  算法: 完播率(0.4) > 点赞(0.25) > 评论(0.2) > 分享(0.15)
  路径: 短视频挂车 → 直播转化 → 商城搜索

小红书/Instagram:
  内容: 图文笔记 | 精美封面 | 真实体验 | 干货分享
  算法: 收藏(0.35) > 评论(0.3) > 点赞(0.2) > 分享(0.15)
  路径: 种草笔记 → 搜索比价 → 平台内转化

X/Twitter:
  内容: 观点输出 | 话题讨论 | 热点追踪 | 简短有力
  算法: 转发(0.4) > 回复(0.3) > 点赞(0.2) > 引用(0.1)
  路径: 话题造势 → 品牌认知 → 搜索转化

微信视频号:
  内容: 私域沉淀 | 社交裂变 | 长内容 | 信任背书
  算法: 社交推荐(0.5) > 算法推荐(0.3) > 搜索(0.2)
  路径: 内容种草 → 私域转化 → 复购裂变
```

---

## ⚙️ 配置说明

### 标签系统配置

```json
{
  "tag_system_enabled": true,
  "tag_domains": {
    "commodity": true,
    "crowd": true,
    "platform": true
  },
  "tag_embedding_mode": "rule"
}
```

### 联想策略权重

```json
{
  "association_weights": {
    "graph_walk": 0.35,
    "semantic": 0.30,
    "cooccurrence": 0.20,
    "rule": 0.15
  }
}
```

### 召回策略权重

```json
{
  "recall_weights": {
    "semantic": 0.40,
    "keyword": 0.15,
    "associative": 0.15,
    "temporal": 0.03,
    "strength": 0.02,
    "tag_association": 0.15,
    "cross_domain": 0.10
  }
}
```

### API服务配置

```json
{
  "tag_system_api": {
    "enabled": false,
    "host": "127.0.0.1",
    "port": 8351
  }
}
```

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定阶段测试
python -m pytest tests/test_tag_system.py -v        # Phase 1: 数据模型
python -m pytest tests/test_tag_domains.py -v       # Phase 2: 三级标签域
python -m pytest tests/test_phase3_engine.py -v     # Phase 3: 联想引擎
python -m pytest tests/test_phase4_signals.py -v    # Phase 4: 信号处理器
python -m pytest tests/test_phase5_recall.py -v     # Phase 5: 召回扩展
python -m pytest tests/test_phase6_api.py -v        # Phase 6: API与配置
```

**测试覆盖**: 205个用例，覆盖数据模型、标签域、联想引擎、信号处理、召回策略、API端点

---

## 📊 性能指标

| 指标 | 目标 | 实测 |
|------|------|------|
| 单域查询延迟 | <50ms | ~2ms |
| 批量嵌入(50标签) | <100ms | ~3ms |
| 标签召回延迟 | <100ms | ~5ms |
| 跨域召回延迟 | <100ms | ~8ms |
| 支持标签数量 | 10万+ | ✅ |

---

## 🗺️ 开发路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 数据模型 (TagNode/TagAssociation/CrossDomainLink/SignalFeature) | ✅ 完成 |
| Phase 2 | 三级标签域 (货品/人群/平台) | ✅ 完成 |
| Phase 3 | 5C联想引擎 (嵌入/质心/匹配) | ✅ 完成 |
| Phase 4 | 信号处理器 (社媒/商单/内容) | ✅ 完成 |
| Phase 5 | 召回引擎扩展 (标签联想/跨域联想) | ✅ 完成 |
| Phase 6 | API与配置 (网关/命令/配置) | ✅ 完成 |

---

## 📄 文档

- [完整设计文档](docs/TAG-ASSOCIATION-SERVICE-DESIGN.md) — 架构设计、算法详解、失效模式分析
- [API参考](#-api参考) — REST API端点说明
- [配置说明](#-配置说明) — 配置项详解

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📝 许可证

MIT License

---

**⭐ 如果这个项目对您有帮助，请考虑给我们一个 Star！**
