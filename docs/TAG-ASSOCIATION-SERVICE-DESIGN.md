# 货品-人群-平台 三级标签泛化联想服务 — 开发设计文档

> **版本**: v1.0 | **日期**: 2025-07-03 | **作者**: qa296  
> **目标**: 基于现有 memory-association-service 架构，重构为面向内容电商+货架电商的三级标签联想服务

---

## 目录

1. [现状分析与评估](#1-现状分析与评估)
2. [需求分解与提炼](#2-需求分解与提炼)
3. [三级标签类目体系设计](#3-三级标签类目体系设计)
4. [5C法则联想引擎](#4-5c法则联想引擎)
5. [架构设计与文件结构](#5-架构设计与文件结构)
6. [数据模型设计](#6-数据模型设计)
7. [核心算法设计](#7-核心算法设计)
8. [失效模式与交叉验证](#8-失效模式与交叉验证)
9. [共性维度提炼](#9-共性维度提炼)
10. [实现路线图](#10-实现路线图)
11. [API设计](#11-api设计)
12. [配置与部署](#12-配置与部署)
13. [测试与验证](#13-测试与验证)
14. [风险与缓解](#14-风险与缓解)

---

## 1. 现状分析与评估

### 1.1 现有架构概览

当前 `memory-association-service` 是一个 AstrBot 聊天插件，核心能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                    现有架构                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Concept      │───→│ Memory       │───→│ Connection   │      │
│  │ (概念节点)    │    │ (记忆条目)    │    │ (概念连接)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         ↑                   ↑                   ↑               │
│         │                   │                   │               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │TopicAnalyzer │    │UserProfiling │    │MemoryRecall  │      │
│  │ (话题分析)    │    │ (用户画像)    │    │ (记忆召回)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                    ┌──────────────┐                             │
│                    │EmbeddingCache│                             │
│                    │ (向量缓存)    │                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 组件能力评估

| 组件 | 当前能力 | 电商标签化适配度 | 改造难度 |
|------|---------|----------------|---------|
| `Concept` | 单层概念节点 | ⭐⭐ 需扩展为三级标签 | 低 |
| `Memory` | 聊天记忆条目 | ⭐⭐⭐ 可映射为标签关联 | 中 |
| `Connection` | 概念间连接 | ⭐⭐⭐ 可扩展为跨域关联 | 低 |
| `MemoryGraph` | 图数据结构 | ⭐⭐⭐ 直接复用 | 低 |
| `TopicAnalyzer` | 聊天话题提取 | ⭐ 需完全重写 | 高 |
| `UserProfilingSystem` | 亲密度+兴趣 | ⭐⭐ 需扩展为人群质心 | 中 |
| `EnhancedMemoryRecall` | 5策略召回 | ⭐⭐⭐ 可扩展新策略 | 中 |
| `EmbeddingCacheManager` | 文本向量缓存 | ⭐⭐⭐ 直接复用 | 低 |

### 1.3 关键洞察

**可复用的**：
- 图数据结构（MemoryGraph）的邻接表、连接管理
- 向量嵌入缓存（EmbeddingCacheManager）的预计算、批量处理、LRU策略
- 多策略召回框架（5策略并行+去重+排序）
- 事件总线（MemoryEventBus）的发布-订阅模式

**需要重写的**：
- 话题分析器 → 电商内容信号处理器
- 用户画像系统 → 人群质心计算器
- 记忆模型 → 标签关联模型

**需要新增的**：
- 三级标签域管理（货品/人群/平台）
- 5C联想引擎
- 跨域关联计算
- 社媒数据特征处理器

---

## 2. 需求分解与提炼

### 2.1 核心问题

> 用什么内容(What Content) × 在什么渠道(What Channel) × 向什么人群(What Crowd) × 推什么商品(What Commodity) × 得到什么回报(What Conversion)

这个问题的本质是**多维标签的联想匹配**：

```
输入：任一维度的标签或特征
  ↓
处理：跨域联想匹配（语义相似度 + 规则约束 + 协同过滤）
  ↓
输出：Top-K 联想结果（5C组合）
```

### 2.2 数据来源映射

| 数据来源 | 提供的特征 | 对应标签域 |
|---------|-----------|-----------|
| AI打标服务 | 用户兴趣质心向量 | 人群域-兴趣质心 |
| 商单系统 | 品牌调性、预算、KPI | 货品域+人群域 |
| TikTok数据 | 短视频互动、受众画像 | 平台域-TikTok |
| Instagram数据 | 图文互动、生活方式 | 平台域-Instagram |
| X/Twitter数据 | 话题热度、观点分布 | 平台域-X |
| 商品库 | 品类、价格、属性 | 货品域 |
| 用户行为 | 浏览、点击、购买 | 人群域-消费行为 |

### 2.3 标签粒度定义

```
粗粒度 (Coarse)   → 用于快速筛选、冷启动
中粒度 (Medium)   → 用于日常联想匹配
细粒度 (Fine)     → 用于精准推荐、A/B测试
```

---

## 3. 三级标签类目体系设计

### 3.1 货品域 (Commodity Domain)

```yaml
货品域:
  品类标签 (Category):
    一级类目:
      - 美妆护肤 (beauty_skincare)
      - 服饰鞋包 (fashion_accessories)
      - 食品饮料 (food_beverage)
      - 3C数码 (electronics)
      - 家居生活 (home_living)
      - 母婴亲子 (baby_parenting)
      - 运动户外 (sports_outdoor)
      - 宠物用品 (pet_supplies)
    
    二级类目 (以美妆为例):
      - 护肤 (skincare)
        - 精华 (serum)
        - 面霜 (cream)
        - 防晒 (sunscreen)
        - 面膜 (mask)
      - 彩妆 (makeup)
        - 口红 (lipstick)
        - 粉底 (foundation)
        - 眼影 (eyeshadow)
    
    三级类目 (以精华为例):
      - 烟酰胺精华 (niacinamide_serum)
      - 玻尿酸精华 (hyaluronic_serum)
      - VC精华 (vitamin_c_serum)
      - A醇精华 (retinol_serum)
  
  价格带标签 (PriceBand):
    - 平价 (budget): 0-99 CNY
    - 中端 (mid_range): 100-499 CNY
    - 轻奢 (affordable_luxury): 500-1999 CNY
    - 高端 (premium): 2000+ CNY
  
  风格标签 (Style):
    - 极简 (minimalist)
    - 国潮 (guochao)
    - 小众设计师 (indie_designer)
    - 快时尚 (fast_fashion)
    - 复古 (vintage)
    - 甜美 (sweet)
    - 酷飒 (cool)
  
  场景标签 (Scenario):
    - 日常通勤 (daily_commute)
    - 约会 (date_night)
    - 运动健身 (workout)
    - 居家办公 (home_office)
    - 旅行 (travel)
    - 聚会 (party)
    - 送礼 (gift)
  
  功效标签 (Efficacy):
    - 美白 (brightening)
    - 抗老 (anti_aging)
    - 保湿 (hydrating)
    - 修护 (repairing)
    - 控油 (oil_control)
    - 祛痘 (acne_treatment)
    - 舒缓 (soothing)
```

### 3.2 人群域 (Crowd Domain)

```yaml
人群域:
  人口统计 (Demographics):
    年龄段:
      - Z世代 (gen_z): 18-24
      - 新锐白领 (young_professional): 25-34
      - 精致妈妈 (quality_mom): 30-40
      - 银发族 (senior): 50+
    
    性别倾向:
      - 女性主导 (female_oriented)
      - 男性主导 (male_oriented)
      - 中性 (neutral)
    
    城市层级:
      - 一线城市 (tier_1)
      - 新一线城市 (new_tier_1)
      - 二线城市 (tier_2)
      - 下沉市场 (lower_tier)
  
  兴趣质心 (InterestCentroid):
    主兴趣簇:
      - 美妆护肤 (beauty_interest)
      - 穿搭时尚 (fashion_interest)
      - 美食探店 (food_interest)
      - 科技数码 (tech_interest)
      - 家居家装 (home_interest)
      - 母婴育儿 (parenting_interest)
      - 健身运动 (fitness_interest)
      - 旅行探店 (travel_interest)
    
    次兴趣簇:
      - 宠物 (pet)
      - 影视娱乐 (entertainment)
      - 学习成长 (learning)
      - 理财投资 (finance)
  
  消费行为 (ConsumptionBehavior):
    消费频次:
      - 高频 (high_frequency): 月3+次
      - 中频 (medium_frequency): 月1-2次
      - 低频 (low_frequency): 季1次
    
    决策因子:
      - 成分党 (ingredient_focused)
      - 颜值党 (aesthetic_focused)
      - 性价比 (value_focused)
      - 品牌忠诚 (brand_loyal)
      - KOL驱动 (kol_driven)
    
    渠道偏好:
      - 直播冲动型 (live_impulse)
      - 搜索目的型 (search_intent)
      - 种草拔草型 (seeding_harvesting)
  
  内容偏好 (ContentPreference):
    内容形式:
      - 短视频 (short_video)
      - 图文笔记 (image_text_note)
      - 直播 (live_stream)
      - 测评长文 (review_article)
    
    内容调性:
      - 专业硬核 (professional)
      - 轻松日常 (casual)
      - 种草安利 (recommendation)
      - 避坑指南 (anti_pitfall)
    
    互动模式:
      - 评论活跃 (comment_active)
      - 点赞收藏 (like_save)
      - 分享传播 (share_spread)
  
  生命周期阶段 (LifecycleStage):
    - 新客 (new_customer)
    - 首购 (first_purchase)
    - 复购 (repeat_purchase)
    - 沉睡 (dormant)
    - 流失 (churned)
```

### 3.3 平台域 (Platform Domain)

```yaml
平台域:
  tiktok_douyin:
    内容特征:
      - 短视频15-60s (short_form_video)
      - 强视觉冲击 (visual_impact)
      - BGM驱动 (bgm_driven)
      - 剧情化 (storytelling)
    
    算法特征:
      - 完播率权重: 0.4
      - 点赞权重: 0.25
      - 评论权重: 0.2
      - 分享权重: 0.15
    
    人群特征:
      - 年龄: 18-35为主
      - 性别: 女性略多
      - 城市: 下沉渗透强
      - 消费力: 中等偏下
    
    电商路径:
      - 短视频挂车 → 直播转化 → 商城搜索
  
  instagram_xiaohongshu:
    内容特征:
      - 图文笔记 (image_text_note)
      - 精美封面 (aesthetic_cover)
      - 真实体验 (authentic_experience)
      - 干货分享 (valuable_content)
    
    算法特征:
      - 收藏权重: 0.35
      - 评论权重: 0.3
      - 点赞权重: 0.2
      - 分享权重: 0.15
    
    人群特征:
      - 年龄: 20-35为主
      - 性别: 女性为主(70%+)
      - 城市: 一二线为主
      - 消费力: 中等偏上
    
    电商路径:
      - 种草笔记 → 搜索比价 → 平台/站外转化
  
  x_twitter:
    内容特征:
      - 观点输出 (opinion_output)
      - 话题讨论 (topic_discussion)
      - 热点追踪 (trending_tracking)
      - 简短有力 (concise_powerful)
    
    算法特征:
      - 转发权重: 0.4
      - 回复权重: 0.3
      - 点赞权重: 0.2
      - 引用权重: 0.1
    
    人群特征:
      - 年龄: 22-40为主
      - 性别: 男性略多
      - 兴趣: 科技/财经/泛知识
      - 消费力: 中等偏上
    
    电商路径:
      - 话题造势 → 品牌认知 → 搜索转化
  
  wechat_channels:
    内容特征:
      - 私域沉淀 (private_domain)
      - 社交裂变 (social_viral)
      - 长内容 (long_form_content)
      - 信任背书 (trust_endorsement)
    
    算法特征:
      - 社交推荐权重: 0.5
      - 算法推荐权重: 0.3
      - 搜索权重: 0.2
    
    人群特征:
      - 年龄: 30+为主
      - 性别: 均衡
      - 城市: 下沉市场渗透强
      - 消费力: 中等
    
    电商路径:
      - 内容种草 → 私域转化 → 复购裂变
```

---

## 4. 5C法则联想引擎

### 4.1 联想匹配模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    5C 联想匹配流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入层 (Input Layer)                                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │ Content │ │ Channel │ │ Crowd   │ │Commodity│              │
│  │ 特征    │ │ 特征    │ │ 质心    │ │ 属性    │              │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│       │           │           │           │                     │
│       └───────────┴───────────┴───────────┘                     │
│                       │                                         │
│                       ↓                                         │
│  编码层 (Encoding Layer)                                        │
│  ┌─────────────────────────────────────────────┐               │
│  │  TagEmbeddingService                        │               │
│  │  - 标签→向量映射                             │               │
│  │  - 特征→向量编码                             │               │
│  │  - 多模态融合（文本+行为+时序）               │               │
│  └─────────────────────┬───────────────────────┘               │
│                         │                                       │
│                         ↓                                       │
│  联想层 (Association Layer)                                     │
│  ┌─────────────────────────────────────────────┐               │
│  │  AssociationEngine                          │               │
│  │  - 语义相似度匹配 (Cosine Similarity)        │               │
│  │  - 协同过滤匹配 (Co-occurrence)              │               │
│  │  - 规则约束过滤 (Business Rules)             │               │
│  │  - 时效性衰减 (Time Decay)                   │               │
│  └─────────────────────┬───────────────────────┘               │
│                         │                                       │
│                         ↓                                       │
│  输出层 (Output Layer)                                          │
│  ┌─────────────────────────────────────────────┐               │
│  │  Top-K 联想结果                              │               │
│  │  - Content × Channel × Crowd × Commodity    │               │
│  │  - 置信度评分                                 │               │
│  │  - 预期转化信号                               │               │
│  └─────────────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 联想策略

#### 策略1：单域查询
```python
# 输入：货品标签
# 输出：匹配的人群+平台+内容
query = TagQuery(domain="commodity", tags=["烟酰胺精华", "美白"])
results = engine.associate(query, top_k=10)

# 示例输出:
# 1. Crowd: Z世代+成分党 → Platform: 小红书 → Content: 测评长文 → Conversion: 0.72
# 2. Crowd: 新锐白领+性价比 → Platform: 抖音 → Content: 短视频 → Conversion: 0.65
```

#### 策略2：双域交叉
```python
# 输入：人群+平台
# 输出：匹配的货品+内容
query = TagQuery(
    domain="crowd_platform",
    crowd_tags=["Z世代", "美妆兴趣"],
    platform_tags=["tiktok_douyin"]
)
results = engine.associate(query, top_k=10)

# 示例输出:
# 1. Commodity: 平价口红 → Content: 剧情化短视频 → Conversion: 0.68
# 2. Commodity: 美甲贴片 → Content: 变装视频 → Conversion: 0.71
```

#### 策略3：全链路5C匹配
```python
# 输入：部分5C维度
# 输出：补全缺失维度
query = TagQuery(
    partial_5c={
        "content": ["测评"],
        "channel": ["tiktok_douyin"],
        "crowd": None,  # 待补全
        "commodity": ["护肤", "精华"],
        "conversion": None  # 待预测
    }
)
results = engine.complete_5c(query)

# 示例输出:
# Crowd: 精致妈妈+成分党
# Conversion: 预期ROI 1:3.5
```

### 4.3 相似度计算

```python
def compute_association_score(
    source_tag: TagNode,
    target_tag: TagNode,
    context: dict
) -> float:
    """
    计算两个标签的联想得分
    
    Score = α * semantic_sim + β * cooccurrence + γ * rule_boost - δ * time_decay
    """
    # 1. 语义相似度（嵌入向量余弦）
    semantic_sim = cosine_similarity(
        source_tag.embedding, 
        target_tag.embedding
    )
    
    # 2. 共现频率（历史数据）
    cooccurrence = get_cooccurrence_freq(
        source_tag.id, 
        target_tag.id,
        context.get("platform")
    )
    
    # 3. 规则加成（业务规则）
    rule_boost = apply_business_rules(
        source_tag, 
        target_tag, 
        context
    )
    
    # 4. 时效性衰减
    time_decay = compute_time_decay(
        source_tag.last_updated,
        target_tag.last_updated
    )
    
    # 加权合并
    weights = {"semantic": 0.4, "cooccurrence": 0.3, "rule": 0.2, "decay": 0.1}
    score = (
        weights["semantic"] * semantic_sim +
        weights["cooccurrence"] * cooccurrence +
        weights["rule"] * rule_boost -
        weights["decay"] * time_decay
    )
    
    return min(1.0, max(0.0, score))
```

---

## 5. 架构设计与文件结构

### 5.1 目标架构

```
memory-association-service/
├── core/
│   ├── models.py              # [修改] 新增 TagNode, TagAssociation, CrossDomainLink
│   ├── memory_system.py       # [修改] 注入标签系统组件
│   ├── memory_graph.py        # [保留] 图数据结构复用
│   └── config.py              # [修改] 新增标签系统配置
│
├── intelligence/
│   ├── tag_system/            # [新增] 三级标签系统
│   │   ├── __init__.py
│   │   ├── tag_node.py        # 标签节点模型
│   │   ├── tag_graph.py       # 标签联想图
│   │   ├── commodity_domain.py # 货品域管理
│   │   ├── crowd_domain.py    # 人群域管理
│   │   ├── platform_domain.py # 平台域管理
│   │   ├── association_engine.py # 5C联想引擎
│   │   └── centroid_calculator.py # 质心计算器
│   │
│   ├── signal_processors/     # [新增] 信号处理器
│   │   ├── __init__.py
│   │   ├── social_signal.py   # 社媒数据特征
│   │   ├── commercial_signal.py # 商单机会特征
│   │   └── content_signal.py  # 内容特征
│   │
│   ├── recall/                # [新增] 联想召回
│   │   ├── __init__.py
│   │   ├── tag_association_recall.py
│   │   └── cross_domain_recall.py
│   │
│   ├── topic_analyzer.py      # [保留] 旧话题分析器（兼容）
│   ├── profiling.py           # [保留] 旧用户画像（兼容）
│   └── temporal.py            # [保留] 时间维度系统
│
├── infrastructure/
│   ├── tag_embedding.py       # [新增] 标签语义嵌入服务
│   ├── tag_database.py        # [新增] 标签数据持久化
│   ├── embedding.py           # [保留] 通用向量缓存
│   ├── database.py            # [保留] 通用数据库
│   ├── events.py              # [保留] 事件总线
│   └── resources.py           # [保留] 资源管理
│
├── api/
│   ├── gateway.py             # [保留] 旧API网关
│   └── tag_gateway.py         # [新增] 标签服务API
│
├── memory/
│   ├── memory_recall.py       # [修改] 新增标签联想召回策略
│   ├── memory_display.py      # [保留]
│   └── visualization.py       # [保留]
│
├── tests/
│   ├── test_tag_system.py     # [新增] 标签系统测试
│   ├── test_association.py    # [新增] 联想引擎测试
│   └── ...                    # [保留] 旧测试
│
├── docs/
│   └── TAG-ASSOCIATION-SERVICE-DESIGN.md  # [新增] 本文档
│
├── main.py                    # [修改] 注册新命令和工具
├── _conf_schema.json          # [修改] 新增配置项
└── requirements.txt           # [修改] 新增依赖
```

### 5.2 模块依赖关系

```
                    ┌──────────────┐
                    │   main.py    │
                    │  (入口注册)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ API层    │ │ 命令层    │ │ LLM工具层 │
        │gateway   │ │commands  │ │llm_tools │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ↓
                 ┌────────────────┐
                 │  智能层         │
                 │ (intelligence) │
                 ├────────────────┤
                 │ tag_system/    │ ← 三级标签
                 │ signal_*       │ ← 信号处理
                 │ recall/        │ ← 联想召回
                 └───────┬────────┘
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 基础设施  │ │ 核心层   │ │ 数据层   │
        │infrast   │ │ core/    │ │ models   │
        └──────────┘ └──────────┘ └──────────┘
```

---

## 6. 数据模型设计

### 6.1 TagNode（标签节点）

```python
@dataclass
class TagNode:
    """三级标签节点"""
    id: str                          # 唯一标识
    name: str                        # 标签名称
    name_en: str                     # 英文名称（用于嵌入）
    
    # 层级信息
    level: int                       # 1=一级, 2=二级, 3=三级
    domain: str                      # commodity/crowd/platform
    parent_id: str | None            # 父标签ID
    
    # 语义信息
    embedding: list[float] | None    # 语义嵌入向量
    description: str                 # 标签描述
    synonyms: list[str]              # 同义词
    
    # 统计信息
    usage_count: int = 0             # 使用次数
    last_used: float = 0             # 最后使用时间戳
    created_at: float = 0            # 创建时间戳
    
    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)
    # 例: {"platform": "tiktok", "category": "beauty", "price_range": "mid"}
```

### 6.2 TagAssociation（标签联想）

```python
@dataclass
class TagAssociation:
    """标签联想关系"""
    id: str                          # 唯一标识
    source_tag_id: str               # 源标签ID
    target_tag_id: str               # 目标标签ID
    
    # 联想强度
    strength: float                  # 联想强度 0-1
    confidence: float                # 置信度 0-1
    
    # 联想类型
    association_type: str            # semantic/cooccurrence/hierarchical/cross_domain
    
    # 上下文
    context: dict[str, Any]          # 联想上下文
    # 例: {"platform": "tiktok", "time_decay": 0.95}
    
    # 统计
    hit_count: int = 0               # 命中次数
    last_hit: float = 0              # 最后命中时间
    
    # 时间
    created_at: float = 0
    last_strengthened: float = 0
```

### 6.3 CrossDomainLink（跨域关联）

```python
@dataclass
class CrossDomainLink:
    """跨域关联"""
    id: str
    source_domain: str               # 源域
    target_domain: str               # 目标域
    
    # 关联规则
    rule_type: str                   # constraint/boost/filter
    rule_expression: str             # 规则表达式
    # 例: "IF crowd.gen_z AND commodity.budget THEN platform.tiktok BOOST 0.3"
    
    # 权重
    weight: float = 1.0
    enabled: bool = True
    
    # 统计
    trigger_count: int = 0
    last_triggered: float = 0
```

### 6.4 SignalFeature（信号特征）

```python
@dataclass
class SignalFeature:
    """信号特征（社媒/商单/内容）"""
    id: str
    source: str                      # social/commercial/content
    
    # 特征向量
    features: dict[str, float]       # 特征键值对
    # 社媒示例: {"engagement_rate": 0.05, "video_completion": 0.72}
    # 商单示例: {"budget_level": 3, "brand_fit": 0.85}
    # 内容示例: {"sentiment": 0.7, "topic_relevance": 0.9}
    
    # 嵌入向量
    embedding: list[float] | None
    
    # 来源信息
    platform: str | None             # tiktok/instagram/x/wechat
    creator_id: str | None
    content_id: str | None
    
    # 时间
    captured_at: float = 0
    expires_at: float | None = 0
```

---

## 7. 核心算法设计

### 7.1 质心计算器 (CentroidCalculator)

```python
class CentroidCalculator:
    """人群兴趣质心计算器"""
    
    def compute_centroid(
        self, 
        user_signals: list[SignalFeature],
        weights: dict[str, float] | None = None
    ) -> list[float]:
        """
        计算用户兴趣质心向量
        
        方法：加权平均 + 时间衰减
        """
        if not user_signals:
            return []
        
        # 默认权重：最近的信号权重更高
        if weights is None:
            weights = self._compute_time_decay_weights(user_signals)
        
        # 加权平均
        centroid = np.zeros_like(user_signals[0].embedding)
        total_weight = 0
        
        for signal, weight in zip(user_signals, weights):
            if signal.embedding:
                centroid += np.array(signal.embedding) * weight
                total_weight += weight
        
        if total_weight > 0:
            centroid /= total_weight
        
        return centroid.tolist()
    
    def find_nearest_tags(
        self,
        centroid: list[float],
        tag_nodes: list[TagNode],
        top_k: int = 10
    ) -> list[tuple[TagNode, float]]:
        """找到与质心最近的标签"""
        scores = []
        for tag in tag_nodes:
            if tag.embedding:
                sim = cosine_similarity(centroid, tag.embedding)
                scores.append((tag, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 7.2 联想引擎 (AssociationEngine)

```python
class AssociationEngine:
    """5C联想引擎"""
    
    def __init__(self, tag_graph: TagAssociationGraph):
        self.tag_graph = tag_graph
        self.config = {
            "semantic_weight": 0.4,
            "cooccurrence_weight": 0.3,
            "rule_weight": 0.2,
            "time_decay_weight": 0.1,
            "min_score_threshold": 0.3,
            "max_results": 20
        }
    
    async def associate(
        self,
        query: TagQuery,
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[AssociationResult]:
        """
        执行联想匹配
        
        Args:
            query: 标签查询（包含部分5C维度）
            top_k: 返回Top-K结果
            filters: 过滤条件
            
        Returns:
            联想结果列表
        """
        results = []
        
        # 1. 获取源标签的嵌入向量
        source_embeddings = await self._encode_query(query)
        
        # 2. 并行执行多种联想策略
        tasks = [
            self._semantic_association(source_embeddings, query),
            self._cooccurrence_association(query),
            self._rule_based_association(query),
        ]
        
        strategy_results = await asyncio.gather(*tasks)
        
        # 3. 融合结果
        for strategy_result in strategy_results:
            results.extend(strategy_result)
        
        # 4. 去重、排序、截断
        results = self._deduplicate_and_rank(results)
        results = results[:top_k]
        
        return results
    
    async def complete_5c(
        self,
        partial_5c: dict[str, list[str] | None]
    ) -> list[Full5CResult]:
        """
        补全5C维度
        
        输入: 部分5C维度
        输出: 完整5C组合
        """
        # 识别缺失维度
        missing_dims = [k for k, v in partial_5c.items() if v is None]
        
        if not missing_dims:
            # 已完整，直接返回
            return [Full5CResult(**partial_5c)]
        
        # 对缺失维度进行联想填充
        results = []
        for dim in missing_dims:
            filled = await self._fill_dimension(dim, partial_5c)
            results.extend(filled)
        
        return results
```

### 7.3 时间衰减函数

```python
def compute_time_decay(
    last_updated: float,
    current_time: float | None = None,
    half_life_days: float = 30.0
) -> float:
    """
    指数时间衰减
    
    half_life_days: 半衰期天数（默认30天）
    返回: 0-1 的衰减系数
    """
    if current_time is None:
        current_time = time.time()
    
    days_elapsed = (current_time - last_updated) / (24 * 3600)
    decay = 0.5 ** (days_elapsed / half_life_days)
    
    return decay
```

---

## 8. 失效模式与交叉验证

### 8.1 失效模式矩阵

```
┌─────────────────────────────────────────────────────────────────┐
│                    失效模式分析矩阵                               │
├─────────────┬─────────────┬─────────────┬───────────────────────┤
│ 失效模式     │ 触发条件     │ 影响范围     │ 缓解策略              │
├─────────────┼─────────────┼─────────────┼───────────────────────┤
│ 冷启动失效   │ 新用户/新品类 │ 联想准确率   │ 默认画像+品类预填充    │
│ 跨域漂移     │ 兴趣≠行为    │ 推荐相关性   │ 消费反馈回路修正       │
│ 平台偏见     │ 单平台过拟合 │ 跨平台泛化   │ 平台独立权重+交叉验证  │
│ 标签膨胀     │ 标签爆炸     │ 图谱稀疏     │ 自动合并+数量上限      │
│ 时效性衰减   │ 过季/过气    │ 推荐陈旧度   │ 时间衰减+季节性降权    │
│ 语义模糊     │ 多义词/同义词 │ 匹配噪声     │ 上下文消歧+同义词合并  │
│ 数据稀疏     │ 低互动品类   │ 质心不准确   │ 迁移学习+外部知识注入  │
│ 冷启动商品   │ 新品无数据   │ 联想空洞     │ 属性迁移+相似品借力    │
└─────────────┴─────────────┴─────────────┴───────────────────────┘
```

### 8.2 交叉验证策略

#### 验证1：冷启动 vs 成熟用户
```
测试组A: 新注册用户（<7天，<10次互动）
测试组B: 活跃用户（>30天，>100次互动）

验证指标:
- 联想准确率 (Precision@5)
- 覆盖率 (能联想的标签比例)
- 用户满意度 (点击/转化率)

预期: 组A准确率比组B低15-20%，通过默认画像缓解
```

#### 验证2：单平台 vs 跨平台
```
测试组A: 仅使用TikTok数据
测试组B: 使用TikTok+Instagram数据

验证指标:
- 联想多样性 (推荐标签的覆盖广度)
- 跨平台一致性 (同一用户在不同平台的联想重叠度)
- 转化效率 (最终购买转化率)

预期: 组B多样性高20%，一致性>60%
```

#### 验证3：规则驱动 vs 数据驱动
```
测试组A: 纯规则联想（业务规则）
测试组B: 纯数据联想（协同过滤）
测试组C: 混合联想（规则+数据）

验证指标:
- 准确率
- 新颖性（推荐了多少用户没见过的标签）
- 业务合规性（是否符合品牌调性）

预期: 组C在准确率和合规性上最优，组B新颖性最高
```

### 8.3 监控指标

```python
MONITORING_METRICS = {
    # 准确率指标
    "precision_at_5": "联想Top-5的准确率",
    "precision_at_10": "联想Top-10的准确率",
    "ndcg_at_10": "归一化折损累积增益",
    
    # 覆盖率指标
    "tag_coverage": "被联想命中的标签比例",
    "domain_coverage": "各域标签的覆盖均衡度",
    "cold_start_coverage": "冷启动场景的覆盖率",
    
    # 性能指标
    "avg_latency_ms": "平均联想延迟",
    "p99_latency_ms": "P99延迟",
    "qps": "每秒查询数",
    
    # 业务指标
    "click_through_rate": "联想推荐点击率",
    "conversion_rate": "联想推荐转化率",
    "revenue_per_recommendation": "单次推荐收益",
    
    # 健康度指标
    "tag_freshness": "标签时效性得分",
    "association_density": "联想图密度",
    "embedding_quality": "嵌入向量质量（通过下游任务评估）"
}
```

---

## 9. 共性维度提炼

经过多维度交叉验证，提取以下6个共性维度：

### 9.1 维度1：语义相似度 (Semantic Similarity)

```
定义: 标签在语义空间中的距离
计算: 余弦相似度(cosine_similarity)
范围: 0-1
权重: 0.4 (最高)

应用场景:
- 货品域: "烟酰胺精华" ≈ "美白精华" (0.85)
- 人群域: "Z世代" ≈ "95后" (0.92)
- 平台域: "抖音" ≈ "TikTok" (0.95)
```

### 9.2 维度2：共现频率 (Co-occurrence Frequency)

```
定义: 两个标签在同一上下文中共同出现的频率
计算: P(A∩B) / P(A)P(B) (PMI)
范围: 0-1 (归一化后)
权重: 0.3

应用场景:
- 货品×人群: "精华" + "成分党" (高共现)
- 内容×平台: "测评" + "小红书" (高共现)
- 商品×转化: "口红" + "冲动购买" (高共现)
```

### 9.3 维度3：领域一致性 (Domain Consistency)

```
定义: 跨域标签在语义空间中的对齐程度
计算: 跨域嵌入对齐后的相似度
范围: 0-1
权重: 0.15

应用场景:
- 货品域"护肤" 与 人群域"美妆兴趣" 的对齐度
- 平台域"小红书" 与 人群域"一二线女性" 的对齐度
```

### 9.4 维度4：时效性 (Freshness)

```
定义: 标签的活跃度和新鲜度
计算: 指数衰减函数 (half_life=30天)
范围: 0-1
权重: 0.1

应用场景:
- 季节性标签: "防晒"在夏季权重高，冬季降权
- 热点标签: "某明星同款"随时间衰减
- 过季商品: "2024款"自动降权
```

### 9.5 维度5：置信度 (Confidence)

```
定义: 联想关系的可靠程度
计算: 基于数据量和一致性的贝叶斯估计
范围: 0-1
权重: 间接影响 (低置信度结果降权)

应用场景:
- 高数据量标签的联想更可信
- 多平台一致的联想更可靠
- 新标签的联想需要更多验证
```

### 9.6 维度6：转化信号 (Conversion Signal)

```
定义: 从内容互动到消费行为的转化强度
计算: 历史转化率 + 预测转化率
范围: 0-1
权重: 业务目标导向

应用场景:
- 评估联想结果的商业价值
- 优化推荐策略的ROI
- 衡量内容-商品匹配度
```

---

## 10. 实现路线图

### Phase 1: 数据模型重构 (Week 1-2)

**目标**: 建立三级标签数据模型，保持向后兼容

```python
# 核心任务
1. core/models.py 新增 TagNode, TagAssociation, CrossDomainLink, SignalFeature
2. core/tag_graph.py 实现标签图结构（复用 MemoryGraph 架构）
3. infrastructure/tag_database.py 实现标签持久化
4. 编写单元测试
```

**验收标准**:
- [ ] TagNode CRUD 测试通过
- [ ] TagAssociation CRUD 测试通过
- [ ] 与现有 Concept/Memory/Connection 兼容

### Phase 2: 三级标签域 (Week 3-4)

**目标**: 实现货品/人群/平台三个标签域的管理

```python
# 核心任务
1. intelligence/tag_system/commodity_domain.py
   - 品类树管理（三级类目）
   - 价格带/风格/场景/功效标签
   
2. intelligence/tag_system/crowd_domain.py
   - 人口统计标签
   - 消费行为标签
   - 内容偏好标签
   - 生命周期标签
   
3. intelligence/tag_system/platform_domain.py
   - 平台特征库
   - 算法参数
   - 人群特征
   - 电商路径
```

**验收标准**:
- [ ] 三个域的标签CRUD测试通过
- [ ] 品类树层级查询测试通过
- [ ] 平台特征查询测试通过

### Phase 3: 5C联想引擎 (Week 5-7)

**目标**: 实现核心联想匹配算法

```python
# 核心任务
1. intelligence/tag_system/association_engine.py
   - 单域查询
   - 双域交叉
   - 全链路5C匹配
   
2. intelligence/tag_system/centroid_calculator.py
   - 质心向量计算
   - 近邻标签查找
   
3. infrastructure/tag_embedding.py
   - 标签→向量映射
   - 多模态融合
```

**验收标准**:
- [ ] 单域查询延迟<50ms
- [ ] 5C匹配Precision@5>0.7
- [ ] 10万标签下性能测试通过

### Phase 4: 信号处理器 (Week 8-9)

**目标**: 集成社媒/商单/内容数据特征

```python
# 核心任务
1. intelligence/signal_processors/social_signal.py
   - TikTok数据解析
   - Instagram数据解析
   - X/Twitter数据解析
   
2. intelligence/signal_processors/commercial_signal.py
   - 商单机会特征提取
   - 品牌调性匹配
   
3. intelligence/signal_processors/content_signal.py
   - 内容主题提取
   - 情感分析
   - 视觉风格识别
```

**验收标准**:
- [ ] 社媒数据解析测试通过
- [ ] 商单特征提取测试通过
- [ ] 内容特征提取测试通过

### Phase 5: 召回引擎扩展 (Week 10)

**目标**: 将标签联想集成到现有召回框架

```python
# 核心任务
1. intelligence/recall/tag_association_recall.py
   - 标签联想召回策略
   
2. intelligence/recall/cross_domain_recall.py
   - 跨域联想召回策略
   
3. memory/memory_recall.py
   - 新增第6、7种召回策略
   - 调整权重配置
```

**验收标准**:
- [ ] 新召回策略与现有5种并行运行
- [ ] 召回融合测试通过
- [ ] A/B测试框架就绪

### Phase 6: API与配置 (Week 11)

**目标**: 提供完整的API接口和配置

```python
# 核心任务
1. api/tag_gateway.py
   - 标签CRUD API
   - 联想查询API
   - 批量导入/导出API
   
2. _conf_schema.json
   - 标签域开关
   - 联想阈值
   - 召回权重
   
3. main.py
   - 注册新命令
   - 注册LLM工具
```

**验收标准**:
- [ ] API端点测试通过
- [ ] 配置热更新测试通过
- [ ] 文档生成

---

## 11. API设计

### 11.1 标签管理API

```yaml
# 标签CRUD
POST   /api/tags                    # 创建标签
GET    /api/tags/{id}               # 获取标签
PUT    /api/tags/{id}               # 更新标签
DELETE /api/tags/{id}               # 删除标签
GET    /api/tags?domain=commodity   # 列表查询

# 标签层级
GET    /api/tags/{id}/children      # 获取子标签
GET    /api/tags/{id}/path          # 获取标签路径（到根）

# 批量操作
POST   /api/tags/batch              # 批量创建
PUT    /api/tags/batch              # 批量更新
```

### 11.2 联想查询API

```yaml
# 单域联想
POST   /api/associate/single
Body: {
  "domain": "commodity",
  "tags": ["烟酰胺精华", "美白"],
  "top_k": 10,
  "filters": {"platform": "tiktok_douyin"}
}

# 双域交叉
POST   /api/associate/cross
Body: {
  "source_domain": "crowd",
  "target_domain": "platform",
  "source_tags": ["Z世代", "美妆兴趣"],
  "top_k": 10
}

# 全链路5C匹配
POST   /api/associate/full-5c
Body: {
  "content": ["测评"],
  "channel": ["tiktok_douyin"],
  "crowd": null,
  "commodity": ["护肤", "精华"],
  "conversion": null,
  "top_k": 5
}

# 质心查询
POST   /api/centroid/compute
Body: {
  "user_id": "user_123",
  "signal_ids": ["sig_1", "sig_2", "sig_3"]
}

POST   /api/centroid/find-tags
Body: {
  "centroid": [0.1, 0.2, ...],
  "top_k": 10
}
```

### 11.3 信号导入API

```yaml
# 社媒数据导入
POST   /api/signals/social
Body: {
  "platform": "tiktok_douyin",
  "creator_id": "creator_123",
  "content_id": "video_456",
  "features": {
    "engagement_rate": 0.05,
    "video_completion": 0.72,
    "like_count": 1200,
    "comment_count": 89,
    "share_count": 45
  }
}

# 商单数据导入
POST   /api/signals/commercial
Body: {
  "brand_name": "某品牌",
  "budget_range": "50k-100k",
  "kpi": "roi_1:3",
  "brand_tone": ["年轻", "时尚"],
  "target_audience": ["Z世代", "女性"]
}

# 内容数据导入
POST   /api/signals/content
Body: {
  "content_type": "short_video",
  "topic": "护肤测评",
  "sentiment": 0.8,
  "visual_style": "clean_minimal",
  "text_tone": "professional"
}
```

---

## 12. 配置与部署

### 12.1 新增配置项

```json
{
  "tag_system_enabled": {
    "description": "启用标签联想系统",
    "type": "bool",
    "default": true
  },
  "tag_domains": {
    "description": "启用的标签域",
    "type": "object",
    "default": {
      "commodity": true,
      "crowd": true,
      "platform": true
    }
  },
  "association_weights": {
    "description": "联想权重配置",
    "type": "object",
    "default": {
      "semantic": 0.4,
      "cooccurrence": 0.3,
      "rule": 0.2,
      "time_decay": 0.1
    }
  },
  "tag_embedding_model": {
    "description": "标签嵌入模型",
    "type": "string",
    "default": "text-embedding-3-small"
  },
  "max_tags_per_domain": {
    "description": "每个域最大标签数",
    "type": "int",
    "default": 10000
  },
  "association_cache_ttl": {
    "description": "联想结果缓存TTL（秒）",
    "type": "int",
    "default": 3600
  },
  "cold_start_strategy": {
    "description": "冷启动策略",
    "type": "string",
    "options": ["default_profile", "category_prior", "popularity"],
    "default": "default_profile"
  }
}
```

### 12.2 依赖更新

```txt
# requirements.txt 新增
sentence-transformers>=2.2.0  # 标签语义嵌入
faiss-cpu>=1.7.0              # 向量索引（可选，大规模标签）
```

### 12.3 部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    部署架构                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ AstrBot主进程 │    │ 标签索引服务  │    │ 数据导入管道  │      │
│  │ (主插件)     │    │ (可选独立)   │    │ (定时任务)   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                    ┌──────────────┐                             │
│                    │  SQLite      │                             │
│                    │  (标签存储)   │                             │
│                    └──────────────┘                             │
│                             │                                   │
│                    ┌──────────────┐                             │
│                    │  LanceDB     │                             │
│                    │  (向量存储)   │                             │
│                    └──────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. 测试与验证

### 13.1 单元测试

```python
# test_tag_system.py
class TestTagNode:
    def test_create_tag(self): ...
    def test_tag_hierarchy(self): ...
    def test_tag_embedding(self): ...

class TestTagAssociation:
    def test_create_association(self): ...
    def test_association_strength(self): ...

class TestCommodityDomain:
    def test_category_tree(self): ...
    def test_price_band_tags(self): ...

class TestCrowdDomain:
    def test_demographics_tags(self): ...
    def test_interest_centroid(self): ...

class TestPlatformDomain:
    def test_platform_features(self): ...
    def test_algorithm_params(self): ...
```

### 13.2 集成测试

```python
# test_association.py
class TestAssociationEngine:
    def test_single_domain_query(self): ...
    def test_cross_domain_query(self): ...
    def test_full_5c_match(self): ...
    def test_cold_start_handling(self): ...

class TestSignalProcessors:
    def test_social_signal_processing(self): ...
    def test_commercial_signal_processing(self): ...
    def test_content_signal_processing(self): ...
```

### 13.3 性能测试

```python
# test_performance.py
class TestPerformance:
    def test_10k_tags_latency(self):
        """10万标签下查询延迟<50ms"""
        
    def test_concurrent_queries(self):
        """并发100查询的吞吐量"""
        
    def test_embedding_batch(self):
        """批量嵌入计算性能"""
```

### 13.4 准确率测试

```python
# 准确率评估
def evaluate_precision_at_k(
    test_cases: list[dict],
    engine: AssociationEngine,
    k: int = 5
) -> float:
    """
    人工标注100组联想结果，计算Precision@K
    
    测试用例格式:
    {
        "query": {"domain": "commodity", "tags": ["烟酰胺精华"]},
        "expected": ["Z世代", "成分党", "小红书", ...],
        "platform": "tiktok_douyin"
    }
    """
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        results = engine.associate(case["query"], top_k=k)
        predicted = [r.tag.name for r in results]
        
        # 计算交集
        intersection = set(predicted) & set(case["expected"])
        correct += len(intersection) / k
    
    return correct / total
```

---

## 14. 风险与缓解

### 14.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 嵌入模型质量不足 | 中 | 高 | 使用预训练模型 + 微调 |
| 图谱规模过大 | 低 | 中 | 分片存储 + 索引优化 |
| 跨域对齐困难 | 中 | 高 | 对比学习 + 共享嵌入空间 |
| 实时性要求高 | 中 | 中 | 缓存 + 异步预计算 |

### 14.2 业务风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 冷启动效果差 | 高 | 中 | 默认画像 + 迁移学习 |
| 标签体系变更 | 中 | 高 | 版本化 + 兼容层 |
| 数据质量问题 | 高 | 中 | 数据校验 + 异常检测 |
| 业务规则冲突 | 低 | 中 | 规则优先级 + 冲突检测 |

### 14.3 运营风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 标签膨胀 | 中 | 中 | 自动合并 + 人工审核 |
| 联想偏差 | 中 | 高 | 监控 + 人工干预 |
| 性能退化 | 低 | 高 | 基准测试 + 告警 |

---

## 附录

### A. 术语表

| 术语 | 定义 |
|------|------|
| 货品域 (Commodity Domain) | 商品相关的标签集合 |
| 人群域 (Crowd Domain) | 目标用户相关的标签集合 |
| 平台域 (Platform Domain) | 电商/社媒平台相关的标签集合 |
| 兴趣质心 (Interest Centroid) | 用户兴趣的向量表示 |
| 5C法则 | Content×Channel×Crowd×Commodity×Conversion |
| 联想匹配 (Association Matching) | 基于标签的跨域匹配 |
| 跨域关联 (Cross-Domain Link) | 不同域标签之间的关联 |

### B. 参考资料

1. 认知心理学：激活扩散理论 (Spreading Activation)
2. 推荐系统：协同过滤 + 内容过滤混合
3. 知识图谱：TransE/TransR 嵌入方法
4. 电商实践：阿里妈妈DIN模型思路

---

> **文档状态**: 初稿完成  
> **下一步**: 按Phase 1-6逐步实现  
> **维护者**: qa296  
