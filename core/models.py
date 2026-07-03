"""
数据模型定义
包含记忆系统的核心数据结构：
- Legacy: Concept, Memory, Connection
- Tag System: TagNode, TagAssociation, CrossDomainLink, SignalFeature
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Concept:
    """概念节点"""

    id: str
    name: str
    created_at: float = None
    last_accessed: float = None
    access_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_accessed is None:
            self.last_accessed = time.time()


@dataclass
class Memory:
    """记忆条目"""

    id: str
    concept_id: str
    content: str
    details: str = ""  # 详细描述
    participants: str = ""  # 参与者
    location: str = ""  # 地点
    emotion: str = ""  # 情感
    tags: str = ""  # 标签
    created_at: float = None
    last_accessed: float = None
    access_count: int = 0
    strength: float = 1.0
    allow_forget: bool = True
    group_id: str = ""  # 群组ID，用于群聊隔离

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_accessed is None:
            self.last_accessed = time.time()
        if self.allow_forget is None:
            self.allow_forget = True


@dataclass
class Connection:
    """概念之间的连接"""

    id: str
    from_concept: str
    to_concept: str
    strength: float = 1.0
    last_strengthened: float = None

    def __post_init__(self):
        if self.last_strengthened is None:
            self.last_strengthened = time.time()


# ============================================================
# 标签联想系统数据模型 (Phase 1)
# ============================================================

# 标签域常量
DOMAIN_COMMODITY = "commodity"   # 货品域
DOMAIN_CROWD = "crowd"           # 人群域
DOMAIN_PLATFORM = "platform"     # 平台域
ALL_DOMAINS = (DOMAIN_COMMODITY, DOMAIN_CROWD, DOMAIN_PLATFORM)

# 标签层级常量
LEVEL_CATEGORY_1 = 1   # 一级类目
LEVEL_CATEGORY_2 = 2   # 二级类目
LEVEL_CATEGORY_3 = 3   # 三级类目

# 联想类型常量
ASSOC_SEMANTIC = "semantic"             # 语义相似
ASSOC_COOCCURRENCE = "cooccurrence"     # 共现关系
ASSOC_HIERARCHICAL = "hierarchical"     # 层级包含
ASSOC_CROSS_DOMAIN = "cross_domain"     # 跨域关联
ALL_ASSOC_TYPES = (
    ASSOC_SEMANTIC, ASSOC_COOCCURRENCE,
    ASSOC_HIERARCHICAL, ASSOC_CROSS_DOMAIN,
)

# 跨域规则类型常量
RULE_CONSTRAINT = "constraint"   # 约束规则
RULE_BOOST = "boost"             # 加成规则
RULE_FILTER = "filter"           # 过滤规则
ALL_RULE_TYPES = (RULE_CONSTRAINT, RULE_BOOST, RULE_FILTER)

# 信号来源常量
SIGNAL_SOCIAL = "social"           # 社媒数据
SIGNAL_COMMERCIAL = "commercial"   # 商单机会
SIGNAL_CONTENT = "content"         # 内容特征
ALL_SIGNAL_SOURCES = (SIGNAL_SOCIAL, SIGNAL_COMMERCIAL, SIGNAL_CONTENT)


@dataclass
class TagNode:
    """三级标签节点

    层级结构示例:
        货品域 → 美妆护肤(1级) → 护肤(2级) → 精华(3级) → 烟酰胺精华(叶)
        人群域 → Z世代(1级) → ...
        平台域 → tiktok_douyin(1级) → ...
    """
    id: str                          # 唯一标识 tag_{domain}_{hash}
    name: str                        # 标签名称（中文）
    name_en: str                     # 英文名称（用于嵌入计算）

    # 层级信息
    level: int = LEVEL_CATEGORY_1    # 1=一级, 2=二级, 3=三级
    domain: str = DOMAIN_COMMODITY   # commodity/crowd/platform
    parent_id: Optional[str] = None     # 父标签ID（顶级为None）

    # 语义信息
    embedding: Optional[list[float]] = None   # 语义嵌入向量
    description: str = ""                  # 标签描述
    synonyms: list[str] = field(default_factory=list)  # 同义词列表

    # 统计信息
    usage_count: int = 0             # 使用次数
    last_used: float = 0.0           # 最后使用时间戳
    created_at: float = 0.0          # 创建时间戳

    # 元数据（灵活扩展）
    metadata: dict[str, Any] = field(default_factory=dict)
    # 例: {"platform": "tiktok", "category": "beauty", "price_range": "mid"}

    def __post_init__(self):
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_used == 0.0:
            self.last_used = now
        # 保证 synonyms 是列表
        if isinstance(self.synonyms, str):
            self.synonyms = [s.strip() for s in self.synonyms.split(",") if s.strip()]

    @property
    def is_root(self) -> bool:
        """是否为顶级标签"""
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        """是否为叶级标签（level=3）"""
        return self.level == LEVEL_CATEGORY_3

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "name_en": self.name_en,
            "level": self.level,
            "domain": self.domain,
            "parent_id": self.parent_id,
            "description": self.description,
            "synonyms": self.synonyms,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class TagAssociation:
    """标签联想关系

    表示两个标签之间的联想关联，支持多种联想类型和上下文感知。
    """
    id: str                          # 唯一标识 assoc_{src}_{tgt}
    source_tag_id: str               # 源标签ID
    target_tag_id: str               # 目标标签ID

    # 联想强度
    strength: float = 0.5            # 联想强度 0-1
    confidence: float = 0.5          # 置信度 0-1

    # 联想类型
    association_type: str = ASSOC_SEMANTIC
    # semantic/cooccurrence/hierarchical/cross_domain

    # 上下文
    context: dict[str, Any] = field(default_factory=dict)
    # 例: {"platform": "tiktok", "time_decay": 0.95}

    # 统计
    hit_count: int = 0               # 命中次数
    last_hit: float = 0.0            # 最后命中时间

    # 时间
    created_at: float = 0.0
    last_strengthened: float = 0.0

    def __post_init__(self):
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_strengthened == 0.0:
            self.last_strengthened = now
        if self.last_hit == 0.0:
            self.last_hit = now

    @property
    def is_cross_domain(self) -> bool:
        """是否为跨域关联"""
        return self.association_type == ASSOC_CROSS_DOMAIN

    def strengthen(self, delta: float = 0.1):
        """强化联想关系"""
        self.strength = min(1.0, self.strength + delta)
        self.last_strengthened = time.time()
        self.hit_count += 1
        self.last_hit = time.time()

    def weaken(self, delta: float = 0.05):
        """弱化联想关系"""
        self.strength = max(0.0, self.strength - delta)
        self.last_strengthened = time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "source_tag_id": self.source_tag_id,
            "target_tag_id": self.target_tag_id,
            "strength": self.strength,
            "confidence": self.confidence,
            "association_type": self.association_type,
            "context": self.context,
            "hit_count": self.hit_count,
            "last_hit": self.last_hit,
            "created_at": self.created_at,
            "last_strengthened": self.last_strengthened,
        }


@dataclass
class CrossDomainLink:
    """跨域关联规则

    定义不同标签域之间的业务规则，用于约束、加成或过滤联想结果。
    例: "IF crowd.gen_z AND commodity.budget THEN platform.tiktok BOOST 0.3"
    """
    id: str                          # 唯一标识
    source_domain: str               # 源域 (commodity/crowd/platform)
    target_domain: str               # 目标域

    # 关联规则
    rule_type: str = RULE_BOOST      # constraint/boost/filter
    rule_expression: str = ""        # 规则表达式（人类可读）

    # 权重
    weight: float = 1.0              # 规则权重
    enabled: bool = True             # 是否启用

    # 统计
    trigger_count: int = 0           # 触发次数
    last_triggered: float = 0.0      # 最后触发时间

    # 时间
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def trigger(self):
        """记录规则触发"""
        self.trigger_count += 1
        self.last_triggered = time.time()

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "rule_type": self.rule_type,
            "rule_expression": self.rule_expression,
            "weight": self.weight,
            "enabled": self.enabled,
            "trigger_count": self.trigger_count,
            "last_triggered": self.last_triggered,
            "created_at": self.created_at,
        }


@dataclass
class SignalFeature:
    """信号特征（社媒/商单/内容）

    统一的数据接口，用于接收外部系统的特征数据。
    """
    id: str                          # 唯一标识
    source: str                      # social/commercial/content

    # 特征键值对
    features: dict[str, float] = field(default_factory=dict)
    # 社媒示例: {"engagement_rate": 0.05, "video_completion": 0.72}
    # 商单示例: {"budget_level": 3, "brand_fit": 0.85}
    # 内容示例: {"sentiment": 0.7, "topic_relevance": 0.9}

    # 嵌入向量
    embedding: Optional[list[float]] = None

    # 来源信息
    platform: Optional[str] = None      # tiktok/instagram/x/wechat
    creator_id: Optional[str] = None
    content_id: Optional[str] = None

    # 时间
    captured_at: float = 0.0
    expires_at: Optional[float] = None

    def __post_init__(self):
        if self.captured_at == 0.0:
            self.captured_at = time.time()

    @property
    def is_expired(self) -> bool:
        """是否已过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def get_feature(self, key: str, default: float = 0.0) -> float:
        """获取单个特征值"""
        return self.features.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "source": self.source,
            "features": self.features,
            "platform": self.platform,
            "creator_id": self.creator_id,
            "content_id": self.content_id,
            "captured_at": self.captured_at,
            "expires_at": self.expires_at,
        }
