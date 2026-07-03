"""智能能力模块"""
try:
    from .topic_analyzer import TopicAnalyzer
except (ImportError, TypeError):
    TopicAnalyzer = None

try:
    from .profiling import UserProfilingSystem
except (ImportError, TypeError):
    UserProfilingSystem = None

try:
    from .temporal import TemporalMemorySystem
except (ImportError, TypeError):
    TemporalMemorySystem = None

from .tag_system import (
    CommodityDomain, CrowdDomain, PlatformDomain,
    CentroidCalculator, AssociationEngine, TagQuery, AssociationResult, Full5CResult,
)
from .signal_processors import (
    SocialSignalProcessor, CommercialSignalProcessor, ContentSignalProcessor,
)
from .recall import TagAssociationRecall, CrossDomainRecall

__all__ = [
    'TopicAnalyzer', 'UserProfilingSystem', 'TemporalMemorySystem',
    'CommodityDomain', 'CrowdDomain', 'PlatformDomain',
    'CentroidCalculator', 'AssociationEngine', 'TagQuery', 'AssociationResult', 'Full5CResult',
    'SocialSignalProcessor', 'CommercialSignalProcessor', 'ContentSignalProcessor',
    'TagAssociationRecall', 'CrossDomainRecall',
]
