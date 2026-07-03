"""基础设施模块"""
from .tag_database import TagDatabase
from .tag_embedding import TagEmbeddingService

try:
    from .database import SmartDatabaseMigration
except ImportError:
    SmartDatabaseMigration = None

try:
    from .resources import resource_manager, ResourceManager
except ImportError:
    resource_manager = None
    ResourceManager = None

try:
    from .embedding import EmbeddingCacheManager
except ImportError:
    EmbeddingCacheManager = None

try:
    from .events import MemoryEventBus, MemoryEvent, MemoryEventType, get_event_bus, initialize_event_bus, shutdown_event_bus
except ImportError:
    MemoryEventBus = None
    MemoryEvent = None
    MemoryEventType = None
    get_event_bus = None
    initialize_event_bus = None
    shutdown_event_bus = None

# 多模态编码器和 LanceDB 存储（可选依赖，导入失败不影响其他功能）
try:
    from .multimodal_encoder import MultimodalEncoder, compute_combined_vector, CLIP_VECTOR_DIM, ZERO_VECTOR
except ImportError:
    MultimodalEncoder = None
    compute_combined_vector = None
    CLIP_VECTOR_DIM = 512
    ZERO_VECTOR = None

try:
    from .lancedb_store import LanceDBVectorStore
except ImportError:
    LanceDBVectorStore = None

__all__ = [
    'SmartDatabaseMigration', 'resource_manager', 'ResourceManager',
    'EmbeddingCacheManager',
    'MemoryEventBus', 'MemoryEvent', 'MemoryEventType',
    'get_event_bus', 'initialize_event_bus', 'shutdown_event_bus',
    'MultimodalEncoder', 'compute_combined_vector', 'CLIP_VECTOR_DIM', 'ZERO_VECTOR',
    'LanceDBVectorStore',
]
