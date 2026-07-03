"""API模块"""
from .tag_gateway import TagAPIGateway

try:
    from .gateway import MemoryAPIGateway, APIResponse, PerformanceMonitor
except (ImportError, TypeError):
    MemoryAPIGateway = None
    APIResponse = None
    PerformanceMonitor = None

__all__ = ['TagAPIGateway', 'MemoryAPIGateway', 'APIResponse', 'PerformanceMonitor']
