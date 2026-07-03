"""
信号处理器模块

提供外部数据源的标准化处理:
- SocialSignalProcessor: 社媒数据 (TikTok/Instagram/X)
- CommercialSignalProcessor: 商单机会数据
- ContentSignalProcessor: 内容特征数据

所有处理器输出统一的 SignalFeature 对象，供联想引擎使用。
"""
from __future__ import annotations

from .social_signal import SocialSignalProcessor
from .commercial_signal import CommercialSignalProcessor
from .content_signal import ContentSignalProcessor

__all__ = [
    "SocialSignalProcessor",
    "CommercialSignalProcessor",
    "ContentSignalProcessor",
]
