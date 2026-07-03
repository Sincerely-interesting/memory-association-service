"""
多模态编码器
基于 CLIP 的图文双编码，支持跨模态向量检索
"""

import io
import time
from typing import Optional, Union, List

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


# CLIP 默认向量维度
CLIP_VECTOR_DIM = 512

# 零向量常量，用于降级场景
ZERO_VECTOR = [0.0] * CLIP_VECTOR_DIM


class MultimodalEncoder:
    """多模态编码器 — 基于 CLIP 的图文联合编码

    使用 CLIP 模型将文本和图片映射到同一 512 维向量空间，
    支持跨模态相似度计算（文字查图片、图片查文字）。

    模型懒加载：首次调用 encode 时才加载，避免启动延迟。
    降级保护：CLIP 不可用时返回零向量，不影响系统正常运行。
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._load_failed = False
        self._load_attempted = False

    def _ensure_loaded(self) -> bool:
        """懒加载 CLIP 模型

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        if self._load_attempted:
            return False

        self._load_attempted = True

        try:
            logger.info(f"正在加载 CLIP 模型: {self.model_name} ...")
            start = time.time()

            from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model.eval()

            elapsed = time.time() - start
            logger.info(f"CLIP 模型加载完成，耗时 {elapsed:.1f}s")
            return True

        except ImportError as e:
            logger.warning(
                f"CLIP 依赖缺失 (transformers/torch)，多模态编码不可用: {e}"
            )
            self._load_failed = True
            return False
        except Exception as e:
            logger.warning(f"CLIP 模型加载失败，多模态编码不可用: {e}")
            self._load_failed = True
            return False

    def encode_text(self, text: str) -> list:
        """将文本编码为 512 维向量

        Args:
            text: 输入文本（中文/英文均可）

        Returns:
            512 维 float 列表，L2 归一化。加载失败时返回零向量。
        """
        if not text or not text.strip():
            return ZERO_VECTOR[:]

        if not self._ensure_loaded():
            return ZERO_VECTOR[:]

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=77,
                truncation=True,
                padding=True,
            )
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)

            # L2 归一化
            features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()

        except Exception as e:
            logger.warning(f"CLIP 文本编码失败: {e}")
            return ZERO_VECTOR[:]

    def encode_image(self, image_source: Union[str, bytes]) -> list:
        """将图片编码为 512 维向量

        Args:
            image_source: 图片来源，支持：
                - HTTP/HTTPS URL
                - 本地文件路径
                - bytes 数据

        Returns:
            512 维 float 列表，L2 归一化。加载失败时返回零向量。
        """
        if not image_source:
            return ZERO_VECTOR[:]

        if not self._ensure_loaded():
            return ZERO_VECTOR[:]

        try:
            import torch
            from PIL import Image

            image = self._load_image(image_source)
            if image is None:
                return ZERO_VECTOR[:]

            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)

            # L2 归一化
            features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()

        except Exception as e:
            logger.warning(f"CLIP 图片编码失败: {e}")
            return ZERO_VECTOR[:]

    def encode_image_batch(self, image_sources: list) -> list:
        """批量编码图片

        Args:
            image_sources: 图片来源列表

        Returns:
            向量列表，与输入一一对应
        """
        return [self.encode_image(src) for src in image_sources]

    def _load_image(self, image_source):
        """加载图片为 PIL Image

        Returns:
            PIL.Image (RGB) 或 None（加载失败时）
        """
        from PIL import Image

        try:
            if isinstance(image_source, bytes):
                return Image.open(io.BytesIO(image_source)).convert("RGB")

            if isinstance(image_source, str):
                if image_source.startswith(("http://", "https://")):
                    import requests

                    resp = requests.get(image_source, timeout=10)
                    resp.raise_for_status()
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    return Image.open(image_source).convert("RGB")

            return None

        except Exception as e:
            logger.debug(f"图片加载失败: {e}")
            return None

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return CLIP_VECTOR_DIM

    @property
    def is_available(self) -> bool:
        """检查编码器是否可用"""
        return self._ensure_loaded()

    def get_status(self) -> dict:
        """获取编码器状态"""
        return {
            "model_name": self.model_name,
            "dimension": CLIP_VECTOR_DIM,
            "is_available": self.is_available,
            "load_failed": self._load_failed,
            "load_attempted": self._load_attempted,
        }


def compute_combined_vector(
    text_vector: list[float],
    image_vector: Optional[list[float]],
    modality: str,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> list[float]:
    """计算图文融合向量

    基于 vector-knowledge 文章 (四) 的 WeightedAverageFusion 策略：
    加权平均 + L2 归一化

    Args:
        text_vector: 文本向量
        image_vector: 图片向量（可为 None）
        modality: 模态类型 "text" / "image" / "multimodal"
        text_weight: 文本权重（默认 0.5）
        image_weight: 图片权重（默认 0.5）

    Returns:
        融合向量，L2 归一化
    """
    try:
        import numpy as np
    except ImportError:
        # numpy 不可用时的降级处理
        if modality == "text" or image_vector is None:
            return text_vector
        if modality == "image":
            return image_vector
        # 简单平均（无归一化）
        return [(t + i) / 2 for t, i in zip(text_vector, image_vector)]

    if modality == "text" or image_vector is None:
        return text_vector

    if modality == "image":
        return image_vector

    # multimodal: 加权平均 + L2 归一化
    t = np.array(text_vector, dtype=np.float32)
    i = np.array(image_vector, dtype=np.float32)

    # 归一化权重
    total = text_weight + image_weight
    w_t = text_weight / total
    w_i = image_weight / total

    combined = w_t * t + w_i * i
    norm = np.linalg.norm(combined)

    if norm > 0:
        combined = combined / norm

    return combined.tolist()
