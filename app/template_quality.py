"""从内存人物裁剪与检测框计算模板更新用的启发式观测质量分数。

本模块只返回有限数值，不保存图像或特征，也不判断身份。HybridSession 在已经确认
身份模板可更新后调用它；输出的 quality 只控制 EMA 权重，不代表身份概率。
"""

from __future__ import annotations

import math

import cv2
import numpy as np


QUALITY_VERSION = "body-quality-v1"
SHARPNESS_SCALE = 100.0
EDGE_MARGIN_RATIO = 0.02
SIZE_SATURATION_HEIGHT = 256.0


def _clip_unit(value: float) -> float:
    """把有限质量分量限制在 0～1，非有限输入按 0 处理。"""
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _intersection_over_target(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """计算两个归一化框的交集占第一个（目标）框面积的比例。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return _clip_unit(intersection / area) if area > 1e-12 else 1.0


def measure_body_quality(crop: np.ndarray, box_xyxy: tuple[float, ...],
                         frame_width: int, frame_height: int,
                         other_boxes: list[tuple[float, ...]]) -> dict[str, float]:
    """计算小尺寸、模糊、贴边、与旁人框重叠四个风险分量和几何平均总分。

    输入是只在内存中的 BGR 人物裁剪、归一化人体框、原帧尺寸及其他人物框。
    输出各个 [0,1] 分量；越高表示这一启发式线索下风险越低。裁剪非法、尺寸非法
    或像素异常时返回全零，调用方据此让 B2 的 alpha 归零。
    """
    zero = {"size": 0.0, "sharpness": 0.0, "edge": 0.0, "overlap": 0.0, "quality": 0.0}
    if (not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.size == 0
            or frame_width <= 0 or frame_height <= 0 or len(box_xyxy) != 4
            or not np.isfinite(crop).all()):
        return zero
    crop_height, crop_width = crop.shape[:2]
    if crop_width <= 0 or crop_height <= 0:
        return zero

    # OSNet 固定把高度缩放到 256；源高度远小于此值时，模型只能放大有限像素。
    size_quality = _clip_unit(crop_height / SIZE_SATURATION_HEIGHT)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_quality = _clip_unit(sharpness / (sharpness + SHARPNESS_SCALE))

    left, top, right, bottom = box_xyxy
    edge_distance = min(left * frame_width, top * frame_height,
                        (1.0 - right) * frame_width, (1.0 - bottom) * frame_height)
    edge_scale = max(1.0, EDGE_MARGIN_RATIO * min(frame_width, frame_height))
    edge_quality = _clip_unit(edge_distance / edge_scale)

    overlap_risk = max((_intersection_over_target(box_xyxy, other)
                        for other in other_boxes), default=0.0)
    overlap_quality = _clip_unit(1.0 - overlap_risk)

    # 几何平均会让任一明显风险降低总分；该值是固定规则的启发式分数，不是概率。
    components = (size_quality, sharpness_quality, edge_quality, overlap_quality)
    quality = math.prod(components) ** 0.25
    return {"size": size_quality, "sharpness": sharpness_quality,
            "edge": edge_quality, "overlap": overlap_quality,
            "quality": _clip_unit(quality)}
