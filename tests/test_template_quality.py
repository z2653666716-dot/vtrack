"""用合成像素检查身体模板质量线索的范围、方向和异常输入处理。"""

import numpy as np

from app.template_quality import measure_body_quality


def test_quality_scores_are_bounded_and_overlap_lowers_quality() -> None:
    """同一清晰裁剪与旁人框重叠时，重叠风险分量和总分应降低。"""
    rng = np.random.default_rng(5)
    crop = rng.integers(0, 256, size=(180, 90, 3), dtype=np.uint8)
    target = (0.2, 0.2, 0.5, 0.8)

    clear = measure_body_quality(crop, target, 640, 480, [])
    overlapping = measure_body_quality(crop, target, 640, 480, [(0.4, 0.2, 0.7, 0.8)])

    assert all(0 <= value <= 1 for value in clear.values())
    assert overlapping["overlap"] < clear["overlap"]
    assert overlapping["quality"] < clear["quality"]


def test_small_blurry_edge_crop_receives_lower_quality() -> None:
    """小而模糊、贴边的裁剪应低于居中清晰裁剪的质量分数。"""
    rng = np.random.default_rng(8)
    clear_crop = rng.integers(0, 256, size=(220, 100, 3), dtype=np.uint8)
    blurry_crop = np.full((24, 12, 3), 128, dtype=np.uint8)

    clear = measure_body_quality(clear_crop, (0.2, 0.2, 0.5, 0.8), 640, 480, [])
    weak = measure_body_quality(blurry_crop, (0.0, 0.0, 0.1, 0.1), 640, 480, [])

    assert weak["size"] < clear["size"]
    assert weak["sharpness"] < clear["sharpness"]
    assert weak["edge"] == 0
    assert weak["quality"] < clear["quality"]


def test_invalid_crop_returns_finite_zero_scores() -> None:
    """空或含无穷值的裁剪不让非有限分数进入 EMA。"""
    assert measure_body_quality(np.empty((0, 0, 3)), (0.2, 0.2, 0.4, 0.8), 640, 480, []) == {
        "size": 0.0, "sharpness": 0.0, "edge": 0.0, "overlap": 0.0, "quality": 0.0
    }
    invalid = np.full((32, 16, 3), np.inf, dtype=np.float32)
    assert measure_body_quality(invalid, (0.2, 0.2, 0.4, 0.8), 640, 480, [])["quality"] == 0
