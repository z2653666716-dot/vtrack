"""验证 B0/B1/B2 的模板存储行为、权重差异和匿名数值诊断。"""

import numpy as np
import pytest

from app.config import AppConfig
from app.hybrid_session import HybridSession
from app.reid import normalize


def make_session(strategy: str) -> tuple[HybridSession, str]:
    """建立不加载模型的 HybridSession，并登记可手算的 640 维模板。"""
    config = AppConfig(identity_mode="hybrid", reid_model="osnet_x0_25",
                       tracker="botsort.yaml", reid_max_templates=5)
    session = HybridSession(config, extractor=object(), face_encoder=object(),
                            clock=lambda: 2.0, body_template_update=strategy,
                            collect_template_diagnostics=True)
    anchor = np.zeros(640, dtype=np.float32)
    anchor[0], anchor[512] = 1.0, 0.2
    identity = session.gallery.register(normalize(anchor))
    session.body_valid[identity.id] = True
    session.seq = 12
    session.current_body_quality[7] = {
        "size": 0.5, "sharpness": 0.6, "edge": 0.7, "overlap": 0.8, "quality": 0.25,
    }
    return session, identity.id


def make_feature() -> np.ndarray:
    """返回与锚点身体方向不同、颜色尾段固定的合成 OSNet+颜色向量。"""
    feature = np.zeros(640, dtype=np.float32)
    feature[1], feature[512] = 1.0, 0.2
    return normalize(feature)


def test_quality_ema_uses_same_maximum_as_fixed_ema_but_scales_by_quality() -> None:
    """B1/B2 仅 alpha 不同；B2 的低质量观测只获得固定上限的四分之一权重。"""
    fixed, fixed_id = make_session("fixed_ema")
    quality, quality_id = make_session("quality_ema")

    fixed._update_body_template(7, fixed_id, make_feature(), "face_confirmed")
    quality._update_body_template(7, quality_id, make_feature(), "face_confirmed")

    fixed_record = fixed.template_update_records[-1]
    quality_record = quality.template_update_records[-1]
    assert fixed_record["candidate_alpha"] == pytest.approx(0.1)
    assert fixed_record["alpha"] == pytest.approx(0.1)
    assert quality_record["candidate_alpha"] == pytest.approx(0.025)
    assert quality_record["alpha"] == pytest.approx(0.025)
    assert fixed_record["updated"] is quality_record["updated"] is True
    assert len(fixed.gallery.items[fixed_id].templates) == 2
    assert len(quality.gallery.items[quality_id].templates) == 2


def test_quality_ema_zero_quality_keeps_templates_unchanged_and_logs_no_vectors() -> None:
    """质量为零时不建动态模板；帧级诊断只含数值与原因，不含媒体或向量。"""
    session, identity_id = make_session("quality_ema")
    session.current_body_quality[7]["quality"] = 0.0
    original = session.gallery.items[identity_id].templates[0].copy()

    changed = session._update_body_template(7, identity_id, make_feature(), "face_confirmed")

    record = session.template_update_records[-1]
    assert changed is False
    assert record["reason"] == "zero_quality_weight"
    assert record["alpha"] == 0
    assert len(session.gallery.items[identity_id].templates) == 1
    assert np.array_equal(session.gallery.items[identity_id].templates[0], original)
    assert not any("feature" in key or "crop" in key or "image" in key
                   for key in record)
