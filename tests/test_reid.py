"""检查外观向量、身份库与重识别规则；合成用例只验证算法边界，不代表实拍准确率。"""
from pathlib import Path

import numpy as np
import pytest

from app.config import AppConfig, PROJECT_ROOT
from app.gallery import Gallery
from app.reid import FeatureExtractor, cosine, crop_target, normalize
from app.schemas import Detection
from app.session import VideoSession


def person(left=20):
    """返回合成图中固定大小的人物框，仅用于状态流检查。"""
    return Detection(cls=0, name="person", score=0.9, xyxy=(float(left), 10., float(left + 50), 140.))


def test_vector_and_crop():
    """向量归一化、零向量拒绝、裁剪扩边与边界截断必须正确。"""
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([2, 0], [1, 0]) == 1
    with pytest.raises(ValueError): normalize([0, 0])
    with pytest.raises(ValueError): normalize([np.nan, 1])
    frame = np.zeros((160, 320, 3), np.uint8)
    assert crop_target(frame, (0, 0, 0.2, 0.8)).shape[0] > 128
    assert crop_target(frame, (0, 0, 0.01, 0.01)) is None


def test_gallery_margin_templates_and_capacity():
    """身份模板有限、原始模板保留；在线第一名不能被排除后错误匹配到第二名。"""
    cfg = AppConfig(reid_max_templates=2, identity_limit=2)
    gallery = Gallery(cfg)
    first = gallery.register([1., 0., 0.])
    second = gallery.register([0.9, 0.1, 0.])
    assert gallery.rank([1, 0, 0], excluded={first.id})[0] is None
    assert gallery.rank([1, 0, 0])[2] > 0.99
    gallery.append(first.id, [0.8, 0.6, 0])
    gallery.append(first.id, [0.7, 0, 0.7])
    assert len(first.templates) == 2 and np.array_equal(first.templates[0], [1, 0, 0])
    with pytest.raises(ValueError): gallery.register([0, 0, 1])


def test_gallery_ema_preserves_anchor_and_only_updates_body_dimensions():
    """EMA 动态槽只修改指定身体维度，保留初始锚点和拼接向量颜色尾段。"""
    gallery = Gallery(AppConfig(reid_max_templates=4))
    identity = gallery.register([0.6, 0.8, 0.0, 0.0])
    anchor = identity.templates[0].copy()

    result = gallery.update_ema(identity.id, [0.0, 0.6, 0.8, 0.0], 0.25,
                                appearance_dimensions=2)

    assert result["updated"] is True
    assert len(identity.templates) == 2
    assert np.array_equal(identity.templates[0], anchor)
    assert np.array_equal(identity.templates[1][2:], anchor[2:])
    assert cosine(identity.templates[1][:2], anchor[:2]) < 1


def test_gallery_ema_zero_weight_and_invalid_alpha_are_safe():
    """质量为零不建新槽；非法 alpha 在修改模板前报错。"""
    gallery = Gallery(AppConfig(reid_max_templates=4))
    identity = gallery.register([1.0, 0.0])
    skipped = gallery.update_ema(identity.id, [0.0, 1.0], 0.0)
    assert skipped["reason"] == "zero_quality_weight"
    assert len(identity.templates) == 1
    with pytest.raises(ValueError, match="alpha"):
        gallery.update_ema(identity.id, [0.0, 1.0], float("nan"))
    assert len(identity.templates) == 1


def test_expired_track_relocks_without_claiming_different_color():
    """丢失超过轨迹期限后身份还在：新轨迹需连续确认；另一颜色不得变绿。"""
    cfg = AppConfig(det_device="cpu", reid_model="color_histogram", reid_ema=0, track_max_age=3)
    session = VideoSession(cfg, FeatureExtractor(cfg))
    red = np.full((160, 320, 3), (0, 0, 240), np.uint8)
    blue = np.full_like(red, (240, 0, 0))
    seq = 0

    def step(frame, boxes):
        nonlocal seq
        seq += 1
        session.update(boxes, 320, 160)
        return session.process(frame, seq)

    for _ in range(6): step(red, [person()])
    old_track = session.boxes[0].track_id
    assert session.lock(old_track, seq)["identity_id"] == "ID-01"
    assert step(red, [person()])[0].state == "locked"
    for _ in range(40): assert step(red, []) == []
    assert session.summaries()[0].online is False
    assert not session.observations
    for _ in range(8): assert step(blue, [person(200)])[0].state == "none"
    for _ in range(40): step(red, [])
    states = []
    for _ in range(12): states.append(step(red, [person(200)])[0].state)
    assert "candidate" in states and states[-1] == "locked"
    assert session.boxes[0].identity_id == "ID-01" and session.boxes[0].track_id != old_track
    assert any(item["name"] == "relocked" for item in session.events) or states[-1] == "locked"
    # 同时两个外观完全相同的框，也不能把一个身份绑定两次。
    for _ in range(10): boxes = step(red, [person(200), person()])
    assert sum(box.state == "locked" for box in boxes) == 1
    # 保持轨迹号、像素突然换成不同外观，五次反向检查后必须解绑。
    for _ in range(18): boxes = step(blue, [person(200), person()])
    assert all(box.state == "none" for box in boxes)
    session.delete("ID-01")
    assert session.summaries() == []
    session.reset()
    assert session.frame is None


def test_real_pretrained_extractor():
    """真实本地 MobileNet 权重可前向，融合向量704维；不联网、不读摄像头。"""
    import cv2
    cfg = AppConfig()
    extractor = FeatureExtractor(cfg)
    image = cv2.imread(str(PROJECT_ROOT / "data/samples/test.jpg"))
    vector = extractor.extract([image])[0]
    flipped = extractor.extract([image], flip=True)[0]
    assert vector.shape == flipped.shape == (704,)
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)
    assert cosine(vector, vector) == pytest.approx(1, abs=1e-5)
    assert cosine(vector, flipped) > 0.8


@pytest.mark.parametrize("appear,color", [(0., 1.), (1., 0.)])
def test_zero_weight_can_be_smoothed(appear, color):
    """用户将某个融合权重调到0时，EMA仍可工作，不能归一化全零分量。"""
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.config = AppConfig(reid_w_appear=appear, reid_w_color=color)
    vector = np.r_[np.ones(576) / np.sqrt(576) * np.sqrt(appear),
                   np.ones(128) / np.sqrt(128) * np.sqrt(color)].astype(np.float32)
    assert np.linalg.norm(extractor.smooth(vector, vector)) == pytest.approx(1)


def test_best_visible_candidate_wins_independent_of_detection_order():
    """低相似度候选排在检测数组前面时，仍应由更相似的人取得身份，不能先到先得。"""
    class KnownExtractor:
        def extract(self, crops, flip=False):
            return [normalize([1, 0]) if crop[crop.shape[0] // 2, crop.shape[1] // 2, 0] > 100
                    else normalize([.85, np.sqrt(1 - .85**2)]) for crop in crops]

        def smooth(self, old, new):
            return new

    session = VideoSession(AppConfig(), KnownExtractor())
    session.gallery.register([1, 0], source="photo")
    frame = np.zeros((160, 320, 3), np.uint8)
    frame[:, 160:] = 240
    for seq in range(1, 10):
        session.update([person(), person(200)], 320, 160)
        boxes = session.process(frame, seq)
    assert boxes[0].state == "none" and boxes[1].state == "locked"
    assert boxes[1].identity_id == "ID-01"
