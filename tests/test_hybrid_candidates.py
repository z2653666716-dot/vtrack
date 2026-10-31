"""多人移动时的候选确认与短暂观测缺失回归；只使用合成框、已知向量，不访问设备。

真实BoT-SORT推进轨迹，受控编码器精确安排可靠脸/模糊脸，验证计数规则而非识别准确率。
"""
from time import monotonic

import numpy as np
import pytest

from app.faces import FaceEncoder, FaceSample
from app.hybrid_session import HybridSession
from app.reid import normalize
from app.schemas import Detection
from tests.test_hybrid_identity import KnownBodies, body_vector, config


class TwoFaces:
    """输出两张各自归属明确的合成人脸；允许目标脸暂时模糊或换成明确的另一人。"""
    associate = staticmethod(FaceEncoder.associate)

    def __init__(self):
        self.target = normalize([1., 0.])
        self.other = normalize([0., 1.])
        self.reason = "READY"
        self.target_visible = True

    def read(self, frame, recognize=True):
        faces = [
            FaceSample((40, 15, 60, 45), self.target if recognize else None,
                       self.reason if recognize else "FACE_OBSERVATION_INTERVAL", .99),
            FaceSample((180, 15, 200, 45), self.other if recognize else None,
                       "READY" if recognize else "FACE_OBSERVATION_INTERVAL", .99),
        ]
        return faces if self.target_visible else faces[1:]


def advance(session, frame, seq, boxes=None):
    """两个身体框相交但脸分别在不重叠区域；经过真实prepare/update/process流程返回框。"""
    coordinates = [(20., 10., 140., 150.), (100., 10., 220., 150.)] if boxes is None else boxes
    detections = [Detection(cls=0, name="person", score=.95, xyxy=coords) for coords in coordinates]
    detections = session.prepare_frame(frame, detections)
    session.update(detections, frame.shape[1], frame.shape[0], frame)
    return session.process(frame, seq)


def stable_scene():
    """返回尚未登记的稳定两人场景，确保测试问题不来自新轨迹需要初始化。"""
    session = HybridSession(config(), KnownBodies(), TwoFaces())
    frame = np.full((160, 320, 3), 100, np.uint8)
    for seq in range(1, 7):
        advance(session, frame, seq)
    return session, frame


def test_overlapping_bodies_with_distinct_faces_keep_bound_identity():
    """身体框相交不等于人脸归属不清；间隔提脸的帧不应让目标每次刚绿就解除。"""
    session, frame = stable_scene()
    target = session.boxes[0].track_id
    session.lock(target, session.seq, "lock")
    for seq in range(7, 19):
        result = advance(session, frame, seq)
        matched = next(box for box in result if box.track_id == target)
        assert matched.state == "locked" and matched.identity_id == "ID-01"
        assert sum(box.state == "locked" for box in result) == 1


def test_candidate_retains_actual_evidence_across_short_blur_gaps():
    """每次可靠脸之间夹一帧模糊，仍须取得六份真实成功证据；模糊帧不能凑数。"""
    session, frame = stable_scene()
    session.register_identity(body_vector(), face_feature=normalize([1., 0.]))
    target = session.boxes[0].track_id
    for index in range(11):
        good = index % 2 == 0
        session.face_encoder.target = normalize([1., 0.]) if good else None
        session.face_encoder.reason = "READY" if good else "FACE_BLURRED"
        result = advance(session, frame, 7 + index)
        matched = next(box for box in result if box.track_id == target)
        expected_hits = index // 2 + 1
        assert session.observations[target].candidate_hits == expected_hits
        assert matched.state == ("locked" if index == 10 else "candidate")
        assert all(box.state != "locked" for box in result if box.track_id != target)


def test_candidate_contradiction_and_long_gap_reset_progress():
    """明确异人脸立即清零；连续三帧无可靠观测也清零，不能无限累积历史成功。"""
    session, frame = stable_scene()
    session.register_identity(body_vector(), face_feature=normalize([1., 0.]))
    target = session.boxes[0].track_id
    for seq in range(7, 12):
        advance(session, frame, seq)
    assert session.observations[target].candidate_hits == 5
    session.face_encoder.target = normalize([0., 1.])
    assert all(box.state != "locked" for box in advance(session, frame, 12))
    assert session.observations[target].candidate_hits == 0
    session.face_encoder.target = normalize([1., 0.])
    advance(session, frame, 13)
    assert session.observations[target].candidate_hits == 1
    session.face_encoder.target = None
    session.face_encoder.reason = "FACE_BLURRED"
    for seq in range(14, 17):
        advance(session, frame, seq)
    assert session.observations[target].candidate_hits == 0


def test_head_only_uncertain_face_has_bounded_continuity():
    """只有头部时没有身体向量可用；中间分数短暂下降可等待，超过期限仍必须撤绿。"""
    session, frame = stable_scene()
    target = session.boxes[0].track_id
    session.lock(target, session.seq, "lock")
    session.face_only_indices = {0}
    # 本测试直接给当前帧证据，避免prepare_frame的每三帧策略掩盖这里要验证的状态条件。
    session.face_encoder.target = normalize([.3, np.sqrt(1 - .3 ** 2)])
    session.detected_faces = session.face_encoder.read(frame)
    session.face_only_ids = {target}
    session.observations[target].motion_ok = True
    result = session.process(frame, 7)
    assert next(box for box in result if box.track_id == target).state == "locked"
    session.observations[target].last_verified = monotonic() - 3
    session.detected_faces = session.face_encoder.read(frame)
    result = session.process(frame, 8)
    assert next(box for box in result if box.track_id == target).state != "locked"


def test_competing_faces_and_elapsed_gap_cannot_use_old_progress(monkeypatch):
    """两人同分不能沿用已有五次强行胜出；即使只缺一帧，时间过长也不能保留陈旧证据。"""
    session, frame = stable_scene()
    session.register_identity(body_vector(), face_feature=normalize([1., 0.]))
    target = session.boxes[0].track_id
    for seq in range(7, 12):
        advance(session, frame, seq)
    assert session.observations[target].candidate_hits == 5
    session.face_encoder.other = normalize([1., 0.])
    assert all(box.state == "none" for box in advance(session, frame, 12))
    assert session.observations[target].candidate_hits == 0
    session.face_encoder.other = normalize([0., 1.])
    advance(session, frame, 13)
    previous = session.observations[target].candidate_last_positive
    session.face_encoder.target = None
    session.face_encoder.reason = "FACE_BLURRED"
    session.clock = lambda: previous + 3
    assert all(box.state != "locked" for box in advance(session, frame, 14))
    assert session.observations[target].candidate_hits == 0


@pytest.mark.parametrize("missing_count", [1, 2, 3])
def test_brief_missing_track_keeps_only_recent_actual_confirmation(missing_count):
    """短漏检回来仍需原六份证据；超过两帧的断点不得复用五次旧候选。"""
    session, frame = stable_scene()
    session.register_identity(body_vector(), face_feature=normalize([1., 0.]))
    target = session.boxes[0].track_id
    for seq in range(7, 12):
        advance(session, frame, seq)
    assert session.observations[target].candidate_hits == 5
    session.face_encoder.target_visible = False
    for offset in range(missing_count):
        advance(session, frame, 12 + offset, [(100., 10., 220., 150.)])
    session.face_encoder.target_visible = True
    result = advance(session, frame, 12 + missing_count)
    if missing_count <= 2:
        assert any(box.track_id == target and box.state == "locked" for box in result)
    else:
        assert all(box.state != "locked" for box in result)


class DistinctBodies:
    """用合成图蓝/红区域区分两个已知身体向量，验证当前身体竞争，不代表视觉准确率。"""
    def extract(self, crops, flip=False):
        return [body_vector(1. if crop[:, :, 0].mean() > crop[:, :, 2].mean() else .1) for crop in crops]


def test_unique_body_can_wait_briefly_during_face_overlap(monkeypatch):
    """脸归属暂时歧义，但运动连贯、身体明显领先旁人时可短等；不得靠身体无限延长。"""
    session = HybridSession(config(), DistinctBodies(), TwoFaces())
    frame = np.full((160, 320, 3), 100, np.uint8)
    frame[:, :100], frame[:, 140:] = (200, 40, 20), (20, 40, 200)
    for seq in range(1, 7):
        advance(session, frame, seq)
    target = session.boxes[0].track_id
    session.lock(target, session.seq, "lock")
    verified = session.observations[target].last_verified
    # 两张脸都落在两个身体框的交叠区：归属确实有歧义，不能直接把其中任一脸分给目标。
    monkeypatch.setattr(session.face_encoder, "read", lambda frame, recognize=True: [
        FaceSample((105, 15, 125, 45), normalize([1., 0.]), "READY", .99),
        FaceSample((115, 15, 135, 45), normalize([0., 1.]), "READY", .99),
    ])
    result = advance(session, frame, 7)
    assert next(box for box in result if box.track_id == target).state == "locked"
    assert session.observations[target].last_verified == verified
    session.clock = lambda: verified + 3
    assert all(box.state != "locked" for box in advance(session, frame, 8))


def test_distant_body_lock_can_learn_first_face_across_sampling_intervals():
    """初始只有身体模板，走近后每三帧才提一次脸；跳过的帧不应把六次可靠采样反复清零。"""
    session = HybridSession(config(), DistinctBodies(), TwoFaces())
    frame = np.full((160, 320, 3), 100, np.uint8)
    frame[:, :100], frame[:, 140:] = (200, 40, 20), (20, 40, 200)
    session.face_encoder.target = None
    session.face_encoder.reason = "FACE_TOO_SMALL"
    for seq in range(1, 7):
        advance(session, frame, seq)
    target = session.boxes[0].track_id
    session.lock(target, session.seq, "lock")
    assert not session.face_templates["ID-01"]
    session.face_encoder.target = normalize([1., 0.])
    session.face_encoder.reason = "READY"
    for seq in range(7, 22):
        result = advance(session, frame, seq)
        assert next(box for box in result if box.track_id == target).state == "locked"
        assert not session.face_templates["ID-01"]  # 前五份可靠采样，间隔帧不能凑数。
    advance(session, frame, 22)
    assert len(session.face_templates["ID-01"]) == 1
    assert session.observations[target].face_seed is None


def test_inconsistent_face_learning_does_not_mix_people_and_cancel_clears_seed():
    """暂存脸样本彼此不一致时重新计数；取消必须清掉尚未登记的临时脸特征。"""
    session = HybridSession(config(), DistinctBodies(), TwoFaces())
    frame = np.full((160, 320, 3), 100, np.uint8)
    frame[:, :100], frame[:, 140:] = (200, 40, 20), (20, 40, 200)
    session.face_encoder.target = None
    for seq in range(1, 7):
        advance(session, frame, seq)
    target = session.boxes[0].track_id
    session.lock(target, session.seq, "lock")
    for seq in range(7, 23):
        session.face_encoder.target = normalize([1., 0.] if ((seq - 7) // 3) % 2 == 0 else [0., 1.])
        advance(session, frame, seq)
    assert not session.face_templates["ID-01"]
    assert session.observations[target].face_seed is not None
    assert session.observations[target].face_learning == 1
    session.clear_lock()
    assert session.observations[target].face_seed is None
