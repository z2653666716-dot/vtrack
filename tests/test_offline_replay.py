"""离线回放绑定规则测试，确保重入换轨迹号后按身份而非旧轨迹号判断。"""

from app.config import AppConfig
from app.hybrid_session import HybridSession
from app.pipeline import Pipeline
from app.schemas import Box
from scripts.evaluate_video import box_record, make_config, select_initialization_box


def _box(track_id: int, identity_id: str | None, state: str) -> Box:
    """构造纯数值测试框，模拟轨迹号变化或身份状态变化。"""
    return Box(cls=0, name="person", score=0.9, xyxy=(0.1, 0.1, 0.5, 0.9),
               track_id=track_id, identity_id=identity_id, state=state)


def test_identity_methods_follow_registered_identity_after_track_id_changes() -> None:
    """身份方案重入后可换轨迹号，但只有登记身份仍 locked 才绑定目标。"""
    rebound = box_record(_box(2, "ID-01", "locked"), target_track_id=1,
                         tracker_only=False, target_identity_id="ID-01")
    different_identity = box_record(_box(2, "ID-02", "locked"), target_track_id=1,
                                    tracker_only=False, target_identity_id="ID-01")
    candidate = box_record(_box(2, None, "candidate"), target_track_id=1,
                           tracker_only=False, target_identity_id="ID-01")

    assert rebound["bound_to_target"] is True
    assert different_identity["bound_to_target"] is False
    assert candidate["bound_to_target"] is False


def test_bytetrack_baseline_remains_bound_to_original_track_only() -> None:
    """ByteTrack 基线没有身份层，仍只按最初登记的轨迹号判断。"""
    same_track = box_record(_box(1, None, "none"), target_track_id=1,
                            tracker_only=True)
    changed_track = box_record(_box(2, "ID-01", "locked"), target_track_id=1,
                               tracker_only=True, target_identity_id="ID-01")

    assert same_track["bound_to_target"] is True
    assert changed_track["bound_to_target"] is False


def test_single_person_initialization_prefers_first_stable_track():
    """单人无标注回放优先从已有轨迹号的检测框初始化，避免选无轨迹候选框。"""
    untracked = Box(cls=0, name="person", score=0.99, xyxy=(0.0, 0.0, 0.4, 0.8),
                    track_id=None, identity_id=None, state="none")
    tracked = Box(cls=0, name="person", score=0.70, xyxy=(0.2, 0.1, 0.6, 0.9),
                  track_id=7, identity_id=None, state="none")

    assert select_initialization_box([untracked, tracked]) is tracked
    assert select_initialization_box([]) is None


def test_hybrid_face_ablation_changes_only_face_evidence_switch():
    """人脸消融组与Hybrid使用相同跟踪器、OSNet和配置阈值。"""
    base = AppConfig(det_device="cpu")
    full = make_config(base, "hybrid", None)
    no_face = make_config(base, "hybrid_no_face", None)

    assert full == no_face
    assert full.tracker == "botsort.yaml"
    assert full.identity_mode == "hybrid"
    assert full.reid_model == "osnet_x0_25"


def test_face_disabled_hybrid_skips_face_encoder_loading():
    """消融会话的帧准备与模型准备都不调用人脸编码器。"""
    class EmptyExtractor:
        person_cache = {}

    base = AppConfig(det_device="cpu", identity_mode="hybrid",
                     reid_model="osnet_x0_25", tracker="botsort.yaml")
    pipeline = Pipeline(detector=object(), config=base)
    pipeline.extractor = EmptyExtractor()
    session = HybridSession(base, use_face_evidence=False)

    pipeline.prepare_hybrid(session)
    assert session.face_encoder is None
    assert session.prepare_frame(None, []) == []
    assert session.detected_faces == []
