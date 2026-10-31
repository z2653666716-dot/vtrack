"""离线回放绑定规则测试，确保重入换轨迹号后按身份而非旧轨迹号判断。"""

from app.schemas import Box
from scripts.evaluate_video import box_record


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
