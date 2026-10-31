"""人脸证据消融报告测试：校验对照公平性和离场帧分母处理。"""

import pytest

from scripts.summarize_ablation import build_ablation_report, parse_absent_ranges


def _prediction(variant: str, face_enabled: bool, bound_frames: set[int]) -> dict:
    """生成同视频、同检测/跟踪和同初始化的纯数值预测样本。"""
    rows = []
    for frame in range(4):
        is_bound = frame in bound_frames
        rows.append({
            "frame_index": frame,
            "video_time_seconds": frame / 30,
            "boxes": [{"track_id": 1, "bbox_xyxy": [0.1, 0.1, 0.5, 0.9],
                       "bound_to_target": is_bound,
                       "state": "locked" if is_bound else "none"}],
        })
    return {
        "schema_version": 1,
        "video": {"id": "clip", "sha256": "a" * 64, "reported_frame_count": 4},
        "variant": variant,
        "registration": {"frame_index": 0, "effective_frame_index": 1, "track_id": 1},
        "run": {
            "config": {
                "tracker": "botsort.yaml", "identity_mode": "hybrid", "reid_model": "osnet_x0_25",
                "det_imgsz": 640, "max_side": 640, "det_conf": 0.35, "det_iou": 0.5,
                "det_device": 0, "identity_device": "cpu", "track_max_age": 30,
                "body_match_threshold": 0.75, "body_hold_threshold": 0.55,
                "reid_confirm_frames": 6, "face_evidence_enabled": face_enabled,
            },
            "model_sha256": {"detector": "det", "body_reid": "body",
                             **({"face_detector": "face-det", "face_recognizer": "face-rec"} if face_enabled else {})},
        },
        "predictions": rows,
    }


def test_ablation_report_excludes_absent_frames_from_visible_rate():
    """人工标为离场的帧不进入可见绑定覆盖率分母，并单独计绑定状态。"""
    full = _prediction("hybrid", True, {0, 1, 2, 3})
    no_face = _prediction("hybrid_no_face", False, {0, 2, 3})

    summary, rows = build_ablation_report(full, no_face, [(2, 3)])

    assert summary["visible_frame_count"] == 3
    assert summary["metrics"]["hybrid"]["bound_visible_frames"] == 3
    assert summary["metrics"]["hybrid_no_face"]["bound_visible_frames"] == 2
    assert summary["metrics"]["hybrid"]["bound_during_manual_absence"] == 1
    assert summary["comparison"]["geometry_and_track_identical_frames"] == 4
    assert summary["comparison"]["bound_flag_difference_frames"] == [1]
    assert len(rows) == 4


def test_absent_range_validation_rejects_overlap_and_bad_boundaries():
    """离场区间采用左闭右开，重复覆盖或越出预测帧范围时拒绝计算。"""
    assert parse_absent_ranges(["1:3", "4:5"], 6) == [(1, 3), (4, 5)]
    with pytest.raises(ValueError, match="不能互相重叠"):
        parse_absent_ranges(["1:4", "3:5"], 6)
    with pytest.raises(ValueError, match="超出有效范围"):
        parse_absent_ranges(["2:7"], 6)
