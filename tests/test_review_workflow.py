"""离线视频复核流程测试：问题备注与定量评测标注彼此独立。"""

from scripts.review_video import (
    build_review_notes_payload,
    choose_automatic_registration,
    prediction_binding_label,
)


def test_review_notes_export_does_not_require_boxes_or_registration():
    """只写问题备注就能导出；结果不含人物框、登记操作或媒体内容。"""
    payload = build_review_notes_payload(
        {"id": "clip", "sha256": "a" * 64},
        {"file_name": "clip_hybrid_predictions.json", "variant": "hybrid"},
        {
            83: {"frame_index": 83, "video_time_seconds": 2.77,
                 "issue_type": "identity_lost", "review_note": "开始移动后未绑定"},
            90: {"frame_index": 90, "video_time_seconds": 3.0,
                 "issue_type": "other", "review_note": "  "},
        },
    )

    assert payload["kind"] == "algorithm_review_notes"
    assert payload["notes"] == [{
        "frame_index": 83,
        "video_time_seconds": 2.77,
        "issue_type": "identity_lost",
        "note": "开始移动后未绑定",
    }]
    assert "registration" not in payload
    assert "bbox_xyxy" not in str(payload)
    assert payload["privacy"]["frames_exported"] is False


def test_registration_is_automatically_the_earliest_target_box():
    """完整评测从最早出现的人工目标框初始化，不需单独点登记按钮。"""
    annotations = {
        235: {"frame_index": 235, "video_time_seconds": 7.8,
              "people": [{"person_id": "person-1", "bbox_xyxy": [0.1, 0.1, 0.4, 0.8]}]},
        4: {"frame_index": 4, "video_time_seconds": 0.13,
            "people": [{"person_id": "person-1", "bbox_xyxy": [0.2, 0.1, 0.5, 0.8]}]},
        20: {"frame_index": 20, "video_time_seconds": 0.67,
             "people": [{"person_id": "person-2", "bbox_xyxy": [0.3, 0.1, 0.6, 0.8]}]},
    }

    assert choose_automatic_registration(annotations, "person-1")["frame_index"] == 4
    assert choose_automatic_registration(annotations, "missing") is None


def test_binding_label_uses_target_binding_flag_not_locked_state():
    """ByteTrack state=none 也可能已绑定；绿色绑定字段才是目标判断依据。"""
    assert prediction_binding_label({"bound_to_target": True, "state": "none"}) == "已绑定目标"
    assert prediction_binding_label({"bound_to_target": False, "state": "locked"}) == "已锁定但非登记目标"
    assert prediction_binding_label({"bound_to_target": False, "state": "candidate"}) == "候选中"
