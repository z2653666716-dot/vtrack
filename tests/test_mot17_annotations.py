"""检查 MOT17 标注转换的可见性、像素缩放与目标登记规则。"""

from scripts.prepare_mot17_annotations import build_annotations, normalized_box


def test_mot_gt_converter_separates_visible_occluded_and_missing_target() -> None:
    """可见目标参加评分；visibility=0 分离为遮挡；无目标记录保持 unknown。"""
    sequence = {"gt_path": "train/MOT17-01-DPM/gt/gt.txt", "frame_count": 3,
                "width": 1920, "height": 1080, "frame_rate": 30}
    video = {"fps": 30, "frame_count": 3, "width": 960, "height": 540,
             "scale_from_gt": 0.5, "sha256": "a" * 64}
    rows = [
        [1, 1, 260, 450, 102, 262, 1, 1, 0.8],
        [1, 2, 900, 300, 100, 250, 1, 1, 1.0],
        [2, 1, 270, 450, 100, 260, 1, 1, 0.0],
        [2, 2, 910, 300, 100, 250, 1, 1, 1.0],
    ]

    result = build_annotations(sequence, video, rows, 1, "01", "DPM")

    assert result["registration"]["frame_index"] == 0
    assert result["annotations"][0]["target_visibility"] == "visible"
    assert result["annotations"][0]["target_visibility_ratio"] == 0.8
    assert result["annotations"][0]["people"][0]["bbox_xyxy"] == [
        260 / 1920, 450 / 1080, 362 / 1920, 712 / 1080
    ]
    assert result["annotations"][1]["target_visibility"] == "occluded"
    assert all(person["person_id"] != "mot-1" for person in result["annotations"][1]["people"])
    assert result["annotations"][2]["target_visibility"] == "unknown"


def test_normalized_box_clips_negative_gt_edges() -> None:
    """画面边缘外的 MOT 框裁到画面后仍输出合法归一化坐标。"""
    assert normalized_box([1, 1, -19, -20, 252, 622, 1, 1, 0.7], 640, 480) == [
        0, 0, 233 / 640, 480 / 480
    ]
    assert normalized_box([1, 1, 700, 10, 50, 20, 1, 1, 0.5], 640, 480) is None
