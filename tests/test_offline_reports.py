"""离线报告脚本的合成数据测试，不读取视频或加载模型。"""

import hashlib
import json
from pathlib import Path

from scripts import summarize_evaluation


def _write_json(path: Path, payload: dict) -> Path:
    """将小型合成输入写入临时目录并返回路径。"""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summary_cli_writes_scene_and_iou_reports(tmp_path, monkeypatch):
    """相同人工标注下的两个方案应生成总体、分场景与 IoU 敏感性报告。"""
    annotation = {
        "schema_version": 1,
        "video": {"id": "synthetic", "sha256": "a" * 64},
        "target_person_id": "target",
        "registration": {"frame_index": 0, "person_id": "target"},
        "annotations": [
            {"frame_index": 0, "video_time_seconds": 0.0, "target_visibility": "visible",
             "scene": "far_to_near", "people": [{"person_id": "target", "bbox_xyxy": [0.1, 0.1, 0.5, 0.9]}]},
            {"frame_index": 1, "video_time_seconds": 1.0, "target_visibility": "visible",
             "scene": "reentry", "people": [{"person_id": "target", "bbox_xyxy": [0.1, 0.1, 0.5, 0.9]},
                                               {"person_id": "other", "bbox_xyxy": [0.5, 0.1, 0.9, 0.9]}]},
            {"frame_index": 2, "video_time_seconds": 2.0, "target_visibility": "unknown",
             "scene": "back_turn", "people": []},
        ],
        "reentry_events": [],
    }
    annotation_path = _write_json(tmp_path / "annotations.json", annotation)
    annotation_sha = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    common_run = {
        "annotation_sha256": annotation_sha,
        "detector_sha256": "b" * 64,
        "config": {"det_imgsz": 640, "det_conf": 0.25, "det_iou": 0.7,
                   "det_device": "cpu", "identity_device": "cpu", "max_side": 960},
    }

    prediction_paths = []
    for variant, second_binding in (("bytetrack", "other"), ("hybrid", "target")):
        rows = []
        for frame_index, timestamp, binding in (
            (0, 0.0, "target"), (1, 1.0, second_binding), (2, 2.0, None),
        ):
            boxes = []
            if binding is not None:
                x1 = 0.1 if binding == "target" else 0.5
                boxes.append({"bbox_xyxy": [x1, 0.1, x1 + 0.4, 0.9],
                              "bound_to_target": True, "track_id": 1,
                              "detection_score": 0.9, "state": "locked"})
            rows.append({"frame_index": frame_index, "video_time_seconds": timestamp,
                         "processing_ms": 10.0, "boxes": boxes})
        payload = {"schema_version": 1, "video": annotation["video"], "variant": variant,
                   "run": common_run, "predictions": rows}
        prediction_paths.append(_write_json(tmp_path / f"{variant}.json", payload))

    output_dir = tmp_path / "reports"
    monkeypatch.setattr("sys.argv", ["summarize_evaluation.py", "--annotations", str(annotation_path),
                                      "--predictions", *(str(path) for path in prediction_paths),
                                      "--output-dir", str(output_dir)])
    assert summarize_evaluation.main() == 0

    report = json.loads((output_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert report["variants"]["bytetrack"]["summary"]["wrong_person_frames"] == 1
    assert report["variants"]["hybrid"]["summary"]["correct_binding_frames"] == 2
    assert report["scenes"]["far_to_near"]["hybrid"]["visible_frames"] == 1
    assert set(report["iou_sensitivity"]) == {"0.3", "0.7"}
    assert (output_dir / "scene_summary.csv").is_file()
    assert (output_dir / "iou_sensitivity.csv").is_file()
    assert (output_dir / "failure_cases.csv").is_file()
