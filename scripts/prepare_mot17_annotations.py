"""将 MOT17 GT 转为 ViTrack 离线评测标注，不导出或生成任何画面。

输入：本地 MOT17 标签 ZIP、对应本地视频、序列编号及目标轨迹编号。
输出：按 ViTrack 视频帧号排列的 JSON 数值标注。GT 框只用于目标选择、框匹配和评分，
回放程序仍从视频解码后用自己的检测框提取特征。
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path
import zipfile

import cv2


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA256，输入本地路径，输出十六进制摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """解析本地数据路径和序列参数；拒绝联网地址与不存在的文件。"""
    parser = argparse.ArgumentParser(description="把 MOT17 GT 转成 ViTrack 数值评测标注")
    parser.add_argument("--labels-zip", required=True, type=Path, help="本地 MOT17Labels.zip")
    parser.add_argument("--video", required=True, type=Path, help="对应的本地 MOT17 视频文件")
    parser.add_argument("--sequence", required=True, choices=("02", "04", "05", "09", "10", "11", "13"))
    parser.add_argument("--detector", default="DPM", choices=("DPM", "FRCNN", "SDP"))
    parser.add_argument("--target-track-id", type=int, default=1)
    parser.add_argument("--output", type=Path, help="输出 JSON；默认写入 outputs/evaluation/MOT17-XX/")
    args = parser.parse_args()
    args.labels_zip = args.labels_zip.expanduser().resolve()
    args.video = args.video.expanduser().resolve()
    args.output = (args.output.expanduser().resolve() if args.output else
                   ROOT / "outputs" / "evaluation" / f"MOT17-{args.sequence}" /
                   f"MOT17-{args.sequence}-{args.detector}_visibility_annotations.json")
    if not args.labels_zip.is_file() or not zipfile.is_zipfile(args.labels_zip):
        parser.error("--labels-zip 必须指向本地有效 ZIP 文件")
    if not args.video.is_file():
        parser.error("--video 必须指向本地视频文件")
    if args.target_track_id <= 0:
        parser.error("--target-track-id 必须是正整数")
    return args


def read_sequence_metadata(archive: zipfile.ZipFile, sequence: str, detector: str) -> tuple[dict, list[list[float]]]:
    """从 ZIP 读取 seqinfo.ini 与 MOT GT 九列，验证字段后返回元数据和记录行。"""
    folder = f"train/MOT17-{sequence}-{detector}/"
    names = set(archive.namelist())
    info_path, gt_path = folder + "seqinfo.ini", folder + "gt/gt.txt"
    if info_path not in names or gt_path not in names:
        raise ValueError(f"标签 ZIP 中缺少 {info_path} 或 {gt_path}")
    parser = configparser.ConfigParser()
    parser.read_string(archive.read(info_path).decode("utf-8-sig"))
    info = parser["Sequence"]
    metadata = {
        "name": info["name"], "frame_rate": float(info["frameRate"]),
        "frame_count": int(info["seqLength"]), "width": int(info["imWidth"]),
        "height": int(info["imHeight"]), "gt_path": gt_path,
    }
    rows = []
    for line_number, line in enumerate(archive.read(gt_path).decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 9:
            raise ValueError(f"GT 第 {line_number} 行不是九列")
        try:
            values = [float(part) for part in parts]
        except ValueError as error:
            raise ValueError(f"GT 第 {line_number} 行含非数字字段") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"GT 第 {line_number} 行含 NaN 或无穷值")
        frame_number, identity = values[:2]
        x, y, width, height, mark, category, visibility = values[2:]
        if (not frame_number.is_integer() or not identity.is_integer()
                or not category.is_integer() or frame_number < 1 or frame_number > metadata["frame_count"]
                or identity <= 0 or width <= 0 or height <= 0 or not 0 <= visibility <= 1):
            raise ValueError(f"GT 第 {line_number} 行帧号、ID、框尺寸或可见比例非法")
        rows.append(values)
    return metadata, rows


def video_metadata(path: Path, expected: dict) -> dict:
    """读取本地视频容器元数据并核验帧数、比例；不解码或保存视频画面。"""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法读取指定的视频文件")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frames != expected["frame_count"]:
        raise ValueError(f"视频尺寸/帧数非法或与 GT 不匹配：video={width}x{height}/{frames}, GT={expected}")
    scale_x, scale_y = width / expected["width"], height / expected["height"]
    if not math.isclose(scale_x, scale_y, rel_tol=1e-4, abs_tol=1e-6):
        raise ValueError("视频与 GT 的横纵缩放比例不同，拒绝生成可能错位的标注")
    if not math.isfinite(fps) or fps <= 0:
        fps = expected["frame_rate"]
    return {"fps": fps, "frame_count": frames, "width": width, "height": height,
            "scale_from_gt": scale_x, "sha256": sha256_file(path)}


def normalized_box(row: list[float], width: int, height: int) -> list[float] | None:
    """把 MOT 左上角加宽高框缩放、裁到画面内并输出 0～1 xyxy。"""
    x, y, box_width, box_height = row[2:6]
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def build_annotations(sequence_info: dict, video: dict, rows: list[list[float]],
                      target_track_id: int, sequence: str, detector: str) -> dict:
    """按零起始视频帧生成目标可见、完全遮挡或未知标注，并选择首个可见登记帧。

    只把 mark>0、类别为 pedestrian(1)、visibility>0 的人物框放入 people，
    防止完全遮挡时的推测 GT 框被拿来判系统框身份。GT 中缺失的目标记 unknown，
    不擅自推断为离场。输入为本地 GT 数值行与视频元数据，输出纯数值 JSON 对象。
    """
    width, height, fps = video["width"], video["height"], video["fps"]
    by_frame: dict[int, list[list[float]]] = {}
    for row in rows:
        if int(row[6]) > 0 and int(row[7]) == 1:
            by_frame.setdefault(int(row[0]) - 1, []).append(row)
    target_frames: dict[int, list[float]] = {}
    for frame_index, frame_rows in by_frame.items():
        target_rows = [row for row in frame_rows if int(row[1]) == target_track_id]
        if len(target_rows) > 1:
            raise ValueError(f"目标 ID 在第 {frame_index} 帧出现多条有效 GT")
        if target_rows:
            target_frames[frame_index] = target_rows[0]
    eligible = [index for index, row in target_frames.items() if row[8] > 0]
    if not eligible:
        raise ValueError("目标 ID 没有可见比例大于 0 的有效 pedestrian 标注")
    registration_frame = min(eligible)

    annotations = []
    for frame_index in range(video["frame_count"]):
        frame_rows = by_frame.get(frame_index, [])
        people = []
        for row in frame_rows:
            # visibility=0 的目标框通常是全遮挡情况下的标注框，不作为可见人物匹配真值。
            if row[8] <= 0:
                continue
            # GT 框以 seqinfo.ini 的画布为单位；归一化时除 GT 尺寸，等价于先缩放到视频再归一化。
            box = normalized_box(row, sequence_info["width"], sequence_info["height"])
            if box is not None:
                people.append({"person_id": f"mot-{int(row[1])}", "bbox_xyxy": box})
        target_row = target_frames.get(frame_index)
        if target_row is None:
            target_visibility, visibility_ratio = "unknown", None
        elif target_row[8] <= 0:
            target_visibility, visibility_ratio = "occluded", 0.0
        else:
            target_visibility, visibility_ratio = "visible", target_row[8]
        annotations.append({
            "frame_index": frame_index,
            "video_time_seconds": round(frame_index / fps, 6),
            "target_visibility": target_visibility,
            "target_visibility_ratio": visibility_ratio,
            "people": people,
            "scene": "mot17_public",
        })
    if not any(person["person_id"] == f"mot-{target_track_id}"
               for person in annotations[registration_frame]["people"]):
        raise ValueError("自动登记帧没有有效目标框，拒绝输出不一致标注")
    return {
        "schema_version": 1,
        "video": {"id": f"MOT17-{sequence}-{detector}", "sha256": video["sha256"],
                  "fps": fps, "frame_count": video["frame_count"],
                  "width": width, "height": height},
        "dataset": {"name": "MOT17 training sequence", "sequence": sequence,
                    "detector_variant": detector, "gt_path": sequence_info["gt_path"],
                    "gt_frame_count": sequence_info["frame_count"],
                    "gt_width": sequence_info["width"], "gt_height": sequence_info["height"],
                    "video_scale_from_gt": video["scale_from_gt"],
                    "gt_row_fields": ["frame", "track_id", "left", "top", "width", "height",
                                      "mark", "class", "visibility"],
                    "scored_mark": 1, "scored_class": 1,
                    "missing_target_gt_means": "unknown_not_absent"},
        "target_person_id": f"mot-{target_track_id}",
        "registration_mode": "mot17_gt_first_visible_target_frame",
        "registration": {"frame_index": registration_frame,
                          "video_time_seconds": round(registration_frame / fps, 6),
                          "person_id": f"mot-{target_track_id}"},
        "sampling": {"kind": "every_decoded_frame", "frame_index_base": 0},
        "annotations": annotations,
        "reentry_events": [],
    }


def main() -> int:
    """入口：读取标签与元数据、校验视频映射、生成并保存数值标注。"""
    args = parse_args()
    try:
        with zipfile.ZipFile(args.labels_zip) as archive:
            sequence_info, rows = read_sequence_metadata(archive, args.sequence, args.detector)
            gt_sha256 = hashlib.sha256(archive.read(sequence_info["gt_path"])).hexdigest()
        video = video_metadata(args.video, sequence_info)
        payload = build_annotations(sequence_info, video, rows, args.target_track_id,
                                    args.sequence, args.detector)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        counts = {state: sum(row["target_visibility"] == state for row in payload["annotations"])
                  for state in ("visible", "occluded", "unknown")}
        print(json.dumps({"status": "created", "output": str(args.output),
                          "registration_frame": payload["registration"]["frame_index"],
                          "visibility_counts": counts,
                          "gt_sha256": gt_sha256}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, KeyError, configparser.Error) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
