"""在指定本地视频上运行 ViTrack 离线算法回放。

程序通过文件路径读取视频，不接受摄像头编号或网络流；每次运行创建新会话，
按人工标注中的同一帧登记目标，调用现有 Pipeline、Tracker 与身份模块，逐帧输出
数值框、绑定状态和真实处理耗时。它不会生成带框视频、截图、裁剪或特征文件。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppConfig, load_config
from app.detector import Detector
from app.hybrid_session import HybridSession
from app.pipeline import Pipeline
from app.schemas import Box
from app.session import SessionError, VideoSession
from app.tracker import Tracker


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VARIANTS = ("bytetrack", "body", "hybrid")
REGISTRATION_IOU = 0.5


class VideoClock:
    """可修改的视频时间源；会话读取它来执行短时身份保持和候选超时。"""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        """无输入，返回当前已处理视频帧的时间戳秒数。"""
        return self.seconds


def sha256_file(path: Path) -> str:
    """分块读取本地文件并返回 SHA-256，用于证明输入和模型与记录一致。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """读取显式视频、标注、算法方案和结果路径；拒绝视频以外的输入。"""
    parser = argparse.ArgumentParser(description="在指定本地视频上运行单方案的 ViTrack 离线回放")
    parser.add_argument("--video", required=True, type=Path, help="本地视频文件路径，不支持摄像头或网络流")
    parser.add_argument("--annotations", required=True, type=Path, help="审阅界面导出的人工标注 JSON")
    parser.add_argument("--variant", required=True, choices=VARIANTS, help="bytetrack、body 或 hybrid")
    parser.add_argument("--output", type=Path, help="预测 JSON 输出路径")
    parser.add_argument("--device", choices=("cpu", "0"), help="覆盖检测设备；默认使用项目 .env 配置")
    args = parser.parse_args()

    args.video = args.video.expanduser().resolve()
    args.annotations = args.annotations.expanduser().resolve()
    if not args.video.is_file() or args.video.suffix.lower() not in VIDEO_SUFFIXES:
        parser.error("--video 必须指向现存的本地视频文件（MP4/MOV/AVI/MKV/WebM/M4V）")
    if not args.annotations.is_file() or args.annotations.suffix.lower() != ".json":
        parser.error("--annotations 必须指向现存的 JSON 标注文件")
    args.output = (args.output.expanduser().resolve() if args.output else
                   ROOT / "outputs" / "evaluation" / f"{args.video.stem}_{args.variant}_predictions.json")
    if args.output in {args.video, args.annotations}:
        parser.error("--output 不能覆盖视频或人工标注文件")
    return args


def load_annotations(path: Path, video_path: Path) -> tuple[dict, dict[int, dict]]:
    """校验审阅 JSON 和视频摘要，输出头信息及按原视频帧号索引的标注。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("不支持的标注格式版本，请用当前 scripts/review_video.py 重新导出")
    video = payload.get("video")
    if not isinstance(video, dict) or video.get("sha256") != sha256_file(video_path):
        raise ValueError("标注中的视频 SHA-256 与当前视频不一致，请选择原视频")
    target_id = payload.get("target_person_id")
    registration = payload.get("registration")
    if not isinstance(target_id, str) or not target_id.strip() or not isinstance(registration, dict):
        raise ValueError("标注缺少目标人物编号或登记操作")
    if registration.get("person_id") != target_id:
        raise ValueError("目标登记编号与视频目标编号不一致")

    annotations = {}
    for row in payload.get("annotations", []):
        if not isinstance(row, dict):
            raise ValueError("每条逐帧标注必须是 JSON 对象")
        frame_index = row.get("frame_index")
        time_seconds = row.get("video_time_seconds")
        if type(frame_index) is not int or frame_index < 0 or frame_index in annotations:
            raise ValueError("标注帧号必须是非负且不重复的整数")
        if (isinstance(time_seconds, bool) or not isinstance(time_seconds, (int, float))
                or not math.isfinite(time_seconds) or time_seconds < 0):
            raise ValueError("人工标注视频时间必须是有限非负数")
        if row.get("target_visibility") not in {"visible", "absent", "unknown"}:
            raise ValueError("target_visibility 只能是 visible、absent 或 unknown")
        people = row.get("people", [])
        if not isinstance(people, list):
            raise ValueError("people 必须是数组")
        seen_people = set()
        for person in people:
            person_id, box = person.get("person_id"), person.get("bbox_xyxy")
            if not isinstance(person_id, str) or not person_id.strip() or person_id in seen_people:
                raise ValueError(f"第 {frame_index} 帧人物编号缺失或重复")
            seen_people.add(person_id)
            if (not isinstance(box, list) or len(box) != 4
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in box)
                    or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                raise ValueError(f"第 {frame_index} 帧人物 {person_id} 的边界框格式错误")
        if row["target_visibility"] == "visible" and target_id not in seen_people:
            raise ValueError(f"第 {frame_index} 帧目标可见但缺少目标人工框")
        annotations[frame_index] = row

    registration_frame = registration.get("frame_index")
    if type(registration_frame) is not int or registration_frame not in annotations:
        raise ValueError("目标登记帧必须存在于逐帧人工标注中")
    if target_id not in {person["person_id"] for person in annotations[registration_frame]["people"]}:
        raise ValueError("目标登记帧中没有登记目标的人物框")
    return payload, annotations


def make_config(base: AppConfig, variant: str, device: str | None) -> AppConfig:
    """从项目配置复制同一检测参数，只切换比较方案所需的跟踪/身份模块。"""
    settings: dict[str, Any] = {}
    if variant == "bytetrack":
        settings.update(tracker="bytetrack.yaml", identity_mode="appearance")
    elif variant == "body":
        settings.update(tracker="botsort.yaml", identity_mode="appearance", reid_model="osnet_x0_25")
    else:
        settings.update(tracker="botsort.yaml", identity_mode="hybrid", reid_model="osnet_x0_25")
    if device is not None:
        settings["det_device"] = device if device == "cpu" else 0
    return base.model_copy(update=settings)


def iou(first: tuple[float, ...] | list[float], second: tuple[float, ...] | list[float]) -> float:
    """输入两个 0～1 归一化人物框，输出交并比，供登记框唯一匹配。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-12)


def match_registration(boxes: list[Box], target_box: list[float]) -> tuple[Box | None, float | None]:
    """找与人工目标框 IoU 最大的系统框；低重叠或近似并列时拒绝猜测。"""
    ranked = sorted(((iou(box.xyxy, target_box), box) for box in boxes), key=lambda pair: pair[0], reverse=True)
    if not ranked or ranked[0][0] < REGISTRATION_IOU:
        return None, ranked[0][0] if ranked else None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.05:
        return None, ranked[0][0]
    return ranked[0][1], ranked[0][0]


def git_revision() -> str | None:
    """读取本地 HEAD 提交号；仓库状态异常时返回 None，不阻断离线推理。"""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True, timeout=5)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def working_tree_is_dirty() -> bool | None:
    """查询当前工作区是否有改动；Git 不可用时返回 None 而不阻止回放。"""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, check=True, timeout=5)
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def model_digests(config: AppConfig, variant: str) -> dict[str, str]:
    """记录本次方案实际依赖的模型摘要，不把原始媒体内容放进报告。"""
    paths = {"detector": config.det_model}
    if variant == "body":
        paths["body_reid"] = config.person_weights
    elif variant == "hybrid":
        paths.update(body_reid=config.person_weights, face_detector=config.face_detector_weights,
                     face_recognizer=config.face_recognizer_weights)
    elif config.reid_model == "mobilenet_v3_small":
        paths["legacy_reid"] = config.reid_weights
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def box_record(box: Box, target_track_id: int | None, tracker_only: bool) -> dict:
    """将单个输出框转为纯数值预测；不序列化任何裁剪或外观特征。"""
    is_target_track = box.track_id is not None and box.track_id == target_track_id
    return {
        "bbox_xyxy": [round(float(value), 7) for value in box.xyxy],
        "class_id": box.cls,
        "detection_score": round(float(box.score), 6),
        "track_id": box.track_id,
        # 纯 ByteTrack 对照只看初始轨迹号；身份方案还要求该帧仍处于 locked 状态。
        "bound_to_target": is_target_track and (tracker_only or box.state == "locked"),
        "state": box.state,
    }


async def run_replay(args: argparse.Namespace, annotation_file: dict, annotations: dict[int, dict],
                     config: AppConfig, detector: Detector) -> dict:
    """逐帧运行现有 Pipeline，遇登记帧按人工目标框初始化同一个目标。"""
    pipeline = Pipeline(detector, config)
    clock = VideoClock()
    session: VideoSession | Tracker
    if args.variant == "bytetrack":
        # 这个对照只保留初始轨迹编号；轨迹结束/换号后不做外观重识别。
        session = Tracker(config)
    else:
        session = pipeline.create_tracker(clock=clock)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法解码指定本地视频")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = float(annotation_file["video"].get("fps") or 30.0)
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    registration = annotation_file["registration"]
    registration_frame = registration["frame_index"]
    target_id = annotation_file["target_person_id"]
    registration_status = {"status": "pending", "frame_index": registration_frame,
                           "iou_threshold": REGISTRATION_IOU, "track_id": None, "reason": None}
    locked_track_id = None
    predictions = []
    previous_timestamp = -1.0
    frame_index = 0
    started_all = perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = timestamp_ms / 1000 if math.isfinite(timestamp_ms) and timestamp_ms >= 0 else frame_index / fps
            if frame_index == 0 and timestamp == 0:
                timestamp = 0.0
            if timestamp <= previous_timestamp and frame_index > 0:
                timestamp = frame_index / fps
            if timestamp <= previous_timestamp:
                raise ValueError(f"视频时间戳不递增，停止在第 {frame_index} 帧")
            previous_timestamp = timestamp
            clock.seconds = timestamp

            frame_started = perf_counter()
            result = await pipeline.process_frame(frame, frame_index, tracker=session)
            processing_ms = (perf_counter() - frame_started) * 1000
            output_boxes = result.boxes

            if frame_index == registration_frame:
                human_frame = annotations[registration_frame]
                target_box = next(person["bbox_xyxy"] for person in human_frame["people"]
                                  if person["person_id"] == target_id)
                selected, overlap_score = match_registration(output_boxes, target_box)
                registration_status["best_iou"] = round(overlap_score, 6) if overlap_score is not None else None
                if selected is None or selected.track_id is None:
                    registration_status.update(status="failed", reason="no_unique_detection_match")
                elif isinstance(session, Tracker):
                    locked_track_id = selected.track_id
                    registration_status.update(status="registered", track_id=locked_track_id)
                else:
                    try:
                        session.lock(selected.track_id, frame_index, "lock")
                        locked_track_id = selected.track_id
                        registration_status.update(status="registered", track_id=locked_track_id)
                        # 当前帧的 FrameResult 已在登记操作前返回，所以同步修正该帧显示状态。
                        output_boxes = [box.model_copy(update={"state": "locked", "identity_id": "ID-01"})
                                        if box.track_id == locked_track_id else box for box in output_boxes]
                    except SessionError as error:
                        registration_status.update(status="failed", reason=error.code)

            predictions.append({
                "frame_index": frame_index,
                "video_time_seconds": round(timestamp, 6),
                "processing_ms": round(processing_ms, 3),
                "boxes": [box_record(box, locked_track_id, args.variant == "bytetrack")
                          for box in output_boxes],
            })
            frame_index += 1
    finally:
        capture.release()
        if isinstance(session, VideoSession):
            await pipeline.reset_tracker(session)
        else:
            session.reset()

    if frame_index == 0:
        raise ValueError("视频没有可解码的画面")
    return {
        "schema_version": 1,
        "video": {"id": annotation_file["video"]["id"], "sha256": annotation_file["video"]["sha256"],
                  "fps": fps, "reported_frame_count": reported_frame_count, "processed_frame_count": frame_index},
        "variant": args.variant,
        "registration": registration_status,
        "run": {
            "git_revision": git_revision(),
            "working_tree_dirty": working_tree_is_dirty(),
            "detector": config.det_model.name,
            "detector_sha256": sha256_file(config.det_model),
            "model_sha256": model_digests(config, args.variant),
            "config": {"tracker": config.tracker, "identity_mode": config.identity_mode,
                       "reid_model": config.reid_model, "det_imgsz": config.det_imgsz,
                       "max_side": config.max_side,
                       "det_conf": config.det_conf, "det_iou": config.det_iou,
                       "det_device": config.det_device, "identity_device": config.identity_device,
                       "track_max_age": config.track_max_age,
                       "face_match_threshold": config.face_match_threshold,
                       "body_match_threshold": config.body_match_threshold,
                       "body_hold_threshold": config.body_hold_threshold,
                       "reid_confirm_frames": config.reid_confirm_frames},
            "registration_iou": REGISTRATION_IOU,
            "annotation_sha256": sha256_file(args.annotations),
            "elapsed_computation_seconds": round(perf_counter() - started_all, 3),
            "prediction_samples": len(predictions),
        },
        "predictions": predictions,
        "privacy": {"video_copied": False, "frames_exported": False, "crops_exported": False,
                    "features_exported": False},
    }


def main() -> int:
    """入口：校验人工标注 → 配置方案 → 加载模型 → 本地回放 → 写出数值 JSON。"""
    args = parse_args()
    try:
        annotation_file, annotations = load_annotations(args.annotations, args.video)
        config = make_config(load_config(), args.variant, args.device)
        detector = Detector(config)
        report = asyncio.run(run_replay(args, annotation_file, annotations, config, detector))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "completed", "variant": args.variant,
                          "registration": report["registration"], "frames": len(report["predictions"]),
                          "output": str(args.output)}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"离线回放失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
