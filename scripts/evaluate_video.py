"""在指定本地视频上运行 ViTrack 离线算法回放。

程序通过文件路径读取视频，不接受摄像头编号或网络流；每次运行创建新会话，
可使用人工标注登记目标，也可在单人视频中自动选择首个稳定人物轨迹，调用现有 Pipeline、Tracker 与身份模块，逐帧输出
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
from app.evaluation import effective_target_visibility
from app.hybrid_session import HybridSession
from app.pipeline import Pipeline
from app.schemas import Box
from app.session import SessionError, VideoSession
from app.tracker import Tracker


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VARIANTS = ("bytetrack", "body", "hybrid_no_face", "hybrid",
            "hybrid_b0", "hybrid_b1", "hybrid_b2")
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
    parser.add_argument("--annotations", type=Path, help="可选：审阅界面导出的人工评测标注；省略时按单人模式自动登记首个稳定人物轨迹")
    parser.add_argument("--variant", required=True, choices=VARIANTS,
                        help="现有对照方案，或 hybrid_b0/b1/b2 身体模板更新对照")
    parser.add_argument("--output", type=Path, help="预测 JSON 输出路径")
    parser.add_argument("--device", choices=("cpu", "0"), help="覆盖检测设备；默认使用项目 .env 配置")
    args = parser.parse_args()

    args.video = args.video.expanduser().resolve()
    args.annotations = args.annotations.expanduser().resolve() if args.annotations else None
    if not args.video.is_file() or args.video.suffix.lower() not in VIDEO_SUFFIXES:
        parser.error("--video 必须指向现存的本地视频文件（MP4/MOV/AVI/MKV/WebM/M4V）")
    if args.annotations and (not args.annotations.is_file() or args.annotations.suffix.lower() != ".json"):
        parser.error("--annotations 必须指向现存的 JSON 标注文件")
    args.output = (args.output.expanduser().resolve() if args.output else
                   ROOT / "outputs" / "evaluation" / f"{args.video.stem}_{args.variant}_predictions.json")
    if args.output == args.video or (args.annotations and args.output == args.annotations):
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
    inferred_visible_frames = 0
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
        if row.get("target_visibility") not in {"visible", "absent", "occluded", "unknown"}:
            raise ValueError("target_visibility 只能是 visible、absent、occluded 或 unknown")
        visibility_ratio = row.get("target_visibility_ratio")
        if visibility_ratio is not None and (
                isinstance(visibility_ratio, bool) or not isinstance(visibility_ratio, (int, float))
                or not math.isfinite(visibility_ratio) or not 0 <= visibility_ratio <= 1):
            raise ValueError(f"第 {frame_index} 帧 target_visibility_ratio 必须在 0 到 1 之间")
        if row["target_visibility"] == "occluded" and visibility_ratio != 0:
            raise ValueError(f"第 {frame_index} 帧完全遮挡标注必须有 visibility ratio 0")
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
        effective_visibility, inferred = effective_target_visibility(
            row["target_visibility"], people, target_id)
        if inferred:
            row["target_visibility"] = "visible"
            inferred_visible_frames += 1
        annotations[frame_index] = row

    # 记录修正数量供报告解释；文件 SHA 仍基于用户原始导出的 JSON 计算。
    payload["_visibility_inferred_frames"] = inferred_visible_frames

    registration_frame = registration.get("frame_index")
    if type(registration_frame) is not int or registration_frame not in annotations:
        raise ValueError("目标登记帧必须存在于逐帧人工标注中")
    if target_id not in {person["person_id"] for person in annotations[registration_frame]["people"]}:
        raise ValueError("目标登记帧中没有登记目标的人物框")
    return payload, annotations


def build_single_person_metadata(video_path: Path) -> dict:
    """为无需人工真值框的单人回放建立视频元数据与自动登记模式。

    输入是已由命令行明确指定的本地视频；输出仅含 SHA-256、帧率、帧数和目标编号，
    不保存视频画面。空 annotations 表示本轮只生成预测供人工复核，不计算准确率。
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法解码指定本地视频")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    digest = sha256_file(video_path)
    return {
        "schema_version": 1,
        "video": {"id": digest[:12], "sha256": digest, "fps": fps,
                  "frame_count": frame_count, "width": width, "height": height},
        "target_person_id": "person-1",
        "registration": None,
        "registration_mode": "single_person_first_stable_track",
        "annotations": [],
        "reentry_events": [],
    }


def select_initialization_box(boxes: list[Box]) -> Box | None:
    """从当前人物检测中选择单人模式的初始化候选框。

    优先取已有稳定 track_id 的框；如果跟踪器尚未分配编号，则暂取置信度最高的框，
    后续帧会用框重合度接上轨迹。空输入返回 None。
    """
    if not boxes:
        return None
    tracked_boxes = [box for box in boxes if box.track_id is not None]
    return max(tracked_boxes or boxes, key=lambda box: box.score)


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


def match_registration(boxes: list[Box], target_box: list[float],
                       minimum_iou: float = REGISTRATION_IOU) -> tuple[Box | None, float | None]:
    """找与人工目标框 IoU 最大的系统框；低重叠或近似并列时拒绝猜测。"""
    ranked = sorted(((iou(box.xyxy, target_box), box) for box in boxes), key=lambda pair: pair[0], reverse=True)
    if not ranked or ranked[0][0] < minimum_iou:
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
    elif variant in {"hybrid", "hybrid_no_face", "hybrid_b0", "hybrid_b1", "hybrid_b2"}:
        paths["body_reid"] = config.person_weights
        if variant == "hybrid":
            paths.update(face_detector=config.face_detector_weights,
                         face_recognizer=config.face_recognizer_weights)
    elif config.reid_model == "mobilenet_v3_small":
        paths["legacy_reid"] = config.reid_weights
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def box_record(box: Box, target_track_id: int | None, tracker_only: bool,
               target_identity_id: str | None = None) -> dict:
    """将输出框转成纯数值预测；基线看轨迹号，身份方案看稳定身份编号。"""
    # 轨迹号会在目标离场重入后变化；身份方案必须按登记时的 identity_id 判断。
    is_target = (box.track_id is not None and box.track_id == target_track_id
                 if tracker_only else
                 box.identity_id is not None and box.identity_id == target_identity_id
                 and box.state == "locked")
    return {
        "bbox_xyxy": [round(float(value), 7) for value in box.xyxy],
        "class_id": box.cls,
        "detection_score": round(float(box.score), 6),
        "track_id": box.track_id,
        # 不把身份编号写入评测文件，只保存能否对应登记目标这一布尔结果。
        "bound_to_target": is_target,
        "state": box.state,
    }


async def run_replay(args: argparse.Namespace, annotation_file: dict, annotations: dict[int, dict],
                     config: AppConfig, detector: Detector) -> dict:
    """逐帧运行现有 Pipeline；按人工框登记，或在单人模式自动初始化首个轨迹。"""
    pipeline = Pipeline(detector, config)
    clock = VideoClock()
    session: VideoSession | Tracker
    if args.variant == "bytetrack":
        # 这个对照只保留初始轨迹编号；轨迹结束/换号后不做外观重识别。
        session = Tracker(config)
    else:
        template_strategy = {"hybrid_b0": "append", "hybrid_b1": "fixed_ema",
                             "hybrid_b2": "quality_ema"}.get(args.variant, "append")
        collect_template_diagnostics = args.variant in {"hybrid_b0", "hybrid_b1", "hybrid_b2"}
        session = pipeline.create_tracker(clock=clock,
                                          use_face_evidence=args.variant != "hybrid_no_face",
                                          body_template_update=template_strategy,
                                          collect_template_diagnostics=collect_template_diagnostics)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法解码指定本地视频")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = float(annotation_file["video"].get("fps") or 30.0)
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    registration = annotation_file.get("registration")
    registration_frame = registration.get("frame_index") if isinstance(registration, dict) else None
    target_id = annotation_file["target_person_id"]
    registration_status = {"status": "not_started", "frame_index": registration_frame,
                           "mode": annotation_file.get("registration_mode", "manual_target_box"),
                           "iou_threshold": REGISTRATION_IOU, "track_id": None, "reason": None}
    locked_track_id = None
    target_identity_id = None
    pending_lock_track_id = None
    pending_registration_box = None
    predictions = []
    template_update_diagnostics = []
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

            if registration_frame is None and registration_status["status"] == "not_started":
                # 纯复核模式没有人工登记框；本视频确定只有一个人，因此取首个稳定人物轨迹。
                selected = select_initialization_box(output_boxes)
                if selected is not None:
                    registration_status.update(frame_index=frame_index,
                                               requested_time_seconds=timestamp,
                                               best_score=round(selected.score, 6))
                    if selected.track_id is None:
                        pending_registration_box = list(selected.xyxy)
                        registration_status.update(status="pending", reason="waiting_for_track_id",
                                                   anchor_iou_threshold=0.3)
                    elif isinstance(session, Tracker):
                        locked_track_id = selected.track_id
                        registration_status.update(status="registered", track_id=locked_track_id,
                                                   effective_frame_index=frame_index, reason=None)
                    else:
                        pending_lock_track_id = selected.track_id
                        locked_track_id = selected.track_id
                        registration_status.update(status="pending", track_id=locked_track_id,
                                                   reason="waiting_for_track_warmup")

            if registration_frame is not None and frame_index == registration_frame:
                registration_status["requested_time_seconds"] = timestamp
                human_frame = annotations[registration_frame]
                target_box = next(person["bbox_xyxy"] for person in human_frame["people"]
                                  if person["person_id"] == target_id)
                selected, overlap_score = match_registration(output_boxes, target_box)
                registration_status["best_iou"] = round(overlap_score, 6) if overlap_score is not None else None
                if selected is None:
                    registration_status.update(status="failed", reason="no_unique_detection_match")
                elif isinstance(session, Tracker):
                    if selected.track_id is None:
                        pending_registration_box = list(selected.xyxy)
                        registration_status.update(status="pending", reason="waiting_for_track_id",
                                                   anchor_iou_threshold=0.3)
                    else:
                        locked_track_id = selected.track_id
                        registration_status.update(status="registered", track_id=locked_track_id,
                                                   effective_frame_index=frame_index, reason=None)
                else:
                    # 某些跟踪器首帧只输出候选框；先保存框位置，待它产生稳定轨迹号。
                    if selected.track_id is None:
                        pending_registration_box = list(selected.xyxy)
                        registration_status.update(status="pending", reason="waiting_for_track_id",
                                                   anchor_iou_threshold=0.3)
                    else:
                        pending_lock_track_id = selected.track_id
                        locked_track_id = selected.track_id
                        registration_status.update(status="pending", track_id=locked_track_id,
                                                   reason="waiting_for_track_warmup")

            # 登记操作先固定人工目标，再等待跟踪器生成轨迹号或满足身份特征预热条件。
            if registration_status["status"] == "pending":
                if pending_registration_box is not None and pending_lock_track_id is None:
                    anchor_box, anchor_iou = match_registration(
                        output_boxes, pending_registration_box, minimum_iou=0.3)
                    if anchor_box is not None:
                        pending_registration_box = list(anchor_box.xyxy)
                        if anchor_box.track_id is not None:
                            pending_lock_track_id = anchor_box.track_id
                            locked_track_id = anchor_box.track_id
                            registration_status.update(track_id=locked_track_id,
                                                       reason="waiting_for_track_warmup",
                                                       anchor_best_iou=round(anchor_iou, 6))

                if (timestamp - registration_status["requested_time_seconds"] > 2.0
                        and pending_lock_track_id is None):
                    pending_registration_box = None
                    registration_status.update(status="failed", reason="TRACK_NOT_INITIALIZED")
                elif (timestamp - registration_status["requested_time_seconds"] > 2.0
                      and pending_lock_track_id is not None
                      and isinstance(session, VideoSession)):
                    pending_lock_track_id = None
                    registration_status.update(status="failed", reason="REID_NOT_WARM")
                elif pending_lock_track_id is not None and isinstance(session, Tracker):
                    registration_status.update(status="registered", effective_frame_index=frame_index,
                                               warmup_delay_seconds=round(
                                                   timestamp - registration_status["requested_time_seconds"], 6),
                                               reason=None)
                    pending_lock_track_id = None
                elif pending_lock_track_id is not None and isinstance(session, VideoSession):
                    if pending_lock_track_id not in session.observations:
                        pending_lock_track_id = None
                        registration_status.update(status="failed", reason="TRACK_GONE")
                    elif any(box.track_id == pending_lock_track_id for box in output_boxes):
                        try:
                            lock_result = session.lock(pending_lock_track_id, frame_index, "lock")
                            target_identity_id = lock_result.get("identity_id")
                            registration_status.update(status="registered", track_id=pending_lock_track_id,
                                                       effective_frame_index=frame_index, reason=None,
                                                       warmup_delay_seconds=round(
                                                           timestamp - registration_status["requested_time_seconds"], 6))
                            output_boxes = [box.model_copy(update={"state": "locked",
                                                                  "identity_id": target_identity_id})
                                            if box.track_id == pending_lock_track_id else box
                                            for box in output_boxes]
                            pending_lock_track_id = None
                        except SessionError as error:
                            if error.code != "REID_NOT_WARM":
                                pending_lock_track_id = None
                                registration_status.update(status="failed", reason=error.code)

            predictions.append({
                "frame_index": frame_index,
                "video_time_seconds": round(timestamp, 6),
                "processing_ms": round(processing_ms, 3),
                "boxes": [box_record(box, locked_track_id, args.variant == "bytetrack",
                                     target_identity_id=target_identity_id)
                          for box in output_boxes],
            })
            frame_index += 1
        if isinstance(session, HybridSession) and collect_template_diagnostics:
            template_update_diagnostics = list(session.template_update_records)
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
                       "face_evidence_enabled": args.variant != "hybrid_no_face",
                       "body_template_update": {"hybrid_b0": "append", "hybrid_b1": "fixed_ema",
                                                "hybrid_b2": "quality_ema"}.get(args.variant),
                       "body_template_alpha_max": (session.body_template_alpha
                                                   if isinstance(session, HybridSession)
                                                   and collect_template_diagnostics else None),
                       "body_quality_version": ("body-quality-v1" if args.variant == "hybrid_b2" else None),
                       "body_match_threshold": config.body_match_threshold,
                       "body_hold_threshold": config.body_hold_threshold,
                       "reid_confirm_frames": config.reid_confirm_frames},
            "registration_iou": REGISTRATION_IOU,
            "annotation_sha256": sha256_file(args.annotations) if args.annotations else None,
            "visibility_inferred_from_target_boxes": annotation_file.get(
                "_visibility_inferred_frames", 0),
            "elapsed_computation_seconds": round(perf_counter() - started_all, 3),
            "prediction_samples": len(predictions),
        },
        "predictions": predictions,
        "body_template_update_diagnostics": template_update_diagnostics,
        "privacy": {"video_copied": False, "frames_exported": False, "crops_exported": False,
                    "features_exported": False},
    }


def main() -> int:
    """入口：校验人工标注 → 配置方案 → 加载模型 → 本地回放 → 写出数值 JSON。"""
    args = parse_args()
    try:
        if args.annotations:
            annotation_file, annotations = load_annotations(args.annotations, args.video)
        else:
            annotation_file, annotations = build_single_person_metadata(args.video), {}
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
