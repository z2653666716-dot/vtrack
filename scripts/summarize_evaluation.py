"""把人工逐帧标注与多个方案的离线预测对齐，生成可复核的比较报告。

本脚本只读取 JSON 数值和文字，不读取视频或模型。它按预先固定的 IoU 门槛
判断系统绑定框对应哪位人工标注人物，定位失败单独记为 unmatched，再调用
app.evaluation 计算目标覆盖、身份判断、错绑和重入结果，导出 JSON/CSV/Markdown。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation import (FrameObservation, effective_target_visibility,
                            evaluate_reentry, summarize_frames)


DEFAULT_IOU = 0.5
SENSITIVITY_IOU = (0.3, 0.7)
AMBIGUITY_MARGIN = 0.05


def sha256_file(path: Path) -> str:
    """分块计算 JSON 文件摘要，用于核对所有预测使用同一份标注。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """读取一个标注文件和至少两个不同算法的预测 JSON。"""
    parser = argparse.ArgumentParser(description="汇总并比较 ViTrack 离线视频评测结果")
    parser.add_argument("--annotations", required=True, type=Path, help="审阅工具导出的人工标注 JSON")
    parser.add_argument("--predictions", required=True, nargs="+", type=Path,
                        help="一个或多个 scripts/evaluate_video.py 产生的预测 JSON")
    parser.add_argument("--output-dir", type=Path, help="结果目录，默认 outputs/evaluation/<video-id>")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="主结果人物框 IoU 门槛，默认 0.5")
    args = parser.parse_args()
    args.annotations = args.annotations.expanduser().resolve()
    args.predictions = [path.expanduser().resolve() for path in args.predictions]
    if not args.annotations.is_file():
        parser.error("--annotations 文件不存在")
    if any(not path.is_file() for path in args.predictions):
        parser.error("至少一个预测 JSON 文件不存在")
    if len(args.predictions) < 2:
        parser.error("算法比较至少需要两个方案的预测结果")
    if isinstance(args.iou, bool) or not math.isfinite(args.iou) or not 0 < args.iou <= 1:
        parser.error("--iou 必须是 0 到 1 之间的有限小数")
    args.output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else None)
    return args


def intersection_over_union(first: list[float], second: list[float]) -> float:
    """输入两个人物框的归一化 xyxy 坐标，输出 0～1 交并比。"""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return overlap / max(area_first + area_second - overlap, 1e-12)


def classify_binding(annotation: dict, prediction: dict, target_id: str,
                     threshold: float) -> tuple[str | None, dict]:
    """把一帧系统绑定框映射为 target/other/unmatched；不信任系统自报身份名。"""
    if annotation["target_visibility"] == "unknown":
        return None, {"bound_boxes": 0, "duplicate_bound": False, "matched_person_id": None,
                      "binding_iou": None, "reason": "human_unknown"}
    bound_boxes = [box for box in prediction.get("boxes", []) if box.get("bound_to_target") is True]
    detail = {"bound_boxes": len(bound_boxes), "duplicate_bound": len(bound_boxes) > 1,
              "matched_person_id": None, "binding_iou": None, "reason": None}
    if not bound_boxes:
        detail["reason"] = "not_bound"
        return None, detail
    if len(bound_boxes) != 1:
        detail["reason"] = "multiple_boxes_bound_to_one_target"
        return "unmatched", detail

    system_box = bound_boxes[0].get("bbox_xyxy")
    people = annotation.get("people", [])
    ranked = sorted(
        ((intersection_over_union(system_box, person["bbox_xyxy"]), person["person_id"])
         for person in people),
        reverse=True,
    )
    if not ranked or ranked[0][0] < threshold:
        detail.update(reason="no_person_box_reached_iou_threshold",
                      binding_iou=round(ranked[0][0], 6) if ranked else None)
        return "unmatched", detail
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < AMBIGUITY_MARGIN:
        detail.update(reason="ambiguous_person_box_iou", binding_iou=round(ranked[0][0], 6))
        return "unmatched", detail
    person_id = ranked[0][1]
    detail.update(matched_person_id=person_id, binding_iou=round(ranked[0][0], 6), reason="matched")
    return ("target" if person_id == target_id else "other"), detail


def target_detection_iou(annotation: dict, prediction: dict, target_id: str) -> float | None:
    """返回预测中最贴近人工目标框的检测 IoU；不要求该框已经绑定身份。"""
    visibility, _inferred = effective_target_visibility(
        annotation["target_visibility"], annotation.get("people", []), target_id)
    if visibility is not True:
        return None
    target = next(person["bbox_xyxy"] for person in annotation["people"]
                  if person["person_id"] == target_id)
    scores = [intersection_over_union(box["bbox_xyxy"], target)
              for box in prediction.get("boxes", [])]
    return max(scores) if scores else 0.0


def load_inputs(annotation_path: Path, prediction_paths: list[Path]) -> tuple[dict, list[dict]]:
    """读取并验证视频 SHA、标注 SHA、方案名称和逐帧预测结构。"""
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation.get("schema_version") != 1:
        raise ValueError("人工标注 JSON 版本不支持")
    ann_sha = sha256_file(annotation_path)
    variants = set()
    predictions = []
    for path in prediction_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"预测 JSON 版本不支持：{path.name}")
        if payload.get("video", {}).get("sha256") != annotation.get("video", {}).get("sha256"):
            raise ValueError(f"视频摘要不一致：{path.name}")
        if payload.get("run", {}).get("annotation_sha256") != ann_sha:
            raise ValueError(f"该预测使用了另一份人工标注：{path.name}")
        variant = payload.get("variant")
        if not isinstance(variant, str) or variant in variants:
            raise ValueError("每个算法方案只能提供一个预测文件，且方案名必须不同")
        variants.add(variant)
        indexed = {}
        previous_frame, previous_time = -1, -1.0
        for row in payload.get("predictions", []):
            frame_index, timestamp = row.get("frame_index"), row.get("video_time_seconds")
            if type(frame_index) is not int or frame_index <= previous_frame:
                raise ValueError(f"{variant} 的预测帧号重复或倒序")
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp) or timestamp <= previous_time):
                raise ValueError(f"{variant} 的视频时间重复、倒序或非法")
            if not isinstance(row.get("boxes"), list):
                raise ValueError(f"{variant} 第 {frame_index} 帧缺少 boxes 数组")
            indexed[frame_index] = row
            previous_frame, previous_time = frame_index, timestamp
        predictions.append({"path": str(path), "payload": payload, "indexed": indexed})

    common_hashes = {item["payload"]["run"].get("annotation_sha256") for item in predictions}
    common_detector = {
        (item["payload"]["run"].get("detector_sha256"),
         item["payload"]["run"].get("config", {}).get("det_imgsz"),
         item["payload"]["run"].get("config", {}).get("det_conf"),
         item["payload"]["run"].get("config", {}).get("det_iou"),
         item["payload"]["run"].get("config", {}).get("det_device"))
        for item in predictions
    }
    common_identity_runtime = {
        (item["payload"]["run"].get("config", {}).get("identity_device"),
         item["payload"]["run"].get("config", {}).get("max_side"))
        for item in predictions
    }
    if common_hashes != {ann_sha}:
        raise ValueError("方案之间没有共用完全相同的人工标注")
    if len(common_detector) != 1:
        raise ValueError("算法比较检测器权重、尺寸、阈值或设备不同；请先固定检测条件")
    if len(common_identity_runtime) != 1:
        raise ValueError("身份模型运行设备或输入尺寸不同；请先固定身份推理条件")
    return annotation, predictions


def evaluate_variant(annotation: dict, item: dict, iou_threshold: float) -> dict:
    """按人工采样帧计算身份指标、检测定位辅助指标、重入事件和失败帧。"""
    target_id = annotation["target_person_id"]
    prediction_map = item["indexed"]
    observations = []
    frame_rows = []
    failure_rows = []
    duplicate_bound_frames = 0
    target_detected = 0
    target_visible_count = 0
    target_ious = []
    visible_unbound = 0
    visible_wrong = 0
    visible_unmatched = 0
    inferred_visibility = 0

    for human_frame in sorted(annotation.get("annotations", []), key=lambda row: row["frame_index"]):
        frame_index = human_frame["frame_index"]
        prediction = prediction_map.get(frame_index)
        if prediction is None:
            raise ValueError(f"{item['payload']['variant']} 缺少已标注第 {frame_index} 帧的系统输出")
        visibility, inferred = effective_target_visibility(
            human_frame["target_visibility"], human_frame.get("people", []), target_id)
        inferred_visibility += inferred
        effective_frame = ({**human_frame, "target_visibility": "visible"}
                           if inferred else human_frame)
        binding, detail = classify_binding(effective_frame, prediction, target_id, iou_threshold)
        processing_ms = prediction.get("processing_ms")
        observations.append(FrameObservation(
            frame_index=frame_index,
            time_seconds=human_frame["video_time_seconds"],
            target_visible=visibility,
            binding=binding,
            processing_ms=processing_ms,
        ))
        detection_iou = target_detection_iou(effective_frame, prediction, target_id)
        if visibility is True:
            target_visible_count += 1
            target_ious.append(detection_iou)
            target_detected += detection_iou >= iou_threshold
            visible_unbound += binding is None
            visible_wrong += binding == "other"
            visible_unmatched += binding == "unmatched"
        duplicate_bound_frames += detail["duplicate_bound"]

        row = {
            "variant": item["payload"]["variant"],
            "scene": human_frame.get("scene", "unspecified"),
            "frame_index": frame_index,
            "video_time_seconds": human_frame["video_time_seconds"],
            "target_visibility": "visible" if inferred else human_frame["target_visibility"],
            "visibility_inferred_from_target_box": inferred,
            "binding": binding or "unbound_or_unknown",
            "matched_person_id": detail["matched_person_id"],
            "binding_iou": detail["binding_iou"],
            "binding_reason": detail["reason"],
            "target_detection_iou": round(detection_iou, 6) if detection_iou is not None else None,
            "processing_ms": processing_ms,
            "review_note": human_frame.get("review_note", ""),
        }
        frame_rows.append(row)
        if visibility is True and binding != "target":
            failure_rows.append(row)
        elif visibility is False and binding is not None:
            failure_rows.append(row)

    summary = summarize_frames(observations)
    summary.update({
        "target_detection_recall": target_detected / target_visible_count if target_visible_count else None,
        "target_detection_iou_threshold": iou_threshold,
        "mean_target_detection_iou": (sum(target_ious) / len(target_ious) if target_ious else None),
        "visible_unbound_frames": visible_unbound,
        "visible_wrong_person_frames": visible_wrong,
        "visible_unmatched_binding_frames": visible_unmatched,
        "multiple_bound_boxes_frames": duplicate_bound_frames,
        "visibility_inferred_from_target_boxes": inferred_visibility,
    })

    reentry_results = []
    for event in annotation.get("reentry_events", []):
        try:
            result = evaluate_reentry(
                observations,
                start_seconds=event["start_seconds"],
                end_seconds=event["end_seconds"],
            )
        except (KeyError, ValueError) as error:
            result = {"success": None, "complete_observation": False, "reason": str(error),
                      "recovery_seconds": None, "confirmation_seconds": None}
        reentry_results.append({"event_id": event.get("event_id"), **result})

    evaluable_reentries = [event for event in reentry_results
                           if event.get("complete_observation") is True]
    successful_reentries = [event for event in reentry_results if event.get("success") is True]
    confirmed_reentries = [event for event in successful_reentries
                           if event.get("confirmation_seconds") is not None]
    summary.update({
        "reentry_events": len(reentry_results),
        "evaluable_reentry_events": len(evaluable_reentries),
        "successful_reentry_events": len(successful_reentries),
        "mean_reentry_recovery_seconds": (
            sum(event["recovery_seconds"] for event in successful_reentries
                if event.get("recovery_seconds") is not None)
            / sum(event.get("recovery_seconds") is not None for event in successful_reentries)
            if any(event.get("recovery_seconds") is not None for event in successful_reentries) else None
        ),
        "mean_reentry_confirmation_seconds": (
            sum(event["confirmation_seconds"] for event in confirmed_reentries)
            / len(confirmed_reentries) if confirmed_reentries else None
        ),
    })

    return {"summary": summary, "reentry_events": reentry_results,
            "frame_rows": frame_rows, "failure_rows": failure_rows}


def percentage_points(value: float | None, baseline: float | None) -> float | None:
    """把两个比例相减转为百分点；缺少任一可评分分母时返回 None。"""
    if value is None or baseline is None:
        return None
    return round((value - baseline) * 100, 3)


def write_csv(path: Path, rows: list[dict]) -> None:
    """按首行字段写 UTF-8-SIG CSV，方便在 Windows Excel 直接打开中文。"""
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict) -> str:
    """生成简短中文报告，指标名称显示含义和比较百分点。"""
    lines = [
        "# ViTrack 离线跟踪对比报告",
        "",
        f"视频摘要：`{report['video_sha256']}`",
        f"人工标注摘要：`{report['annotation_sha256']}`",
        f"可见状态修正：{report.get('visibility_inferred_from_target_boxes', 0)} 帧根据人工标注的目标人物框从 unknown 计为 visible。",
        f"框匹配主门槛：IoU ≥ {report['iou_threshold']:.2f}；无法匹配或存在多人歧义的框不计作身份错误。",
        "",
        "| 方案 | 目标正确覆盖率 | 匹配绑定框身份正确率 | 匹配绑定框错绑率 | 目标检测召回率 | 可见目标帧 | 错绑帧 | 未绑定帧 | 无法对应绑定框 | 重入成功/可评 | 恢复/确认延迟(s) | 平均处理耗时(ms) | 耗时样本 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in report["variants"].items():
        metrics = result["summary"]
        def percent(value: float | None) -> str:
            return "无样本" if value is None else f"{value * 100:.2f}%"
        def number(value: float | None) -> str:
            return "无样本" if value is None else f"{value:.2f}"
        recovery = number(metrics["mean_reentry_recovery_seconds"])
        confirmation = number(metrics["mean_reentry_confirmation_seconds"])
        lines.append(
            f"| {variant} | {percent(metrics['visible_correct_binding_ratio'])} "
            f"| {percent(metrics['bound_correct_ratio'])} | {percent(metrics['wrong_person_bound_ratio'])} "
            f"| {percent(metrics['target_detection_recall'])} | {metrics['visible_frames']} "
            f"| {metrics['wrong_person_frames']} | {metrics['visible_unbound_frames']} "
            f"| {metrics['unmatched_binding_frames']} "
            f"| {metrics['successful_reentry_events']}/{metrics['evaluable_reentry_events']} "
            f"| {recovery}/{confirmation} | {number(metrics['mean_processing_ms'])} "
            f"| {metrics['timed_frames']} |"
        )
    lines.extend(["", "百分点差值以 `bytetrack` 方案为参照。正值表示该指标比例更高；错绑率下降通常更好。"])
    baseline = report["variants"].get("bytetrack")
    if baseline:
        base_metrics = baseline["summary"]
        lines.extend(["", "## 相对 ByteTrack 的百分点变化", ""])
        for variant, result in report["variants"].items():
            if variant == "bytetrack":
                continue
            metrics = result["summary"]
            coverage = percentage_points(metrics["visible_correct_binding_ratio"],
                                         base_metrics["visible_correct_binding_ratio"])
            identity = percentage_points(metrics["bound_correct_ratio"], base_metrics["bound_correct_ratio"])
            false_bind = percentage_points(metrics["wrong_person_bound_ratio"],
                                           base_metrics["wrong_person_bound_ratio"])
            lines.append(f"- {variant}：目标正确覆盖率 {coverage if coverage is not None else '不可计算'} 个百分点；"
                         f"匹配绑定框正确率 {identity if identity is not None else '不可计算'} 个百分点；"
                         f"错绑率 {false_bind if false_bind is not None else '不可计算'} 个百分点。")
    if report.get("iou_sensitivity"):
        lines.extend(["", "## IoU 匹配门槛敏感性", "", "| IoU 门槛 | 方案 | 目标正确覆盖率 | 身份正确率 | 无法对应绑定框数 |",
                      "|---:|---|---:|---:|---:|"])
        for threshold, variants in report["iou_sensitivity"].items():
            for variant, metrics in variants.items():
                coverage = metrics["visible_correct_binding_ratio"]
                identity = metrics["bound_correct_ratio"]
                lines.append(f"| {threshold} | {variant} | "
                             f"{'无样本' if coverage is None else f'{coverage * 100:.2f}%'} | "
                             f"{'无样本' if identity is None else f'{identity * 100:.2f}%'} | "
                             f"{metrics['unmatched_binding_frames']} |")
    if report.get("scenes"):
        lines.extend(["", "## 分场景目标正确覆盖率", "", "| 场景 | 方案 | 正确覆盖率 | 可见目标帧 | 错绑帧 | 未绑定帧 |", "|---|---|---:|---:|---:|---:|"])
        for scene, variants in report["scenes"].items():
            for variant, metrics in variants.items():
                ratio = metrics["visible_correct_binding_ratio"]
                ratio_text = "无样本" if ratio is None else f"{ratio * 100:.2f}%"
                lines.append(f"| {scene} | {variant} | {ratio_text} | {metrics['visible_frames']} "
                             f"| {metrics['visible_wrong_person_frames']} | {metrics['visible_unbound_frames']} |")
    lines.extend([
        "",
        "重入恢复均值分别报告稳定正确帧串的起点延迟与凑齐确认帧的延迟。",
        "",
        "## 结论边界",
        "",
        "结果只说明当前视频与人工标注条件下的表现。帧率为离线计算样本，不能解释成手机端实时 FPS；"
        "框无法唯一对应人工人物时作为 unmatched 单列，不作为认错人。该视频若用于调参，就不能再充当独立最终测试集。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    """入口：校验共同输入 → 比较每个算法 → 写 JSON、CSV 和 Markdown。"""
    args = parse_args()
    try:
        annotation, prediction_inputs = load_inputs(args.annotations, args.predictions)
        output_dir = args.output_dir or ROOT / "outputs" / "evaluation" / annotation["video"]["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        scenes: dict[str, dict] = {}
        all_frames, all_failures = [], []
        for item in prediction_inputs:
            variant = item["payload"]["variant"]
            scored = evaluate_variant(annotation, item, args.iou)
            results[variant] = {
                "summary": scored["summary"],
                "reentry_events": scored["reentry_events"],
                "run": item["payload"]["run"],
                "prediction_file": Path(item["path"]).name,
            }
            all_frames.extend(scored["frame_rows"])
            all_failures.extend(scored["failure_rows"])

        # 場景分組只取人工逐幀選定的標籤；缺少該欄位的舊標註歸入 unspecified。
        scene_names = sorted({row.get("scene", "unspecified")
                              for row in annotation.get("annotations", [])})
        for scene in scene_names:
            scene_annotation = {
                **annotation,
                "annotations": [row for row in annotation.get("annotations", [])
                                if row.get("scene", "unspecified") == scene],
                "reentry_events": [],
            }
            scenes[scene] = {}
            for item in prediction_inputs:
                scored_scene = evaluate_variant(scene_annotation, item, args.iou)
                scenes[scene][item["payload"]["variant"]] = scored_scene["summary"]

        sensitivity = {
            str(threshold): {
                item["payload"]["variant"]: evaluate_variant(annotation, item, threshold)["summary"]
                for item in prediction_inputs
            }
            for threshold in SENSITIVITY_IOU
        }

        report = {
            "schema_version": 1,
            "status": "completed_for_this_video_not_general_accuracy_claim",
            "video_id": annotation["video"]["id"],
            "video_sha256": annotation["video"]["sha256"],
            "annotation_sha256": sha256_file(args.annotations),
            "iou_threshold": args.iou,
            "ambiguity_margin": AMBIGUITY_MARGIN,
            "visibility_inferred_from_target_boxes": sum(
                row.get("target_visibility") == "unknown"
                and any(person.get("person_id") == annotation["target_person_id"]
                        for person in row.get("people", []))
                for row in annotation.get("annotations", [])),
            "variants": results,
            "scenes": scenes,
            "iou_sensitivity": sensitivity,
            "comparison_note": "A single video compares these runs only; it does not establish general accuracy or statistical significance.",
        }
        (output_dir / "evaluation_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_csv(output_dir / "evaluation_summary.csv", [
            {"variant": variant, **result["summary"]} for variant, result in results.items()
        ])
        write_csv(output_dir / "frame_results.csv", all_frames)
        write_csv(output_dir / "failure_cases.csv", all_failures)
        scene_rows = [
            {"scene": scene, "variant": variant, **metrics}
            for scene, variants in scenes.items()
            for variant, metrics in variants.items()
        ]
        write_csv(output_dir / "scene_summary.csv", scene_rows)
        sensitivity_rows = [
            {"iou_threshold": threshold, "variant": variant, **metrics}
            for threshold, variants in sensitivity.items()
            for variant, metrics in variants.items()
        ]
        write_csv(output_dir / "iou_sensitivity.csv", sensitivity_rows)
        (output_dir / "evaluation_summary.md").write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps({"status": report["status"], "variants": list(results),
                          "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"报告生成失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
