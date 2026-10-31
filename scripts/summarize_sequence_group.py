"""汇总多条完整视频的 ViTrack 离线评测报告。

输入是 summarize_evaluation.py 为每条视频生成的 JSON 报告；本脚本只读取数值，
分别输出按帧加权和按视频等权的汇总，避免把相邻视频帧误当作独立样本。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


VARIANTS = ("hybrid_b0", "hybrid_b1", "hybrid_b2")


def parse_args() -> argparse.Namespace:
    """读取若干单视频报告与结果目录；缺失输入会在开始汇总前报错。"""
    parser = argparse.ArgumentParser(description="汇总多视频模板更新对照实验")
    parser.add_argument("--reports", nargs="+", required=True, type=Path,
                        help="一个或多个 evaluation_summary.json")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="JSON、CSV 和 Markdown 汇总输出目录")
    args = parser.parse_args()
    args.reports = [path.expanduser().resolve() for path in args.reports]
    args.output_dir = args.output_dir.expanduser().resolve()
    if any(not path.is_file() for path in args.reports):
        parser.error("所有 --reports 输入都必须是已有文件")
    if len(args.reports) < 2:
        parser.error("至少需要两条完整视频，才能生成视频等权汇总")
    return args


def load_reports(paths: list[Path]) -> list[dict]:
    """读取并检查单视频报告，输入路径列表，输出报告对象列表。"""
    reports = []
    seen = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        video_id = payload.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError(f"报告视频标识缺失或重复：{path}")
        if not isinstance(payload.get("variants"), dict):
            raise ValueError(f"报告缺少 variants：{path}")
        for variant in VARIANTS:
            if variant not in payload["variants"]:
                raise ValueError(f"{video_id} 缺少方案 {variant}")
        seen.add(video_id)
        reports.append(payload)
    return reports


def ratio(numerator: int, denominator: int) -> float | None:
    """输入计数与分母；没有可评分样本时返回 None 而非伪造 0%。"""
    return numerator / denominator if denominator else None


def finite_number(value: object) -> float | None:
    """把报告中的有限数值规范为 float；非法数值视作缺失。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def build_group_report(reports: list[dict]) -> dict:
    """合并每条视频的计数和诊断。

    帧加权结果先合计分子/分母再求比率；视频等权结果先计算每条视频比率，
    再对有分母的视频取平均。输入单视频报告列表，输出可序列化的汇总对象。
    """
    per_video = []
    variants = {}
    for variant in VARIANTS:
        video_metrics = []
        totals = {key: 0 for key in (
            "visible_frames", "correct_binding_frames", "wrong_person_frames",
            "visible_unbound_frames", "unmatched_binding_frames", "occluded_frames",
            "occluded_frames_with_binding", "timed_frames", "successful_reentry_events",
            "evaluable_reentry_events", "actual_template_updates", "eligible_opportunities",
        )}
        registered_videos = failed_registration_videos = 0
        processing_weighted_sum = 0.0
        processing_sample_count = 0
        quality_values = []
        alpha_values = []
        for report in reports:
            entry = report["variants"][variant]
            metrics = entry["summary"]
            updates = entry.get("body_template_updates", {})
            registration = entry.get("registration") or {}
            registration_status = registration.get("status", "not_recorded")
            registered_videos += registration_status == "registered"
            failed_registration_videos += registration_status == "failed"
            row = {
                "video_id": report["video_id"],
                "video_sha256": report["video_sha256"],
                "registration_status": registration_status,
                "registration_reason": registration.get("reason"),
                "visible_frames": metrics["visible_frames"],
                "correct_binding_frames": metrics["correct_binding_frames"],
                "visible_correct_binding_ratio": metrics["visible_correct_binding_ratio"],
                "wrong_person_frames": metrics["wrong_person_frames"],
                "visible_unbound_frames": metrics["visible_unbound_frames"],
                "unmatched_binding_frames": metrics["unmatched_binding_frames"],
                "occluded_frames": metrics["occluded_frames"],
                "occluded_frames_with_binding": metrics["occluded_frames_with_binding"],
                "timed_frames": metrics["timed_frames"],
                "mean_processing_ms": metrics.get("mean_processing_ms"),
                "successful_reentry_events": metrics.get("successful_reentry_events", 0),
                "evaluable_reentry_events": metrics.get("evaluable_reentry_events", 0),
                "eligible_opportunities": updates.get("eligible_opportunities", 0),
                "actual_template_updates": updates.get("actual_template_updates", 0),
                "quality_median": updates.get("quality_median"),
                "alpha_median": updates.get("alpha_median"),
            }
            video_metrics.append(row)
            for key in totals:
                totals[key] += int(row[key] or 0)
            average_ms = finite_number(row["mean_processing_ms"])
            if average_ms is not None and row["timed_frames"]:
                processing_weighted_sum += average_ms * row["timed_frames"]
                processing_sample_count += row["timed_frames"]
            if finite_number(row["quality_median"]) is not None:
                quality_values.append(float(row["quality_median"]))
            if finite_number(row["alpha_median"]) is not None:
                alpha_values.append(float(row["alpha_median"]))
        per_video.extend({"variant": variant, **row} for row in video_metrics)
        valid_coverages = [finite_number(row["visible_correct_binding_ratio"])
                           for row in video_metrics if row["visible_frames"] > 0]
        valid_coverages = [value for value in valid_coverages if value is not None]
        registered_rows = [row for row in video_metrics
                           if row["registration_status"] == "registered" and row["visible_frames"] > 0]
        registered_coverages = [finite_number(row["visible_correct_binding_ratio"])
                                for row in registered_rows]
        registered_coverages = [value for value in registered_coverages if value is not None]
        registered_visible = sum(row["visible_frames"] for row in registered_rows)
        registered_correct = sum(row["correct_binding_frames"] for row in registered_rows)
        variants[variant] = {
            **totals,
            "visible_correct_binding_ratio_frame_weighted": ratio(
                totals["correct_binding_frames"], totals["visible_frames"]),
            "visible_correct_binding_ratio_video_macro": mean(valid_coverages) if valid_coverages else None,
            "wrong_person_ratio_of_visible_frames": ratio(totals["wrong_person_frames"], totals["visible_frames"]),
            "mean_processing_ms_frame_weighted": (
                processing_weighted_sum / processing_sample_count if processing_sample_count else None),
            "processing_sample_count": processing_sample_count,
            "videos_with_visible_samples": len(valid_coverages),
            # 同时单列已登记成功子集，帮助区分身份更新策略与目标登记失败的影响。
            "registered_only_videos": len(registered_rows),
            "registered_only_visible_frames": registered_visible,
            "registered_only_correct_binding_frames": registered_correct,
            "registered_only_coverage_frame_weighted": ratio(registered_correct, registered_visible),
            "registered_only_coverage_video_macro": mean(registered_coverages) if registered_coverages else None,
            "registration_success_videos": registered_videos,
            "registration_failed_videos": failed_registration_videos,
            "registration_status_recorded_videos": registered_videos + failed_registration_videos,
            "quality_median_across_video_medians": mean(quality_values) if quality_values else None,
            "alpha_median_across_video_medians": mean(alpha_values) if alpha_values else None,
        }
    return {
        "schema_version": 1,
        "status": "exploratory_group_summary_not_statistical_generalization",
        "video_count": len(reports),
        "videos": [{"video_id": report["video_id"],
                    "video_sha256": report["video_sha256"],
                    "annotation_sha256": report["annotation_sha256"]}
                   for report in reports],
        "variants": variants,
        "per_video": per_video,
        "interpretation": (
            "Frame-weighted metrics summarize these exact sequences; adjacent frames are correlated. "
            "Video-macro metrics give each sequence equal weight. The sequence set, not frame count, "
            "defines the number of independent video samples. No significance claim is made."
        ),
    }


def write_outputs(report: dict, output_dir: Path) -> None:
    """将一个汇总对象写成 JSON、便于 Excel 查看的 CSV 和简短 Markdown。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sequence_group_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = list(report["per_video"][0])
    with (output_dir / "per_video_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["per_video"])
    lines = [
        "# ViTrack 多视频身体模板更新对照汇总", "",
        f"视频数：{report['video_count']}。结果状态：探索性汇总，不代表统计显著或普遍泛化。", "",
        "帧加权覆盖率 = 汇总正确绑定帧 / 汇总可见目标帧；视频等权覆盖率为每条有可见帧的视频覆盖率的算术平均。",
        "相邻视频帧高度相关，因此帧数不是独立样本数。处理耗时是离线单帧后端统计，不代表端到端延迟或手机 FPS。", "",
        "| 方案 | 视频数 | 登记成功/失败 | 全集帧加权覆盖率 | 登记成功子集覆盖率 | 子集视频数 | 全集视频等权覆盖率 | 错绑帧 | 未绑定帧 | 实际模板写入 | 平均处理耗时(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in report["variants"].items():
        frame_ratio = metrics["visible_correct_binding_ratio_frame_weighted"]
        macro_ratio = metrics["visible_correct_binding_ratio_video_macro"]
        registered_ratio = metrics["registered_only_coverage_frame_weighted"]
        lines.append(
            f"| {variant} | {metrics['videos_with_visible_samples']} "
            f"| {metrics['registration_success_videos']}/{metrics['registration_failed_videos']} "
            f"| {frame_ratio:.2%} | {registered_ratio:.2%} "
            f"| {metrics['registered_only_videos']} | {macro_ratio:.2%} | {metrics['wrong_person_frames']} "
            f"| {metrics['visible_unbound_frames']} | {metrics['actual_template_updates']} "
            f"| {metrics['mean_processing_ms_frame_weighted']:.2f} |"
        )
    lines.extend(["", "## 各视频结果", "",
                  "| 视频 | 方案 | 登记 | 登记原因 | 可见帧 | 正确绑定 | 覆盖率 | 错绑 | 未绑定 | 模板写入 |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in report["per_video"]:
        lines.append(
            f"| {row['video_id']} | {row['variant']} | {row['registration_status']} "
            f"| {row['registration_reason'] or '—'} | {row['visible_frames']} "
            f"| {row['correct_binding_frames']} | {row['visible_correct_binding_ratio']:.2%} "
            f"| {row['wrong_person_frames']} | {row['visible_unbound_frames']} "
            f"| {row['actual_template_updates']} |"
        )
    lines.extend(["", "## 解释边界", "", report["interpretation"],
                  "MOT17 项目内保留序列不是 MOTChallenge 官方 test split。MOT17 使用 CC BY-NC-SA 3.0；本地数据不随代码或报告分发。"])
    (output_dir / "sequence_group_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """命令行入口：加载单视频 JSON，汇总后写出三种报告格式。"""
    args = parse_args()
    try:
        report = build_group_report(load_reports(args.reports))
        write_outputs(report, args.output_dir)
        print(json.dumps({"status": report["status"], "video_count": report["video_count"],
                          "output_dir": str(args.output_dir)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
