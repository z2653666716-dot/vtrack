"""汇总同一视频中的人脸证据消融结果。

本脚本读取两份纯数值预测：Hybrid 完整方案和关闭人脸证据的方案。它验证两次运行
是否使用同一视频、检测器、跟踪器、身体模型和初始化方式，再按人工给出的目标离场区间
计算可见帧绑定覆盖率。脚本不读取视频画面，也不把绑定覆盖率冒充为身份准确率。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SHARED_CONFIG_KEYS = (
    "tracker", "identity_mode", "reid_model", "det_imgsz", "max_side", "det_conf",
    "det_iou", "det_device", "identity_device", "track_max_age", "body_match_threshold",
    "body_hold_threshold", "reid_confirm_frames",
)


def parse_absent_ranges(values: list[str], frame_count: int) -> list[tuple[int, int]]:
    """解析人工标记的离场区间，返回左闭右开的帧号范围。

    输入示例为 ``["266:389"]``，含义是第266帧起离场，第389帧重新出现；
    输出为可合并计算的整数区间。右端点代表重新出现帧，因此不计入离场。
    """
    ranges = []
    for value in values:
        try:
            start_text, end_text = value.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except (ValueError, TypeError) as error:
            raise ValueError(f"离场区间格式应为 起始帧:重现帧，例如 266:389；收到 {value!r}") from error
        if start < 0 or end <= start or end > frame_count:
            raise ValueError(f"离场区间 {value!r} 超出有效范围 0:{frame_count}")
        ranges.append((start, end))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError("多段离场区间不能互相重叠")
    return ranges


def _bound_frames(rows: list[dict]) -> set[int]:
    """从逐帧预测中取出被系统标为绑定目标的帧号集合。"""
    return {
        row["frame_index"] for row in rows
        if any(box.get("bound_to_target") is True for box in row.get("boxes", []))
    }


def _boxes_match(first: list[dict], second: list[dict]) -> bool:
    """判断两个方案每帧的人物框数量、轨迹号和坐标是否完全一致。"""
    if len(first) != len(second):
        return False
    return all(
        left.get("track_id") == right.get("track_id")
        and left.get("bbox_xyxy") == right.get("bbox_xyxy")
        for left, right in zip(first, second)
    )


def build_ablation_report(full: dict, no_face: dict,
                          absent_ranges: list[tuple[int, int]]) -> tuple[dict, list[dict]]:
    """对齐两份预测并计算人脸证据开关的差异。

    输入是完整 Hybrid 与人脸关闭方案的 JSON 对象，以及人工确认的离场帧区间；
    输出为汇总字典和逐帧数值行。只有配置相同且唯一变化是人脸证据时才继续。
    """
    if full.get("schema_version") != 1 or no_face.get("schema_version") != 1:
        raise ValueError("预测文件版本必须为 schema_version=1")
    if full.get("variant") != "hybrid" or no_face.get("variant") != "hybrid_no_face":
        raise ValueError("请分别提供 hybrid 完整方案和 hybrid_no_face 人脸消融方案")
    if full.get("video", {}).get("sha256") != no_face.get("video", {}).get("sha256"):
        raise ValueError("两份预测不是同一个视频")
    full_run, noface_run = full.get("run", {}), no_face.get("run", {})
    full_config, noface_config = full_run.get("config", {}), noface_run.get("config", {})
    full_shared_config = {key: value for key, value in full_config.items()
                          if key != "face_evidence_enabled"}
    noface_shared_config = {key: value for key, value in noface_config.items()
                            if key != "face_evidence_enabled"}
    if full_shared_config != noface_shared_config:
        changed = sorted(key for key in set(full_shared_config) | set(noface_shared_config)
                         if full_shared_config.get(key) != noface_shared_config.get(key))
        raise ValueError("除人脸证据开关外配置不一致：" + ", ".join(changed))
    if full_config.get("face_evidence_enabled") is not True:
        raise ValueError("完整 Hybrid 预测没有声明启用人脸证据")
    if noface_config.get("face_evidence_enabled") is not False:
        raise ValueError("人脸消融预测没有声明关闭人脸证据")
    full_registration, noface_registration = full.get("registration", {}), no_face.get("registration", {})
    for key in ("frame_index", "effective_frame_index", "track_id"):
        if full_registration.get(key) != noface_registration.get(key):
            raise ValueError(f"两次运行初始化不一致：{key}")
    full_models, noface_models = full_run.get("model_sha256", {}), noface_run.get("model_sha256", {})
    for key in ("detector", "body_reid"):
        if full_models.get(key) != noface_models.get(key):
            raise ValueError(f"两次运行的共享模型不一致：{key}")
    if not full_models.get("face_detector") or not full_models.get("face_recognizer"):
        raise ValueError("完整 Hybrid 预测缺少人脸模型校验值")
    if "face_detector" in noface_models or "face_recognizer" in noface_models:
        raise ValueError("人脸关闭组不应加载或记录人脸模型")

    full_rows, noface_rows = full.get("predictions", []), no_face.get("predictions", [])
    if not full_rows or len(full_rows) != len(noface_rows):
        raise ValueError("两份预测必须包含相同且非空的帧数")
    full_by_frame = {row.get("frame_index"): row for row in full_rows}
    noface_by_frame = {row.get("frame_index"): row for row in noface_rows}
    expected_frames = list(range(len(full_rows)))
    if sorted(full_by_frame) != expected_frames or sorted(noface_by_frame) != expected_frames:
        raise ValueError("预测帧号必须从0开始连续且两份完全对应")
    ranges = parse_absent_ranges(
        [f"{start}:{end}" for start, end in absent_ranges], len(full_rows))
    absent = {index for start, end in ranges for index in range(start, end)}
    visible = set(expected_frames) - absent

    full_bound = _bound_frames(full_rows)
    noface_bound = _bound_frames(noface_rows)
    per_frame = []
    geometry_same_frames = 0
    state_diff_frames = []
    binding_diff_frames = []
    for frame_index in expected_frames:
        full_row, noface_row = full_by_frame[frame_index], noface_by_frame[frame_index]
        if full_row.get("video_time_seconds") != noface_row.get("video_time_seconds"):
            raise ValueError(f"第 {frame_index} 帧视频时间不一致")
        full_boxes, noface_boxes = full_row.get("boxes", []), noface_row.get("boxes", [])
        geometry_same = _boxes_match(full_boxes, noface_boxes)
        geometry_same_frames += int(geometry_same)
        full_states = [box.get("state") for box in full_boxes]
        noface_states = [box.get("state") for box in noface_boxes]
        full_is_bound, noface_is_bound = frame_index in full_bound, frame_index in noface_bound
        if full_states != noface_states:
            state_diff_frames.append(frame_index)
        if full_is_bound != noface_is_bound:
            binding_diff_frames.append(frame_index)
        per_frame.append({
            "frame_index": frame_index,
            "video_time_seconds": full_row.get("video_time_seconds"),
            "target_visible_by_manual_interval": frame_index in visible,
            "hybrid_bound_to_target": full_is_bound,
            "hybrid_no_face_bound_to_target": noface_is_bound,
            "state_changed": full_states != noface_states,
            "boxes_and_tracks_identical": geometry_same,
        })

    visible_count = len(visible)
    full_visible_bound = len(full_bound & visible)
    noface_visible_bound = len(noface_bound & visible)
    full_absent_bound = len(full_bound & absent)
    noface_absent_bound = len(noface_bound & absent)
    full_rate = full_visible_bound / visible_count if visible_count else None
    noface_rate = noface_visible_bound / visible_count if visible_count else None
    summary = {
        "schema_version": 1,
        "kind": "face_evidence_ablation",
        "video": {
            "id": full["video"].get("id"),
            "sha256": full["video"].get("sha256"),
            "reported_frame_count": full["video"].get("reported_frame_count"),
            "decoded_frame_count": len(full_rows),
        },
        "manual_absent_ranges_half_open": [
            {"start_frame": start, "reappearance_frame": end,
             "absent_frame_count": end - start}
            for start, end in ranges
        ],
        "visible_frame_count": visible_count,
        "registration": full.get("registration"),
        "comparison": {
            "changed_factor": "face_evidence_enabled",
            "shared_config": {key: full_config.get(key) for key in SHARED_CONFIG_KEYS},
            "shared_model_hashes": {key: full_models.get(key) for key in ("detector", "body_reid")},
            "human_face_model_hashes": {
                key: full_models[key] for key in ("face_detector", "face_recognizer")
            },
            "geometry_and_track_identical_frames": geometry_same_frames,
            "state_difference_frames": state_diff_frames,
            "bound_flag_difference_frames": binding_diff_frames,
        },
        "metrics": {
            "name": "visible_frame_binding_coverage",
            "hybrid": {"bound_visible_frames": full_visible_bound,
                       "visible_frames": visible_count, "ratio": full_rate,
                       "bound_during_manual_absence": full_absent_bound,
                       "bound_during_absence_frames": sorted(full_bound & absent)},
            "hybrid_no_face": {"bound_visible_frames": noface_visible_bound,
                               "visible_frames": visible_count, "ratio": noface_rate,
                               "bound_during_manual_absence": noface_absent_bound,
                               "bound_during_absence_frames": sorted(noface_bound & absent)},
            "difference_percentage_points_hybrid_minus_no_face": (
                (full_rate - noface_rate) * 100 if full_rate is not None and noface_rate is not None else None
            ),
        },
        "interpretation_limit": (
            "此值统计系统标记绑定的可见帧比例；单段视频不能代表泛化准确率。"
            "若人工发现绿框实际错人，应从相应方案的可见绑定帧中扣除。"
            "离场区间内的绑定单独列出，不计入可见帧分母或分子。"
        ),
        "privacy": {"frames_exported": False, "crops_exported": False,
                    "features_exported": False},
    }
    return summary, per_frame


def parse_args() -> argparse.Namespace:
    """读取两份预测、人工离场区间和结果目录，不接受视频或媒体输入。"""
    parser = argparse.ArgumentParser(description="比较 Hybrid 是否受益于人脸证据")
    parser.add_argument("--hybrid", required=True, type=Path, help="人脸证据开启的 hybrid 预测 JSON")
    parser.add_argument("--no-face", required=True, type=Path, help="关闭人脸证据的 hybrid_no_face 预测 JSON")
    parser.add_argument("--absent-range", required=True, action="append",
                        help="人工确认的离场范围，左闭右开；例如 266:389 表示266离场、389重现，可重复填写")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "evaluation" / "face_ablation")
    args = parser.parse_args()
    args.hybrid = args.hybrid.expanduser().resolve()
    args.no_face = args.no_face.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.hybrid.is_file() or not args.no_face.is_file():
        parser.error("两份预测 JSON 必须存在")
    return args


def main() -> int:
    """读取预测并输出 JSON、CSV、Markdown 汇总；不会重新打开视频。"""
    args = parse_args()
    try:
        full = json.loads(args.hybrid.read_text(encoding="utf-8"))
        no_face = json.loads(args.no_face.read_text(encoding="utf-8"))
        full_count = len(full.get("predictions", []))
        ranges = parse_absent_ranges(args.absent_range, full_count)
        summary, rows = build_ablation_report(full, no_face, ranges)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "ablation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output_dir / "ablation_frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metrics = summary["metrics"]
        full_rate = metrics["hybrid"]["ratio"]
        noface_rate = metrics["hybrid_no_face"]["ratio"]
        markdown = (
            "# 人脸证据消融结果\n\n"
            f"- 解码帧数：{summary['video']['decoded_frame_count']}；容器报告帧数：{summary['video']['reported_frame_count']}。\n"
            f"- 人工离场帧：{sum(item['absent_frame_count'] for item in summary['manual_absent_ranges_half_open'])}；"
            f"可见帧：{summary['visible_frame_count']}。\n"
            f"- Hybrid：{metrics['hybrid']['bound_visible_frames']}/{metrics['hybrid']['visible_frames']}"
            f"（{full_rate:.2%}）；离场区间仍标为绑定 {metrics['hybrid']['bound_during_manual_absence']} 帧。\n"
            f"- 关闭人脸：{metrics['hybrid_no_face']['bound_visible_frames']}/{metrics['hybrid_no_face']['visible_frames']}"
            f"（{noface_rate:.2%}）；离场区间仍标为绑定 {metrics['hybrid_no_face']['bound_during_manual_absence']} 帧。\n"
            f"- 可见帧比例差（Hybrid − 关闭人脸）：{metrics['difference_percentage_points_hybrid_minus_no_face']:.2f} 个百分点。\n"
            f"- 两方案框与轨迹完全一致：{summary['comparison']['geometry_and_track_identical_frames']}"
            f"/{summary['video']['decoded_frame_count']} 帧；状态差异帧：{summary['comparison']['state_difference_frames']}；"
            f"绑定标记差异帧：{summary['comparison']['bound_flag_difference_frames']}。\n\n"
            f"> {summary['interpretation_limit']}\n"
        )
        (args.output_dir / "ablation_summary.md").write_text(markdown, encoding="utf-8")
        print(json.dumps({"status": "completed", "output_dir": str(args.output_dir),
                          "visible_frame_count": summary["visible_frame_count"],
                          "hybrid_rate": full_rate, "no_face_rate": noface_rate}, ensure_ascii=False))
        return 0
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        print(f"消融汇总失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
