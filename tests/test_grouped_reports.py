"""验证多视频报告按帧加权和按视频等权的计算口径。"""

from scripts.summarize_sequence_group import build_group_report


def make_video(video_id: str, visible: int, correct: int) -> dict:
    """构造精简单视频报告，模拟评测汇总器的输出结构。"""
    return {
        "video_id": video_id,
        "video_sha256": f"video-{video_id}",
        "annotation_sha256": f"annotation-{video_id}",
        "variants": {
            name: {
                "summary": {
                    "visible_frames": visible,
                    "correct_binding_frames": correct,
                    "visible_correct_binding_ratio": correct / visible,
                    "wrong_person_frames": 0,
                    "visible_unbound_frames": visible - correct,
                    "unmatched_binding_frames": 0,
                    "occluded_frames": 0,
                    "occluded_frames_with_binding": 0,
                    "timed_frames": visible,
                    "mean_processing_ms": 10.0,
                    "successful_reentry_events": 0,
                    "evaluable_reentry_events": 0,
                },
                "body_template_updates": {
                    "eligible_opportunities": 1,
                    "actual_template_updates": 1,
                    "quality_median": 0.5,
                    "alpha_median": 0.1,
                },
            }
            for name in ("hybrid_b0", "hybrid_b1", "hybrid_b2")
        },
    }


def test_group_summary_separates_frame_weighted_and_video_macro_rates():
    """大视频按帧数影响帧加权结果；视频宏平均仍让两条视频等权。"""
    report = build_group_report([
        make_video("short", visible=10, correct=5),
        make_video("long", visible=90, correct=90),
    ])
    metrics = report["variants"]["hybrid_b0"]
    assert metrics["visible_frames"] == 100
    assert metrics["correct_binding_frames"] == 95
    assert metrics["visible_correct_binding_ratio_frame_weighted"] == 0.95
    assert metrics["visible_correct_binding_ratio_video_macro"] == 0.75
    assert metrics["actual_template_updates"] == 2


def test_group_summary_keeps_missing_video_score_as_none():
    """某视频没有可见目标帧时不伪造 0% 并纳入视频等权平均。"""
    empty = make_video("empty", visible=1, correct=0)
    scored = make_video("scored", visible=4, correct=2)
    empty["variants"]["hybrid_b0"]["summary"]["visible_frames"] = 0
    empty["variants"]["hybrid_b0"]["summary"]["correct_binding_frames"] = 0
    empty["variants"]["hybrid_b0"]["summary"]["visible_correct_binding_ratio"] = None
    report = build_group_report([empty, scored])
    metrics = report["variants"]["hybrid_b0"]
    assert metrics["visible_correct_binding_ratio_frame_weighted"] == 0.5
    assert metrics["visible_correct_binding_ratio_video_macro"] == 0.5
    assert metrics["videos_with_visible_samples"] == 1
