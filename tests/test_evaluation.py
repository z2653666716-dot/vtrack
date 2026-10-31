"""用可手算的小样本检查离线评测统计规则，不读取或生成任何视频画面。

测试输入只包含帧号、视频时间、人工可见性判断、绑定类别和可选耗时。
这些测试保护 app.evaluation 的离线评测逻辑，供后续回放工具复用。
"""

import math

import pytest

from app.evaluation import FrameObservation, evaluate_reentry, summarize_frames, validate_frames


def observation(
    frame_index: int,
    time_seconds: float,
    target_visible: bool | None,
    binding: str | None,
    processing_ms: float | None = None,
) -> FrameObservation:
    """建立一条短小的人工评测记录，参数依次对应一帧的字段。"""
    return FrameObservation(
        frame_index=frame_index,
        time_seconds=time_seconds,
        target_visible=target_visible,
        binding=binding,
        processing_ms=processing_ms,
    )


def test_summary_keeps_binding_errors_and_denominators_separate() -> None:
    """正确、未绑定、错绑、无法对应、离场和未知帧应按定义分开统计。"""
    frames = [
        observation(0, 0.0, True, "target", 30),
        observation(1, 0.1, True, None, 50),
        observation(2, 0.2, True, "other", 40),
        observation(3, 0.3, False, "other", 20),
        observation(4, 0.4, True, "unmatched", 10),
        observation(5, 0.5, None, "target", 99),  # 人工无法判断：不计入身份分母
        observation(6, 0.6, True, None),
    ]

    result = summarize_frames(frames)

    assert result["total_frames"] == 7
    assert result["ignored_frames"] == 1
    assert result["scored_frames"] == 6
    assert result["visible_frames"] == 5
    assert result["absent_frames"] == 1
    assert result["correct_binding_frames"] == 1
    assert result["wrong_person_frames"] == 2  # 目标可见与离场时错绑都计入
    assert result["unmatched_binding_frames"] == 1
    assert result["bound_frames"] == 4
    assert result["matched_bound_frames"] == 3
    assert result["visible_unbound_frames"] == 2
    assert result["visible_wrong_person_frames"] == 1
    assert result["visible_unmatched_binding_frames"] == 1
    assert result["visible_correct_binding_ratio"] == pytest.approx(1 / 5)
    # 定位无法对应的第四个框单独统计，不把它直接算成身份错误。
    assert result["bound_correct_ratio"] == pytest.approx(1 / 3)
    assert result["wrong_person_frame_ratio"] == pytest.approx(2 / 6)
    assert result["wrong_person_bound_ratio"] == pytest.approx(2 / 3)
    assert result["unmatched_bound_ratio"] == pytest.approx(1 / 4)
    # 未知帧不参与耗时均值；没有耗时的可评分帧也不伪造为 0ms。
    assert result["timed_frames"] == 5
    assert result["mean_processing_ms"] == pytest.approx(30)


def test_summary_returns_none_for_empty_denominators() -> None:
    """空样本、全未知样本和没有任何已绑定框时不能伪造 0% 成绩。"""
    empty = summarize_frames([])
    unknown = summarize_frames([observation(0, 0, None, None)])
    visible_unbound = summarize_frames([observation(0, 0, True, None)])

    assert empty["total_frames"] == 0
    assert empty["visible_correct_binding_ratio"] is None
    assert empty["bound_correct_ratio"] is None
    assert unknown["scored_frames"] == 0
    assert unknown["visible_correct_binding_ratio"] is None
    assert visible_unbound["visible_correct_binding_ratio"] == 0
    assert visible_unbound["bound_correct_ratio"] is None
    assert visible_unbound["mean_processing_ms"] is None


@pytest.mark.parametrize(
    "frames",
    [
        [observation(0, 0, True, None), observation(0, 1, True, None)],
        [observation(1, 1, True, None), observation(0, 2, True, None)],
        [observation(0, 1, True, None), observation(1, 1, True, None)],
        [observation(0, 2, True, None), observation(1, 1, True, None)],
        [observation(-1, 0, True, None)],
        [observation(True, 0, True, None)],
        [observation(0, 0, False, "target")],
        [observation(0, 0, True, "someone-else")],
    ],
)
def test_validation_rejects_duplicate_order_and_contradictory_labels(
    frames: list[FrameObservation],
) -> None:
    """重复/倒序帧号、时间以及不可能的标注必须在统计前报错。"""
    with pytest.raises(ValueError):
        validate_frames(frames)


@pytest.mark.parametrize(
    "frame",
    [
        observation(0, math.nan, True, None),
        observation(0, math.inf, True, None),
        observation(0, -0.1, True, None),
        observation(0, True, True, None),
        observation(0, 0, True, None, math.nan),
        observation(0, 0, True, None, math.inf),
        observation(0, 0, True, None, -1),
        observation(0, 0, True, None, True),
        observation(0, 0, 1, None),
    ],
)
def test_validation_rejects_invalid_numbers_and_boolean_as_number(
    frame: FrameObservation,
) -> None:
    """时间和耗时应是有限非负数；bool 不能冒充整数或浮点数。"""
    with pytest.raises(ValueError):
        validate_frames([frame])


def test_reentry_resets_after_wrong_binding_and_separates_two_delays() -> None:
    """中途错绑会清零连续帧；恢复起点与凑齐确认帧的延迟分别返回。"""
    frames = [
        observation(0, 0, False, None),
        observation(1, 1, True, None),  # 重入窗口起点
        observation(2, 2, True, "target"),
        observation(3, 3, True, "other"),  # 失败中断前一段正确观测
        observation(4, 4, True, "target"),
        observation(5, 5, True, "target"),
        observation(6, 6, True, "target"),  # 第三次连续正确，完成确认
        observation(7, 7, True, None),
    ]

    result = evaluate_reentry(frames, 1, 7, required_hits=3, max_gap_seconds=1)

    assert result["complete_observation"] is True
    assert result["success"] is True
    assert result["recovery_seconds"] == 3  # 正确连续段第一帧：4 - 1
    assert result["confirmation_seconds"] == 5  # 第三帧确认：6 - 1
    assert result["sampled_frames"] == 7


def test_reentry_reports_failure_only_for_complete_window() -> None:
    """完整窗口未达到连续帧数才算失败；缺样不能冒充算法失败。"""
    complete = [
        observation(0, 0, False, None),
        observation(1, 1, True, None),
        observation(2, 2, True, "target"),
        observation(3, 3, True, None),
        observation(4, 4, True, "target"),
        observation(5, 5, True, None),
    ]
    sparse = [
        observation(0, 0, False, None),
        observation(1, 1, True, None),
        observation(2, 2, True, "target"),
        observation(3, 4, True, "target"),  # 采样间隔 2 秒，超过允许值
        observation(4, 5, True, None),
    ]
    unknown = [
        observation(0, 0, False, None),
        observation(1, 1, True, None),
        observation(2, 2, True, None),
        observation(3, 3, None, None),  # 人工无法判断会中断连续确认
        observation(4, 4, True, None),
        observation(5, 5, True, None),
    ]

    failed = evaluate_reentry(complete, 1, 5, required_hits=2, max_gap_seconds=1)
    undersampled = evaluate_reentry(sparse, 1, 5, required_hits=2, max_gap_seconds=1)
    indeterminate = evaluate_reentry(unknown, 1, 5, required_hits=2, max_gap_seconds=1)

    assert failed["complete_observation"] is True
    assert failed["success"] is False
    assert failed["recovery_seconds"] is None
    assert undersampled["complete_observation"] is False
    assert undersampled["success"] is None
    assert indeterminate["ignored_frames"] == 1
    assert indeterminate["complete_observation"] is False
    assert indeterminate["success"] is None


@pytest.mark.parametrize(
    "frames,start,end",
    [
        ([observation(0, 0, False, None), observation(1, 2, True, None)], 1, 2),
        ([observation(0, 0, False, None), observation(1, 1, True, None)], 1, 2),
        ([observation(0, 0, False, None), observation(1, 1, False, None),
          observation(2, 2, True, None)], 1, 2),
    ],
)
def test_reentry_rejects_incomplete_or_invalid_window_start(
    frames: list[FrameObservation], start: float, end: float,
) -> None:
    """起点采样缺失、结束范围未覆盖或起点不是目标可见均不能打分。"""
    with pytest.raises(ValueError):
        evaluate_reentry(frames, start, end)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required_hits": 0},
        {"required_hits": True},
        {"max_gap_seconds": 0},
        {"max_gap_seconds": math.inf},
    ],
)
def test_reentry_rejects_invalid_thresholds(kwargs: dict) -> None:
    """确认帧数和最大采样间隔是可解释、有效的正数参数。"""
    frames = [observation(0, 0, True, None), observation(1, 1, True, None)]
    with pytest.raises(ValueError):
        evaluate_reentry(frames, 0, 1, **kwargs)
