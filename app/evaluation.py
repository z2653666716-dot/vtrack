"""离线评测的统计规则，不参与 ViTrack 实时检测与身份判断。

后续视频评测入口会把人工标注与系统结果对齐，生成 FrameObservation，
再调用 summarize_frames 和 evaluate_reentry。这里只接收数值与判断结果，
不读取视频、不加载模型、不保存图片或身份特征，也不改变现有运行参数。
"""

from dataclasses import dataclass
import math
from typing import Literal, Sequence


# target：已绑定的框确实属于登记目标；other：属于其他人。
# unmatched：系统已经绑定，但标注无法支持这个框；不能直接认定为认错人。
# None：未绑定，包括白框、识别中的候选框，以及未检测到目标。
Binding = Literal["target", "other", "unmatched"] | None


@dataclass(frozen=True)
class FrameObservation:
    """一帧的评测输入，由未来的标注对齐模块创建。

    frame_index 是原视频帧号，time_seconds 是视频时间，不是电脑推理时间。
    target_visible 由人工标注决定：True 可见、False 离场、None 无法判断。
    binding 由系统已绑定框与人物标注的对应关系产生，不由系统自报身份决定。
    processing_ms 可记录该帧处理耗时；它不是手机采集到显示的端到端延迟。
    """

    frame_index: int
    time_seconds: float
    target_visible: bool | None
    binding: Binding
    processing_ms: float | None = None
    target_occluded: bool = False
    target_visibility_ratio: float | None = None
    bound_box_count: int = 0


def effective_target_visibility(
    visibility: str,
    people: Sequence[dict],
    target_person_id: str,
) -> tuple[bool | None, bool]:
    """把人工可见性文字转为评测值，并修正常见的“目标框已画但仍为 unknown”。

    输入为人工标注状态、该帧人物框列表和目标人物编号；输出为 True/False/None
    及是否由明确的目标人物框推断为可见。目标框本身是人工确认目标出现在画面中的证据。
    """
    if visibility == "unknown" and any(
        isinstance(person, dict) and person.get("person_id") == target_person_id
        for person in people
    ):
        return True, True
    values = {"visible": True, "absent": False, "unknown": None, "occluded": None}
    if visibility not in values:
        raise ValueError("目标可见状态只能是 visible、absent、occluded 或 unknown")
    # 完全遮挡时即使标注仍提供推测框，也不能据此推断目标可见。
    if visibility == "occluded":
        return None, False
    return values[visibility], False


def validate_frames(frames: Sequence[FrameObservation]) -> None:
    """检查逐帧输入；成功无返回值，错误抛出 ValueError。

    两个统计函数都会调用本函数。拒绝重复帧、倒序时间和矛盾标注，
    防止重复输入增加成功次数，或把错误时间换算成看似合理的恢复速度。
    """
    previous_index, previous_time = -1, -1.0
    for frame in frames:
        if type(frame.frame_index) is not int or frame.frame_index <= previous_index:
            raise ValueError("原视频帧号必须是严格递增的非负整数")
        _check_number(frame.time_seconds, "视频时间")
        if frame.time_seconds <= previous_time:
            raise ValueError("视频时间必须严格递增，不能重复或倒序")
        if frame.target_visible is not None and type(frame.target_visible) is not bool:
            raise ValueError("目标可见性只能是 True、False 或 None")
        if type(frame.target_occluded) is not bool:
            raise ValueError("target_occluded 必须是布尔值")
        if frame.target_occluded and frame.target_visible is not None:
            raise ValueError("完全遮挡帧不能同时标为可见或离场")
        if frame.target_visibility_ratio is not None:
            _check_number(frame.target_visibility_ratio, "目标可见比例")
            if frame.target_visibility_ratio > 1:
                raise ValueError("目标可见比例必须在 0 到 1 之间")
            if frame.target_occluded and frame.target_visibility_ratio != 0:
                raise ValueError("完全遮挡帧的可见比例必须为 0")
            if frame.target_visible is True and frame.target_visibility_ratio == 0:
                raise ValueError("目标可见帧的可见比例必须大于 0")
        if type(frame.bound_box_count) is not int or frame.bound_box_count < 0:
            raise ValueError("已绑定框数量必须是非负整数")
        if frame.binding not in (None, "target", "other", "unmatched"):
            raise ValueError("绑定判断只能是 target、other、unmatched 或 None")
        if frame.target_visible is False and frame.binding == "target":
            raise ValueError("目标离场时，不能标记为正确绑定目标")
        if frame.processing_ms is not None:
            _check_number(frame.processing_ms, "处理耗时")
        previous_index, previous_time = frame.frame_index, frame.time_seconds


def _check_number(value: float, label: str) -> None:
    """检查时间或耗时是有限非负数；拒绝布尔值、NaN 和无穷大。"""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(label + "必须是有限的非负数")


def _ratio(numerator: int, denominator: int) -> float | None:
    """内部计算比例；没有可评样本时返回 None，不用 0 假装已经测过。"""
    return numerator / denominator if denominator else None


def summarize_frames(frames: Sequence[FrameObservation]) -> dict:
    """输入按时间排列的逐帧判断，输出计数、分母和比例字典。

    不可判断帧整体排除。可见帧中，正确、错误、无法对应和未绑定四类互斥；
    目标离场帧仍检查是否误绑旁人。仅看已绑定帧的正确率会掩盖频繁失锁，
    所以同时报告“可见帧正确绑定比例”和“可对应人物的绑定框正确比例”。
    无法与人工人物框对应的绑定框单独计数，不把定位失败直接当成身份认错。
    """
    validate_frames(frames)
    occluded = [frame for frame in frames if frame.target_occluded]
    scored = [frame for frame in frames if frame.target_visible is not None]
    visible = [frame for frame in scored if frame.target_visible]
    correct = sum(frame.binding == "target" for frame in scored)
    wrong = sum(frame.binding == "other" for frame in scored)
    unmatched = sum(frame.binding == "unmatched" for frame in scored)
    bound = correct + wrong + unmatched
    # 只有能和人工人物框建立唯一对应的绑定框，才回答“身份认对没有”。
    # 无法匹配的框另行报告，避免把框偏移误算为错认人。
    matched_bound = correct + wrong
    timings = [frame.processing_ms for frame in frames
               if not frame.target_occluded and frame.target_visible is not None
               and frame.processing_ms is not None]
    visibility_groups = (
        ("(0,0.25]", 0.0, 0.25), ("(0.25,0.5]", 0.25, 0.5),
        ("(0.5,0.75]", 0.5, 0.75), ("(0.75,1]", 0.75, 1.0),
    )
    visibility_strata = {}
    for label, lower, upper in visibility_groups:
        samples = [frame for frame in visible
                   if frame.target_visibility_ratio is not None
                   and lower < frame.target_visibility_ratio <= upper]
        correct_in_bin = sum(frame.binding == "target" for frame in samples)
        visibility_strata[label] = {
            "frames": len(samples),
            "correct_binding_frames": correct_in_bin,
            "correct_binding_ratio": _ratio(correct_in_bin, len(samples)),
        }
    # 延迟独立统计样本数；没有耗时的帧仍参与身份评测，不能填 0 拉低平均值。
    return {
        "total_frames": len(frames),
        "ignored_frames": len(frames) - len(scored),
        "unknown_frames": len(frames) - len(scored) - len(occluded),
        "occluded_frames": len(occluded),
        "occluded_frames_with_binding": sum(frame.bound_box_count > 0 for frame in occluded),
        "occluded_bound_boxes": sum(frame.bound_box_count for frame in occluded),
        "scored_frames": len(scored),
        "visible_frames": len(visible),
        "absent_frames": len(scored) - len(visible),
        "correct_binding_frames": correct,
        "wrong_person_frames": wrong,
        "unmatched_binding_frames": unmatched,
        "bound_frames": bound,
        "matched_bound_frames": matched_bound,
        "visible_unbound_frames": sum(frame.binding is None for frame in visible),
        "visible_wrong_person_frames": sum(frame.binding == "other" for frame in visible),
        "visible_unmatched_binding_frames": sum(frame.binding == "unmatched" for frame in visible),
        "visible_correct_binding_ratio": _ratio(correct, len(visible)),
        "bound_correct_ratio": _ratio(correct, matched_bound),
        "wrong_person_frame_ratio": _ratio(wrong, len(scored)),
        "wrong_person_bound_ratio": _ratio(wrong, matched_bound),
        "unmatched_binding_frame_ratio": _ratio(unmatched, len(scored)),
        "unmatched_bound_ratio": _ratio(unmatched, bound),
        "timed_frames": len(timings),
        "mean_processing_ms": sum(timings) / len(timings) if timings else None,
        "visible_binding_by_visibility_ratio": visibility_strata,
    }


def evaluate_reentry(
    frames: Sequence[FrameObservation],
    start_seconds: float,
    end_seconds: float,
    required_hits: int = 3,
    max_gap_seconds: float = 0.5,
) -> dict:
    """评估人工指定的一次重入窗口，返回恢复与持续确认结果。

    输入：逐帧判断、重新可见时刻、观察截止时刻、连续正确帧数、最大采样间隔。
    窗口必须由标注提供，不能把系统绿框重现当作“目标重新出现”。本函数不推断
    离场原因，也不自动把遮挡算作离开重入；由未来评测入口核验事件标注。

    输出 recovery_seconds 为首次满足稳定条件时，这一组正确帧的起点延迟；
    confirmation_seconds 为凑齐这些帧的延迟。两者均使用视频时间。
    """
    validate_frames(frames)
    _check_number(start_seconds, "重入开始时间")
    _check_number(end_seconds, "重入截止时间")
    _check_number(max_gap_seconds, "最大采样间隔")
    if end_seconds <= start_seconds or max_gap_seconds <= 0:
        raise ValueError("截止时间必须晚于开始时间，最大采样间隔必须大于 0")
    if type(required_hits) is not int or required_hits < 1:
        raise ValueError("连续正确帧数必须是正整数")
    if not frames or frames[0].time_seconds > start_seconds or frames[-1].time_seconds < end_seconds:
        raise ValueError("输入未覆盖完整观察窗口，不能把提前结束的片段算作重入失败")

    hits, streak_start, previous_time = 0, None, None
    maximum_gap, ignored = 0.0, 0
    # 与窗口起点同刻的可评可见标注必须存在，避免采样跳过最初一段而低报恢复时间。
    window = [frame for frame in frames if start_seconds <= frame.time_seconds <= end_seconds]
    if not window or not math.isclose(window[0].time_seconds, start_seconds, abs_tol=1e-6, rel_tol=0):
        raise ValueError("观察窗口必须从一帧已标注为目标可见的采样帧开始")
    if window[0].target_visible is not True:
        raise ValueError("重入开始帧必须明确标注为目标可见")
    recovery, confirmation = None, None
    for frame in window:
        # 固定视频时间避免“电脑跑得慢，目标就离场更久”的评测混淆。
        gap = frame.time_seconds - (previous_time if previous_time is not None else start_seconds)
        maximum_gap = max(maximum_gap, gap)
        if gap > max_gap_seconds:
            hits, streak_start = 0, None
        if frame.target_visible is None:
            ignored += 1
        if frame.target_visible is True and frame.binding == "target":
            if hits == 0:
                streak_start = frame.time_seconds
            hits += 1
            if hits >= required_hits and recovery is None:
                recovery = streak_start - start_seconds
                confirmation = frame.time_seconds - start_seconds
        else:
            # 错绑、白框、离场或无法判断都中断连续确认，不能跳过失败帧凑次数。
            hits, streak_start = 0, None
        previous_time = frame.time_seconds
    maximum_gap = max(maximum_gap, end_seconds - window[-1].time_seconds)
    # 稀疏或有不明标注的窗口仍可报告已经观察到的成功；没成功时不能断言失败。
    complete = maximum_gap <= max_gap_seconds and ignored == 0
    success = True if recovery is not None else (False if complete else None)
    return {
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "required_hits": required_hits,
        "max_gap_seconds": max_gap_seconds,
        "sampled_frames": len(window),
        "ignored_frames": ignored,
        "complete_observation": complete,
        "success": success,
        "recovery_seconds": recovery,
        "confirmation_seconds": confirmation,
    }
