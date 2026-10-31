"""实验身份保持：BoT-SORT连续运动、OSNet行人特征与SFace人脸证据分别承担不同职责。

Pipeline只在identity_mode=hybrid时创建本类，main的appearance基线不改变。
输入当前帧/检测/用户登记，输出匿名身份状态；模板全在内存，停止或删除时清除。
"""
from dataclasses import dataclass
from time import monotonic

import cv2

from .reid import cosine, crop_target, normalize
from .schemas import Box, Detection
from .session import Observation, SessionError, VideoSession


def overlap(first, second):
    """输入两个人物框，输出交并比；用于判断跨人遮挡和连续运动是否有歧义。"""
    width = max(0., min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0., min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / max(area(first) + area(second) - intersection, 1e-8)


@dataclass
class HybridObservation(Observation):
    """在原有轨迹上增加可靠观测时间、连续性和谨慎的人脸补充计数，无历史图片。"""
    last_verified: float = 0.
    motion_ok: bool = False
    face_learning: int = 0
    diagnostic_frame: int = -30
    candidate_start_frame: int = 0
    candidate_last_positive: float = 0.
    candidate_gaps: int = 0
    last_box: tuple | None = None
    last_seen_at: float = 0.
    observation_gap: int = 0
    face_seed: object = None
    # 只有用户手动选中的连续轨迹才能在尚无人脸模板时补充距离变化；重入新轨迹默认无此许可。
    body_learning_allowed: bool = False


class HybridSession(VideoSession):
    """由Pipeline串行调用；身份重入需要新证据，短时连续跟踪不再要求复现登记姿势。"""

    def __init__(self, config, extractor=None, face_encoder=None, clock=None):
        super().__init__(config, extractor, clock=clock)
        self.face_encoder = face_encoder
        self.face_templates, self.body_valid, self.current_faces = {}, {}, {}
        self.current_body_features = {}
        self.unknown_body_overlap = set()
        self.detected_faces = None
        self.face_only_indices, self.face_only_ids = set(), set()
        self.head_only_ids = set()
        self.diagnostic("hybrid_policy", face_match=config.face_match_threshold, face_reject=config.face_reject_threshold,
                        body_match=config.body_match_threshold, body_hold=config.body_hold_threshold,
                        grace_seconds=config.identity_hold_seconds, face_interval=3, candidate_gap_frames=2)

    def prepare_frame(self, frame, detections):
        """全帧只检测一次人脸；近距离只见脸、YOLO未给人物框时，补充明确标为person-face的观测。"""
        # 搜索目标时逐帧提脸；已有连续绑定时每3帧做一次较重的人脸前向，中间不复用旧脸向量。
        bound = any(track.identity_id for track in self.observations.values() if track.last_frame == self.frame_count)
        self.detected_faces = self.face_encoder.read(frame, recognize=not bound or self.frame_count % 3 == 0)
        output = list(detections)
        self.face_only_indices = set()
        height, width = frame.shape[:2]
        for face in self.detected_faces:
            left, top, right, bottom = face.xyxy
            x, y = (left + right) / 2, (top + bottom) / 2
            if any(item.cls == 0 and item.xyxy[0] <= x <= item.xyxy[2] and item.xyxy[1] <= y <= item.xyxy[3] for item in detections):
                continue
            if right <= left or bottom <= top:
                continue
            self.face_only_indices.add(len(output))
            # 这不是YOLO的person分数；名称明确区分人脸补充观测，避免混用两种检测来源。
            output.append(Detection(cls=0, name="person-face", score=float(face.confidence),
                                    xyxy=(float(left), float(top), float(right), float(bottom))))
        return output

    def update(self, detections, width, height, frame=None):
        """先关联，再检查最近真实位置；仅原轨迹短漏检、位置连贯且未超时才保留进度。"""
        fields = ("last_frame", "last_box", "last_seen_at", "hits", "candidate", "candidate_hits",
                  "candidate_start_frame", "candidate_last_positive", "candidate_gaps")
        previous = {key: {name: getattr(track, name, 0) for name in fields}
                    for key, track in self.observations.items()} if self.size == (width, height) else {}
        boxes = super().update(detections, width, height, frame)
        self.face_only_ids = {box.track_id for index, box in enumerate(boxes) if index in self.face_only_indices and box.track_id is not None}
        for key, track in list(self.observations.items()):
            if not isinstance(track, HybridObservation):
                self.observations[key] = HybridObservation(**vars(track))
        for box in boxes:
            if box.track_id is not None:
                track = self.observations[box.track_id]
                old, now = previous.get(box.track_id), self.clock()
                gap = self.frame_count - old["last_frame"] - 1 if old else 0
                track.motion_ok = bool(old and old["last_box"] and gap <= 2 and
                                       overlap(box.xyxy, old["last_box"]) > .15 and
                                       (gap == 0 or now - old["last_seen_at"] <= self.config.identity_hold_seconds))
                track.observation_gap = gap
                if gap > 2 or (old is not None and not track.motion_ok):
                    track.body_learning_allowed = False
                    track.face_learning, track.face_seed = 0, None
                if gap > 0 and track.motion_ok:
                    track.hits = old["hits"] + 1
                    if old["candidate"]:
                        # 漏掉的帧只占等待额度，不能增加可靠成功次数；之后还会经过当前人脸/多人竞争检查。
                        for name in ("candidate", "candidate_hits", "candidate_start_frame", "candidate_last_positive"):
                            setattr(track, name, old[name])
                        track.candidate_gaps = old["candidate_gaps"] + gap
                track.last_box, track.last_seen_at = box.xyxy, now
        return boxes

    def body_score(self, feature, identity_id):
        """仅比较OSNet的512维行人分量，颜色不作为强制身份门槛，降低姿态/照明影响。"""
        if feature is None or not self.body_valid.get(identity_id):
            return -1.
        return max(cosine(feature[:512], template[:512]) for template in self.gallery.items[identity_id].templates)

    def evidence(self, track_id, feature, identity_id, crowded, continuing=False):
        """输入本帧证据和连续性，输出可否绑定/维持及原因；人脸质量不足不等于确认换人。"""
        track, identity = self.observations[track_id], self.gallery.items[identity_id]
        sample = self.current_faces.get(track_id)
        templates = self.face_templates.get(identity_id, [])
        face_score = max(cosine(sample.feature, item) for item in templates) if templates and sample is not None and sample.feature is not None else None
        body_score = self.body_score(feature, identity_id)
        negative = max((cosine(feature[:512], item[:512]) for item in identity.distractors), default=-1.) if feature is not None else -1.
        face_positive = face_score is not None and bool(templates) and face_score >= self.config.face_match_threshold
        body_positive = body_score >= self.config.body_match_threshold and body_score - negative >= self.config.reid_margin
        other_body = max((self.body_score(value, identity_id) for key, value in self.current_body_features.items()
                          if key != track_id), default=-1.)
        body_unique = (body_positive and body_score - other_body >= self.config.reid_margin
                       and track_id not in self.unknown_body_overlap)
        mode, reason = "none", "NO_RELIABLE_IDENTITY"
        if face_positive:
            mode, reason = "face", "FACE_CONFIRMED"
        elif face_score is not None and templates and face_score < self.config.face_reject_threshold and sample.reason != "FACE_PROFILE":
            reason = "FACE_CONTRADICTION"
        elif face_score is not None and templates:
            # 可用人脸但尚未通过匹配时，只允许短暂等待，不能让相似衣服无限盖过人脸冲突。
            reason = "FACE_UNCERTAIN"
            if continuing and track.motion_ok and not crowded:
                if track_id in self.head_only_ids and sample.reason == "FACE_PROFILE":
                    mode, reason = "continuity", "CONTINUOUS_PROFILE_TRACK"
                elif (track_id in self.head_only_ids or body_score >= self.config.body_hold_threshold) and self.clock() - track.last_verified <= self.config.identity_hold_seconds:
                    mode, reason = "continuity", "SHORT_FACE_UNCERTAINTY"
        elif (continuing and track.motion_ok and crowded and body_unique and
              self.clock() - track.last_verified <= self.config.identity_hold_seconds):
            # 人脸归属不清时不强行借用另一张脸；原轨迹的身体还明显领先旁人，只允许有限等待。
            # continuity不会刷新last_verified，因此不能靠同一份身体依据无限延长遮挡等待。
            mode, reason = "continuity", "SHORT_UNIQUE_BODY_OVERLAP"
        elif continuing and track.motion_ok and not crowded:
            if (track_id in self.head_only_ids and templates and sample is not None
                    and sample.reason in {"FACE_EXTREME_ANGLE", "FACE_OBSERVATION_INTERVAL"}):
                # 近距离只有头部时，不能把脸裁剪送给行人模型后要求“衣着相同”。
                # 已确认目标的脸仍持续检出、运动连贯且无其他人交叠时，侧脸可以继续跟踪，明确标记continuity。
                mode, reason = "continuity", "CONTINUOUS_FACE_TRACK"
            elif body_positive:
                mode, reason = "body", "BODY_CONFIRMED"
            elif body_score >= self.config.body_hold_threshold and body_score - negative >= self.config.reid_margin:
                mode, reason = "continuity", "CONTINUOUS_BODY_TRACK"
            elif face_score is None and self.clock() - track.last_verified <= self.config.identity_hold_seconds:
                mode, reason = "continuity", "SHORT_OBSERVATION_GAP"
        elif not continuing and not templates and body_positive and not crowded:
            mode, reason = "body", "BODY_CONFIRMED"
        if mode == "none" and crowded and not face_positive:
            reason = "AMBIGUOUS_OVERLAP"
        return {"accepted": mode != "none", "mode": mode, "reason": reason,
                "body_score": round(body_score, 4), "face_score": round(face_score, 4) if face_score is not None else None,
                "face_quality": sample.reason if sample is not None else "NO_FACE",
                "face_threshold": self.config.face_match_threshold, "body_threshold": self.config.body_match_threshold,
                "body_hold_threshold": self.config.body_hold_threshold,
                "other_body_score": round(other_body, 4), "body_unique": bool(body_unique),
                "motion_ok": bool(track.motion_ok), "observation_gap": track.observation_gap,
                "score": face_score if mode == "face" else body_score}

    def remember_body_transition(self, key, feature, identity_id, check):
        """输入当前身体依据，输出是否足以补脸，并谨慎记住过渡距离的身体外观。

        由remember_confirmed调用。仅用户手动选择的原连续轨迹、身体匹配至少0.85且领先旁人时
        才允许补模板；同时仍须与最初身体达到维持门槛，避免相邻模板无限漂移到另一个人。
        Gallery.append保留原模板且限制总数，所有向量仅在会话内存。
        """
        track = self.observations[key]
        if self.face_templates.get(identity_id) or feature is None or not self.body_valid.get(identity_id):
            return False
        anchor = self.gallery.items[identity_id].templates[0]
        # 新入镜白框尚无轨迹号/身体特征时，不能把“还没比较”误当成目标已领先。
        unmeasured_person = any(box.cls == 0 and box.track_id != key
                                and box.track_id not in self.current_body_features for box in self.boxes)
        reliable = bool(track.body_learning_allowed and track.motion_ok and track.observation_gap == 0
                        and check["mode"] == "body" and check["body_score"] >= .85
                        and check.get("body_unique", False) and not unmeasured_person
                        and cosine(feature[:512], anchor[:512]) >= self.config.body_hold_threshold)
        if reliable:
            # 远景到近景逐步变化时先保留强匹配的过渡外观，不能等已经有人脸才更新身体。
            # 不降低重入或补脸门槛；突然变化、旁人同分、只有头部和失踪重入均不能借此学习。
            self.gallery.append(identity_id, feature)
        return reliable

    def remember_confirmed(self, key, feature, identity_id, check):
        """输入当前身份依据，补充可靠模板；首份人脸之前仅允许手动原轨迹的受限身体过渡。"""
        track = self.observations[key]
        sample = self.current_faces.get(key)
        if check["mode"] in {"face", "body"}:
            track.last_verified = self.clock()
        if check["mode"] == "face":
            if feature is not None:
                if not self.body_valid.get(identity_id):
                    self.gallery.items[identity_id].templates = [normalize(feature).copy()]
                    self.body_valid[identity_id] = True
                else:
                    self.gallery.append(identity_id, feature)
            templates = self.face_templates[identity_id]
            if sample is not None and sample.feature is not None and check["face_score"] >= max(.55, self.config.face_match_threshold):
                if (cosine(sample.feature, templates[0]) >= self.config.face_match_threshold and
                        max(cosine(sample.feature, item) for item in templates) < .97 and self.config.reid_max_templates > 1):
                    templates.append(sample.feature.copy())
                    if len(templates) > self.config.reid_max_templates:
                        del templates[1]
        # 远处登记时脸太小：走近后仍需6份互相一致的清晰正脸，身体还须明显领先旁人。
        # 每三帧主动跳过的前向不代表观测失败；这些帧既不加次数，也不抹掉先前的可靠采样。
        if self.face_templates.get(identity_id):
            track.face_learning, track.face_seed = 0, None
            return
        strong_body = self.remember_body_transition(key, feature, identity_id, check)
        if strong_body and sample is not None and sample.feature is not None and sample.reason == "READY":
            if track.face_seed is None or cosine(sample.feature, track.face_seed) < self.config.face_match_threshold:
                track.face_seed, track.face_learning = sample.feature.copy(), 1
            else:
                track.face_learning += 1
            if track.face_learning >= 6:
                self.face_templates[identity_id] = [track.face_seed.copy()]
                track.face_seed, track.face_learning = None, 0
                self.diagnostic("face_enrolled_after_body_confirmation", identity_id=identity_id, track_id=key)
        elif not (strong_body and sample is not None and sample.reason == "FACE_OBSERVATION_INTERVAL"):
            track.face_learning, track.face_seed = 0, None

    def reset_candidate(self, track_id, reason):
        """清除某轨迹的确认进度并记下原因；不删除已经登记的人脸/身体模板。"""
        track = self.observations[track_id]
        if track.candidate is not None:
            self.diagnostic("candidate_reset", track_id=track_id, reason=reason, hits=track.candidate_hits,
                            gaps=track.candidate_gaps)
        track.candidate, track.candidate_hits = None, 0
        track.candidate_start_frame, track.candidate_last_positive, track.candidate_gaps = 0, 0., 0

    def progress_candidate(self, track_id, identity_id, check, allow_gap=False):
        """输入本帧真实匹配结果，输出是否继续显示候选；短缺失只保留进度，绝不增加成功次数。

        process先完成全局候选竞争再调用。最多容忍连续两帧中性缺失，且仍受最近证据时间、
        总确认窗口和运动连续性约束；明确异人脸、跨人歧义或另一候选胜出会直接清零。
        """
        track, now = self.observations[track_id], self.clock()
        window = max(self.config.reid_confirm_frames * 2, self.config.reid_confirm_frames + 2)
        if track.candidate is not None and self.frame_count - track.candidate_start_frame >= window:
            self.reset_candidate(track_id, "CONFIRMATION_WINDOW_EXPIRED")
        if track.candidate is not None and track.candidate_gaps > 2:
            self.reset_candidate(track_id, "OBSERVATION_GAP_LIMIT")
        # 只有通过原有人脸/身体门槛的当前帧才能计数，不能用连续运动猜测凑满六次。
        if check is not None and check["accepted"]:
            if track.candidate != identity_id:
                self.reset_candidate(track_id, "CANDIDATE_CHANGED")
                track.candidate = identity_id
                track.candidate_start_frame = self.frame_count
            elif track.candidate_gaps and now - track.candidate_last_positive > self.config.identity_hold_seconds:
                self.reset_candidate(track_id, "OBSERVATION_GAP_TIMEOUT")
                track.candidate, track.candidate_start_frame = identity_id, self.frame_count
            track.candidate_hits += 1
            track.candidate_last_positive, track.candidate_gaps = now, 0
            if track.candidate_hits == 1:
                self.diagnostic("candidate_started", track_id=track_id, identity_id=identity_id,
                                hits=1, required=self.config.reid_confirm_frames)
            return True
        if allow_gap and track.candidate == identity_id:
            track.candidate_gaps += 1
            if (track.candidate_gaps <= 2 and
                    now - track.candidate_last_positive <= self.config.identity_hold_seconds):
                if track.candidate_gaps == 1:
                    self.diagnostic("candidate_waiting", track_id=track_id, hits=track.candidate_hits,
                                    reason=check["reason"], gaps=track.candidate_gaps)
                return True
            reason = "OBSERVATION_GAP_LIMIT" if track.candidate_gaps > 2 else "OBSERVATION_GAP_TIMEOUT"
        else:
            reason = check["reason"] if check is not None and not check["accepted"] else "CANDIDATE_COMPETITION"
        self.reset_candidate(track_id, reason)
        return False

    def process(self, frame, seq):
        """按帧提取证据，先验证已有绑定，再为未绑定轨迹竞争同一个目标，最后输出状态。"""
        self.frame, self.seq = frame, seq
        self.events, self.pending_events = self.pending_events, []
        all_people = [box for box in self.boxes if box.cls == 0]
        people = [box for box in all_people if box.track_id is not None]
        crops = [(box.track_id, crop_target(frame, box.xyxy)) for box in people]
        crops = [(key, crop) for key, crop in crops if crop is not None]
        features = dict(zip([key for key, _ in crops], self.extractor.extract([crop for _, crop in crops])))
        # 只有脸的补充框不能作为全身OSNet身份依据，避免把两种输入混为一谈。
        features = {key: value for key, value in features.items() if key not in self.face_only_ids}
        faces = self.detected_faces if self.detected_faces is not None else self.face_encoder.read(frame)
        self.detected_faces = None
        self.current_faces = self.face_encoder.associate(faces, all_people, frame.shape[1], frame.shape[0])
        self.head_only_ids = set(self.face_only_ids)
        for box in people:
            sample = self.current_faces.get(box.track_id)
            if sample is not None and (sample.xyxy[3] - sample.xyxy[1]) / max((box.xyxy[3] - box.xyxy[1]) * frame.shape[0], 1) > .45:
                self.head_only_ids.add(box.track_id)
        features = {key: value for key, value in features.items() if key not in self.head_only_ids}
        self.current_body_features = features
        # 新白框或仅头部的人没有可比较的身体向量；不能因为“没分数”就宣称目标胜过了他。
        self.unknown_body_overlap = {box.track_id for box in people if any(
            other is not box and other.track_id not in features and overlap(box.xyxy, other.xyxy) > .15
            for other in all_people)}
        # 身体框交叠并不必然代表串人：两张脸仍各自唯一归属时，间隔提脸帧也可以连续跟踪。
        # associate会把同时落在两个人框内的脸排除，无轨迹号的新白框也参与该歧义判断。
        crowded = {box.track_id: box.track_id not in self.current_faces and
                   any(other.track_id != box.track_id and overlap(box.xyxy, other.xyxy) > .15 for other in all_people)
                   for box in people}
        visible = {box.track_id for box in people}
        if visible != self._visible_tracks:
            self.diagnostic("visible_tracks_changed", visible=sorted(visible), missing=sorted(self._visible_tracks - visible))
            self._visible_tracks = visible
        checks, claimed = {}, set()
        for box in people:
            key, track = box.track_id, self.observations[box.track_id]
            feature = features.get(key)
            track.feature = feature
            if track.identity_id not in self.gallery.items:
                track.identity_id = None
            if track.identity_id:
                check = self.evidence(key, feature, track.identity_id, crowded[key], continuing=True)
                if track.identity_id in claimed:
                    check.update(accepted=False, reason="IDENTITY_ALREADY_CLAIMED")
                if not check["accepted"]:
                    self.diagnostic("identity_unbound", track_id=key, identity_id=track.identity_id, check=check)
                    self.events.append({"type": "event", "name": "unbound", "identity_id": track.identity_id})
                    track.identity_id = None
                    track.body_learning_allowed = False
                    track.face_learning, track.face_seed = 0, None
                    self.reset_candidate(key, check["reason"])
                else:
                    claimed.add(track.identity_id)
                    checks[key] = check
                    self.remember_confirmed(key, feature, track.identity_id, check)
                    if self.frame_count - track.diagnostic_frame >= 30:
                        self.diagnostic("identity_kept", track_id=key, identity_id=track.identity_id, check=check)
                        track.diagnostic_frame = self.frame_count
        # 单目标库中只有一个身份；同帧多个可疑人物必须有领先差距，不能按检测顺序抢占。
        waiting, candidate_checks = [], {}
        identity_id = next(iter(self.gallery.items), None)
        if identity_id and identity_id not in claimed:
            for box in people:
                key, track = box.track_id, self.observations[box.track_id]
                if track.identity_id is None and track.hits >= 3:
                    check = self.evidence(key, features.get(key), identity_id, crowded[key])
                    candidate_checks[key] = check
                    if check["accepted"]:
                        waiting.append((check["score"], key, check))
                    else:
                        self.log_rejection(key, check)
            waiting.sort(key=lambda item: item[0], reverse=True)
        winner = waiting[0] if waiting and (len(waiting) == 1 or waiting[0][0] - waiting[1][0] >= self.config.reid_margin) else None
        output = []
        for box in self.boxes:
            track = self.observations.get(box.track_id)
            check = checks.get(box.track_id)
            state = "locked" if check else "none"
            if track is not None and track.identity_id is None:
                if winner and winner[1] == box.track_id:
                    check = winner[2]
                    self.progress_candidate(box.track_id, identity_id, check)
                    state = "candidate"
                    if track.candidate_hits >= self.config.reid_confirm_frames:
                        for other in self.observations.values():
                            if other is not track and other.identity_id == identity_id:
                                other.identity_id = None
                        track.identity_id = identity_id
                        state = "locked"
                        self.remember_confirmed(box.track_id, features.get(box.track_id), identity_id, check)
                        self.events.append({"type": "event", "name": "relocked", "identity_id": identity_id})
                        self.diagnostic("identity_confirmed", track_id=box.track_id, identity_id=identity_id, mode=check["mode"], frames=track.candidate_hits)
                else:
                    assessment = candidate_checks.get(box.track_id)
                    if assessment is not None and assessment["accepted"]:
                        # 单人分数过线不等于全局竞争胜出；并列或落后时不能继续累计成功证据。
                        assessment = {**assessment, "accepted": False, "reason": "CANDIDATE_COMPETITION"}
                    # 仅在没有别的可靠候选、没有跨人歧义、运动仍连贯时等待短缺失。
                    # 多个可靠候选分数相近属于竞争，不是缺帧，不能沿用先前进度强行选人。
                    unique_body_gap = bool(assessment and assessment.get("body_unique") and
                                           assessment["reason"] == "AMBIGUOUS_OVERLAP")
                    neutral = (assessment is not None and assessment["reason"] in
                               {"NO_RELIABLE_IDENTITY", "FACE_UNCERTAIN"}) or unique_body_gap
                    allow_gap = (not waiting and identity_id not in claimed and neutral and
                                 track.motion_ok and (not crowded.get(box.track_id, True) or unique_body_gap))
                    if self.progress_candidate(box.track_id, identity_id, assessment, allow_gap):
                        state = "candidate"
            output.append(box.model_copy(update={"state": state, "identity_id": track.identity_id if state == "locked" else None,
                "sim": max(-1., min(1., check["score"])) if check and check["score"] >= 0 else None,
                "verification": check["mode"] if check else "none"}))
        self.boxes = output
        online = {box.identity_id for box in output if box.identity_id}
        self.events.extend({"type": "event", "name": "lost", "identity_id": key} for key in self.was_online - online)
        self.was_online, self.updated_at = online, self.clock()
        self.write_events()
        self.recorder.write(frame, output, seq, self.events)
        return output

    def register_identity(self, feature, face_feature=None, body_valid=True, **details):
        """统一单目标注册；人脸与身体模板分别存放，脸部特写不能伪装成全身模板。"""
        identity = super().register_identity(feature, **details)
        self.face_templates[identity.id] = [normalize(face_feature).copy()] if face_feature is not None else []
        self.body_valid[identity.id] = body_valid
        return identity

    def lock(self, track_id, seq, action="toggle"):
        """沿用明确锁定/取消语义；登记时有唯一高质量脸便补充专用人脸特征。"""
        previous = next(iter(self.gallery.items), None)
        current = self.observations.get(track_id)
        already_bound = current is not None and current.identity_id in self.gallery.items
        if track_id in self.head_only_ids and action != "unlock" and not already_bound:
            sample = self.current_faces.get(track_id)
            if sample is None or sample.feature is None:
                raise SessionError("FACE_NOT_READY", "目前只看到人脸，但质量不足，请正对镜头片刻后再锁定")
        reply = super().lock(track_id, seq, action)
        identity_id = reply.get("identity_id")
        if identity_id and identity_id != previous:
            sample = self.current_faces.get(track_id)
            if sample is not None and sample.feature is not None:
                self.face_templates[identity_id] = [sample.feature.copy()]
            self.body_valid[identity_id] = track_id not in self.head_only_ids
            self.observations[track_id].last_verified = self.clock()
            self.observations[track_id].body_learning_allowed = self.body_valid[identity_id]
        return reply

    def clear_lock(self, reason="cancel"):
        super().clear_lock(reason)
        self.face_templates.clear()
        self.body_valid.clear()
        self.current_body_features.clear()
        self.unknown_body_overlap.clear()
        for track in self.observations.values():
            if isinstance(track, HybridObservation):
                track.face_learning, track.face_seed = 0, None
                track.body_learning_allowed = False

    def delete(self, identity_id):
        super().delete(identity_id)
        self.face_templates.pop(identity_id, None)
        self.body_valid.pop(identity_id, None)
        for track in self.observations.values():
            if isinstance(track, HybridObservation):
                track.face_learning, track.face_seed = 0, None
                track.body_learning_allowed = False

    def reset(self):
        super().reset()
        self.face_templates.clear()
        self.body_valid.clear()
        self.current_faces.clear()
        self.current_body_features.clear()
        self.unknown_body_overlap.clear()
        self.detected_faces = None
        self.face_only_ids.clear()
        self.head_only_ids.clear()
        if self.extractor is not None and hasattr(self.extractor, "person_cache"):
            self.extractor.person_cache.clear()

    def register_photo_frame(self, frame, detector):
        """照片登记入口：单人全身用身体+可用人脸；只有脸时允许专用人脸登记，不能多脸猜人。"""
        height, width = frame.shape[:2]
        scale = min(1., 960 / max(height, width))
        if scale < 1:
            frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
        height, width = frame.shape[:2]
        people = [item for item in detector.detect(frame) if item.cls == 0]
        faces = self.face_encoder.read(frame)
        good = [face for face in faces if face.feature is not None]
        if len(people) > 1 or len(good) > 1:
            raise SessionError("MULTI_PERSON_PHOTO", "照片中有多个人，请选择只有本人的照片")
        sample = None
        if people:
            box = Box.from_detection(people[0], width, height).model_copy(update={"track_id": 1})
            sample = self.face_encoder.associate(faces, [box], width, height).get(1)
            crop = crop_target(frame, box.xyxy)
        elif len(good) == 1:
            sample = good[0]
            left, top, right, bottom = sample.xyxy
            crop = frame[top:bottom, left:right].copy()
        else:
            raise SessionError("NO_USABLE_FACE", "照片中没有清晰可用的人脸或人物，请换一张清晰单人照")
        if crop is None:
            raise SessionError("PHOTO_TARGET_SMALL", "照片中的人物太小，请选择更清晰的照片")
        face = sample.feature if sample is not None else None
        feature = self.extractor.extract([crop], flip=True)[0]
        thumb_height = 160
        thumb_width = min(320, max(1, round(crop.shape[1] * thumb_height / crop.shape[0])))
        ok, encoded = cv2.imencode(".jpg", cv2.resize(crop, (thumb_width, thumb_height)))
        if not ok:
            raise SessionError("BAD_PHOTO", "缩略图处理失败")
        body_valid = bool(people)
        if body_valid and sample is not None:
            body_valid = (sample.xyxy[3] - sample.xyxy[1]) / max(people[0].xyxy[3] - people[0].xyxy[1], 1) <= .45
        if not body_valid and face is None:
            raise SessionError("NO_USABLE_FACE", "目前只有头部，但人脸质量不足，请选择清晰的正脸照片登记")
        identity = self.register_identity(feature, face_feature=face, body_valid=body_valid,
                                          source="photo", name="照片注册", thumbnail=encoded.tobytes())
        self.diagnostic("photo_registered", identity_id=identity.id, face_registered=face is not None, body_registered=body_valid)
        return {"ok": True, "identity_id": identity.id, "thumb": self.thumbnail_url(identity.id),
                "code": "PHOTO_REGISTERED", "quality": {"person_count": len(people), "face_registered": face is not None},
                "message": "照片已登记（人脸身份），等待视频匹配" if face is not None else "照片已按行人外观登记；人脸不够清晰，尚未建立人脸模板"}
