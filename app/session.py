"""一路视频的短时轨迹、长期身份与当前内存帧，由 Pipeline 串行调用。
身份库在轨迹丢失后保留，视频停止或断开后清空；不自动保存媒体。
"""
from dataclasses import dataclass
from collections import deque
from secrets import token_urlsafe
from time import monotonic
from datetime import datetime, timezone
import json
import logging
from uuid import uuid4

from .config import PROJECT_ROOT

from .gallery import Gallery
from .reid import crop_target
from .tracker import Tracker
from .recorder import Recorder


class SessionError(ValueError):
    """向接口提供可以直接显示的错误码与中文提示。"""
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class Observation:
    """一条轨迹的连续观测、EMA 特征与候选计数，不保存历史画面。"""
    hits: int = 0
    last_frame: int = 0
    identity_id: str | None = None
    feature: object = None
    candidate: str | None = None
    candidate_hits: int = 0
    bad_hits: int = 0
    last_rejection: str | None = None
    rejection_frame: int = -60


class VideoSession:
    """输入检测、帧与控制请求，输出带身份状态的框、身份条和事件。"""

    def __init__(self, config, extractor=None, clock=None):
        self.config, self.extractor = config, extractor
        # 在线服务继续使用真实 monotonic 时钟；离线回放可注入按视频时间推进的时钟。
        self.clock = clock or monotonic
        self.tracker = Tracker(config)
        self.token = token_urlsafe(32)
        self.gallery = Gallery(config)
        self.observations = {}
        self.boxes = []
        self.seq = -1
        self.frame_count = 0
        self.updated_at = self.clock()
        self.frame = None
        self.size = None
        self.events = []
        self.pending_events = []
        self.was_online = set()
        self.recorder = Recorder(config)
        self.event_path = PROJECT_ROOT / "outputs/events" / ("tracking_" + uuid4().hex + ".jsonl")
        # trace_id是只供诊断关联的随机别名，绝不使用可控制会话的token作为日志编号。
        self.trace_id = uuid4().hex[:12]
        self._interaction_times = deque(maxlen=10)
        self._visible_tracks = set()
        self.diagnostic("session_started", reid_model=config.reid_model, high=config.reid_tau_high,
                        confirm_frames=config.reid_confirm_frames, color_min=config.reid_color_min)

    def diagnostic(self, event, **details):
        """内部调用：将匿名会话号、帧号与受控标量写入已有轮换日志，不保存图片或特征。"""
        logging.getLogger(__name__).info("TRACK_DIAG %s", json.dumps(
            {"event": event, "session": self.trace_id, "seq": self.seq, **details},
            ensure_ascii=False, allow_nan=False, separators=(",", ":")))

    def record_interaction(self, details):
        """接收接口已验证的枚举/数字，每会话每秒最多10条；不记录用户输入的任意文字。"""
        now = self.clock()
        if len(self._interaction_times) == 10 and now - self._interaction_times[0] < 1:
            return {"ok": True, "recorded": False}
        self._interaction_times.append(now)
        self.diagnostic("client_gesture", client=details)
        return {"ok": True, "recorded": True}

    def log_rejection(self, track_id, assessment):
        """同一轨迹/原因最多每60帧记一次，原因变化立即记录，防止逐帧刷满日志。"""
        track = self.observations[track_id]
        reason = assessment["reason"]
        if reason != track.last_rejection or self.frame_count - track.rejection_frame >= 60:
            self.diagnostic("candidate_rejected", track_id=track_id, check=assessment)
            track.last_rejection, track.rejection_frame = reason, self.frame_count

    def update(self, detections, width, height, frame=None):
        """输入当前检测与尺寸，输出原始跟踪框；空帧也推进失踪生命周期。"""
        if self.size != (width, height):
            self.observations.clear()
            self.size = (width, height)
        self.boxes = self.tracker.update(detections, width, height, image=frame, extractor=self.extractor) if frame is not None else self.tracker.update(detections, width, height)
        self.frame_count += 1
        self.updated_at = self.clock()
        for box in self.boxes:
            if box.track_id is None:
                continue
            track = self.observations.setdefault(box.track_id, Observation())
            continuous = track.last_frame == self.frame_count - 1
            track.hits = track.hits + 1 if continuous else 1
            if not continuous:
                track.candidate, track.candidate_hits = None, 0
            track.last_frame = self.frame_count
        self.observations = {key: value for key, value in self.observations.items()
                             if self.frame_count - value.last_frame <= self.config.track_max_age}
        return self.boxes

    def process(self, frame, seq):
        """输入 BGR 帧与序号，批量提取特征、匹配、反向校验，输出身份化的框。
        人物逐帧批量提取特征；绿框也必须有本帧证据，不能只沿用旧轨迹号或旧EMA。
        """
        self.frame, self.seq = frame, seq
        self.events, self.pending_events = self.pending_events, []
        crops, keys = [], []
        for box in self.boxes:
            if box.cls != 0 or box.track_id is None:
                continue  # 行人外观编码器不能用来声称识别任意物体身份。
            track = self.observations[box.track_id]
            crop = crop_target(frame, box.xyxy)
            if crop is not None:
                crops.append(crop)
                keys.append(box.track_id)
        fresh = {}
        if self.extractor is not None:
            fresh = dict(zip(keys, self.extractor.extract(crops)))
            for key, feature in fresh.items():
                track = self.observations[key]
                track.feature = self.extractor.smooth(track.feature, feature)
        visible = {box.track_id for box in self.boxes if box.track_id is not None}
        if visible != self._visible_tracks:
            self.diagnostic("visible_tracks_changed", visible=sorted(visible),
                            appeared=sorted(visible - self._visible_tracks),
                            missing=sorted(self._visible_tracks - visible), detected=len(self.boxes))
            self._visible_tracks = visible
        claimed = set()
        # 当前可见绑定有优先权，不允许另一条相似轨迹占据同一个身份。
        for key in sorted(visible):
            track = self.observations[key]
            if track.identity_id not in self.gallery.items or track.identity_id in claimed:
                track.identity_id = None
            check = (self.gallery.assess(fresh[key], track.identity_id) if key in fresh else
                     {"accepted": False, "reason": "NO_FRESH_FEATURE"}) if track.identity_id else None
            if track.identity_id and not check["accepted"]:
                # ByteTrack在交叉/遮挡时可能沿用别人的旧号；当前外观不足即回白，而不是等低于0.45五次。
                self.events.append({'type': 'event', 'name': 'unbound', 'identity_id': track.identity_id})
                self.diagnostic("identity_unbound", track_id=key, identity_id=track.identity_id,
                                reason="fresh_identity_check_failed", check=check,
                                ema_score=round(self.gallery.score(track.feature, track.identity_id), 4)
                                if track.feature is not None else None)
                track.identity_id = None
                track.candidate, track.candidate_hits = None, 0
                track.feature = fresh.get(key)
            if track.identity_id:
                claimed.add(track.identity_id)
        # 多个人都与同一身份相似时，先全局比较再分配，不能让检测数组中排在前面的人抢先变绿。
        contenders = {}
        for box in self.boxes:
            track = self.observations.get(box.track_id)
            if box.cls != 0 or track is None or track.identity_id or track.hits < 3 or box.track_id not in fresh:
                continue
            identity_id, best, second = self.gallery.rank(track.feature, claimed)
            raw_check = self.gallery.assess(fresh[box.track_id], identity_id) if identity_id else None
            ema_check = self.gallery.assess(track.feature, identity_id) if identity_id else None
            if (identity_id and best >= self.config.reid_tau_high and best - second >= self.config.reid_margin
                    and raw_check["accepted"] and ema_check["accepted"]):
                contenders.setdefault(identity_id, []).append((best, box.track_id))
            elif identity_id:
                check = raw_check if not raw_check["accepted"] else ema_check
                if check["accepted"]:
                    check = {"accepted": False, "reason": "IDENTITY_MARGIN_LOW", "best": round(best, 4),
                             "second": round(second, 4), "margin": self.config.reid_margin}
                self.log_rejection(box.track_id, check)
        winners = {}
        for identity_id, candidates in contenders.items():
            candidates.sort(reverse=True)
            # 身份之间要有差距，竞争同一身份的两个可见人物之间同样要有差距。
            if len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= self.config.reid_margin:
                winners[candidates[0][1]] = identity_id
            else:
                for score, key in candidates:
                    self.log_rejection(key, {"accepted": False, "reason": "COMPETING_TRACKS",
                                            "best": round(score, 4), "margin": self.config.reid_margin})
        output = []
        for box in self.boxes:
            track = self.observations.get(box.track_id)
            state, similarity = 'none', None
            if track is not None and track.identity_id is None and box.track_id not in fresh:
                # 裁剪太小等导致缺少本帧观测，也会中断连续确认，不能把不连续帧累计起来。
                track.candidate, track.candidate_hits = None, 0
            if box.cls == 0 and track is not None and track.feature is not None:
                if track.identity_id:
                    similarity = self.gallery.score(track.feature, track.identity_id)
                    state = 'locked'  # 上方已用本帧外观逐帧校验，通过后才允许继续绿框。
                if track.identity_id is None and track.hits >= 3 and box.track_id in fresh:
                    identity_id, best, second = self.gallery.rank(track.feature, claimed)
                    similarity = best if identity_id else None
                    if identity_id and winners.get(box.track_id) == identity_id:
                        track.candidate_hits = track.candidate_hits + 1 if track.candidate == identity_id else 1
                        track.candidate = identity_id
                        if track.candidate_hits == 1:
                            self.diagnostic("confirmation_started", track_id=box.track_id, identity_id=identity_id,
                                            required=self.config.reid_confirm_frames)
                        state = 'candidate'
                        if track.candidate_hits >= self.config.reid_confirm_frames:
                            # 新号接管身份后，旧失踪轨迹不得复活成重复绿框。
                            for other in self.observations.values():
                                if other is not track and other.identity_id == identity_id:
                                    other.identity_id = None
                            track.identity_id = identity_id
                            claimed.add(identity_id)
                            # 自动重匹配不扩充模板，避免一次误认逐渐污染登记的身份。
                            self.events.append({'type': 'event', 'name': 'relocked', 'identity_id': identity_id})
                            self.diagnostic("identity_confirmed", track_id=box.track_id, identity_id=identity_id,
                                            frames=track.candidate_hits)
                            track.candidate, track.candidate_hits = None, 0
                            state = 'locked'
                    else:
                        track.candidate, track.candidate_hits = None, 0
            output.append(box.model_copy(update={'state': state, 'sim': similarity,
                                                'identity_id': track.identity_id if state == 'locked' else None}))
        self.boxes = output
        online = {box.identity_id for box in output if box.identity_id}
        self.events.extend({'type': 'event', 'name': 'lost', 'identity_id': key}
                           for key in sorted(self.was_online - online))
        self.was_online = online
        # 冷启动特征提取可能耗时，过期判断以结果完成时间为准，不从推理开始计时。
        self.updated_at = self.clock()
        self.write_events()
        self.recorder.write(frame, output, seq, self.events)
        return output

    def start_record(self, confirm):
        """只有允许保存且请求确认时才启动；默认返回明确拒绝，不会创建文件。"""
        if self.frame is None:
            raise SessionError("TRACK_GONE", "请先开始检测，收到画面后再录制")
        height, width = self.frame.shape[:2]
        try:
            return {"ok": True, **self.recorder.start(width, height, confirm)}
        except PermissionError as error:
            raise SessionError("MEDIA_SAVE_FORBIDDEN", str(error)) from error

    def stop_record(self):
        """停止本会话录制并返回文件名，供页面显示。"""
        return {"ok": True, **self.recorder.stop()}

    def record_file(self, filename):
        """仅允许下载本会话已结束的录制文件；文件名必须精确匹配，不拼接输入路径。"""
        path = self.recorder.path
        if path is None or filename != path.name or not path.is_file():
            raise SessionError("RECORD_NOT_FOUND", "录制文件不存在")
        if self.recorder.writer is not None:
            raise SessionError("RECORD_ACTIVE", "请先停止录制再下载")
        return path

    def write_events(self):
        """只记录事件名、匿名编号、时间和帧号；无图片、向量或控制会话凭据。"""
        if not self.events:
            return
        try:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            if self.event_path.exists() and self.event_path.stat().st_size > 2 * 1024 * 1024:
                return  # 单会话事件上限，防止反复进出无限占用磁盘。
            with self.event_path.open("a", encoding="utf-8") as stream:
                for event in self.events:
                    stream.write(json.dumps({**event, "seq": self.seq,
                        "at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
        except OSError:
            # 磁盘满等问题不应让内存中的检测和匹配一起停掉。
            logging.getLogger(__name__).exception("事件记录失败")

    def clear_lock(self, reason="cancel"):
        """解除当前目标及其搜索任务，丢弃旧身份/候选；取消后旧目标不会自己重新变绿。"""
        if self.gallery.items:
            self.diagnostic("identity_cleared", reason=reason, identities=list(self.gallery.items))
        self.gallery.items.clear()
        for track in self.observations.values():
            track.identity_id = None
            track.candidate, track.candidate_hits = None, 0
            track.bad_hits = 0
            track.feature = None
        self.boxes = [box.model_copy(update={'state': 'none', 'identity_id': None, 'sim': None}) for box in self.boxes]
        self.pending_events.clear()
        self.was_online.clear()

    def register_identity(self, feature, **details):
        """先成功构造新身份，再原子替换旧目标；照片注册和手动锁定共用单目标规则。"""
        replacement = Gallery(self.config)
        replacement.next_id = self.gallery.next_id  # 编号不复用，避免旧卡片误指向新身份。
        identity = replacement.register(feature, **details)
        self.clear_lock(reason="replace")
        self.gallery = replacement
        self.diagnostic("identity_registered", identity_id=identity.id, source=identity.source)
        return identity

    def lock(self, track_id, seq, action='toggle'):
        """输入轨迹号、帧号和lock/unlock/toggle，输出明确的操作结果；同时只保留一个目标。"""
        self.diagnostic("lock_requested", track_id=track_id, client_seq=seq, action=action,
                        frame_age_ms=round((self.clock() - self.updated_at) * 1000, 2))
        def reject(code, message):
            """所有业务拒绝同时留下受控错误码，再由HTTP接口返回中文提示。"""
            self.diagnostic("lock_rejected", track_id=track_id, code=code, action=action)
            raise SessionError(code, message)

        box = next((item for item in self.boxes if item.track_id == track_id), None)
        if box is None or self.clock() - self.updated_at > 3 or seq > self.seq or self.seq - seq > 30:
            reject('TRACK_GONE', '目标已离开画面或画面已过期，请重新选择')
        if box.cls != 0:
            reject('UNSUPPORTED_TARGET', '身份锁定目前支持人物，请选择 person 框')
        track = self.observations[track_id]
        if action not in {'lock', 'unlock', 'toggle'}:
            reject('BAD_REQUEST', '锁定操作类型不正确')
        if track.identity_id in self.gallery.items:
            if action in {'unlock', 'toggle'}:
                self.clear_lock()
                self.diagnostic("lock_result", track_id=track_id, action="unlocked")
                return {'ok': True, 'action': 'unlocked', 'identity_id': None, 'message': '已取消锁定'}
            # 明确lock请求的重复发送不取消，只有用户再次长按绿框才发送unlock。
            self.diagnostic("lock_result", track_id=track_id, action="locked", repeated=True)
            return {'ok': True, 'action': 'locked', 'identity_id': track.identity_id, 'message': '目标已锁定'}
        if action == 'unlock':
            self.diagnostic("lock_result", track_id=track_id, action="unlocked", no_change=True)
            return {'ok': True, 'action': 'unlocked', 'identity_id': None, 'message': '该目标当前未锁定'}
        # 切换目标时旧EMA会被清除，但稳定命中次数仍有效；本次注册会重新提取当前裁剪。
        if track.hits < 5 or self.extractor is None:
            reject('REID_NOT_WARM', '请让目标在画面中停留片刻后再按住；过小的人物框可能无法提取特征')
        if track.identity_id not in self.gallery.items:
            crop = crop_target(self.frame, box.xyxy)
            if crop is None:
                reject('REID_NOT_WARM', '目标太小，请靠近一些后重试')
            # 同一帧其他人的特征作为登记反例。只取明显不相交的框，避免把同一人的重复框当反例。
            other_crops = []
            for other in self.boxes:
                if other.cls != 0 or other.track_id == track_id:
                    continue
                left, top = max(box.xyxy[0], other.xyxy[0]), max(box.xyxy[1], other.xyxy[1])
                right, bottom = min(box.xyxy[2], other.xyxy[2]), min(box.xyxy[3], other.xyxy[3])
                if right > left and bottom > top:
                    continue
                other_crop = crop_target(self.frame, other.xyxy)
                if other_crop is not None:
                    other_crops.append(other_crop)
            features = self.extractor.extract([crop, *other_crops], flip=True)
            feature = features[0]
            try:
                identity = self.register_identity(feature, distractors=features[1:])
            except ValueError as error:
                raise SessionError('IDENTITY_LIMIT', str(error)) from error
            track.identity_id = identity.id
            track.feature = feature.copy()
            track.bad_hits = 0
            self.pending_events.append({'type': 'event', 'name': 'lock_created', 'identity_id': identity.id})
        self.diagnostic("lock_result", track_id=track_id, action="locked", identity_id=track.identity_id)
        return {'ok': True, 'action': 'locked', 'identity_id': track.identity_id,
                'message': f'{track.identity_id} 已锁定，已解除其他目标'}

    def select(self, x, y, seq):
        """在触点命中框里挑面积最小者；空白处不锁定远处人物。"""
        hits = [box for box in self.boxes if box.track_id is not None
                and box.xyxy[0] <= x <= box.xyxy[2] and box.xyxy[1] <= y <= box.xyxy[3]]
        if not hits:
            raise SessionError('TRACK_GONE', '请按住目标框内')
        box = min(hits, key=lambda item: (item.xyxy[2] - item.xyxy[0]) * (item.xyxy[3] - item.xyxy[1]))
        return self.lock(box.track_id, seq)

    def summaries(self):
        """无输入，输出身份条；仅将本帧绑定身份标为在线。"""
        return [summary.model_copy(update={"thumb": self.thumbnail_url(summary.id)
                if self.gallery.items[summary.id].thumbnail else None})
                for summary in self.gallery.summaries({box.identity_id for box in self.boxes if box.identity_id})]

    def thumbnail_url(self, identity_id):
        """输出当前会话专用缩略图地址，访问凭据不可用于其他视频。"""
        return f"/api/identities/{identity_id}/thumbnail?session_id={self.token}"

    def thumbnail(self, identity_id):
        """返回该身份的内存 JPEG；不存在或已删除时拒绝，绝不从任意磁盘路径读取。"""
        item = self.gallery.items.get(identity_id)
        if item is None or item.thumbnail is None:
            raise SessionError("IDENTITY_NOT_FOUND", "身份缩略图不存在或已清除")
        return item.thumbnail

    def delete(self, identity_id):
        """删除身份特征和缩略图内存，清除绑定与候选。"""
        if identity_id not in self.gallery.items:
            raise SessionError('IDENTITY_NOT_FOUND', '身份不存在')
        del self.gallery.items[identity_id]
        for track in self.observations.values():
            if track.identity_id == identity_id:
                track.identity_id = None
            if track.candidate == identity_id:
                track.candidate, track.candidate_hits = None, 0

    def reset(self):
        """视频停止后清除图像和身份内存；轨迹号仍由 Tracker 保证不复用。"""
        self.diagnostic("session_stopped", identities=len(self.gallery.items), tracks=len(self.observations))
        self.recorder.stop()
        self.tracker.reset()
        self.observations.clear()
        self.gallery.items.clear()
        self.boxes = []
        self.frame = None
        self.events.clear()
        self.pending_events.clear()
        self.was_online.clear()
        self._visible_tracks.clear()
        self._interaction_times.clear()

    def snapshot(self):
        """仅输出跟踪数值，不含画面或向量。"""
        return self.tracker.snapshot()
