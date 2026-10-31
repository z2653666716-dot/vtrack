"""串联一帧图像的解码、检测、坐标转换和结果组装，属于 ViTrack 的帧处理流水线。

浏览器推帧接口调用 process_message；IP 视频源后台任务调用 process_frame，共用检测与结果格式。
本模块串联 Detector、会话独立跟踪和外观重识别，不自行打开设备。
"""

import asyncio
import anyio
import base64
import binascii
from collections import deque
from io import BytesIO
import logging
import threading
from time import perf_counter

import cv2
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile

from .config import AppConfig
from .detector import Detector
from .schemas import Box, ErrorResponse, FrameMessage, FrameResult
from .tracker import Tracker
from .session import VideoSession, SessionError
from .reid import FeatureExtractor
from .reid import crop_target
from .hybrid_session import HybridSession
from .faces import FaceEncoder


logger = logging.getLogger(__name__)

# 接收上限与前端的 MAX_SIDE 不同：前者限制输入，后者是计划发送尺寸。
MAX_JPEG_BYTES = 4 * 1024 * 1024
MAX_FRAME_SIDE = 4096
FPS_WINDOW = 30


class PipelineError(ValueError):
    """表示可以向客户端解释的帧处理错误，供未来接口统一捕获。"""

    def __init__(self, code: str, message: str) -> None:
        """输入错误码与中文提示，输出异常对象；由下方流水线抛出。"""
        super().__init__(message)
        self.code = code
        self.message = message

    def to_response(self) -> ErrorResponse:
        """把当前异常转为统一错误数据；未来接口调用后再选择 HTTP 或 WS 发送。"""
        return ErrorResponse(code=self.code, message=self.message)


class Pipeline:
    """持有一个检测器，按顺序处理共享模型上的任务。

    输入：已初始化的 Detector；输出：提供两个异步帧处理入口的流水线实例。
    线程锁保护实际工作，调用方取消等待也不会提前释放仍在推理的模型。
    """

    def __init__(self, detector: Detector, config: AppConfig | None = None) -> None:
        """保存检测器/可选配置，初始化处理锁、有限长度统计窗口和最新结果。

        不重新加载模型；未来应用启动流程创建本对象，健康接口可读取最新统计。
        """
        self.detector = detector
        self.config = config or getattr(detector, "config", None) or AppConfig()
        self._process_lock = threading.Lock()
        self._completion_times: deque[float] = deque(maxlen=FPS_WINDOW)
        self.latest_result: FrameResult | None = None
        self.processed_frames = 0
        self.sessions = {}
        self.extractor = None
        self.face_encoder = None
        self.recommended_max_side = self.config.max_side
        self.slow_frames = 0

    def create_tracker(self, clock=None) -> VideoSession:
        """创建新视频会话；离线回放可传视频时钟，在线调用保留真实系统时钟。"""
        session = (HybridSession(self.config, clock=clock) if self.config.identity_mode == "hybrid"
                   else VideoSession(self.config, clock=clock))
        self.sessions[session.token] = session
        return session

    def prepare_hybrid(self, session):
        """在已有处理锁内加载一次本地行人/人脸模型，并给会话复用；不联网或访问设备。"""
        if self.extractor is None:
            self.extractor = FeatureExtractor(self.config)
        if self.face_encoder is None:
            self.face_encoder = FaceEncoder(self.config)
        session.extractor, session.face_encoder = self.extractor, self.face_encoder
        self.extractor.person_cache.clear()

    async def session_action(self, session_id, action, *args):
        """输入随机会话凭据与操作名，在线程中串行执行，输出操作结果。"""
        def run():
            with self._process_lock:
                session = self.sessions.get(session_id)
                if session is None:
                    raise SessionError("SESSION_GONE", "视频会话已结束，请开始检测后再操作")
                return getattr(session, action)(*args)
        return await asyncio.to_thread(run)

    async def register_photo(self, session_id, content, content_type):
        """输入已限大小的照片请求，在线程中解码、找最大人物、注册特征，返回提示与质量。"""
        def run():
            from .photos import decode_photo
            with self._process_lock:
                session = self.sessions.get(session_id)
                if session is None:
                    raise SessionError("SESSION_GONE", "请先开始检测，再选择照片")
                session.diagnostic("photo_received", bytes=len(content))
                try:
                    frame = decode_photo(content, content_type)
                except SessionError as error:
                    session.diagnostic("photo_rejected", reason=error.code)
                    raise
                if isinstance(session, HybridSession):
                    self.prepare_hybrid(session)
                    try:
                        return session.register_photo_frame(frame, self.detector)
                    except SessionError as error:
                        session.diagnostic("photo_rejected", reason=error.code)
                        raise
                height, width = frame.shape[:2]
                scale = min(1, self.config.max_side / max(width, height))
                if scale < 1:
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                height, width = frame.shape[:2]
                people = [item for item in self.detector.detect(frame) if item.cls == 0]
                if not people:
                    session.diagnostic("photo_rejected", reason="NO_TARGET_IN_PHOTO")
                    raise SessionError("NO_TARGET_IN_PHOTO", "照片中没有检测到人物")
                target = max(people, key=lambda item: (item.xyxy[2] - item.xyxy[0]) * (item.xyxy[3] - item.xyxy[1]))
                box = Box.from_detection(target, width, height)
                crop = crop_target(frame, box.xyxy)
                if crop is None:
                    session.diagnostic("photo_rejected", reason="PHOTO_TARGET_SMALL")
                    raise SessionError("PHOTO_TARGET_SMALL", "照片中的人物太小，请选择更清晰的人物照片")
                if self.extractor is None:
                    self.extractor = FeatureExtractor(self.config)
                feature = self.extractor.extract([crop], flip=True)[0]
                # 缩略图只编码成内存字节，不写 outputs/identities，也不缓存原照片。
                thumb_height = 160
                thumb_width = max(1, round(crop.shape[1] * thumb_height / crop.shape[0]))
                ok, encoded = cv2.imencode(".jpg", cv2.resize(crop, (min(320, thumb_width), thumb_height)))
                if not ok:
                    raise SessionError("BAD_PHOTO", "缩略图处理失败，请换一张照片")
                try:
                    identity = session.register_identity(feature, name="照片注册", source="photo", thumbnail=encoded.tobytes())
                except ValueError as error:
                    raise SessionError("IDENTITY_LIMIT", str(error)) from error
                session.diagnostic("photo_registered", identity_id=identity.id, people=len(people),
                                   crop_width=int(crop.shape[1]), crop_height=int(crop.shape[0]))
                return {"ok": True, "identity_id": identity.id,
                        "thumb": session.thumbnail_url(identity.id),
                        "quality": {"detection_score": target.score, "person_count": len(people),
                                    "crop_width": crop.shape[1], "crop_height": crop.shape[0]},
                        "code": "MULTI_PERSON_PHOTO" if len(people) > 1 else "PHOTO_REGISTERED",
                        "message": "照片有多个人，已自动登记最大的人物" if len(people) > 1 else "照片已登记，将尝试匹配画面中的人物"}
        return await asyncio.to_thread(run)

    async def reset_tracker(self, tracker: Tracker | VideoSession | None) -> None:
        """接收已结束会话的跟踪器，等待在途处理完成后清理；避免取消等待后工作线程继续改旧状态。"""
        if tracker is not None:
            # 先撤销控制凭据；即使连接被框架取消，旧 HTTP 操作也不能再次访问它。
            self.sessions.pop(getattr(tracker, "token", None), None)
            # Starlette 在断开连接时可取消整个作用域；清理在保护区等待底层线程结束。
            with anyio.CancelScope(shield=True):
                await asyncio.to_thread(self._reset_tracker, tracker)

    def _reset_tracker(self, tracker: Tracker | VideoSession) -> None:
        """清理线程取得同一处理锁，再丢弃轨迹；不会触及其他会话的跟踪器。"""
        with self._process_lock:
            tracker.reset()
            if self.extractor is not None:
                self.extractor.person_cache.clear()
            self.sessions.pop(getattr(tracker, "token", None), None)

    async def process_message(self, message: FrameMessage, tracker: Tracker | VideoSession | None = None) -> FrameResult:
        """处理浏览器的一帧消息，由未来 WebSocket 接口 await 调用。

        输入：经过检查的 FrameMessage 和可选会话 Tracker；输出：保留原 seq 的 FrameResult。
        to_thread 将解码和推理交给后台线程，等待期间不占住事件循环。
        """
        if not isinstance(message, FrameMessage):
            raise PipelineError("BAD_FRAME", "请先将帧消息解析为 FrameMessage")
        return await asyncio.to_thread(self._run, message, message.seq, tracker)

    async def process_frame(self, frame_bgr: np.ndarray, seq: int, tracker: Tracker | VideoSession | None = None) -> FrameResult:
        """处理直接提供的图像，由 IP 视频源后台任务调用共同入口。

        输入：BGR 数组、非负帧号、可选会话 Tracker；输出：与浏览器路径相同的 FrameResult。
        本方法不读取摄像头，调用方等待处理期间应保留该帧内容。
        """
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise PipelineError("BAD_FRAME", "帧序号 seq 必须是非负整数")
        return await asyncio.to_thread(self._run, frame_bgr, seq, tracker)

    def _run(self, source: FrameMessage | np.ndarray, seq: int, tracker: Tracker | VideoSession | None = None) -> FrameResult:
        """后台线程共用的同步流程，供两个异步入口调用。

        输入：消息或图像，以及帧号；输出：检测结果与统计。发生错误时抛出 PipelineError。
        锁覆盖解码、推理、组装与统计，保证同一个模型及统计状态不会被交叉修改。
        """
        with self._process_lock:
            started = perf_counter()
            frame = self._decode_message(source) if isinstance(source, FrameMessage) else source
            self._check_frame(frame)
            height, width = frame.shape[:2]
            try:
                detections = self.detector.detect(frame)
                # 跟踪状态由调用方的会话持有；共享的只是模型与处理锁，不把不同视频混在一起。
                if isinstance(tracker, HybridSession):
                    self.prepare_hybrid(tracker)
                    detections = tracker.prepare_frame(frame, detections)
                    boxes = tracker.update(detections, width, height, frame)
                else:
                    boxes = tracker.update(detections, width, height) if tracker is not None else [
                    Box.from_detection(item, width, height) for item in detections
                    ]
                if isinstance(tracker, VideoSession):
                    if self.extractor is None:
                        self.extractor = FeatureExtractor(self.config)
                    tracker.extractor = self.extractor
                    boxes = tracker.process(frame, seq)
                result = FrameResult(seq=seq, boxes=boxes,
                                     session_id=getattr(tracker, "token", None),
                                     identities=tracker.summaries() if isinstance(tracker, VideoSession) else [],
                                     events=tracker.events if isinstance(tracker, VideoSession) else [])
            except Exception as error:
                logger.exception("检测、跟踪或结果转换失败")
                raise PipelineError("INFERENCE_FAILED", "目标检测失败，请检查服务日志") from error

            finished = perf_counter()
            result.latency_ms = round((finished - started) * 1000, 2)
            self._completion_times.append(finished)
            if len(self._completion_times) > 1:
                elapsed = self._completion_times[-1] - self._completion_times[0]
                if elapsed > 0:
                    result.fps = round((len(self._completion_times) - 1) / elapsed, 2)
            # FPS 来自实际处理完成的时间间隔，第一帧没有间隔，保持默认 0。
            self.processed_frames += 1
            # 等冷启动窗口过去，持续30帧低于12FPS才降低传输尺寸；不反复升降造成抖动。
            self.slow_frames = self.slow_frames + 1 if self.processed_frames > 30 and 0 < result.fps < 12 else 0
            if self.slow_frames >= 30:
                self.recommended_max_side = min(self.recommended_max_side, 480)
            result.capture_max_side = self.recommended_max_side
            self.latest_result = result
            return result

    @staticmethod
    def _decode_message(message: FrameMessage) -> np.ndarray:
        """输入 Base64 帧消息，输出 OpenCV BGR 图像，由 _run 在线程内调用。

        顺序：限制文本长度 → Base64 解码 → 检查 JPEG 头信息及尺寸 → OpenCV 解码。
        Pillow 只读取头部元信息，先拦截超大尺寸，实际像素仍由 cv2.imdecode 解码。
        """
        max_encoded_length = 4 * ((MAX_JPEG_BYTES + 2) // 3)
        size_message = f"JPEG 超过 {MAX_JPEG_BYTES / (1024 * 1024):g} MiB，请降低发送尺寸或图片质量"
        if len(message.img) > max_encoded_length:
            raise PipelineError("FRAME_TOO_LARGE", size_message)
        try:
            jpeg_bytes = base64.b64decode(message.img, validate=True)
        except (binascii.Error, ValueError) as error:
            raise PipelineError("BAD_FRAME", "图片必须是纯 Base64 文本，不含 data:image 前缀") from error
        if not jpeg_bytes:
            raise PipelineError("BAD_FRAME", "图片内容为空")
        if len(jpeg_bytes) > MAX_JPEG_BYTES:
            raise PipelineError("FRAME_TOO_LARGE", size_message)

        try:
            # 直接使用 JPEG 头解析器，避免第三方接管通用 Image.open 后自动尝试其他格式。
            with JpegImageFile(BytesIO(jpeg_bytes)) as image:
                width, height = image.size
        except (OSError, ValueError, SyntaxError) as error:
            raise PipelineError("BAD_FRAME", "图片内容损坏或不是可识别的 JPEG") from error
        if width > MAX_FRAME_SIDE or height > MAX_FRAME_SIDE:
            raise PipelineError("FRAME_TOO_LARGE", f"图片宽高不能超过 {MAX_FRAME_SIDE} 像素")
        if (width, height) != (message.w, message.h):
            raise PipelineError(
                "BAD_FRAME",
                f"图片实际尺寸为 {width}×{height}，与消息声明的 {message.w}×{message.h} 不一致",
            )
        try:
            # 以编码后的像素尺寸为准，避免 EXIF 自动旋转改变宽高和坐标对应关系。
            flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), flags)
        except cv2.error as error:
            raise PipelineError("BAD_FRAME", "JPEG 像素解码失败") from error
        if frame is None or frame.shape[:2] != (height, width):
            raise PipelineError("BAD_FRAME", "JPEG 无法完整解码为声明尺寸的图像")
        return frame

    @staticmethod
    def _check_frame(frame: np.ndarray) -> None:
        """检查两个入口共用的图像要求；输入 BGR 数组，成功不返回数据，失败抛出异常。"""
        if not isinstance(frame, np.ndarray):
            raise PipelineError("BAD_FRAME", "帧必须是 OpenCV/NumPy 图像数组")
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
                or frame.shape[0] == 0 or frame.shape[1] == 0):
            raise PipelineError("BAD_FRAME", "帧必须是非空的 uint8 三通道 BGR 图像")
        if max(frame.shape[:2]) > MAX_FRAME_SIDE:
            raise PipelineError("FRAME_TOO_LARGE", f"图片宽高不能超过 {MAX_FRAME_SIDE} 像素")
