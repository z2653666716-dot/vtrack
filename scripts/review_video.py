"""ViTrack 本地离线视频审阅工具。

本脚本使用 Tkinter 显示用户通过文件选择器指定的视频，只在本机内存解码。
审阅者可暂停、逐帧跳转、为人物画框、加载算法预测叠加层、标记场景并写文字备注，最后
分别导出供算法分析的问题备注，或供定量评测使用的人工人物框；两种导出互不阻塞。
脚本不打开摄像头、不启动网络服务，也不把视频上传或复制到其他位置。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 540
SCENE_LABELS = ("unspecified", "far_to_near", "reentry", "back_turn", "multi_person", "other")
ISSUE_LABELS = ("其他", "漏检", "检测到但未绑定", "身份认错",
                "轨迹编号变化", "框抖动", "身份丢失")
ISSUE_CODES = dict(zip(ISSUE_LABELS, ("other", "missed_detection", "unbound_target",
                                     "wrong_identity", "track_id_changed", "box_jitter",
                                     "identity_lost")))
ISSUE_NAMES = {code: label for label, code in ISSUE_CODES.items()}


def choose_automatic_registration(annotations: dict[int, dict], target_person_id: str) -> dict | None:
    """用最早一帧人工目标框自动确定可重复的评测初始化时刻。

    输入是按帧号存放的人工框记录和目标编号；找到目标框时返回该帧记录，
    找不到时返回 None。这个时刻只用于公平重放，不让复核备注依赖人工登记操作。
    """
    candidates = [row for row in annotations.values()
                  if any(person.get("person_id") == target_person_id
                         for person in row.get("people", []))]
    return min(candidates, key=lambda row: row["frame_index"]) if candidates else None


def build_review_notes_payload(video: dict, prediction: dict | None,
                               annotations: dict[int, dict]) -> dict:
    """整理供算法问题分析的纯文字复核记录。

    输入为视频摘要、可选方案信息和按帧保存的表单内容；输出仅包含帧号、时间、
    问题类别与文字，不含人工人物框或任何图像/特征数据。
    """
    notes = []
    for frame_index in sorted(annotations):
        row = annotations[frame_index]
        note = str(row.get("review_note", "")).strip()
        if not note:
            continue
        notes.append({
            "frame_index": frame_index,
            "video_time_seconds": row.get("video_time_seconds"),
            "issue_type": row.get("issue_type", "other"),
            "note": note,
        })
    return {
        "schema_version": 1,
        "kind": "algorithm_review_notes",
        "video": video,
        "prediction": prediction,
        "notes": notes,
        "privacy": {"frames_exported": False, "boxes_exported": False,
                    "features_exported": False},
    }


class VideoReviewApp:
    """管理本地视频、播放位置、人工框和导出标注。

    输入是用户在本地文件选择器中指定的视频；输出是用户选择位置保存的 JSON。
    帧像素只保留在 current_frame 内存变量中，类别信息和框坐标写入 annotations。
    """

    def __init__(self, root: tk.Tk) -> None:
        """创建窗口与控件；由 main() 调用，不读取视频或启动其他设备。"""
        self.root = root
        self.root.title("ViTrack 离线视频审阅")
        self.root.geometry("1320x850")
        self.root.minsize(1000, 700)

        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.video_sha256: str | None = None
        self.predictions: dict[int, dict] = {}
        self.prediction_path: Path | None = None
        self.prediction_variant: str | None = None
        self.fps = 0.0
        self.frame_count = 0
        self.frame_index = 0
        self.video_time_seconds = 0.0
        self.current_frame = None
        self.photo = None
        self.image_rect = (0, 0, 1, 1)
        self.annotations: dict[int, dict] = {}
        self.target_person_id = "person-1"
        self.reentry_start = None
        self.reentry_events = []
        self.drag_start = None
        self.playing = False
        self.updating_scale = False

        self.visibility_var = tk.StringVar(value="unknown")
        self.scene_var = tk.StringVar(value="unspecified")
        self.person_id_var = tk.StringVar(value="person-1")
        self.is_target_var = tk.BooleanVar(value=True)
        self.note_var = tk.StringVar()
        self.issue_var = tk.StringVar(value="其他")
        self.position_var = tk.StringVar(value="尚未打开视频")

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.seek_frame(self.frame_index - 1))
        self.root.bind("<Right>", lambda _event: self.seek_frame(self.frame_index + 1))

    def _build_layout(self) -> None:
        """创建视频画布、播放栏和标注栏；所有按钮都只操作当前本地视频。"""
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开本地视频", command=self.open_video).pack(side="left")
        ttk.Button(toolbar, text="加载方案预测", command=self.open_predictions).pack(side="left", padx=8)
        ttk.Button(toolbar, text="导出 JSON 标注", command=self.export_annotations).pack(side="left", padx=8)
        ttk.Button(toolbar, text="导出问题复核", command=self.export_review_notes).pack(side="left", padx=8)
        ttk.Label(toolbar, textvariable=self.position_var).pack(side="left", padx=14)

        content = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(content, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                                background="#17202a", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render_frame())
        self.canvas.bind("<ButtonPress-1>", self._begin_box)
        self.canvas.bind("<B1-Motion>", self._drag_box)
        self.canvas.bind("<ButtonRelease-1>", self._finish_box)

        side = ttk.Frame(content, width=300, padding=(12, 4))
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="可选定量标注（仅算指标时使用）",
                  font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        ttk.Label(side, text="目标可见状态").pack(anchor="w", pady=(14, 2))
        ttk.Combobox(side, textvariable=self.visibility_var, state="readonly",
                     values=("visible", "absent", "unknown")).pack(fill="x")

        ttk.Label(side, text="场景标签（用于分场景统计）").pack(anchor="w", pady=(8, 2))
        ttk.Combobox(side, textvariable=self.scene_var, state="readonly",
                     values=SCENE_LABELS).pack(fill="x")

        ttk.Label(side, text="人物编号（同一人前后一致）").pack(anchor="w", pady=(12, 2))
        ttk.Entry(side, textvariable=self.person_id_var).pack(fill="x")
        ttk.Checkbutton(side, text="这是登记目标", variable=self.is_target_var).pack(anchor="w", pady=4)
        ttk.Label(side, text="在画面上按住鼠标拖动，为当前人物画框。\n多人同框时，每个人分别画框。",
                  justify="left").pack(anchor="w", pady=(4, 8))
        ttk.Button(side, text="移除选中的人物框", command=self.remove_selected_box).pack(fill="x")
        self.people_list = tk.Listbox(side, height=6, exportselection=False)
        self.people_list.pack(fill="x", pady=(6, 10))

        ttk.Label(side, text="问题复核（无需画框或登记）",
                  font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        ttk.Label(side, text="暂停后选问题类型并写现象，导出问题复核即可交给助手分析。",
                  justify="left", foreground="#555", wraplength=270).pack(anchor="w", pady=(0, 4))
        ttk.Label(side, text="问题类型").pack(anchor="w", pady=(4, 2))
        ttk.Combobox(side, textvariable=self.issue_var, state="readonly",
                     values=ISSUE_LABELS).pack(fill="x")
        ttk.Label(side, text="问题备注").pack(anchor="w", pady=(6, 2))
        ttk.Entry(side, textvariable=self.note_var).pack(fill="x")
        ttk.Button(side, text="保存当前帧标注", command=self.save_current_annotation).pack(fill="x", pady=(8, 4))
        ttk.Label(side, text="完整评测时，登记帧由最早的目标人物框自动确定。",
                  justify="left", foreground="#555", wraplength=270).pack(anchor="w", pady=3)
        ttk.Button(side, text="标记重入起点 / 结束", command=self.mark_reentry).pack(fill="x", pady=3)
        ttk.Label(side, text="重入按钮：第一次记录目标重现起点，\n再次按下记录观察结束时刻。",
                  justify="left", foreground="#555").pack(anchor="w", pady=(2, 8))
        self.annotation_count_var = tk.StringVar(value="已标注 0 帧")
        ttk.Label(side, textvariable=self.annotation_count_var).pack(anchor="w", pady=4)

        controls = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        controls.pack(fill="x")
        ttk.Button(controls, text="上一帧", command=lambda: self.seek_frame(self.frame_index - 1)).pack(side="left")
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle_play)
        self.play_button.pack(side="left", padx=4)
        ttk.Button(controls, text="下一帧", command=lambda: self.seek_frame(self.frame_index + 1)).pack(side="left")
        self.position_scale = ttk.Scale(controls, from_=0, to=1, orient="horizontal", command=self._scale_seek)
        self.position_scale.pack(side="left", fill="x", expand=True, padx=12)

    def open_video(self) -> None:
        """让用户明确选择本机视频，再校验格式并读取第一帧；不接受摄像头编号或流地址。"""
        selected = filedialog.askopenfilename(
            title="选择本地测试视频",
            initialdir=str(PROJECT_ROOT / "resources"),
            filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"), ("全部文件", "*.*")],
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            messagebox.showerror("无法打开", "请选择存在的本地视频文件（MP4、MOV、AVI、MKV、WebM 或 M4V）。")
            return

        self.stop_playback()
        self._release_capture()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("无法打开", "OpenCV 无法解码所选文件，请检查视频格式。")
            return
        self.capture = capture
        self.video_path = path
        self.fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_index = 0
        self.annotations.clear()
        self.reentry_start = None
        self.reentry_events.clear()
        self.target_person_id = "person-1"
        self.person_id_var.set("person-1")
        self.is_target_var.set(True)
        self.video_sha256 = self._file_sha256(path)
        self.predictions.clear()
        self.prediction_path = None
        self.prediction_variant = None
        ok, frame = capture.read()
        if not ok:
            self._release_capture()
            messagebox.showerror("无法打开", "视频中没有可读取的画面。")
            return
        self.current_frame = frame
        self.video_time_seconds = self._read_video_time(0)
        self._load_annotation_for_frame()
        self._update_position()
        self._render_frame()
        self.position_scale.configure(to=max(1, self.frame_count - 1))

    def open_predictions(self) -> None:
        """加载同一视频的数值预测 JSON，在播放画面上叠加跟踪框供人工复核。"""
        if self.video_sha256 is None:
            messagebox.showwarning("请先打开视频", "先打开离线视频，再加载对应方案的预测 JSON。")
            return
        selected = filedialog.askopenfilename(
            title="选择离线算法预测 JSON",
            initialdir=str(PROJECT_ROOT / "outputs" / "evaluation"),
            filetypes=[("JSON 文件", "*.json")],
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError("预测文件版本不支持")
            if payload.get("video", {}).get("sha256") != self.video_sha256:
                raise ValueError("预测文件对应的不是当前视频（SHA-256 不一致）")
            indexed = {}
            for row in payload.get("predictions", []):
                frame_index = row.get("frame_index")
                if type(frame_index) is not int or frame_index < 0 or frame_index in indexed:
                    raise ValueError("预测文件中存在重复或非法帧号")
                if not isinstance(row.get("boxes"), list):
                    raise ValueError(f"第 {frame_index} 帧缺少 boxes 数组")
                indexed[frame_index] = row
        except (OSError, json.JSONDecodeError, ValueError) as error:
            messagebox.showerror("无法加载预测", str(error))
            return
        self.predictions = indexed
        self.prediction_path = path
        self.prediction_variant = str(payload.get("variant", "unknown"))
        self._render_frame()
        messagebox.showinfo("预测已加载", f"方案：{payload.get('variant', 'unknown')}；"
                            f"包含 {len(indexed)} 帧预测。播放或逐帧查看时会叠加预测框。")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """分块读取用户已选视频计算摘要，只返回 SHA-256，不复制或保存媒体字节。"""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _read_video_time(self, fallback_index: int) -> float:
        """优先读取解码器报告的视频时间，异常时用帧号/FPS 估算视频时间。"""
        if self.capture is not None:
            milliseconds = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            if math.isfinite(milliseconds) and milliseconds > 0:
                return milliseconds / 1000.0
        return fallback_index / self.fps if self.fps > 0 else 0.0

    def seek_frame(self, frame_index: int) -> None:
        """跳到指定原视频帧，读取新画面并载入该帧已有的标注。"""
        if self.capture is None or self.frame_count <= 0:
            return
        self.save_current_annotation(silent=True)
        target = max(0, min(int(frame_index), self.frame_count - 1))
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            return
        ok, frame = self.capture.read()
        if not ok:
            return
        self.frame_index = target
        self.current_frame = frame
        self.video_time_seconds = self._read_video_time(target)
        self._load_annotation_for_frame()
        self._update_position()
        self._render_frame()

    def toggle_play(self) -> None:
        """开始或暂停本地逐帧播放；每次计时由视频 FPS 决定，不调用网络接口。"""
        if self.capture is None:
            return
        if self.playing:
            self.stop_playback()
        else:
            self.playing = True
            self.play_button.configure(text="暂停")
            self._play_next()

    def _play_next(self) -> None:
        """读取下一帧并安排下一次 Tk 更新；到视频末尾自动暂停。"""
        if not self.playing or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop_playback()
            return
        self.save_current_annotation(silent=True)
        self.frame_index = min(self.frame_index + 1, self.frame_count - 1)
        self.current_frame = frame
        self.video_time_seconds = self._read_video_time(self.frame_index)
        self._load_annotation_for_frame()
        self._update_position()
        self._render_frame()
        delay_ms = max(1, round(1000 / self.fps))
        self.root.after(delay_ms, self._play_next)

    def stop_playback(self) -> None:
        """暂停播放并恢复播放按钮文字。"""
        self.playing = False
        if hasattr(self, "play_button"):
            self.play_button.configure(text="播放")

    def _scale_seek(self, value: str) -> None:
        """响应时间轴拖动；程序自己更新滑块时跳过回调，避免递归 seek。"""
        if self.updating_scale or self.capture is None:
            return
        self.stop_playback()
        self.seek_frame(round(float(value)))

    def _update_position(self) -> None:
        """刷新当前帧、视频时间及已标注帧数。"""
        if self.video_path is None:
            return
        self.position_var.set(
            f"{self.video_path.name}  |  帧 {self.frame_index}/{max(0, self.frame_count - 1)}"
            f"  |  视频时间 {self.video_time_seconds:.2f}s"
        )
        self.annotation_count_var.set(f"已标注 {len(self.annotations)} 帧")
        self.updating_scale = True
        self.position_scale.set(self.frame_index)
        self.updating_scale = False

    def _render_frame(self) -> None:
        """在画布内等比例显示当前帧并叠加人工人物框；不生成或保存渲染图片。"""
        if self.current_frame is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        height, width = self.current_frame.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        shown_width, shown_height = max(1, round(width * scale)), max(1, round(height * scale))
        left, top = (canvas_width - shown_width) // 2, (canvas_height - shown_height) // 2
        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")
        self.image_rect = (left, top, shown_width, shown_height)

        annotation = self.annotations.get(self.frame_index, {})
        for person in annotation.get("people", []):
            x1, y1, x2, y2 = person["bbox_xyxy"]
            color = "#ffd166" if person.get("person_id") == self.target_person_id else "#58d68d"
            box = (left + x1 * shown_width, top + y1 * shown_height,
                   left + x2 * shown_width, top + y2 * shown_height)
            self.canvas.create_rectangle(*box, outline=color, width=2)
            self.canvas.create_text(box[0] + 3, box[1] + 3,
                                    text=person.get("person_id", "person"), fill=color, anchor="nw")
        # 算法框用不同颜色显示，并标记绑定状态；只在屏幕绘制，不生成带框视频文件。
        prediction = self.predictions.get(self.frame_index, {})
        for box_data in prediction.get("boxes", []):
            coordinates = box_data.get("bbox_xyxy")
            if not isinstance(coordinates, list) or len(coordinates) != 4:
                continue
            x1, y1, x2, y2 = coordinates
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in coordinates):
                continue
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                continue
            color = "#2ecc71" if box_data.get("bound_to_target") is True else "#55c7ec"
            box = (left + x1 * shown_width, top + y1 * shown_height,
                   left + x2 * shown_width, top + y2 * shown_height)
            label = f"预测 track={box_data.get('track_id')} {box_data.get('state', '')}"
            self.canvas.create_rectangle(*box, outline=color, width=2, dash=(5, 3))
            self.canvas.create_text(box[0] + 3, box[1] + 3, text=label, fill=color, anchor="nw")
        if self.drag_start is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(), self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ffffff", width=2, dash=(4, 2))

    def _begin_box(self, event: tk.Event) -> None:
        """暂停播放并开始在当前画面拖出一个人物框。"""
        if self.current_frame is None:
            return
        self.stop_playback()
        if not self._point_inside_image(event.x, event.y):
            return
        self.drag_start = (event.x, event.y)

    def _drag_box(self, event: tk.Event) -> None:
        """拖动鼠标时显示临时矩形，帮助审阅者观察框的位置。"""
        if self.drag_start is not None:
            self._render_frame()

    def _finish_box(self, event: tk.Event) -> None:
        """把拖出的画布矩形换算为 0～1 归一化坐标，保存人物标注后重绘。"""
        if self.drag_start is None or self.current_frame is None:
            return
        start_x, start_y = self.drag_start
        self.drag_start = None
        if not self._point_inside_image(event.x, event.y):
            self._render_frame()
            return
        left, top, shown_width, shown_height = self.image_rect
        x1 = max(0.0, min(1.0, (min(start_x, event.x) - left) / shown_width))
        y1 = max(0.0, min(1.0, (min(start_y, event.y) - top) / shown_height))
        x2 = max(0.0, min(1.0, (max(start_x, event.x) - left) / shown_width))
        y2 = max(0.0, min(1.0, (max(start_y, event.y) - top) / shown_height))
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            self._render_frame()
            return
        person_id = self.person_id_var.get().strip()
        if not person_id:
            messagebox.showwarning("缺少人物编号", "请先填写人物编号，例如 person-1。")
            self._render_frame()
            return
        self._store_current_fields()
        people = self.annotations.setdefault(self.frame_index, self._empty_annotation())["people"]
        people[:] = [person for person in people if person["person_id"] != person_id]
        people.append({"person_id": person_id, "bbox_xyxy": [x1, y1, x2, y2]})
        if self.is_target_var.get():
            self.target_person_id = person_id
            # 人工已为目标编号画框，目标在本帧可见；同步状态避免导出时漏计可见样本。
            self.visibility_var.set("visible")
            self.annotations[self.frame_index]["target_visibility"] = "visible"
        self._refresh_people_list()
        self._update_position()
        self._render_frame()

    def _point_inside_image(self, x: float, y: float) -> bool:
        """判断画布坐标是否落在实际视频图像内，不让黑边变成人物标注区域。"""
        left, top, width, height = self.image_rect
        return left <= x <= left + width and top <= y <= top + height

    def _empty_annotation(self) -> dict:
        """返回当前帧的空标注记录，统一字段以便导出程序读取。"""
        return {
            "frame_index": self.frame_index,
            "video_time_seconds": round(self.video_time_seconds, 6),
            "target_visibility": "unknown",
            "scene": "unspecified",
            "people": [],
            "review_note": "",
            "issue_type": "other",
        }

    def _store_current_fields(self, force: bool = False) -> None:
        """把右侧表单存入当前帧记录；未编辑的播放帧不会被误记成已审阅。"""
        if self.current_frame is None:
            return
        annotation = self.annotations.get(self.frame_index)
        has_manual_content = (
            self.visibility_var.get() != "unknown"
            or bool(self.note_var.get().strip())
            or self.scene_var.get() != "unspecified"
            or (annotation is not None and bool(annotation.get("people")))
        )
        if annotation is None and not force and not has_manual_content:
            return
        if annotation is None:
            annotation = self.annotations.setdefault(self.frame_index, self._empty_annotation())
        annotation["video_time_seconds"] = round(self.video_time_seconds, 6)
        annotation["target_visibility"] = self.visibility_var.get()
        annotation["scene"] = self.scene_var.get()
        annotation["review_note"] = self.note_var.get().strip()
        annotation["issue_type"] = ISSUE_CODES.get(self.issue_var.get(), "other")

    def save_current_annotation(self, silent: bool = False) -> None:
        """保存当前帧的状态和文字备注；silent=True 供播放/跳转时自动保留编辑。"""
        if self.current_frame is None:
            return
        self._store_current_fields(force=not silent)
        self._update_position()
        if not silent:
            messagebox.showinfo("已保存", f"已保存第 {self.frame_index} 帧标注。")

    def _load_annotation_for_frame(self) -> None:
        """切换帧时载入已有标注；尚未标注的帧使用 unknown 和空备注。"""
        annotation = self.annotations.get(self.frame_index)
        self.visibility_var.set(annotation["target_visibility"] if annotation else "unknown")
        self.scene_var.set(annotation.get("scene", "unspecified") if annotation else "unspecified")
        self.note_var.set(annotation["review_note"] if annotation else "")
        code = annotation.get("issue_type", "other") if annotation else "other"
        self.issue_var.set(ISSUE_NAMES.get(code, ISSUE_LABELS[0]))
        self._refresh_people_list()

    def _refresh_people_list(self) -> None:
        """刷新当前帧的人物框列表，供用户选择并删除画错的框。"""
        if not hasattr(self, "people_list"):
            return
        self.people_list.delete(0, tk.END)
        annotation = self.annotations.get(self.frame_index, {})
        for person in annotation.get("people", []):
            is_target = "（目标）" if person["person_id"] == self.target_person_id else ""
            self.people_list.insert(tk.END, f"{person['person_id']}{is_target}")

    def remove_selected_box(self) -> None:
        """删除当前帧选中的人工人物框，不改动其他视频帧。"""
        selected = self.people_list.curselection()
        annotation = self.annotations.get(self.frame_index)
        if not selected or annotation is None:
            return
        person = annotation["people"].pop(selected[0])
        if person["person_id"] == self.target_person_id:
            annotation["target_visibility"] = "unknown"
            self.visibility_var.set("unknown")
        self._refresh_people_list()
        self._render_frame()

    def export_review_notes(self) -> None:
        """单独导出问题帧和文字备注，不要求人物框、可见性或登记帧。"""
        if self.video_path is None or self.video_sha256 is None:
            messagebox.showwarning("尚未打开视频", "请先打开本机上的测试视频。")
            return
        self._store_current_fields()
        video = {"id": self.video_sha256[:12], "sha256": self.video_sha256,
                 "fps": self.fps, "frame_count": self.frame_count}
        prediction = None
        if self.prediction_path is not None:
            prediction = {"file_name": self.prediction_path.name,
                          "variant": self.prediction_variant or "unknown"}
        payload = build_review_notes_payload(video, prediction, self.annotations)
        if not payload["notes"]:
            messagebox.showinfo("没有复核备注", "先暂停在有问题的帧，填写备注并保存当前帧。")
            return
        output_name = filedialog.asksaveasfilename(
            title="导出问题复核记录（不含画面和人物框）",
            initialdir=str(PROJECT_ROOT / "outputs" / "reviews"),
            initialfile=f"{self.video_path.stem}_review_notes.json",
            defaultextension=".json", filetypes=[("JSON 文件", "*.json")],
        )
        if not output_name:
            return
        output = Path(output_name).expanduser().resolve()
        if output == self.video_path:
            messagebox.showerror("路径冲突", "复核记录不能覆盖视频。")
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        messagebox.showinfo("问题复核已导出", f"已导出 {len(payload['notes'])} 条文字记录：\n{output}")

    def mark_reentry(self) -> None:
        """按下两次记录一次重入事件的起止帧，便于统计稳定恢复时间。"""
        if self.reentry_start is None:
            self._store_current_fields()
            annotation = self.annotations.get(self.frame_index)
            if (annotation is None or annotation["target_visibility"] != "visible"
                    or not any(person["person_id"] == self.target_person_id
                               for person in annotation["people"])):
                messagebox.showwarning(
                    "重入起点缺少目标标注",
                    "请先把当前帧标为 visible，并为目标人物画框，再记录重入起点。",
                )
                return
            self.reentry_start = (self.frame_index, self.video_time_seconds)
            messagebox.showinfo("重入起点", "已记录当前帧为目标重新出现的起点；请跳到观察窗口结束帧再按一次。")
            return
        start_frame, start_time = self.reentry_start
        if self.frame_index <= start_frame:
            messagebox.showwarning("结束帧无效", "重入观察结束帧必须晚于起点。")
            return
        # 明确保存窗口结束采样点；如果用户尚未判断，unknown 会让评测报告窗口未完整。
        self._store_current_fields(force=True)
        event_id = f"reentry-{len(self.reentry_events) + 1}"
        self.reentry_events.append({
            "event_id": event_id,
            "start_frame": start_frame,
            "end_frame": self.frame_index,
            "start_seconds": round(start_time, 6),
            "end_seconds": round(self.video_time_seconds, 6),
            "complete": True,
        })
        self.reentry_start = None
        messagebox.showinfo("重入窗口已记录", f"已添加 {event_id}。")

    def export_annotations(self) -> None:
        """将元数据、人工人物框和文字备注写为 JSON；绝不将当前帧图像写入文件。"""
        if self.video_path is None or self.video_sha256 is None:
            messagebox.showwarning("尚未打开视频", "请先打开本机上的测试视频。")
            return
        self._store_current_fields()
        automatic_registration = choose_automatic_registration(
            self.annotations, self.target_person_id)
        if automatic_registration is None:
            messagebox.showwarning("缺少目标人物框", "定量评测需要至少一帧人工确认的目标框；问题复核请用“导出问题复核”。")
            return
        # 统一选最早的人工目标框作为各方案的初始化时刻，避免用户分别指定造成比较不公平。
        missing_target_boxes = [
            annotation["frame_index"]
            for annotation in self.annotations.values()
            if annotation["target_visibility"] == "visible"
            and not any(person["person_id"] == self.target_person_id for person in annotation["people"])
        ]
        if missing_target_boxes:
            messagebox.showwarning(
                "可见目标缺少人工框",
                "以下标注帧标为目标可见，但没有目标人物框："
                + ", ".join(map(str, missing_target_boxes[:12]))
                + (" …" if len(missing_target_boxes) > 12 else "")
                + "。请补画目标框或将状态改为 unknown。",
            )
            return
        suggested = f"{self.video_path.stem}_annotations.json"
        output_name = filedialog.asksaveasfilename(
            title="保存匿名标注 JSON",
            initialdir=str(PROJECT_ROOT / "outputs" / "annotations"),
            initialfile=suggested,
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json")],
        )
        if not output_name:
            return
        output = Path(output_name).expanduser().resolve()
        if output == self.video_path:
            messagebox.showerror("路径冲突", "标注文件不能覆盖视频。")
            return
        payload = {
            "schema_version": 1,
            "video": {
                "id": self.video_sha256[:12],
                "sha256": self.video_sha256,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "width": int(self.current_frame.shape[1]),
                "height": int(self.current_frame.shape[0]),
            },
            "sampling": {"kind": "reviewed_frames", "interval_seconds": None},
            "target_person_id": self.target_person_id,
            "registration": {
                "frame_index": automatic_registration["frame_index"],
                "video_time_seconds": automatic_registration["video_time_seconds"],
                "person_id": self.target_person_id,
                "mode": "automatic_first_target_box",
            },
            "annotations": [self.annotations[index] for index in sorted(self.annotations)],
            "reentry_events": self.reentry_events,
            "privacy": {"video_copied": False, "frames_exported": False, "features_exported": False},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        messagebox.showinfo("标注已导出", f"已保存 {len(self.annotations)} 帧的数值标注：\n{output}")

    def _release_capture(self) -> None:
        """释放视频解码器并丢弃当前画面内存引用。"""
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.current_frame = None
        self.photo = None

    def close(self) -> None:
        """关闭窗口前停止 Tk 定时器并释放本地视频解码器。"""
        self.stop_playback()
        self._release_capture()
        self.root.destroy()


def main() -> None:
    """程序入口：创建本地窗口并等待用户操作；不会自动选择或读取任何视频。"""
    root = tk.Tk()
    VideoReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
