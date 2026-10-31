# 离线视频评测与算法比较

本文规定怎样用同一段视频、同一组人工标注比较 ViTrack 的跟踪方案。目标是分开衡量检测框定位、身份判断和重入恢复，避免把框抖动、漏检或无法判断都混成一个“准确率”。当前内容是评测协议；没有人工标注前不产生识别提升结论。

## 比较对象

| 名称 | 跟踪与身份方式 | 这组比较能回答什么 |
| --- | --- | --- |
| ByteTrack 基线 | ByteTrack 依靠检测框与运动连续性；初始目标关联到的轨迹号视为目标，不增加跨轨迹身份恢复 | 轨迹中断或换号时，纯运动关联能保留目标多久 |
| 身体外观方案 | BoT-SORT + OSNet 身体特征，不使用 SFace 人脸证据 | 增加身体外观后，对远近变化和重入有何影响 |
| ViTrack 联合方案 | BoT-SORT + OSNet + YuNet/SFace，并使用当前多帧确认和模板策略 | 当前完整身份流程的实际结果 |

正式比较入口已实现为 `scripts/evaluate_video.py`：`bytetrack` 使用 ByteTrack 轨迹号，`body` 使用 BoT-SORT + OSNet 外观身份流程，`hybrid` 使用当前完整人脸/身体联合会话。三者均复用项目 `Pipeline`，同一人工标注登记帧各自新建会话。是否能在当前电脑完成推理仍取决于模型文件和运行环境，模型加载失败会明确报错，不生成伪造预测。

上述方案之间存在多个模块变化，因此它们适合回答“整套方案是否更适合这个任务”，不能据此单独宣称“SFace 带来了全部提升”。要归因到人脸模块，后续还需增加一组只开关人脸证据、其余设置相同的消融比较。公开论文中的 MOT 或人脸识别成绩来自不同数据、标注和任务定义，只用于解释方法背景，不直接与本项目数字相减。

## 固定实验条件

每次比较需记录视频 SHA-256、代码提交号、模型文件 SHA-256、检测输入尺寸与阈值、设备、采样规则、人工标注版本、目标登记帧和登记框。所有方案使用相同的检测权重、视频解码帧、目标登记时刻和评估采样点；需要研究检测器时再单独设实验组。每个方案都从空会话开始，按相同人工登记事件建立目标，不能让某方案继承另一方案的模板或状态。

同一视频可用于调试与开发，但不能再被称为独立最终测试集。视频相邻帧高度相似，不能随机拆帧作训练集和测试集。要报告泛化表现，需另留未用于调参的视频，并按完整视频划分。

建议同时报告三类结果：

1. **目标正确覆盖率**：目标人工标注为可见的采样帧中，系统绑定框能与目标人物框唯一匹配的比例。它综合体现检测、定位与身份保持，不应单独称为身份准确率。
2. **匹配绑定框身份正确率**：系统绑定框能与某个人工人物框唯一匹配时，匹配到目标人物的比例。定位不够好、无法唯一对应的框不进入此分母，另列“无法对应绑定框数/比例”。
3. **错绑比例与漏绑帧数**：错绑比例的分母是能与人工人物框匹配的已绑定框；目标可见但未绑定另计。目标已离场时仍绑定到其他人也记为错绑。

重入事件单独统计成功事件数/可评分事件数、恢复起点延迟和稳定确认延迟。处理耗时以 `perf_counter` 实测并独立统计，不用电脑推理速度代替视频时间或手机端延迟。

## 人工框与系统框的对应规则

人工为每个可辨认的人分配本视频内稳定的 `person_id`，每个标注采样帧记录其边界框。目标可见性由人工标注为 `visible`、`absent` 或 `unknown`；未知帧不计入正确率分母。系统输出框与人工框依据交并比（IoU）作一对一最大匹配，默认 IoU 至少 0.5 才建立对应。一个人工人物不能同时对应两个系统框，一个系统框也不能同时对应两个人。

匹配后，已绑定框对应目标人物记为 `target`，对应另一人记为 `other`。系统框没有达到 IoU 门槛或存在同等合理的多人归属时记为 `unmatched`，单独报告，不能直接当作认错人。目标可见但没有任何目标绑定框时记为未绑定。多人重叠导致人工无法确认身份时，该帧标成 `unknown`，不要猜测。

结果需另做 IoU 0.3、0.5、0.7 的敏感性分析，确认结论是否依赖单一框匹配门槛。主结果仍预先固定为 0.5，不能看到比较结果后再挑最有利的门槛。

## 文件格式草案

人工标注 JSON 保存视频信息、采样规则、目标登记操作、人物框、可见性与人工审阅备注。系统预测 JSON 保存每个方案的代码/模型/配置摘要、帧时间、输出框、绑定状态和处理耗时。两者以视频 SHA-256、帧号和视频时间对齐；报告程序从两类记录生成逐帧 `FrameObservation`，再调用 `app/evaluation.py`。

当前本地审阅窗口可从项目根目录启动：

```powershell
.\envs\vitrack\python.exe -X utf8 scripts/review_video.py
```

点击“打开本地视频”后选择文件。空格播放/暂停，左右方向键逐帧移动，也可拖动时间轴；在画面上拖鼠标为人物画框。每个人使用稳定的 `person_id`，目标人物框勾选“这是登记目标”。每个评测采样帧手动设定可见状态和场景标签，再保存。场景标签可选 `far_to_near`、`reentry`、`back_turn`、`multi_person`、`other`；它们由人工按帧选择，用于分场景汇总。登记帧必须先有目标框；重入起点也必须标为 `visible` 并画出目标框。按“标记重入起点 / 结束”两次创建事件窗口。

算法跑完后，在审阅界面打开同一视频，再点“加载方案预测”选择某个预测 JSON。青色虚线表示其他预测框，绿色虚线表示系统绑定到目标的框。暂停、逐帧查看，给失败帧添加文字备注并选择场景；人工框仍用于评测真值，虚线预测框不会被写成真值。导出的数据只有数值和文字。

导出前会检查每个 `visible` 帧都含目标人物框。程序只保存帧号、时间、归一化框、状态和备注。视频通过 OpenCV 在本机选择和解码，摘要以分块方式计算；不创建检测视频、缩略图或人物裁剪，也不把视频发送给服务端。

示例只包含数值和文字，不包含视频帧、截图、裁剪图或特征向量：

```json
{
  "schema_version": 1,
  "video": {"id": "clip-a", "sha256": "<64位摘要>", "fps": 30.0, "frame_count": 900},
  "sampling": {"kind": "timestamp_interval", "interval_seconds": 0.2},
  "target_person_id": "person-1",
  "registration": {"frame_index": 120, "video_time_seconds": 4.0, "person_id": "person-1"},
  "annotations": [
    {"frame_index": 120, "video_time_seconds": 4.0, "target_visibility": "visible",
     "people": [{"person_id": "person-1", "bbox_xyxy": [0.2, 0.1, 0.6, 0.9]}],
     "review_note": "远景登记"}
  ],
  "reentry_events": [
    {"event_id": "reentry-1", "start_frame": 450, "end_frame": 600,
     "start_seconds": 15.0, "end_seconds": 20.0, "complete": true}
  ]
}
```

审阅备注是人工输入的文字，可以描述“转身后丢失”或“框偏向旁人”；它不参与自动正确性判断。导出文件不得包含本机绝对路径或原始媒体字节。标注文件中的 SHA-256 用于确认视频一致，不用于恢复视频。

## 运行三种方案与生成对比报告

先完成逐帧人工标注，并记下登记帧。之后对同一个视频和同一份标注分别运行以下命令；每个方案都会从全新会话开始，预测只包含框、状态、时间和耗时：

```powershell
$python = ".\envs\vitrack\python.exe"
$video = "resources\你的测试视频.mp4"
$annotations = "outputs\annotations\你的测试视频_annotations.json"

& $python -X utf8 scripts/evaluate_video.py --video $video --annotations $annotations --variant bytetrack
& $python -X utf8 scripts/evaluate_video.py --video $video --annotations $annotations --variant body
& $python -X utf8 scripts/evaluate_video.py --video $video --annotations $annotations --variant hybrid

& $python -X utf8 scripts/summarize_evaluation.py `
  --annotations $annotations `
  --predictions outputs\evaluation\你的测试视频_bytetrack_predictions.json `
               outputs\evaluation\你的测试视频_body_predictions.json `
               outputs\evaluation\你的测试视频_hybrid_predictions.json
```

审阅时可依次将每个预测 JSON 载入界面核对暂停帧和备注。汇总器要求三份预测使用相同视频、相同人工标注摘要、相同检测权重/尺寸/阈值/设备和身份推理设备；不同跟踪与身份算法本身是比较变量。默认 IoU 0.5 为主结果，同时固定输出 0.3 与 0.7 的门槛敏感性检查。报告保存在 `outputs/evaluation/<video-id>/`，包括 `evaluation_summary.json`、`.csv`、`.md`、`frame_results.csv`、`failure_cases.csv`、`scene_summary.csv` 和 `iou_sensitivity.csv`。

## 结果解释边界

单个短视频只能说明这些方案在该视频和标注规则下的表现，不能支撑普遍准确率或统计显著性结论。最终简历应写成“在自录场景的固定评测中，某指标从 A 变化到 B，条件为……”，同时保留测试规模与失败类型；拿到可重复的多视频测试集后再总结泛化结论。
