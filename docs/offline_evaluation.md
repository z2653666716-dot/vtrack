# 离线视频评测与算法比较

本文规定怎样用同一段视频、同一组人工标注比较 ViTrack 的跟踪方案。目标是分开衡量检测框定位、身份判断和重入恢复，避免把框抖动、漏检或无法判断都混成一个“准确率”。当前内容是评测协议；没有人工标注前不产生识别提升结论。

## 比较对象

| 名称 | 跟踪与身份方式 | 这组比较能回答什么 |
| --- | --- | --- |
| ByteTrack 基线 | ByteTrack 依靠检测框与运动连续性；初始目标关联到的轨迹号视为目标，不增加跨轨迹身份恢复 | 轨迹中断或换号时，纯运动关联能保留目标多久 |
| 身体外观方案 | BoT-SORT + OSNet 身体特征，不使用 SFace 人脸证据 | 增加身体外观后，对远近变化和重入有何影响 |
| ViTrack 联合方案 | BoT-SORT + OSNet + YuNet/SFace，并使用当前多帧确认和模板策略 | 当前完整身份流程的实际结果 |
| 人脸证据消融 | 与 ViTrack 联合方案使用同一 `HybridSession`、检测器、BoT-SORT、OSNet 和确认门槛，只关闭人脸证据 | 当前视频中的变化是否来自人脸证据开关 |

正式比较入口已实现为 `scripts/evaluate_video.py`：`bytetrack` 使用 ByteTrack 轨迹号，`body` 使用 BoT-SORT + OSNet 外观身份流程，`hybrid` 使用当前完整人脸/身体联合会话。三者均复用项目 `Pipeline`，同一人工标注登记帧各自新建会话。是否能在当前电脑完成推理仍取决于模型文件和运行环境，模型加载失败会明确报错，不生成伪造预测。

ByteTrack、body 和完整 hybrid 是整套方案对比，不能单独把它们的差异归因给某个模型。`hybrid_no_face` 才是人脸证据消融：沿用同一身份会话与 OSNet，只关闭人脸编码、匹配和人脸补充检测框。公开论文中的 MOT 或人脸识别成绩来自不同数据、标注和任务定义，只用于解释方法背景，不直接与本项目数字相减。

## 固定实验条件

每次比较需记录视频 SHA-256、代码提交号、模型文件 SHA-256、检测输入尺寸与阈值、设备、采样规则、人工标注版本、自动确定的目标登记帧和对应人工框。所有方案使用相同检测权重、视频解码帧、登记时刻和评估采样点；需要研究检测器时再单独设实验组。每个方案都从空会话开始，不能让某方案继承另一方案的模板或状态。

同一视频可用于调试与开发，但不能再被称为独立最终测试集。视频相邻帧高度相似，不能随机拆帧作训练集和测试集。要报告泛化表现，需另留未用于调参的视频，并按完整视频划分。

建议同时报告三类结果：

1. **目标正确覆盖率**：目标人工标注为可见的采样帧中，系统绑定框能与目标人物框唯一匹配的比例。它综合体现检测、定位与身份保持，不应单独称为身份准确率。
2. **匹配绑定框身份正确率**：系统绑定框能与某个人工人物框唯一匹配时，匹配到目标人物的比例。定位不够好、无法唯一对应的框不进入此分母，另列“无法对应绑定框数/比例”。
3. **错绑比例与漏绑帧数**：错绑比例的分母是能与人工人物框匹配的已绑定框；目标可见但未绑定另计。目标已离场时仍绑定到其他人也记为错绑。

重入事件单独统计成功事件数/可评分事件数、恢复起点延迟和稳定确认延迟。处理耗时以 `perf_counter` 实测并独立统计，不用电脑推理速度代替视频时间或手机端延迟。

## 人工框与系统框的对应规则

人工为每个可辨认的人分配本视频内稳定的 `person_id`，每个标注采样帧记录其边界框。目标可见性由人工标注为 `visible`、`absent` 或 `unknown`；未知帧不计入正确率分母。系统输出框与人工框依据交并比（IoU）作一对一最大匹配，默认 IoU 至少 0.5 才建立对应。一个人工人物不能同时对应两个系统框，一个系统框也不能同时对应两个人。

匹配后，已绑定框对应目标人物记为 `target`，对应另一人记为 `other`。ByteTrack 基线按登记时的轨迹号判断绑定；body/hybrid 身份方案按锁定后的稳定身份编号判断，因此目标重入后轨迹号改变但身份编号仍匹配时，仍可计为目标绑定。系统框没有达到 IoU 门槛或存在同等合理的多人归属时记为 `unmatched`，单独报告，不能直接当作认错人。目标可见但没有任何目标绑定框时记为未绑定。多人重叠导致人工无法确认身份时，该帧标成 `unknown`，不要猜测。

结果需另做 IoU 0.3、0.5、0.7 的敏感性分析，确认结论是否依赖单一框匹配门槛。主结果仍预先固定为 0.5，不能看到比较结果后再挑最有利的门槛。

## 文件格式草案

人工标注 JSON 保存视频信息、采样规则、目标登记操作、人物框、可见性与人工审阅备注。系统预测 JSON 保存每个方案的代码/模型/配置摘要、帧时间、输出框、绑定状态和处理耗时。两者以视频 SHA-256、帧号和视频时间对齐；报告程序从两类记录生成逐帧 `FrameObservation`，再调用 `app/evaluation.py`。

当前本地审阅窗口可从项目根目录启动：

```powershell
.\envs\vitrack\python.exe -X utf8 scripts/review_video.py
```

点击“打开本地视频”后选择文件。空格播放/暂停，左右方向键逐帧移动，也可拖动时间轴。**日常问题复核不需要画人物框、不需要登记目标**：加载一种方案预测，播放或逐帧查看；发现异常就暂停，选问题类型、写一句现象说明，点击“保存当前帧标注”，最后点“导出问题复核”。导出的 JSON 只包含视频 SHA-256、预测方案名、帧号、视频时间、问题类别和文字，不包含框、画面或特征。可以把这个 JSON 发给助手分析。

若要计算准确率等指标，才使用“导出 JSON 标注”。这条路径需要少量人工真值：在你认为有代表性的帧上为目标人物画框；单人视频只需把目标编号保持为 `person-1`。导出时程序自动选择最早出现的 `person-1` 人工框作为共同登记时刻，不需要再点登记按钮。其余未标注帧不参与计分。只想描述错误时，使用前面的“导出问题复核”，不必补齐定量标注。场景标签可选 `far_to_near`、`reentry`、`back_turn`、`multi_person`、`other`；不确定时留空。只有要计算重入耗时时，才需在重现起点和观察结束位置标注目标框并按两次“标记重入起点 / 结束”。

算法跑完后，在审阅界面打开同一视频，再点“加载方案预测”选择某个预测 JSON。青色虚线表示其他预测框，绿色虚线表示系统绑定到目标的框。暂停并记录问题时，备注独立导出；它不会被当作真值或自动参与评分。完整定量标注另由“导出 JSON 标注”保存。

“导出问题复核”只保存问题文字和定位信息；“导出 JSON 标注”才会检查每个 `visible` 帧是否含目标人物框，并保存帧号、时间、归一化框、状态和备注。视频通过 OpenCV 在本机选择和解码，摘要以分块方式计算；不创建检测视频、缩略图或人物裁剪，也不把视频发送给服务端。

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

只为人工复核重新生成预测时，单人视频不需要标注文件。省略 `--annotations` 后，回放会从首个稳定人物轨迹自动初始化；该模式只生成预测，不计算准确率。多人视频不能用此自动选择模式。每个方案都会从新会话开始，预测只包含数值框、状态、时间和耗时：

```powershell
$python = ".\envs\vitrack\python.exe"
$video = "resources\你的测试视频.mp4"

& $python -X utf8 scripts/evaluate_video.py --video $video --variant bytetrack
& $python -X utf8 scripts/evaluate_video.py --video $video --variant body
& $python -X utf8 scripts/evaluate_video.py --video $video --variant hybrid_no_face
& $python -X utf8 scripts/evaluate_video.py --video $video --variant hybrid
```

同一个视频要归因人脸证据时，完整 `hybrid` 和 `hybrid_no_face` 使用同一登记方式与其他配置。传入人工确认的离场起始帧和重现帧，运行汇总器：

```powershell
& $python -X utf8 scripts/summarize_ablation.py `
  --hybrid outputs\evaluation\你的测试视频_hybrid_predictions.json `
  --no-face outputs\evaluation\你的测试视频_hybrid_no_face_predictions.json `
  --absent-range 266:389
```

`--absent-range 266:389` 表示第266帧起离场，第389帧重新出现；起始包含、重现帧不包含。多个离场区间可重复传入参数。汇总器检查视频 SHA、模型哈希、共享配置、帧时间和登记时刻，生成逐帧 CSV 及 JSON/Markdown 报告。它报告可见帧绑定覆盖率和离场期间仍然显示绑定的帧数，不会把没有人工逐帧真值的结果称作身份准确率。

审阅时可依次将每个预测 JSON 载入界面核对暂停帧和备注。上述自动初始化预测没有人工真值，不能直接交给汇总器计算准确率。要生成定量报告，先导出带人物框的人工 JSON，再为三个方案分别增加 `--annotations $annotations` 重跑；之后才运行汇总器。汇总器要求三份预测使用相同视频、相同人工标注摘要、相同检测权重/尺寸/阈值/设备和身份推理设备；不同跟踪与身份算法本身是比较变量。默认 IoU 0.5 为主结果，同时固定输出 0.3 与 0.7 的门槛敏感性检查。

## 结果解释边界

单个短视频只能说明这些方案在该视频和标注规则下的表现，不能支撑普遍准确率或统计显著性结论。最终简历应写成“在自录场景的固定评测中，某指标从 A 变化到 B，条件为……”，同时保留测试规模与失败类型；拿到可重复的多视频测试集后再总结泛化结论。

## 多序列身体模板更新实验（B0/B1/B2）

`hybrid_b0` 使用现有追加模板逻辑；`hybrid_b1` 保留初始锚点并用固定权重 EMA 更新身体动态模板；`hybrid_b2` 保持 B1 的资格条件与时机，只令 EMA 权重乘以观测质量分数。质量分数由人物裁剪尺寸、清晰度、画面边缘风险和与旁人框重叠构成，只用于降低低质量观测的更新权重，不是身份置信度。

准备 MOT17 训练序列时，视频从 [MOTChallenge 官方序列归档](https://motchallenge.net/data/MOT17/) 取得，标签使用 `MOT17Labels.zip`。原始视频、标签和生成的完整帧级标注只保存在本地 `work/` 与被忽略的 `outputs/`，不可提交到 Git、源码包或公开报告仓库。MOT17 官方条款为 CC BY-NC-SA 3.0；引用时应注明 MOTChallenge 及其论文，任何对数据本身的再分发都需遵守原条款。MOT17-05/09 已用于开发；本项目记录里未发现 02/04/10/11/13 被用于调参，因此本次把这些完整序列作为项目内冻结保留集。它们仍属于 MOT17 training split，**不是官方 test split，也不证明跨数据集泛化**。官方页面已将 MOTChallenge 在线评测标为关闭；公开 test split 没有可用于本流程的 GT。

每段视频固定选一条 GT 行人轨迹：在 `mark>0`、`class=1`、`visibility>0` 的记录中，选可见标注帧数最多的 track ID，并列取较小 ID。该规则只读人工 GT，不看任何方案预测；选出的 ID 和自动登记帧写入标注文件。当前五条序列对应的 ID 为：02→19、04→1、10→4、11→1、13→39。一个视频只选一人，因此结果主要用于方案的初步保留评估，不能把帧数误当成多目标数量或独立样本数。

先固定代码和参数，再对同一视频分别运行三种方案。登记帧由标注工具自动取目标首次满足条件的可见帧；所有方案使用同一检测器、视频帧、目标 ID、模型与配置。不要在查看保留结果后调整阈值或质量映射；如需修改，应另留新的未见视频。

```powershell
$python = ".\envs\vitrack\python.exe"
$sequence = "02"
$targetTrack = 19
$video = "work\public_datasets\mot17_holdout\MOT17-$sequence-DPM-raw.mp4"
$labels = "work\public_datasets\mot17_holdout\MOT17Labels.zip"
$annotations = "outputs\evaluation\MOT17-$sequence\MOT17-$sequence-DPM_visibility_annotations.json"
& $python -X utf8 scripts/prepare_mot17_annotations.py --labels-zip $labels --video $video --sequence $sequence --target-track-id $targetTrack --output $annotations
foreach ($variant in @("hybrid_b0", "hybrid_b1", "hybrid_b2")) {
  & $python -X utf8 scripts/evaluate_video.py --video $video --annotations $annotations --variant $variant --output "outputs\evaluation\quality_template\MOT17-$sequence`_$variant`_predictions.json"
}
& $python -X utf8 scripts/summarize_evaluation.py --annotations $annotations --predictions `
  "outputs\evaluation\quality_template\MOT17-$sequence`_hybrid_b0_predictions.json" `
  "outputs\evaluation\quality_template\MOT17-$sequence`_hybrid_b1_predictions.json" `
  "outputs\evaluation\quality_template\MOT17-$sequence`_hybrid_b2_predictions.json" `
  --output-dir "outputs\evaluation\quality_template\MOT17-$sequence\report"
```

逐视频报告包含 JSON、CSV 和 Markdown。把至少两条完整视频的 `evaluation_summary.json` 传给 `scripts/summarize_sequence_group.py --reports ... --output-dir ...`，可再生成多视频 JSON、CSV、Markdown。汇总同时给出全量尝试的帧加权与视频等权覆盖率，以及登记成功子集的覆盖率和登记成功/失败数：全量结果反映端到端流程，登记成功子集更适合分析后续身份更新策略。相邻帧高度相关，推断样本量时仍以完整视频数为准，不报告统计显著性。

保留集应原样报告所有运行结果，包括登记失败、未绑定和错绑。若要继续开发，先把保留集降级为开发数据，不能继续把同一组视频称作未见测试。处理耗时是电脑离线端单帧后端耗时，不代表手机 FPS 或端到端延迟。MOT17 标签文件提供训练集真值；公开榜单上的 MOT17 test split 没有公开 GT，不能用其可视化视频估算身份准确率。
