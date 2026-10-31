# 测试与验证

## 自动测试

```powershell
.\envs\vitrack\python.exe -X utf8 -m pytest tests --basetemp=work/pytest -q
```

状态规则使用合成框和已知向量，覆盖候选计数、人脸归属、身体竞争、轨迹中断、距离变化、模板漂移、取消和会话清理。真实模型测试加载本地模型与公开样例。浏览器检查使用独立Edge会话和合成输入，并拒绝物理摄像头、麦克风权限；未安装Edge时相应检查跳过。

测试名称和断言描述具体约束。例如`test_distance_reentry.py`模拟相邻身体外观相近、最终与原模板差异较大的过程，再检查补脸和新轨迹重入，同时检查突然变化和旁人竞争不能触发错误学习。

启动入口的配置隔离、热点未开和端口占用由`test_launcher.py`检查。源码包的文件边界由`test_package.py`检查。

## 基准

```powershell
.\envs\vitrack\python.exe -X utf8 scripts/bench.py --frames 10
.\envs\vitrack\python.exe -X utf8 scripts/verify_stream.py --duration 60 --report outputs/acceptance/stream_check.json
```

`bench.py`比较本地模型、输入尺寸和CPU/GPU组合，只衡量检测器。`verify_stream.py`用公开样例推进浏览器到后端的链路，记录帧数、耗时、在途帧和内存变化；使用当前配置选定的身份模式。二者均不能当作手机真实拍摄的FPS或身份准确率。

正式比较应固定硬件、后端、分辨率与视频输入，并报告误认、漏认和重入恢复情况。当前没有这样的完整基准数据，单图增强只用于诊断。

离线人物跟踪对比方案、指标分母、人工框匹配及导出数据格式见[离线视频评测与算法比较](offline_evaluation.md)。公开论文成绩与本地自录视频不直接比较；训练集和最终测试素材按完整视频分开。

## 源码包

```powershell
.\envs\vitrack\python.exe -X utf8 scripts/package_source.py --output outputs/delivery/vitrack_source.zip
.\envs\vitrack\python.exe -X utf8 scripts/verify_package.py outputs/delivery/vitrack_source.zip --identity-mode hybrid --report outputs/delivery/package_check.json
```

打包按文件白名单收集源码、模型和明确指定的公开样例，附SHA256清单；排除Python环境、私有配置、证书、运行结果和本机工作记录。验证脚本独立解压、核对摘要，用包内模型执行公开样例检测、锁定和保存限制检查，最后关闭临时服务。

验证复用当前Python环境，不代表新电脑已经安装依赖；解压后运行`run.py`仍需按安装说明准备环境、网络和证书。

## 人工观察

已有Android手机上的照片登记、双人轻微移动与距离重入检查。尚需系统评估不同光照、遮挡、相似穿着、快速交叉和换衣场景。人工观察只记实际条件和结果，不由诊断事件次数推算准确率。
