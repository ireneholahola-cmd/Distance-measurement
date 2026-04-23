# 路面风险融合功能说明

## 01_技术路线

本次功能围绕 `detect_3d_with_surface.py` 的风险状态输出做增强，不重写原有检测链路。检测脚本仍负责目标检测、路面坑洼/裂缝识别、风险场计算和 `decision_status` 判定；新增能力只在风险状态形成后接入。

声音预警在 `detect_3d_with_surface.py` 中显式初始化。检测脚本在启动阶段直接创建 `RiskSoundAlerter`，并在每帧生成 `frame_record` 后按风险来源播放中文语音：`HIGH` 播放 `sounds/danger_warning.wav`，`MEDIUM` 车辆风险播放 `sounds/vehicle_warning.wav`，`MEDIUM` 坑洼/路面风险播放 `sounds/pothole_warning.wav`。声音逻辑在 `risk_alerts/sound_processing` 中，并加入5秒冷却时间；即使关闭 JSONL 输出，语音仍然有效。

中文视觉提示框通过 PIL 绘制。`risk_alerts/warning_prompt` 提供 `draw_chinese_risk_prompt()`，在主视频画面 `im0` 完成原有路面标注后调用。`HIGH` 显示“警告：发生危险”，`MEDIUM` 显示“提示：请注意”，`LOW/CLEAR` 不显示。提示框绘制在主视频画面上，因此实时窗口、保存图片/视频、以及与右侧 BEV 风险图拼接后的输出都会带有提示。

字体加载采用跨平台自动发现：优先读取 `DRIVESAFE_CHINESE_FONT` 指定路径；Windows 下查找微软雅黑、黑体、宋体等字体；macOS 下查找苹方、华文黑体、宋体等字体；Linux 下查找 Noto CJK、Source Han、文泉驿等常见字体；最后用仓库已有的 `deep_sort/DeepSORT_Monet_traffic/simsunttc/simsun.ttc` 兜底，保证多数环境都能正确绘制中文。

路面模型继续使用原有 `code/models` 目录，保留模型文件名：`best.pt`、`best_night.pt`、`crack_best.pt`。检测脚本和路面检测器默认仍从该目录加载模型。

## 02_问题与解决方案

实现声音报警时，要求避免依赖 Python 启动阶段的自动导入行为。解决方案是把语音提醒直接接入 `detect_3d_with_surface.py`：在每帧结构化记录生成后显式触发语音处理，不再依赖任何启动钩子或运行时补丁。

声音连续播放容易在视频逐帧处理中重复叠音。解决方案是在声音处理模块中加入默认 5 秒冷却，并设置风险优先级：高风险可以打断中风险，中风险不会在冷却期内反复播放。

中文不能用 OpenCV 默认 `putText` 稳定显示。解决方案是改用 PIL 绘制中文文本，再把图像转回 OpenCV 的 BGR 格式，保证中文提示框能正确叠加到视频帧上。

跨平台字体路径不一致。解决方案是做多级字体查找：环境变量优先，其次按 Windows、macOS、Linux 分别查找常见中文字体，最后使用仓库内置宋体文件兜底。

新增模块一开始分散为声音和视觉两个顶层目录，后续维护不够清晰。解决方案是整理为统一的 `risk_alerts` 包，并按职责拆成 `sound_processing` 和 `warning_prompt` 两个子模块。

运行测试时发现当前 Python 环境缺少 `streamlit`，虽然依赖文件中已经声明。解决方案是补装当前环境缺失依赖；后续新环境应优先执行 `pip install -r requirements.txt`。

在 Windows 默认编码环境下，`requirements_common.txt` 顶部中文注释会按 GBK 读取并触发解码错误。解决方案是将该文件注释整理为 ASCII，不改变任何依赖项。

模型目录如果随意改名，默认加载路径和说明文档都需要同步更新，否则会导致路面检测模型加载失败。最终方案是保持原有 `code/models` 路径，减少迁移成本。
