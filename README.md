# Distance-measurement Surface Fusion Fork

这个 fork 的核心目标是：保留原 `detect_3d.py` 的主体结构，只新增少量路面融合 hook，方便同事直接对照原脚本维护。

原始入口 `detect_3d.py` 没有改动；`code/` 目录里的坑洼、裂缝相关源码也没有改动。融合版入口是 `detect_3d_with_surface.py`，复杂逻辑放在 `road_surface_fusion/`。

## 和原代码不一样的地方

### 1. 新增融合入口

新增或重建：

- `detect_3d_with_surface.py`
- `road_surface_fusion/`

`detect_3d_with_surface.py` 基本以 `detect_3d.py` 为底稿，只在固定位置加入路面相关 hook：

- 顶部新增 `road_surface_fusion` 的必要 import
- 初始化阶段新增路面检测、分析、融合、可视化和 JSONL writer 对象
- 每帧 `depth_map` 生成后新增路面检测和分析
- 动态风险计算后新增路面风险场叠加
- BEV 绘制时新增路面风险点
- 主画面显示/保存前新增坑洼、裂缝框和 mask 绘制
- 原 `vehicle_info.txt` 日志附近新增 JSONL 结构化输出

### 2. 保留原脚本结构

融合入口没有改成一套全新的流程，而是继续保留原脚本里的：

- `detect()` 主函数
- 主循环结构
- YOLO 检测流程
- DeepSort 跟踪流程
- Depth Anything 深度估计流程
- 3D box 绘制逻辑
- BEV 绘制逻辑
- 图片、视频保存逻辑
- `vehicle_info.txt` 文本日志逻辑

这样同事看 `detect_3d.py` 和 `detect_3d_with_surface.py` 的 diff 时，主要看到的是少量融合点，而不是一份重新设计过的新脚本。

### 3. 行人进入风险评估

原脚本动态风险过滤类别是：

```python
valid_classes = {1, 2, 3, 5, 7}
```

融合版改为：

```python
valid_classes = {0, 1, 2, 3, 5, 7}
```

也就是把 `person=0` 加入风险评估，其它动态目标处理逻辑尽量保持原样。

### 4. 新增路面风险融合

动态目标风险仍然按原脚本计算。融合版只在动态风险计算完成后追加路面风险：

```python
surface_risk_map, surface_vis_map, surface_risk = road_fuser.build_surface_maps(surface_analysis, risk_engine)
total_risk_map += surface_risk_map
vis_risk_map = np.maximum(vis_risk_map, surface_vis_map)
combined_risk = max(dynamic_risk, surface_risk)
```

如果本帧最高风险来自路面，并且检测到了坑洼或裂缝，则 HUD 风险来源显示为：

```python
max_risk_id = "ROAD"
```

### 5. 新增 JSONL 结构化日志

原来的 `vehicle_info.txt` 保留。融合版在原日志附近额外输出结构化 JSONL：

```text
runs/detect/exp/structured/frame_results.jsonl
```

实际路径受 `--project`、`--name`、`--exist-ok` 影响。

每行是一帧结果，主要字段包括：

- `targets`
- `surface_hazards`
- `dynamic_risk`
- `surface_risk`
- `combined_risk`
- `warning_text`
- `surface_summary`

JSONL schema 和写文件逻辑集中在：

```text
road_surface_fusion/structured_output.py
```

## 新增模块

`road_surface_fusion/` 负责承载复杂逻辑，避免 `detect_3d_with_surface.py` 变得不像原脚本。

主要文件：

- `detector.py`：加载和调用路面坑洼、裂缝模型
- `surface_analysis.py`：解析检测结果，估计位置、距离、严重程度和路面风险
- `risk_fusion.py`：生成路面风险场，并提供 `max(dynamic_risk, surface_risk)` 融合规则
- `visualization.py`：绘制主画面坑洼/裂缝结果和 BEV 路面风险点
- `structured_output.py`：维护 JSONL schema 和按帧写出逻辑
- `depth_runtime.py`：保留深度后端封装，当前融合入口仍沿用原脚本的 `DepthEstimator`

`road_surface_fusion/__init__.py` 统一导出入口需要的对象：

```python
from road_surface_fusion import (
    RoadSurfaceAnalyzer,
    RoadSurfaceDetector,
    RoadSurfaceRiskFuser,
    RoadSurfaceVisualizer,
    StructuredOutputWriter,
    build_frame_record,
)
```

## 运行方式

在仓库目录运行：

```powershell
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu
```

快速烟测 10 帧：

```powershell
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu --max-frames 10
```

关闭 JSONL：

```powershell
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu --max-frames 10 --no-save-jsonl
```

## 新增命令行参数

融合版在原参数基础上新增：

- `--road-model-dir`：路面模型目录，默认 `code/models`
- `--road-conf-thres`：路面检测置信度阈值，默认 `0.25`
- `--depth-backend`：深度后端参数，当前只允许 `depth-anything`
- `--max-frames`：最多处理多少帧，默认不限制，方便烟测
- `--save-jsonl`：开启 JSONL，默认开启
- `--no-save-jsonl`：关闭 JSONL
- `--structured-dir`：JSONL 输出子目录，默认 `structured`
- `--no-view-img`：关闭 OpenCV 窗口，方便无界面测试

注意：`--nosave` 只关闭图片/视频保存，不关闭 JSONL。要完全不输出 JSONL，需要同时传 `--no-save-jsonl`。

## 模型和依赖

路面模型默认从以下目录读取：

```text
code/models
```

目录下应包含：

- `best.pt`
- `best_night.pt`
- `crack_best.pt`

Depth Anything 仍沿用原脚本的深度估计逻辑。如果本地没有缓存模型，运行时可能访问 Hugging Face 下载或校验模型文件。无网络环境下需要提前准备好缓存。

`requirements.txt` 当前包含 `-r requirements_common.txt`，老版 `check_requirements()` 对这种写法解析不稳定，所以融合入口启动检查改为：

```python
check_requirements('requirements_common.txt', exclude=('pycocotools', 'thop'))
```

这只影响 `detect_3d_with_surface.py`，不改 `detect_3d.py`。

## 已验证

已完成的本地检查：

- `detect_3d_with_surface.py` 和 `road_surface_fusion/*.py` AST 语法解析通过
- `python detect_3d_with_surface.py --help` 通过
- `detect_3d.py` 与 `detect_3d_with_surface.py` 的 diff 约 155 行，差异集中在融合 hook、JSONL、参数和 `person=0`
- 10 帧烟测通过：

```powershell
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu --max-frames 10
```

烟测生成：

```text
runs/detect/exp7/structured/frame_results.jsonl
```

检查结果：

- JSONL 共 10 行
- 每行可以用 Python `json.loads` 正常解析
- 每行包含 `targets`、`surface_hazards`、`dynamic_risk`、`surface_risk`、`combined_risk`、`warning_text`

## 本次 fork 的边界

本次改动没有做这些事：

- 没有修改 `detect_3d.py`
- 没有修改 `code/` 源码
- 没有把融合入口改成新的 SDK 或服务
- 没有新增 HTTP、WebSocket、REST API
- 没有重构原始 YOLO、DeepSort、Depth Anything、BEV 保存流程

这个 fork 只是在原 3D 风险脚本旁边新增一个路面融合入口，并把坑洼、裂缝、路面分析、可视化和 JSONL schema 放到 `road_surface_fusion/` 里维护。

## 建议提交信息

```text
refactor: keep surface fusion entry close to detect_3d
```

或者：

```text
feat: add road surface fusion hooks and structured jsonl output
```
