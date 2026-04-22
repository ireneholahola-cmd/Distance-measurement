# 驭安DriveSafe 主动安全预警系统

## 项目概述

驭安DriveSafe是一个基于计算机视觉和深度学习的主动安全预警系统，专注于实时车辆检测、距离测量、风险评估和可视化。该系统使用先进的3D目标检测和深度估计技术，为驾驶员提供周围环境的实时风险分析，帮助提高驾驶安全性。

## 核心功能

- **实时目标检测**：使用YOLOv10模型检测车辆、行人和其他障碍物
- **3D边界框估计**：基于深度信息和相机参数计算目标的3D位置和尺寸
- **深度估计**：使用Depth Anything v2模型估计场景深度
- **目标跟踪**：使用DeepSort算法跟踪目标并分配唯一ID
- **风险场计算**：基于论文公式计算车辆周围的风险场
- **鸟瞰图可视化**：提供场景的鸟瞰视图，显示目标位置和风险分布
- **速度估计**：基于目标运动计算速度
- **路面平整度检测**：检测路面坑洼和裂缝，评估路面风险
- **用户友好界面**：使用Streamlit构建的直观界面

## 技术栈

- **深度学习框架**：PyTorch
- **目标检测**：YOLOv10
- **深度估计**：Depth Anything v2
- **目标跟踪**：DeepSort
- **图像处理**：OpenCV
- **用户界面**：Streamlit
- **其他库**：NumPy, SciPy, Transformers

## 项目结构

```
├── app.py                    # Streamlit应用程序
├── detect_3d.py              # 主检测和处理逻辑
├── detect_3d_with_surface.py # 带路面检测的主检测逻辑
├── depth_model.py            # 深度估计模型
├── bbox3d_utils.py           # 3D边界框估计和可视化
├── risk_field.py             # 风险场计算
├── road_surface_fusion/      # 路面平整度检测和风险融合
├── yolov10/                  # YOLOv10模型
├── deep_sort/                # DeepSort目标跟踪
├── models/                   # 模型权重
├── data/                     # 测试数据
├── utils/                    # 工具函数
├── requirements.txt          # 依赖项
├── requirements_common.txt   # 通用依赖项
└── requirements_gpu.txt      # GPU依赖项
```

## 安装说明

### 1. 克隆仓库

```bash
git clone <repository-url>
cd Distance-measurement
```

### 2. 安装依赖

根据您的环境选择合适的依赖文件：

#### CPU环境

```bash
pip install -r requirements_common.txt
```

#### GPU环境

```bash
pip install -r requirements_gpu.txt
```

### 3. 下载模型权重

系统默认使用以下模型：
- YOLOv10s：用于目标检测
- Depth Anything v2 Small：用于深度估计

模型会在首次运行时自动下载。

## 使用方法

### 1. 启动应用

```bash
streamlit run app.py
```

### 2. 配置参数

在应用界面中，您可以：
- 选择视频源（摄像头或视频文件）
- 上传自定义视频文件
- 调整摄像头编号（默认0）

### 3. 开始检测

点击「开始识别」按钮启动系统，您将看到：
- 左侧：实时视频流，带有3D边界框
- 右侧：风险场视图，显示目标位置和风险分布

### 4. 使用路面平整度检测

要启用路面平整度检测功能，请使用以下命令：

```bash
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu
```

参数说明：
- `--source`：视频源（摄像头或视频文件）
- `--no-view-img`：禁用实时显示
- `--nosave`：禁用结果保存
- `--device`：运行设备（cpu或cuda）

### 5. 查看结果

系统会实时显示：
- 检测到的目标及其3D边界框
- 目标距离和风险评分
- 鸟瞰图中的目标位置和运动轨迹
- 风险场热力图

## 技术实现细节

### 1. 目标检测与跟踪

- 使用YOLOv10模型进行目标检测，支持多种目标类别
- 使用DeepSort算法进行目标跟踪，分配唯一ID并保持跟踪
- 过滤有效车辆类型（bicycle, car, motorcycle, bus, truck）

### 2. 深度估计

- 使用Depth Anything v2模型估计场景深度
- 支持不同模型大小（small, base, large）以平衡性能和精度
- 自动处理设备兼容性（CPU/GPU/MPS）

### 3. 3D边界框估计

- 基于深度值和相机参数计算3D位置
- 使用Kalman滤波器平滑3D框参数
- 考虑目标类型和尺寸的先验知识

### 4. 风险场计算

- 基于论文公式计算运动物体和静止物体的安全势能
- 考虑车速、路面附着系数等因素
- 生成热力图可视化风险分布

### 5. 可视化

- 3D边界框绘制，带有深度感知效果
- 鸟瞰图显示目标位置和运动轨迹
- 风险场热力图，颜色编码风险级别
- 未来运动预测扇形图

### 6. 路面平整度检测

- 使用专门的模型检测路面坑洼和裂缝
- 支持不同场景（白天/夜间）的路面检测
- 基于深度信息评估路面风险程度
- 将路面风险融合到整体风险场中
- 在主画面和鸟瞰图中可视化路面风险

## 性能优化

- **设备自适应**：根据可用硬件自动选择最佳设备（CPU/GPU）
- **模型选择**：提供不同大小的模型以适应不同性能需求
- **批处理**：优化推理流程，提高处理速度
- **内存管理**：有效管理GPU内存使用

## 应用场景

- **高级驾驶辅助系统（ADAS）**：实时监测周围环境，提供碰撞预警
- **自动驾驶**：为自动驾驶系统提供环境感知数据
- **交通监控**：监测交通流量和异常情况
- **智能停车场**：帮助驾驶员识别停车位和障碍物

## 未来发展

- 集成更多传感器数据（如雷达、LiDAR）
- 提高极端天气条件下的性能
- 增加更多场景的适应性
- 开发移动端应用
- 集成到实际车辆系统中

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork仓库
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- YOLOv10团队：提供高效的目标检测模型
- Depth Anything团队：提供先进的深度估计模型
- DeepSort团队：提供可靠的目标跟踪算法
- 所有开源贡献者

---

**驭安DriveSafe** - 为安全驾驶保驾护航


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
