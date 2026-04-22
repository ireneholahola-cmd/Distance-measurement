# 驭安DriveSafe 主动安全预警系统 - 深度教程

## 项目概述

驭安DriveSafe是一个基于计算机视觉和深度学习的主动安全预警系统，专注于实时车辆检测、距离测量、风险评估和可视化。该系统使用先进的3D目标检测和深度估计技术，为驾驶员提供周围环境的实时风险分析，帮助提高驾驶安全性。

## 系统架构

### 核心组件关系

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  目标检测模块     │────>│  深度估计模块     │────>│  3D边界框模块     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  目标跟踪模块     │────>│  轨迹预测模块     │────>│  风险场计算模块   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  路面检测模块     │────>│  风险融合模块     │────>│  可视化模块       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 详细安装指南

### 1. 环境要求

- Python 3.8+ 
- PyTorch 2.2.0+ 
- CUDA 11.7+ (推荐用于GPU加速)
- 至少8GB RAM
- 至少2GB GPU内存 (如果使用GPU)

### 2. 安装步骤

#### 步骤1: 克隆仓库

```bash
git clone <repository-url>
cd Distance-measurement
```

#### 步骤2: 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n drivesafe python=3.9
conda activate drivesafe

# 或者使用venv
python -m venv drivesafe
# Windows
drivesafe\Scripts\activate
# Linux/Mac
source drivesafe/bin/activate
```

#### 步骤3: 安装依赖

根据您的环境选择合适的依赖文件：

##### CPU环境

```bash
pip install -r requirements_common.txt
```

##### GPU环境

```bash
# 首先安装与CUDA版本匹配的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 然后安装其他依赖
pip install -r requirements_gpu.txt
```

#### 步骤4: 下载模型权重

系统默认使用以下模型：
- YOLOv10s：用于目标检测
- Depth Anything v2 Small：用于深度估计
- 路面检测模型：用于检测坑洼和裂缝

模型会在首次运行时自动下载。如果您在无网络环境下运行，需要提前下载并放置到相应目录：

- YOLOv10模型：`yolov10s.pt` (根目录)
- 路面模型：`code/models/` 目录下
  - `best.pt` (白天路面检测)
  - `best_night.pt` (夜间路面检测)
  - `crack_best.pt` (裂缝检测)

## 核心功能模块详解

### 1. 目标检测与跟踪

#### 功能说明
- 使用YOLOv10模型检测车辆、行人和其他障碍物
- 使用DeepSort算法跟踪目标并分配唯一ID
- 过滤有效车辆类型（bicycle, car, motorcycle, bus, truck）

#### 代码位置
- 检测逻辑：`detect_3d.py` 和 `detect_3d_with_surface.py`
- 跟踪逻辑：`deep_sort/deep_sort/` 目录

#### 使用示例

```python
# 检测目标
results = model(image)

# 跟踪目标
detections = [Detection(bbox, confidence, class_id) for bbox, confidence, class_id in results]
tracker.update(detections)

# 遍历跟踪结果
for track in tracker.tracks:
    if not track.is_confirmed() or track.time_since_update > 1:
        continue
    bbox = track.to_tlbr()
    track_id = track.track_id
    # 处理跟踪结果
```

### 2. 深度估计

#### 功能说明
- 使用Depth Anything v2模型估计场景深度
- 支持不同模型大小（small, base, large）以平衡性能和精度
- 自动处理设备兼容性（CPU/GPU/MPS）

#### 代码位置
- 深度估计：`depth_model.py`

#### 使用示例

```python
from depth_model import DepthEstimator

# 初始化深度估计器
depth_estimator = DepthEstimator(model_name="v2_small")

# 估计深度
depth_map = depth_estimator.estimate_depth(image)
```

### 3. 3D边界框估计

#### 功能说明
- 基于深度值和相机参数计算3D位置
- 使用Kalman滤波器平滑3D框参数
- 考虑目标类型和尺寸的先验知识

#### 代码位置
- 3D边界框：`bbox3d_utils.py`

#### 使用示例

```python
from bbox3d_utils import BBox3DEstimator

# 初始化3D边界框估计器
bbox3d_estimator = BBox3DEstimator()

# 计算3D边界框
for detection in detections:
    bbox_3d = bbox3d_estimator.estimate_bbox3d(detection, depth_map)
    # 处理3D边界框
```

### 4. 轨迹预测

#### 功能说明
- 基于卡尔曼滤波的线性/多项式轨迹预测
- 支持时间衰减因子，给近期预测点分配更高权重
- 与风险场集成，提前识别碰撞风险

#### 代码位置
- 运动引擎：`utils/motion_engine.py`
- 轨迹预测：`trajectory_prediction/` 目录

#### 使用示例

```python
from utils.motion_engine import MotionEngine

# 初始化运动引擎
motion_engine = MotionEngine(fps=30)

# 预测轨迹
trajectories = motion_engine.predict_trajectories(tracks, steps=5)

# 计算时间衰减因子
time_decay = motion_engine.calculate_time_decay(steps=5)
```

### 5. 风险场计算

#### 功能说明
- 基于论文公式计算运动物体和静止物体的安全势能
- 考虑车速、路面附着系数等因素
- 生成热力图可视化风险分布
- 集成轨迹预测风险，实现提前预警

#### 代码位置
- 风险场计算：`risk_field.py`

#### 使用示例

```python
from risk_field import RiskFieldEngine

# 初始化风险场引擎
trajectory_prediction/` 目录

#### 使用示例

```python
from risk_field import RiskFieldEngine

# 初始化风险场引擎
risk_engine = RiskFieldEngine()

# 计算轨迹风险
total_risk, trajectory_field = risk_engine.calculate_trajectory_risk(
    ego_pos=(0, 0),
    ego_vel=(0, 0),
    trajectories=list(trajectories.values()),
    velocities_list=velocities_list
)
```

### 6. 路面平整度检测

#### 功能说明
- 使用专门的模型检测路面坑洼和裂缝
- 支持不同场景（白天/夜间）的路面检测
- 基于深度信息评估路面风险程度
- 将路面风险融合到整体风险场中

#### 代码位置
- 路面检测：`road_surface_fusion/` 目录

#### 使用示例

```python
from road_surface_fusion import RoadSurfaceDetector, RoadSurfaceAnalyzer

# 初始化路面检测器和分析器
detector = RoadSurfaceDetector(model_dir="code/models")
analyzer = RoadSurfaceAnalyzer()

# 检测路面
results = detector.detect(image)

# 分析路面风险
analysis = analyzer.analyze(results, depth_map)
```

## 系统使用指南

### 1. Streamlit应用

#### 启动方式

```bash
streamlit run app.py
```

#### 功能说明
- 提供直观的Web界面
- 支持视频文件和摄像头输入
- 实时显示检测结果和风险场
- 可调整参数和查看详细信息

### 2. 命令行工具

#### 基本检测

```bash
python detect_3d.py --source lanechange.mp4 --no-view-img --nosave --device cpu
```

#### 带路面检测的检测

```bash
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu
```

#### 轨迹预测示例

```bash
python trajectory_prediction/example_usage.py
```

### 3. 主UI应用

```bash
python main_ui.py
```

## 高级配置

### 1. 相机参数配置

相机参数文件位于 `config/camera_params.yaml`，包含相机内参和外参。根据实际相机型号和安装位置调整这些参数，以获得更准确的3D位置估计。

### 2. 模型选择

#### YOLOv10模型
- `yolov10s.pt`：轻量级模型，适合实时应用
- `yolov10m.pt`：中等大小模型，平衡速度和精度
- `yolov10l.pt`：大型模型，精度更高但速度较慢

#### Depth Anything模型
- `v2_small`：轻量级模型，适合实时应用
- `v2_base`：中等大小模型
- `v2_large`：大型模型，精度更高

### 3. 风险场参数调整

在 `risk_field.py` 中可以调整以下参数：
- `width_meter`：风险场宽度（米）
- `depth_meter`：风险场深度（米）
- `grid_res`：网格分辨率（米）
- `sigma_x`、`sigma_z`：高斯分布参数，影响风险场的扩散范围

## 代码结构详解

### 根目录文件

| 文件名称 | 功能描述 |
|---------|--------|
| `app.py` | Streamlit应用程序 |
| `detect_3d.py` | 主检测和处理逻辑 |
| `detect_3d_with_surface.py` | 带路面检测的主检测逻辑 |
| `depth_model.py` | 深度估计模型 |
| `bbox3d_utils.py` | 3D边界框估计和可视化 |
| `risk_field.py` | 风险场计算 |
| `main_ui.py` | 主用户界面 |
| `requirements.txt` | 依赖项 |
| `requirements_common.txt` | 通用依赖项 |
| `requirements_gpu.txt` | GPU依赖项 |

### 模块目录

| 目录名称 | 功能描述 |
|---------|--------|
| `road_surface_fusion/` | 路面平整度检测和风险融合 |
| `deep_sort/` | DeepSort目标跟踪 |
| `yolov10/` | YOLOv10模型 |
| `models/` | 模型权重 |
| `utils/` | 工具函数，包含新增的 `motion_engine.py` |
| `trajectory_prediction/` | 轨迹预测模块 |
| `data/` | 测试数据 |
| `code/` | 路面检测模型 |

## 常见问题与解决方案

### 1. 模型下载失败

**问题**：首次运行时模型下载失败。

**解决方案**：
- 检查网络连接
- 手动下载模型并放置到相应目录
- 对于无网络环境，提前准备好模型文件

### 2. GPU内存不足

**问题**：运行时出现GPU内存不足错误。

**解决方案**：
- 使用更小的模型（如YOLOv10s和Depth Anything v2 Small）
- 降低输入图像分辨率
- 使用CPU模式运行：`--device cpu`

### 3. 检测精度问题

**问题**：检测精度不满足要求。

**解决方案**：
- 使用更大的模型（如YOLOv10l和Depth Anything v2 Large）
- 调整置信度阈值
- 确保相机参数正确配置

### 4. 运行速度慢

**问题**：实时处理速度慢。

**解决方案**：
- 使用更小的模型
- 启用GPU加速
- 降低输入图像分辨率
- 调整批处理大小

## 进阶开发指南

### 1. 添加新的目标类别

要添加新的目标类别，需要：
1. 准备包含新类别的数据集
2. 重新训练YOLOv10模型
3. 更新类别名称列表

### 2. 集成新的传感器

要集成新的传感器（如雷达、LiDAR），需要：
1. 实现传感器数据读取接口
2. 开发传感器数据与视觉数据的融合算法
3. 更新风险场计算逻辑

### 3. 自定义风险评估规则

要自定义风险评估规则，需要修改 `risk_field.py` 中的风险计算逻辑，根据具体应用场景调整参数和算法。

### 4. 扩展可视化功能

要扩展可视化功能，可以修改 `road_surface_fusion/visualization.py` 文件，添加新的可视化元素和交互方式。

## 性能基准

### 处理速度

| 硬件配置 | 模型组合 | 处理速度 (FPS) |
|---------|---------|--------------|
| CPU (Intel i7-10700) | YOLOv10s + Depth Anything Small | ~15 FPS |
| GPU (NVIDIA RTX 3080) | YOLOv10s + Depth Anything Small | ~60 FPS |
| GPU (NVIDIA RTX 3080) | YOLOv10m + Depth Anything Base | ~30 FPS |

### 内存使用

| 模型组合 | CPU内存 (GB) | GPU内存 (GB) |
|---------|------------|------------|
| YOLOv10s + Depth Anything Small | ~4 | ~2 |
| YOLOv10m + Depth Anything Base | ~6 | ~4 |
| YOLOv10l + Depth Anything Large | ~8 | ~6 |

## 应用场景示例

### 1. 高级驾驶辅助系统（ADAS）

**配置**：
- 硬件：车载计算机 + 摄像头
- 模型：YOLOv10s + Depth Anything Small
- 处理速度：≥30 FPS

**功能**：
- 实时监测前方车辆和行人
- 计算碰撞风险
- 提供视觉和听觉预警
- 辅助驾驶员避免碰撞

### 2. 交通监控

**配置**：
- 硬件：监控摄像头 + 边缘计算设备
- 模型：YOLOv10m + Depth Anything Base
- 处理速度：≥15 FPS

**功能**：
- 监测交通流量
- 识别异常行为
- 检测交通事故
- 提供实时监控数据

### 3. 智能停车场

**配置**：
- 硬件：停车场摄像头 + 服务器
- 模型：YOLOv10s + Depth Anything Small
- 处理速度：≥20 FPS

**功能**：
- 识别停车位占用情况
- 引导驾驶员找到空闲车位
- 检测停车场内的障碍物
- 提供停车场管理数据

## 未来发展方向

1. **多传感器融合**：集成雷达、LiDAR等传感器数据，提高系统可靠性。

2. **极端天气适应性**：提高在雨、雪、雾等恶劣天气条件下的性能。

3. **场景理解**：增强对复杂交通场景的理解能力，如交叉路口、环岛等。

4. **移动端部署**：开发移动端应用，将系统部署到智能手机和平板电脑。

5. **自动驾驶集成**：将系统集成到自动驾驶系统中，提供环境感知和风险评估功能。

6. **行为预测**：预测其他道路使用者的行为，提前识别潜在风险。

7. **个性化配置**：根据驾驶员的驾驶风格和偏好，调整风险评估和预警策略。

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
