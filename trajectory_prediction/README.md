# 轨迹预测模块

## 功能介绍

本模块实现了基于卡尔曼滤波的轻量目标运动轨迹预测功能，用于预测目标在未来一段时间内的运动轨迹，并将预测结果集成到风险场分析中，以提前识别潜在的碰撞风险。

## 目录结构

```
trajectory_prediction/
├── trajectory_predictor.py  # 轨迹预测核心实现
├── example_usage.py        # 使用示例
└── README.md               # 本说明文件
```

## 实现原理

1. **轨迹预测**：利用DeepSORT内部的卡尔曼滤波器状态，通过线性预测模型预测目标未来的运动轨迹。
2. **风险评估**：基于预测轨迹与自车的距离，计算目标的风险等级。
3. **风险场集成**：将轨迹预测的风险集成到现有的风险场分析中，提前触发预警。
4. **可视化**：在视频中绘制预测轨迹，直观展示目标的运动趋势。

## 核心组件

### 1. MotionEngine (新增)

运动引擎，负责目标轨迹预测和运动分析，位于 `utils/motion_engine.py`。

**主要方法**：
- `predict_trajectory(track, steps=5, method='linear')`：预测单个目标的未来轨迹
- `predict_trajectories(tracks, steps=5, method='linear')`：预测多个目标的未来轨迹
- `calculate_time_decay(steps=5)`：计算时间衰减因子
- `get_motion_info(track)`：获取目标的运动信息

### 2. TrajectoryPredictor

轨迹预测器，负责预测目标的未来运动轨迹。

**主要方法**：
- `predict_trajectories(tracks, steps=10)`：预测多个目标的未来轨迹
- `convert_2d_to_3d(points_2d, depth=10)`：将2D像素坐标转换为3D物理坐标
- `calculate_risk(trajectories, ego_position=(0, 0, 0), time_horizon=5)`：计算轨迹的风险

### 3. RiskFieldIntegrator

风险场集成器，将轨迹预测结果集成到风险场中。

**主要方法**：
- `integrate(tracks, risk_field, ego_position=(0, 0, 0))`：将轨迹预测集成到风险场中

### 4. TrajectoryVisualizer

轨迹可视化器，用于绘制预测轨迹。

**主要方法**：
- `draw_trajectories(image, trajectories, color=(0, 255, 0), thickness=2)`：在图像上绘制预测轨迹

## 集成步骤

### 1. 修改DeepSORT的Track类

已在 `deep_sort/deep_sort/sort/track.py` 文件中添加了 `predict_future_trajectory` 方法，用于预测目标的未来轨迹。

### 2. 在主程序中集成轨迹预测

在检测与跟踪主程序（如 `detect_3d.py` 或 `risk_fusion.py`）中，添加以下代码：

```python
from utils.motion_engine import MotionEngine
from road_surface_fusion.visualization import RoadSurfaceVisualizer

# 初始化运动引擎和可视化器
motion_engine = MotionEngine(fps=30)  # 根据实际帧率设置
visualizer = RoadSurfaceVisualizer()

# 在主循环中
while True:
    # 执行检测和跟踪
    # ...
    
    # 预测轨迹
    trajectories = motion_engine.predict_trajectories(tracks, steps=5)
    
    # 计算时间衰减因子
    time_decay = motion_engine.calculate_time_decay(steps=5)
    
    # 计算风险
    # 这里可以使用risk_field.py中的新方法
    # ...
    
    # 绘制轨迹包络带
    frame = visualizer.draw_trajectories(frame, trajectories, risk_scores)
    
    # 显示结果
    # ...
```

### 3. 与风险场集成

在 `risk_fusion.py` 中，将轨迹预测的风险集成到风险场分析中：

```python
# 集成轨迹预测风险
from utils.motion_engine import MotionEngine
from risk_field import RiskFieldEngine

# 初始化
motion_engine = MotionEngine(fps=30)
risk_engine = RiskFieldEngine()

# 在风险场计算中
risk_field = calculate_risk_field(detections, tracks)

# 预测轨迹
 trajectories = motion_engine.predict_trajectories(tracks, steps=5)

# 计算速度列表
velocities_list = []
for track in tracks:
    if track.is_confirmed():
        vx, vy = track.mean[4], track.mean[5]
        # 复制速度到每个预测点
        velocities = [(vx, vy) for _ in range(5)]
        velocities_list.append(velocities)

# 计算轨迹风险
total_risk, trajectory_field = risk_engine.calculate_trajectory_risk(
    ego_pos=(0, 0),
    ego_vel=(0, 0),  # 根据实际情况设置
    trajectories=list(trajectories.values()),
    velocities_list=velocities_list
)

# 集成风险
integrated_risk = risk_field + trajectory_field
```

### 4. 可视化升级

使用新增的 `draw_trajectory_envelope` 方法绘制轨迹包络带：

```python
from road_surface_fusion.visualization import RoadSurfaceVisualizer

visualizer = RoadSurfaceVisualizer()

# 绘制轨迹包络带
frame = visualizer.draw_trajectories(frame, trajectories, risk_scores)
```

## 依赖项

本模块依赖以下库：
- numpy
- cv2 (OpenCV)

这些依赖项已经在项目的 `requirements_common.txt` 中声明。

## 使用示例

运行示例脚本 `example_usage.py` 可以看到轨迹预测的效果：

```bash
python trajectory_prediction/example_usage.py
```

## 注意事项

1. 轨迹预测基于线性模型，适用于目标做匀速直线运动的场景。
2. 对于复杂运动的目标，预测精度可能会降低。
3. 预测步数不宜过长，建议在5-10步之间，以保证预测的准确性。
4. 风险评估基于预测轨迹与自车的距离，可根据实际需求调整风险阈值。
5. 时间衰减因子用于给近期预测点分配更高的权重，远期预测点分配更低的权重。

## 未来改进

1. 实现多项式预测模型，提高对非线性运动的预测精度。
2. 结合场景信息（如道路结构、交通规则）优化预测结果。
3. 添加目标行为预测，识别潜在的危险行为（如突然变道、急刹车等）。
4. 实现针对特定场景的意图修正逻辑，如变道预测修正。
