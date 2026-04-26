# 驭安DriveSafe 主动安全预警系统 — 技术方案详细实现报告

## 一、项目概述

### 1.1 项目定位

驭安DriveSafe是一套基于YOLOv10目标检测、Depth Anything v2深度估计、DeepSORT多目标跟踪以及驾驶风险场理论的主动安全预警系统。系统通过单目摄像头实时感知前方交通场景，融合3D空间信息与物理风险场模型，为驾驶员提供直观的风险可视化和分级预警。

### 1.2 核心能力

| 能力 | 描述 |
|------|------|
| 实时目标检测 | 基于YOLOv10，检测车辆、行人、自行车等交通参与者 |
| 3D空间感知 | 基于Depth Anything v2深度估计 + 相机内参反投影，获取目标3D位置 |
| 多目标跟踪 | 基于DeepSORT + 3D欧氏距离匹配，实现稳定ID跟踪 |
| 风险场计算 | 基于驾驶风险场理论，计算SCF（Surrogate Conflict Field） |
| 路面风险融合 | 基于YOLOv10路面检测，识别坑洼/裂缝等路面隐患并融合到风险场 |
| BEV鸟瞰图可视化 | 实时生成雷达风格的风险场热力图 |
| Web UI | 基于Streamlit的交互式操作界面 |

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      驭安DriveSafe 系统架构                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 视频输入  │───▶│  YOLOv10     │───▶│  DeepSORT            │  │
│  │ 摄像头/   │    │  目标检测     │    │  多目标跟踪           │  │
│  │ 视频文件  │    │  (2D BBox)   │    │  (ID + 3D Pose)      │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│       │                                          │              │
│       │              ┌──────────────┐            │              │
│       ├─────────────▶│ Depth Anything│            │              │
│       │              │ v2 深度估计   │────────────┤              │
│       │              └──────────────┘            │              │
│       │                                          ▼              │
│       │              ┌──────────────┐    ┌──────────────┐      │
│       ├─────────────▶│ 路面检测模块  │───▶│ 3D BBox      │      │
│       │              │ (YOLOv10)    │    │ 估算 + 卡尔曼 │      │
│       │              └──────────────┘    │ 滤波平滑     │      │
│       │                                  └──────┬───────┘      │
│       │                                         │              │
│       │                                         ▼              │
│       │                                  ┌──────────────┐      │
│       │                                  │ 风险场引擎    │      │
│       │                                  │ (SCF计算)     │      │
│       │                                  └──────┬───────┘      │
│       │                                         │              │
│       │              ┌──────────────┐            │              │
│       │              │ 路面风险融合  │◀───────────┤              │
│       │              │ (Risk Fusion)│            │              │
│       │              └──────┬───────┘            │              │
│       │                     │                    │              │
│       ▼                     ▼                    ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    可视化 & 预警输出层                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │  │
│  │  │ 3D框绘制  │  │ BEV鸟瞰图 │  │ 风险热力图│  │ 警报推送 │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                         │
│                    │ Streamlit Web UI │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
app.py (Streamlit UI)
  └── detect_3d.py (核心检测流程)
        ├── models/yolov10/ (YOLOv10模型)
        ├── deep_sort/ (DeepSORT跟踪)
        ├── depth_model.py (Depth Anything v2)
        ├── bbox3d_utils.py (3D框估算 + BEV可视化)
        ├── risk_field.py (风险场引擎)
        ├── road_surface_fusion/ (路面风险融合)
        │     ├── detector.py
        │     ├── surface_analysis.py
        │     ├── risk_fusion.py
        │     └── visualization.py
        └── trajectory_prediction/ (轨迹预测)
              └── trajectory_predictor.py
```

---

## 三、软件设计实现

### 3.1 目标检测模块

**技术选型：YOLOv10**

YOLOv10是YOLO系列的最新版本，相比前代具有以下优势：
- 无NMS（Non-Maximum Suppression）设计，推理速度更快
- 更小的模型尺寸，适合边缘部署
- 更高的检测精度

**实现细节：**

```python
# 模型加载与推理核心流程
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)

# 推理
pred = model(img, augment=opt.augment)[0]

# NMS后处理
pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                           classes=opt.classes, agnostic=opt.agnostic_nms)
```

**检测类别过滤：** 系统仅关注交通相关目标，过滤有效类别：

```python
valid_classes = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
valid_outputs = [out for out in outputs if int(out[5]) in valid_classes]
```

### 3.2 深度估计模块

**技术选型：Depth Anything V2**

Depth Anything V2是Meta开源的单目深度估计模型，基于Vision Transformer架构，具有强大的零样本泛化能力。

**实现架构：**

```python
class DepthEstimator:
    def __init__(self, model_size='small', device=None):
        # 模型尺寸映射
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base':  'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        # 使用HuggingFace pipeline加载
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)

    def estimate_depth(self, image):
        # BGR → RGB → PIL → 深度图
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        depth_result = self.pipe(pil_image)
        depth_map = depth_result["depth"]
        # 归一化到 [0, 1]
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        return depth_map
```

**关键设计决策：**
- 选择Small模型：在精度和速度之间取得平衡，CPU环境下约100ms/帧
- 自动设备回退：MPS不兼容时自动降级到CPU
- 区域深度提取：使用中心50%区域的中值深度，避免边缘噪声

```python
def get_depth_in_region(self, depth_map, bbox, method='median', scale=0.5):
    # 缩放到中心50%区域，取中值深度
    new_width = width * scale
    new_height = height * scale
    # ...边界裁剪...
    region = depth_map[y1:y2, x1:x2]
    return float(np.median(region))
```

### 3.3 3D边界框估算模块

**技术方案：2D检测 + 深度反投影 + 先验尺寸约束**

**核心流程：**

1. **2D框中心反投影**：利用相机内参矩阵K，将2D像素坐标反投影到3D空间

```python
def _backproject_point(self, x, y, depth):
    point_2d = np.array([x, y, 1.0])
    point_3d = np.linalg.inv(self.K) @ point_2d * depth
    return point_3d
```

2. **先验尺寸约束**：使用预定义的各类别平均尺寸

```python
DEFAULT_DIMS = {
    'car': np.array([1.52, 1.64, 3.85]),      # H, W, L (m)
    'truck': np.array([3.07, 2.63, 11.17]),
    'bus': np.array([3.07, 2.63, 11.17]),
    'motorcycle': np.array([1.50, 0.90, 2.20]),
    'person': np.array([1.75, 0.60, 0.60]),
}
```

3. **朝向估计**：基于2D框宽高比和位置关系推断朝向

```python
def _estimate_orientation(self, bbox_2d, location, class_name):
    theta_ray = np.arctan2(location[0], location[2])
    aspect_ratio = width / height
    if aspect_ratio > 1.5:
        alpha = np.pi / 2 if center < image_center else -np.pi / 2
    else:
        alpha = 0.0
    rot_y = alpha + theta_ray
    return rot_y
```

4. **卡尔曼滤波平滑**：11维状态向量 [x, y, z, w, h, l, yaw, vx, vy, vz, vyaw]

```python
kf = KalmanFilter(dim_x=11, dim_z=7)
# Z轴噪声增大（深度估计噪声大）
kf.R[2, 2] = 2.0
kf.R[6, 6] = 0.3  # 朝向不确定性
```

5. **时间滤波**：指数移动平均（EMA）进一步平滑

```python
alpha = 0.7  # 当前测量权重
filtered_box['location'] = current * (1-weight) + history[i] * weight
```

### 3.4 多目标跟踪模块

**技术选型：DeepSORT + 3D欧氏距离匹配**

DeepSORT在SORT基础上增加了外观特征（ReID）匹配，显著减少了ID Switch问题。

**增强设计：** 在标准DeepSORT基础上，引入3D位姿信息进行匹配：

```python
# 传入3D数据以启用3D欧氏距离匹配
outputs = deepsort.update(xywhs, confss, im0, classes, bbox_3d=bbox_3d_list)
```

**速度估算：** 基于帧间位移计算

```python
def Estimated_speed(locations, fps, width):
    speed = math.sqrt((dx**2 + dy**2)) * width / fps * 3.6 * 2
```

### 3.5 风险场引擎

**理论基础：驾驶风险场理论**

驾驶风险场理论将交通环境中的风险抽象为物理场，每个交通参与者都是一个风险源，其风险势能与速度、质量、距离相关。

**核心公式实现：**

1. **虚拟质量计算**（论文公式13）：

```python
def calc_M_b(vehicle_type, m, v):
    psi_T = 1.000 if vehicle_type == 'car' else 1.443  # 载货车系数
    g_v = 0.002368 * (v ** 4) - 0.3224 * (v ** 3) + 12.57 * (v ** 2) - 1492
    return psi_T * m * g_v
```

2. **道路条件影响因子**（论文公式14）：

```python
def calc_R_b(psi_delta, psi_rho_tau, mu):
    phi_mu = -87290 * (mu ** 3) + 151900 * (mu ** 2) - 84630 * mu + 21780
    return psi_delta * psi_rho_tau * phi_mu
```

3. **运动物体安全势能**（论文附录A公式A10）：

```python
def SPE_v_bi(x_phi, y_phi, x_obs, y_obs, v_b, phi_v, K, k1, k3, R_b, M_b, R_i, M_i, DR_i):
    r_bi = np.sqrt((x_phi - x_obs)**2 + (y_phi - y_obs)**2)
    cos_theta_b = np.dot(vec_r, vec_v) / r_bi
    numerator = K * R_b * M_b * R_i * M_i * (1 + DR_i) * k3
    denominator = (k1 - 1) * (r_bi ** (k1 - 1))
    return (numerator / denominator) * power_term
```

4. **SCF（Surrogate Conflict Field）计算**：

```python
def calculate_scf(self, ego_pos, target_pos, v_ego, v_target, weather_factor=1.0):
    ego_field = self.get_gaussian_field(ego_pos[0], ego_pos[1], v_ego[0], v_ego[1], ...)
    target_field = self.get_gaussian_field(target_pos[0], target_pos[1], v_target[0], v_target[1], ...)
    overlap_field = ego_field * target_field
    scf_value = np.sum(overlap_field)
    return scf_value, ego_field, target_field, overlap_field
```

**高斯风险场模型：** 使用旋转二维高斯分布模拟车辆风险场

```python
def get_gaussian_field(self, x_c, z_c, v_x=0, v_z=0, sigma_x=0.8, sigma_z=2.0, ...):
    # 速度拉伸：纵向sigma随速度增大
    eff_sigma_z = sigma_z + v_mag * v_stretch_factor
    # 旋转：根据速度方向旋转协方差矩阵
    angle = np.arctan2(v_x, v_z)
    Sigma_global = R @ Sigma_local @ R.T
    # 马氏距离计算
    mahal_sq = np.sum(term1 * points, axis=1)
    grid_vals = np.exp(-0.5 * mahal_sq)
    return grid_vals.reshape(self.grid_h, self.grid_w)
```

### 3.6 路面风险融合模块

**架构设计：**

```
road_surface_fusion/
├── detector.py          # YOLOv10路面缺陷检测
├── surface_analysis.py  # 路面分析（几何/严重度评估）
├── risk_fusion.py       # 风险融合（动态+路面）
├── visualization.py     # 可视化（主画面+BEV叠加）
└── structured_output.py # 结构化输出（JSONL）
```

**融合策略：** 取动态风险和路面风险的最大值

```python
def fuse_risk(self, dynamic_risk: float, surface_risk: float) -> float:
    return max(dynamic_risk, surface_risk)
```

**路面风险场生成：** 为每个路面隐患生成高斯风险场

```python
for hazard in analysis.hazards:
    type_weight = 1.25 if hazard.hazard_type == "pothole" else 0.9
    sigma_x = 0.5 + min(1.2, hazard.area_ratio * 60.0 + 0.2)
    sigma_z = 1.2 + hazard.severity * 2.5
    field = risk_engine.get_gaussian_field(hazard.x_m, hazard.z_m, ...)
```

### 3.7 BEV鸟瞰图可视化

**BirdEyeView类设计：**

- 画布尺寸：400×625像素
- 基础比例：25 px/m
- 动态缩放：根据最远目标自动调整视野范围

```python
class BirdEyeView:
    def __init__(self, size=(400, 625), scale=25, camera_height=1.5):
        self.origin_x = self.width // 2
        self.origin_y = self.height - 50

    def update_scale(self, max_distance_m):
        available_height = self.height - 100
        new_scale = available_height / max_distance_m
        self.scale = self.scale * 0.9 + new_scale * 0.1  # 平滑过渡
```

**可视化层次：**

1. 网格背景（5m间距）
2. 轨迹渐隐线
3. 未来预测扇形
4. 车辆矩形标记
5. 风险热力图叠加（JET色图 + 高斯模糊发光 + 等高线）
6. HUD信息（速度、状态、SCF值）

---

## 四、用户界面设计

### 4.1 技术选型：Streamlit

选择Streamlit作为UI框架的原因：
- Python原生，与检测代码无缝集成
- 实时图像显示能力强
- 部署简单，支持本地和云端
- 交互组件丰富（滑块、按钮、文件上传等）

### 4.2 界面布局

```
┌──────────────────────────────────────────────────────────────┐
│                  🚗 驭安DriveSafe 主动安全预警系统              │
│              基于YOLOv10和深度学习的安全驾驶辅助系统              │
├───────────────────────────────┬──────────────────────────────┤
│                               │                              │
│  📹 实时视频流                 │  ⚠️ 风险场视图               │
│  [视频检测画面]                │  [BEV鸟瞰热力图]             │
│                               │                              │
│  📊 实时统计                   │  🚨 实时警报                 │
│  [检测目标] [风险等级] [高危]   │  [分级警报列表]              │
│                               │                              │
├───────────────────────────────┴──────────────────────────────┤
│  ⚙️ 侧边栏设置                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 📹 视频源选择  │ 🎯 检测参数  │ 🖥️ 运行设置  │ 🔊 预警  │  │
│  └────────────────────────────────────────────────────────┘  │
│  [🚀 开始识别]  [⏹️ 停止识别]                                 │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 CSS样式系统

**亮色主题设计：**

- 主色调：天蓝色 `#0099cc`
- 背景：半透明白色 `rgba(255, 255, 255, 0.9)` + 毛玻璃效果
- 卡片：白色半透明 + 天蓝色边框
- 侧边栏：浅灰蓝 `rgba(245, 247, 250, 0.95)`

**动态效果：**

- 卡片悬浮上移 + 阴影增强
- 警报滑入动画
- 状态指示灯脉冲动画
- 渐变分隔线

### 4.4 回调机制

系统通过回调函数实现检测流程与UI的解耦：

```python
def callback(im0, risk_img, frame_idx=None, total_frames=None, detections=None):
    # 视频帧显示
    video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)
    # 风险场显示
    risk_image_placeholder.image(risk_rgb, channels="RGB", use_container_width=True)
    # 统计数据更新
    st.session_state.stats['total_detections'] = len(detections)
    # 风险分级
    if risk_score > risk_threshold_high:
        st.session_state.risk_alerts.append({...})
    # 进度更新
    progress_bar.progress(min(progress, 1.0))
```

---

## 五、数据来源

### 5.1 目标检测数据

**预训练模型：** YOLOv10s（COCO预训练）

COCO数据集包含80个类别，其中交通相关类别：
- person (1)
- bicycle (2)
- car (3)
- motorcycle (4)
- bus (6)
- truck (8)

**自定义训练数据：** 项目提供了自定义训练的模型

```
code/models/
├── best.pt          # 自定义训练的最佳模型
├── best_night.pt    # 夜间场景微调模型
└── crack_best.pt    # 路面裂缝检测模型
```

### 5.2 深度估计数据

**预训练模型：** Depth Anything V2 Small

训练数据涵盖多种场景（室内、室外、驾驶场景），具有强大的零样本泛化能力，无需针对特定场景微调。

### 5.3 多目标跟踪数据

**ReID模型：** `mars-small128.pb`

基于MARS（Multiple Attribute Re-identification Dataset）训练的人员/车辆重识别模型，用于DeepSORT的外观特征提取。

### 5.4 测试视频数据

项目内置多种测试场景：

```
lanechange.mp4      # 变道场景
cutin.txt           # 加塞场景
intersection1.txt   # 交叉路口场景1
intersection2.txt   # 交叉路口场景2
carfellow.txt       # 跟车场景
```

---

## 六、模型训练

### 6.1 YOLOv10训练

**训练框架：** Ultralytics

```python
# 训练命令示例
yolo train model=yolov10s.pt data=custom.yaml epochs=100 imgsz=640 batch=16
```

**自定义数据集构建：**

1. 数据标注：使用LabelImg/CVAT标注2D边界框
2. 数据格式：YOLO格式（txt标注文件）
3. 数据增强：Mosaic、MixUp、HSV变换、随机裁剪
4. 类别定义：在data yaml中配置

**夜间场景微调：**

针对夜间低光照场景，收集夜间驾驶数据并微调模型：

```
code/models/best_night.pt  # 夜间场景微调权重
```

### 6.2 路面缺陷检测训练

**独立模型：** `crack_best.pt`

```python
# 路面检测模型配置
road_detector = RoadSurfaceDetector(
    model_dir=opt.road_model_dir,  # code/models/
    preferred_device=device.type
)
```

### 6.3 DeepSORT ReID训练

**训练数据：** MARS数据集

```python
# deep_sort/deep/train.py
# 使用Triplet Loss训练ReID特征提取网络
```

---

## 七、改进过程

### 7.1 版本演进

| 版本 | 改进内容 |
|------|---------|
| V1.0 | 基础YOLOv5检测 + 2D距离估算 |
| V2.0 | 引入DeepSORT多目标跟踪 + 速度估算 |
| V3.0 | 引入Depth Anything深度估计 + 3D框估算 |
| V4.0 | 实现驾驶风险场理论 + BEV可视化 |
| V5.0 | 路面风险融合 + 结构化输出 |
| V6.0 | Streamlit Web UI + 亮色主题 |

### 7.2 关键改进点

#### 7.2.1 从2D到3D的距离估算

**问题：** 早期版本仅基于2D边界框大小估算距离，精度低且不稳定。

**解决方案：** 引入Depth Anything V2深度估计模型，结合相机内参反投影获取3D位置。

**效果：** 距离估算误差从30%+降低到15%以内。

#### 7.2.2 风险场可视化优化

**问题：** 早期BEV图使用matplotlib实时渲染，帧率极低（~2 FPS）。

**解决方案：** 改用OpenCV直接绘制BEV图，避免matplotlib的figure创建/销毁开销。

**效果：** BEV渲染帧率从2 FPS提升到30+ FPS。

#### 7.2.3 3D框平滑

**问题：** 深度估计逐帧波动导致3D框抖动严重。

**解决方案：** 双重滤波——卡尔曼滤波 + 指数移动平均。

```python
# 卡尔曼滤波：状态预测 + 测量更新
kf.predict()
kf.update(measurement)

# EMA时间滤波：平滑历史轨迹
alpha = 0.7
filtered = current * alpha + history * (1 - alpha)
```

**效果：** 3D框位置抖动减少80%以上。

#### 7.2.4 BEV动态缩放

**问题：** 固定比例BEV图无法同时展示近距离和远距离目标。

**解决方案：** 根据最远目标距离动态调整缩放比例，平滑过渡。

```python
def update_scale(self, max_distance_m):
    new_scale = available_height / max_distance_m
    self.scale = self.scale * 0.9 + new_scale * 0.1  # 平滑过渡
```

#### 7.2.5 UI从暗色到亮色

**问题：** 暗色UI在日间使用时不协调。

**解决方案：** 全面调整CSS配色方案，从深色背景切换到浅色背景。

| 元素 | 暗色 | 亮色 |
|------|------|------|
| 主背景 | `rgba(10,15,25,0.85)` | `rgba(255,255,255,0.9)` |
| 主色调 | `#00d4ff` | `#0099cc` |
| 文字 | `#e0e0e0` | `#333333` |
| 卡片 | `rgba(255,255,255,0.05)` | `rgba(255,255,255,0.9)` |

---

## 八、遇到的困难与解决方法

### 8.1 Python字符串格式化与CSS冲突

**问题：** CSS中的`%`符号与Python的`%`格式化冲突，导致`TypeError: not enough arguments for format string`。

**解决方案：** 将CSS中所有`%`转义为`%%`。

```python
# 错误写法
st.markdown('...width: 50%;...' % bg_base64)

# 正确写法
st.markdown('...width: 50%%;...' % bg_base64)
```

### 8.2 f-string与CSS大括号冲突

**问题：** CSS的`{}`被Python的f-string解析为表达式分隔符，导致`SyntaxError: invalid decimal literal`。

**解决方案：** 将f-string改为普通字符串，使用`%`格式化。

```python
# 错误写法
st.markdown(f'...{bg_base64}...')

# 正确写法
st.markdown('...%s...' % bg_base64)
```

### 8.3 Depth Anything显存不足

**问题：** Depth Anything V2与YOLOv10同时加载到GPU时，显存不足（需6GB+）。

**解决方案：** 自动降级策略——Depth Anything优先使用CPU，仅在显存充足时使用GPU。

```python
if self.device == 'mps':
    self.pipe_device = 'cpu'  # MPS兼容性问题，强制CPU
```

### 8.4 Matplotlib版本兼容性

**问题：** 不同版本的Matplotlib获取图像数据的方式不同。

**解决方案：** 兼容处理。

```python
if hasattr(fig.canvas, 'buffer_rgba'):
    buf = fig.canvas.buffer_rgba()  # Matplotlib 3.8+
else:
    buf = fig.canvas.tostring_rgb()  # 旧版
```

### 8.5 Streamlit实时视频流卡顿

**问题：** Streamlit的`st.image()`每次调用都会重新渲染整个组件，导致视频流卡顿。

**解决方案：** 使用`st.empty()`占位符 + 回调模式，减少不必要的重渲染。

```python
video_placeholder = st.empty()
# 在回调中更新
video_placeholder.image(im0_rgb, channels="RGB", use_container_width=True)
```

### 8.6 统计数据显示随机数

**问题：** 实时统计区域出现3-6的随机数。

**原因：** 当没有检测到目标时，统计数据未被重置，残留了上一帧的值。

**解决方案：** 在回调函数中添加else分支，当detections为None时重置所有统计数据。

```python
if detections is not None:
    st.session_state.stats['total_detections'] = len(detections)
else:
    st.session_state.stats['total_detections'] = 0
    st.session_state.stats['high_risk_count'] = 0
    st.session_state.stats['current_risk'] = 0.0
```

### 8.7 相机内参缺失

**问题：** 大多数测试视频没有标定数据，无法精确反投影。

**解决方案：** 自动估算相机内参。

```python
if cam_params is None:
    est_f = vid_w * 1.0   # 假设标准镜头，焦距≈图像宽度
    est_cx = vid_w / 2     # 主点在图像中心
    est_cy = vid_h / 2
    K = np.array([[est_f, 0, est_cx], [0, est_f, est_cy], [0, 0, 1]])
```

### 8.8 风险场归一化不稳定

**问题：** 风险场值域跨度大（0~10^6），直接归一化导致大部分区域接近0。

**解决方案：** 分段归一化——低值区间线性映射到0~0.3，高值区间映射到0.3~1.0。

```python
p30 = np.percentile(U_total_clipped, 30)
U_norm[mask_low] = (U_total_clipped[mask_low] / p30) * 0.3
U_norm[mask_high] = ((U_total_clipped[mask_high] - p30) / denom) * 0.7 + 0.3
```

---

## 九、系统部署

### 9.1 环境要求

**硬件要求：**

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | Intel i5 8代+ | Intel i7 12代+ |
| 内存 | 8 GB | 16 GB+ |
| GPU | 无（CPU推理） | NVIDIA GTX 1660+ (6GB VRAM) |
| 存储 | 5 GB | 10 GB+ |

**软件要求：**

```
Python 3.8+
PyTorch 2.2+
CUDA 11.8+ (GPU推理)
```

### 9.2 安装步骤

```bash
# 1. 克隆项目
git clone <repository_url>
cd Distance-measurement

# 2. 创建虚拟环境
conda create -n drivesafe python=3.10
conda activate drivesafe

# 3. 安装依赖（CPU版本）
pip install -r requirements.txt

# 或GPU版本
pip install -r requirements_gpu.txt

# 4. 下载模型权重
# YOLOv10s权重已包含在项目根目录：yolov10s.pt
# Depth Anything V2首次运行时自动从HuggingFace下载

# 5. 启动Web UI
streamlit run app.py
```

### 9.3 Docker部署

项目提供了Dockerfile：

```dockerfile
# 基于项目根目录的 _backup/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# 构建镜像
docker build -t drivesafe .

# 运行容器
docker run -p 8501:8501 drivesafe
```

### 9.4 命令行运行

```bash
# 基础运行（视频文件）
python detect_3d.py --source lanechange.mp4 --weights yolov10s.pt

# 摄像头实时检测
python detect_3d.py --source 0 --weights yolov10s.pt --device cuda

# 带路面检测的完整版
python detect_3d_with_surface.py --source lanechange.mp4 --save-jsonl

# 自定义参数
python detect_3d.py \
    --source input.mp4 \
    --weights yolov10s.pt \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --img-size 640 \
    --device cuda
```

### 9.5 性能优化建议

1. **GPU加速**：将`--device cuda`可提升3-5倍推理速度
2. **模型选择**：YOLOv10s（速度优先）vs YOLOv10x（精度优先）
3. **深度估计**：Small模型约100ms/帧，Base模型约200ms/帧
4. **隔帧深度估计**：显存紧张时可隔帧运行深度估计
5. **图像尺寸**：640（默认）vs 1280（高精度），影响推理速度约2倍

---

## 十、项目文件结构

```
Distance-measurement/
├── app.py                          # Streamlit Web UI主入口
├── detect_3d.py                    # 核心检测流程（基础版）
├── detect_3d_with_surface.py       # 核心检测流程（路面融合版）
├── depth_model.py                  # Depth Anything V2深度估计
├── bbox3d_utils.py                 # 3D框估算 + BEV可视化
├── risk_field.py                   # 风险场引擎
├── main_ui.py                      # 备用UI入口
├── yolov10s.pt                     # YOLOv10s预训练权重
├── requirements.txt                # CPU依赖
├── requirements_gpu.txt            # GPU依赖
├── requirements_common.txt         # 共享依赖
│
├── models/                         # YOLOv5/v10模型定义
│   ├── yolov10/                    # YOLOv10模块
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── val.py
│   ├── common.py
│   ├── yolo.py
│   └── experimental.py
│
├── deep_sort/                      # DeepSORT跟踪模块
│   ├── deep_sort.py
│   ├── deep/                       # ReID特征提取
│   │   ├── model.py
│   │   └── train.py
│   ├── sort/                       # SORT跟踪核心
│   │   ├── tracker.py
│   │   ├── track.py
│   │   └── kalman_filter.py
│   └── configs/
│       └── deep_sort.yaml
│
├── road_surface_fusion/            # 路面风险融合模块
│   ├── detector.py
│   ├── surface_analysis.py
│   ├── risk_fusion.py
│   ├── visualization.py
│   └── structured_output.py
│
├── trajectory_prediction/          # 轨迹预测模块
│   └── trajectory_predictor.py
│
├── utils/                          # 工具函数
│   ├── datasets.py
│   ├── general.py
│   ├── distance.py
│   ├── motion_engine.py
│   └── plots.py
│
├── counter/                        # 计数模块
│   ├── counter_frame.py
│   └── draw_counter.py
│
├── code/models/                    # 自定义训练模型
│   ├── best.pt
│   ├── best_night.pt
│   └── crack_best.pt
│
├── picture/                        # UI素材
│   ├── logo.png
│   ├── background1.jpg
│   └── background2.jpg
│
└── plan/                           # 项目文档
    └── technical_implementation.md
```

---

## 十一、总结与展望

### 11.1 技术成果

1. 实现了从2D检测到3D空间感知的完整流水线
2. 基于驾驶风险场理论的风险量化模型
3. 雷达风格的BEV实时可视化
4. 路面风险与动态风险的融合预警
5. 交互式Web UI操作界面

### 11.2 已知局限

1. **深度估计精度**：单目深度估计为相对深度，绝对距离精度有限
2. **速度估算**：基于帧间位移的速度估算精度依赖帧率和标定质量
3. **声音预警**：UI中预留了声音预警选项但尚未实现
4. **实时性能**：CPU模式下帧率约5-10 FPS，需GPU加速达到实时
5. **夜间场景**：低光照条件下检测和深度估计精度下降

### 11.3 未来改进方向

1. **多传感器融合**：引入激光雷达/毫米波雷达提升3D感知精度
2. **声音预警实现**：使用winsound/pygame实现分级声音报警
3. **语音播报**：集成TTS引擎实现语音预警
4. **边缘部署优化**：TensorRT/ONNX加速，支持Jetson等嵌入式平台
5. **OTA更新**：模型热更新机制
6. **V2X通信**：车路协同扩展
