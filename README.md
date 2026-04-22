# 驭安DriveSafe 主动安全预警系统

## 项目概述

驭安DriveSafe是一个基于计算机视觉和深度学习的主动安全预警系统，专注于实时车辆检测、距离测量、风险评估和可视化。该系统使用先进的3D目标检测和深度估计技术，为驾驶员提供周围环境的实时风险分析，帮助提高驾驶安全性。

## 核心功能

- **实时目标检测**：使用YOLOv10模型检测车辆、行人和其他障碍物
- **3D边界框估计**：基于深度信息和相机参数计算目标的3D位置和尺寸
- **深度估计**：使用Depth Anything v2模型估计场景深度
- **目标跟踪**：使用DeepSort算法跟踪目标并分配唯一ID
- **轨迹预测**：基于卡尔曼滤波的线性/多项式轨迹预测
- **风险场计算**：基于论文公式计算车辆周围的风险场
- **路面平整度检测**：检测路面坑洼和裂缝，评估路面风险
- **鸟瞰图可视化**：提供场景的鸟瞰视图，显示目标位置和风险分布
- **速度估计**：基于目标运动计算速度
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
├── main_ui.py                # 主用户界面
├── road_surface_fusion/      # 路面平整度检测和风险融合
├── deep_sort/                # DeepSort目标跟踪
├── yolov10/                  # YOLOv10模型
├── models/                   # 模型权重
├── utils/                    # 工具函数，包含motion_engine.py
├── trajectory_prediction/    # 轨迹预测模块
├── data/                     # 测试数据
├── code/                     # 路面检测模型
├── requirements.txt          # 依赖项
├── requirements_common.txt   # 通用依赖项
├── requirements_gpu.txt      # GPU依赖项
├── README.md                 # 项目说明（本文档）
└── README_DETAILED.md        # 深度教程
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
# 首先安装与CUDA版本匹配的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 然后安装其他依赖
pip install -r requirements_gpu.txt
```

### 3. 下载模型权重

系统默认使用以下模型：
- YOLOv10s：用于目标检测
- Depth Anything v2 Small：用于深度估计
- 路面检测模型：用于检测坑洼和裂缝

模型会在首次运行时自动下载。

## 使用方法

### 1. 启动Streamlit应用

```bash
streamlit run app.py
```

### 2. 运行带路面检测的检测

```bash
python detect_3d_with_surface.py --source lanechange.mp4 --no-view-img --nosave --device cpu
```

### 3. 运行轨迹预测示例

```bash
python trajectory_prediction/example_usage.py
```

### 4. 运行主UI

```bash
python main_ui.py
```

## 深度教程

如需更详细的项目信息、安装指南、代码结构分析和高级使用方法，请参考 [README_DETAILED.md](README_DETAILED.md) 文件。

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
