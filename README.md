# Distance-measurement 路面风险融合版
## 目录
- 第 15 行：1. 项目说明
- 第 23 行：2. 新增内容
- 第 51 行：3. 运行入口
- 第 87 行：4. 命令行接口
- 第 109 行：4.4 UI 同事关注
- 第 160 行：5. 新增模块接口
- 第 208 行：6. 主流程逻辑
- 第 226 行：7. 风险计算逻辑
- 第 255 行：8. 输出效果
- 第 274 行：9. 已验证内容
- 第 285 行：10. 注意事项

## 1. 项目说明
本版本在不直接修改 `code` 目录原始实现、尽量不改原 `Distance-measurement` 接口的前提下，新增了一套“动态目标风险 + 路面风险”融合入口。

运行 `detect_3d_with_surface.py` 时，系统除了保留原有车辆、行人检测和 3D 风险视图，还会：
- 在主画面绘制坑洼、裂缝检测结果
- 在 BEV 风险图中加入路面静态风险
- 在最终风险判断中考虑路面坑洼/裂缝

## 2. 新增内容
新增文件：
- `Distance-measurement/detect_3d_with_surface.py`
- `Distance-measurement/road_surface_fusion/__init__.py`
- `Distance-measurement/road_surface_fusion/detector.py`
- `Distance-measurement/road_surface_fusion/surface_analysis.py`
- `Distance-measurement/road_surface_fusion/risk_fusion.py`
- `Distance-measurement/road_surface_fusion/visualization.py`
- `Distance-measurement/road_surface_fusion/depth_runtime.py`
- `仓库根目录/detect_3d_with_surface.py`

未直接修改：
- `Distance-measurement/detect_3d.py`
- `code/` 目录原始代码

新增文件的职责可以粗分为三层：
- 入口层：`detect_3d_with_surface.py`
  负责命令行参数、模型初始化、整条处理流程调度。
- 能力层：`road_surface_fusion/*.py`
  负责路面检测、风险分析、风险融合和可视化。
- 启动层：仓库根目录 `detect_3d_with_surface.py`
  负责把根目录命令转发到 `Distance-measurement` 目录。

这种拆法和这次实际集成步骤是一一对应的：
- 先梳理 `Distance-measurement` 原始入口，再抽取 `code` 中路面模型的推理能力
- 再把路面检测、风险分析、风险融合、可视化拆进 `road_surface_fusion`，避免原入口继续膨胀
- 最后用新的融合入口和根目录启动脚本把整条流程拼起来，同时兼容原有使用习惯

## 3. 运行入口
推荐在仓库根目录运行：
```powershell
python detect_3d_with_surface.py --source ..\code\test\00493.jpg --no-view-img --nosave --device cpu
```

也可以进入 `Distance-measurement` 后运行：
```powershell
cd Distance-measurement
python detect_3d_with_surface.py --source ..\code\test\00493.jpg --no-view-img --nosave --device cpu
```

根目录启动脚本会自动切换到 `Distance-measurement`、补充 `sys.path`，再转发执行真正入口脚本。
常见运行示例：

```powershell
python detect_3d_with_surface.py --source Distance-measurement\lanechange.mp4
```

```powershell
python detect_3d_with_surface.py --source 0
```

```powershell
python detect_3d_with_surface.py --source Distance-measurement\lanechange.mp4 --road-conf-thres 0.35
```

```powershell
python detect_3d_with_surface.py --source Distance-measurement\lanechange.mp4 --depth-backend depth-anything
```

运行建议：
- 本地联调用 `--no-view-img` 时，更适合让外部 UI 接管显示
- 批处理测试建议加 `--nosave`
- 需要留样结果时，建议显式传 `--project` 和 `--name`

## 4. 命令行接口
### 4.1 原有接口
沿用原脚本常用参数：`--weights`、`--source`、`--img-size`、`--conf-thres`、`--iou-thres`、`--device`、`--view-img`、`--no-view-img`、`--save-txt`、`--save-conf`、`--nosave`、`--classes`、`--agnostic-nms`、`--augment`、`--update`、`--project`、`--name`、`--exist-ok`、`--config_deepsort`。

### 4.2 新增接口
- `--camera-params`：相机内参文件，默认 `config/camera_params.yaml`
- `--road-model-dir`：路面模型目录，默认指向 `code/models`
- `--road-conf-thres`：路面坑洼/裂缝检测置信度阈值，默认 `0.25`
- `--depth-backend`：深度后端，当前仅支持 `depth-anything`

补充：当前深度后端统一为 `depth-anything`；运行环境需要能访问模型，或者本地已经缓存对应权重。

### 4.3 参数行为补充
- `--camera-params`
  如果文件存在，则优先读取真实相机内参；如果读取失败，脚本会按输入分辨率估算一个近似矩阵。
- `--road-model-dir`
  目录下默认需要有 `best.pt`、`best_night.pt`、`crack_best.pt`。
- `--road-conf-thres`
  路面检测阈值，值越大越保守，值越小越容易检出更多候选区域。
- `--depth-backend depth-anything`
  默认值，优先尝试真实深度模型。

### 4.4 UI 同事关注
这次改动**不会直接影响**现有 UI，同事如果还继续调用原来的 `Distance-measurement/detect_3d.py` 或 `code` 里的原始界面，行为不变。

只有在 UI 主动切到新入口 `detect_3d_with_surface.py` 时，界面才会新增以下内容：
- 主画面增加坑洼、裂缝框和半透明 mask
- 主画面左上角增加路面摘要面板
- 主画面右上角增加综合风险状态面板
- 右侧 BEV 图会显示 `P` / `C` 路面风险点

UI 如果通过命令行集成，优先关注这些参数：
- 输入相关：`--source`
- 显示控制：`--view-img`、`--no-view-img`
- 输出控制：`--nosave`、`--project`、`--name`
- 路面能力：`--road-model-dir`、`--road-conf-thres`
- 深度后端：`--depth-backend`

UI 侧当前可依赖的输入输出约定：
- 输入仍然是图片、视频、摄像头编号或流地址，通过 `--source` 传入
- 图片保存时，输出文件名为 `原文件名_with_risk.xxx`
- 视频保存时，输出为“主画面 + BEV”拼接视频
- `--no-view-img` 时不会弹 OpenCV 窗口，适合外部 UI 自己接管展示

当前**没有新增**这些接口：
- 没有 HTTP / WebSocket / REST API
- 没有单独的 JSON 结果导出接口
- `detect_3d_with_surface.py` 仍以脚本运行为主，不是稳定的函数式 UI SDK

如果 UI 同事需要做 Python 侧集成，可以直接 import：
- `RoadSurfaceDetector`
- `RoadSurfaceAnalyzer`
- `RoadSurfaceRiskFuser`
- `RoadSurfaceVisualizer`
- `RobustDepthEstimator`
一个最小化导入示例：

```python
from road_surface_fusion import (
    RoadSurfaceDetector,
    RoadSurfaceAnalyzer,
    RoadSurfaceRiskFuser,
    RoadSurfaceVisualizer,
    RobustDepthEstimator,
)
```

如果 UI 同事未来要做更细粒度联调，建议优先把集成点放在这两层：
- 脚本层：直接调用 `detect_3d_with_surface.py`
  适合快速接通演示流程。
- 模块层：按需调用 `RoadSurfaceDetector`、`RoadSurfaceAnalyzer`、`RoadSurfaceVisualizer`
  适合未来把结果挂进你们自己的 UI 状态管理或消息总线。

## 5. 新增模块接口
### 5.1 `RoadSurfaceDetector`
文件：`road_surface_fusion/detector.py`
职责：加载 `best.pt`、`best_night.pt`、`crack_best.pt`，判断白天/夜间并切换坑洼模型，同时执行裂缝模型。
接口：`RoadSurfaceDetector(model_dir=None, preferred_device=None)`，`detect(image, conf_thres=None, skip_frame_check=False) -> (main_results, aux_results, model_label)`。

返回值说明：
- `main_results`：坑洼主模型结果
- `aux_results`：裂缝辅助模型结果
- `model_label`：当前主模型标签，值为 `day` 或 `night`

### 5.2 `RoadSurfaceAnalyzer`
文件：`road_surface_fusion/surface_analysis.py`
职责：解析 box 或 mask，估计坑洼/裂缝距离与横向位置，输出结构化风险对象和整体路面风险分数。
接口：`RoadSurfaceAnalyzer(danger_zone_ratio=0.4)`，`analyze(image, depth_map, camera_matrix, main_results, aux_results, model_label) -> SurfaceAnalysisResult`。
关键输出：`hazards`、`pothole_count`、`crack_count`、`road_risk_score`、`road_danger_level`、`warning_text`。

补充说明：
- `hazards` 里会放每一个坑洼或裂缝的结构化对象
- `road_risk_score` 是路面整体风险分数
- `road_danger_level` 是离散等级，便于直接给 UI 做状态显示
- `warning_text` 是给面板展示准备的简短提示

### 5.3 `RoadSurfaceRiskFuser`
文件：`road_surface_fusion/risk_fusion.py`
职责：把路面风险映射到 BEV 风险场，输出路面热力图和风险分数，并与动态目标风险融合。
接口：`build_surface_maps(analysis, risk_engine) -> (surface_risk_map, surface_vis_map, surface_risk)`，`fuse_risk(dynamic_risk, surface_risk) -> combined_risk`。
当前规则：`combined_risk = max(dynamic_risk, surface_risk)`。

### 5.4 `RoadSurfaceVisualizer`
文件：`road_surface_fusion/visualization.py`
职责：在主画面绘制坑洼/裂缝 mask、框、距离和摘要面板，在 BEV 图中绘制路面风险点、风险状态和简写标签。
接口：`draw_on_frame(image, analysis) -> np.ndarray`，`draw_on_bev(bev_visualizer, analysis) -> None`。

可视化约定：
- `draw_on_frame` 返回的是叠加后的图像
- `draw_on_bev` 直接在传入的 BEV 画布上绘制，不单独返回新对象

### 5.5 `RobustDepthEstimator`
文件：`road_surface_fusion/depth_runtime.py`
职责：统一封装 `depth-anything` 深度后端，并保持与主流程一致的调用接口。
接口：`RobustDepthEstimator(model_size="small", device="cpu", offline_first=False, backend="depth-anything")`、`estimate_depth(image)`、`get_depth_in_region(depth_map, bbox, method="median", scale=0.5)`。

对外约定：
- `estimate_depth(image)` 返回归一化深度图
- `get_depth_in_region(...)` 用于给目标框或风险区域取局部深度值
- 当前统一使用 `depth-anything`

## 6. 主流程逻辑
每一帧的处理顺序如下：
1. 主模型检测车、人等动态目标。
2. 生成当前帧深度图。
3. 路面模型检测坑洼和裂缝。
4. 将路面检测结果转换成 `SurfaceHazard`。
5. 在主画面叠加路面检测结果。
6. DeepSORT 跟踪动态目标。
7. 估计目标 3D 位置并生成动态风险场。
8. 将坑洼/裂缝映射为 BEV 静态风险场。
9. 融合动态风险和路面风险。
10. 输出主画面与 BEV 拼接结果。

这条流程的实现原则是：
- 动态目标风险尽量沿用原项目现有逻辑
- 路面风险作为新增静态风险源插入，不强改原风控框架
- 让主画面、BEV、最终决策三处同时感知路面信息

## 7. 风险计算逻辑
路面风险的计算基于：
- 检测置信度
- 区域面积占比
- 检测位置是否靠近图像底部
- 深度估计得到的距离
- 风险类型是坑洼还是裂缝

当前实现中：
- 坑洼权重高于裂缝
- 越大、越近、越靠近车前区域，风险越高
- 总路面风险不是平均值，而是“最高风险 + 密度增益 + 近距离增益”

动态目标方面，本次融合版显式保留了：
```python
TRACKED_CLASSES = {0, 1, 2, 3, 5, 7}
```
即把 `person` 也纳入了跟踪和风险评估。
最终融合逻辑当前比较直接：

```python
combined_risk = max(dynamic_risk, surface_risk)
```

这么做的原因是：
- 规则简单，便于排查
- 不会把高路面风险被低动态风险“平均掉”
- 对预警系统来说更偏保守，工程上更稳

## 8. 输出效果
主画面会看到：
- 车辆和行人的检测框
- 坑洼和裂缝框
- 若模型返回 mask，则显示半透明着色
- 左上角路面摘要面板
- 右上角综合风险状态面板

BEV 图会看到：
- 动态目标框和轨迹
- 风险热力图
- `P` 表示坑洼，`C` 表示裂缝
- 底部显示路面风险等级与分数

保存行为说明：
- 图片输入时，输出文件名为 `原文件名_with_risk.xxx`
- 视频输入时，输出为主画面和 BEV 拼接后的视频
- 指定 `--nosave` 时，不会落盘保存结果

## 9. 已验证内容
已完成：
- 新增脚本和新增模块通过语法检查
- 图片输入做过实际运行验证
- 根目录运行方式验证可直接启动
- 运行产物和 `__pycache__` 已清理，当前状态适合直接提交

补充说明：
- 当前版本统一使用 `depth-anything`
- 如果模型不可用，程序会直接报错，需先保证模型下载或本地缓存可用

## 10. 注意事项
- 默认依赖 `code/models/best.pt`、`best_night.pt`、`crack_best.pt`
- 如果模型文件缺失，程序会直接报错
- 默认深度后端是 `depth-anything`
- 当前版本统一依赖 `Depth Anything`，首次运行前需确认模型可用
- 原始 `Distance-measurement/detect_3d.py` 和 `code` 目录没有被直接修改

后续扩展时，最常见的改动位置是：
- 调整路面检测阈值：`--road-conf-thres`
- 替换模型目录：`--road-model-dir`
- 调整风险评分：`road_surface_fusion/surface_analysis.py`
- 调整融合策略：`road_surface_fusion/risk_fusion.py`
- 调整画面样式：`road_surface_fusion/visualization.py`
