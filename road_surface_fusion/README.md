# 路面风险融合模块

这个目录是给 `Distance-measurement` 新增的路面风险融合模块，目标是在 **不修改原始检测入口** 的情况下，把 `code` 项目里的坑洼/裂缝检测能力融合进来。

## 模块组成

- `detector.py`
  负责加载 `code/models` 下的坑洼、夜间坑洼、裂缝模型。
- `surface_analysis.py`
  负责把检测结果转换成结构化路面风险。
- `risk_fusion.py`
  负责把路面风险映射到 BEV 风险场，并与动态目标风险融合。
- `visualization.py`
  负责在主画面和 BEV 画面中绘制坑洼、裂缝和汇总信息。
- `depth_runtime.py`
  负责深度后端选择。默认优先使用 `depth-anything`，不可用时自动回退。

## 调用入口

外部统一通过：

- `Distance-measurement/detect_3d_with_surface.py`

进行调用。

如果你在仓库根目录启动，也可以直接运行：

```powershell
python detect_3d_with_surface.py --source ..\code\test\00493.jpg --no-view-img --nosave --device cpu
```
