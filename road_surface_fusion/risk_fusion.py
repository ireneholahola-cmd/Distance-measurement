from __future__ import annotations

from typing import Tuple

import numpy as np

from .surface_analysis import SurfaceAnalysisResult


class RoadSurfaceRiskFuser:
    def build_surface_maps(self, analysis: SurfaceAnalysisResult, risk_engine) -> Tuple[np.ndarray, np.ndarray, float]:
        total_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w), dtype=np.float32)
        vis_map = np.zeros((risk_engine.grid_h, risk_engine.grid_w), dtype=np.float32)
        peak_score = 0.0

        for hazard in analysis.hazards:
            type_weight = 1.25 if hazard.hazard_type == "pothole" else 0.9
            sigma_x = 0.5 + min(1.2, hazard.area_ratio * 60.0 + 0.2)
            sigma_z = 1.2 + hazard.severity * 2.5
            weight = hazard.severity * type_weight

            field = risk_engine.get_gaussian_field(
                hazard.x_m,
                hazard.z_m,
                0.0,
                0.0,
                sigma_x=sigma_x,
                sigma_z=sigma_z,
                v_stretch_factor=0.0,
                weather_factor=1.0,
            )
            vis_field = risk_engine.get_visualization_field(
                hazard.x_m,
                hazard.z_m,
                0.0,
                0.0,
                sigma_x=sigma_x * 1.8,
                sigma_z=sigma_z * 1.8,
                weather_factor=1.0,
            )

            total_map += field * weight
            vis_map = np.maximum(vis_map, vis_field * weight)
            peak_score = max(peak_score, float(np.max(field) * weight))

        fused_score = max(peak_score, analysis.road_risk_score)
        return total_map, vis_map, fused_score

    def fuse_risk(self, dynamic_risk: float, surface_risk: float) -> float:
        return max(dynamic_risk, surface_risk)
