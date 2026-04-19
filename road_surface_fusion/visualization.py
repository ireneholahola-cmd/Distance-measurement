from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from .surface_analysis import SurfaceAnalysisResult


class RoadSurfaceVisualizer:
    def __init__(self):
        self.colors: Dict[str, Tuple[int, int, int]] = {
            "pothole": (0, 165, 255),
            "crack": (255, 0, 255),
        }

    def draw_on_frame(self, image: np.ndarray, analysis: SurfaceAnalysisResult) -> np.ndarray:
        if image is None:
            return image

        annotated = image.copy()
        overlay = annotated.copy()

        for hazard in analysis.hazards:
            color = self.colors.get(hazard.hazard_type, (255, 255, 255))
            if hazard.mask is not None and np.any(hazard.mask):
                overlay[hazard.mask > 0] = color

        if analysis.hazards:
            annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

        for hazard in analysis.hazards:
            color = self.colors.get(hazard.hazard_type, (255, 255, 255))
            x1, y1, x2, y2 = hazard.bbox
            thickness = 2 if hazard.hazard_type == "pothole" else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"{hazard.label} {hazard.confidence:.2f} {hazard.distance_m:.1f}m"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_top = max(y1 - text_h - 6, 0)
            cv2.rectangle(
                annotated,
                (x1, text_top),
                (x1 + text_w + 6, text_top + text_h + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, text_top + text_h + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        self._draw_summary_panel(annotated, analysis)
        return annotated

    def draw_on_bev(self, bev_visualizer, analysis: SurfaceAnalysisResult) -> None:
        canvas = bev_visualizer.bev_image
        for hazard in analysis.hazards:
            px = bev_visualizer.origin_x + int(hazard.x_m * bev_visualizer.scale)
            py = bev_visualizer.origin_y - int(hazard.z_m * bev_visualizer.scale)

            if px < 0 or py < 0 or px >= bev_visualizer.width or py >= bev_visualizer.height:
                continue

            color = self.colors.get(hazard.hazard_type, (255, 255, 255))
            radius = max(4, int(4 + hazard.severity * 8))
            cv2.circle(canvas, (px, py), radius, color, -1)
            cv2.circle(canvas, (px, py), radius + 2, (255, 255, 255), 1)
            short_label = "P" if hazard.hazard_type == "pothole" else "C"
            cv2.putText(
                canvas,
                short_label,
                (px - 4, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        status = self._status_name(analysis.road_danger_level)
        cv2.putText(
            canvas,
            f"ROAD: {status}",
            (10, bev_visualizer.height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"P:{analysis.pothole_count} C:{analysis.crack_count} SCORE:{analysis.road_risk_score:.2f}",
            (10, bev_visualizer.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_summary_panel(self, image: np.ndarray, analysis: SurfaceAnalysisResult) -> None:
        panel_lines = [
            f"Road model: {analysis.model_label}",
            f"Pothole: {analysis.pothole_count}  Crack: {analysis.crack_count}  Score: {analysis.road_risk_score:.2f}",
            f"Surface: {self._status_name(analysis.road_danger_level)}",
        ]

        panel_x = 15
        panel_y = 20
        panel_w = 460
        panel_h = 82

        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        for index, line in enumerate(panel_lines):
            cv2.putText(
                image,
                line,
                (panel_x + 10, panel_y + 22 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _status_name(self, danger_level: int) -> str:
        if danger_level >= 3:
            return "HIGH"
        if danger_level == 2:
            return "MEDIUM"
        if danger_level == 1:
            return "LOW"
        return "CLEAR"
