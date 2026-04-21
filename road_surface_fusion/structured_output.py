from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .surface_analysis import SurfaceAnalysisResult, SurfaceHazard


def _round_float(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _normalize_risk_source(source: Optional[Any]) -> Optional[Any]:
    if source is None:
        return None
    if isinstance(source, (np.integer, int)):
        return int(source)
    return str(source)


def _build_dimensions_payload(dimensions: Optional[Any]) -> Optional[Dict[str, float]]:
    if dimensions is None:
        return None

    dims = np.asarray(dimensions).reshape(-1).tolist()
    if len(dims) != 3:
        return None

    return {
        "height": _round_float(dims[0]),
        "width": _round_float(dims[1]),
        "length": _round_float(dims[2]),
    }


def build_target_record(target: Dict[str, Any]) -> Dict[str, Any]:
    bbox_2d = target.get("bbox_2d")
    if bbox_2d is None and "box_3d_draw" in target:
        bbox_2d = target["box_3d_draw"].get("bbox_2d")

    return {
        "id": int(target.get("id", -1)),
        "class_name": str(target.get("class_name", "unknown")),
        "object_type": str(target.get("type", "unknown")),
        "confidence": _round_float(target.get("confidence", 0.0)),
        "bbox_2d": [int(value) for value in bbox_2d] if bbox_2d is not None else None,
        "position_m": {
            "x": _round_float(target.get("x", 0.0)),
            "z": _round_float(target.get("z", 0.0)),
        },
        "speed_mps": _round_float(target.get("speed", 0.0)),
        "yaw_rad": _round_float(target.get("yaw", 0.0)),
        "dimensions_m": _build_dimensions_payload(target.get("dims")),
        "risk_score": _round_float(target.get("scf", 0.0)),
    }


def build_surface_hazard_record(hazard: SurfaceHazard) -> Dict[str, Any]:
    return {
        "hazard_id": str(hazard.hazard_id),
        "hazard_type": str(hazard.hazard_type),
        "label": str(hazard.label),
        "confidence": _round_float(hazard.confidence),
        "bbox": [int(value) for value in hazard.bbox],
        "centroid_px": [int(value) for value in hazard.centroid_px],
        "pixel_area": _round_float(hazard.pixel_area),
        "area_ratio": _round_float(hazard.area_ratio),
        "bottom_ratio": _round_float(hazard.bottom_ratio),
        "distance_m": _round_float(hazard.distance_m),
        "position_m": {
            "x": _round_float(hazard.x_m),
            "z": _round_float(hazard.z_m),
        },
        "severity": _round_float(hazard.severity),
        "near_zone": bool(hazard.near_zone),
    }


def build_frame_record(
    *,
    source: str,
    stream_index: int,
    frame_index: int,
    source_frame_index: int,
    image_shape,
    dynamic_targets: Iterable[Dict[str, Any]],
    surface_analysis: SurfaceAnalysisResult,
    dynamic_risk: float,
    surface_risk: float,
    combined_risk: float,
    decision_status: str,
    warning_text: str,
    max_risk_source: Optional[Any],
) -> Dict[str, Any]:
    image_h, image_w = image_shape[:2]

    return {
        "source": str(source),
        "stream_index": int(stream_index),
        "frame_index": int(frame_index),
        "source_frame_index": int(source_frame_index),
        "image": {
            "width": int(image_w),
            "height": int(image_h),
        },
        "decision_status": str(decision_status),
        "warning_text": str(warning_text),
        "dynamic_risk": _round_float(dynamic_risk),
        "surface_risk": _round_float(surface_risk),
        "combined_risk": _round_float(combined_risk),
        "max_risk_source": _normalize_risk_source(max_risk_source),
        "targets": [build_target_record(target) for target in dynamic_targets],
        "surface_hazards": [build_surface_hazard_record(hazard) for hazard in surface_analysis.hazards],
        "surface_summary": {
            "pothole_count": int(surface_analysis.pothole_count),
            "crack_count": int(surface_analysis.crack_count),
            "near_hazard_count": int(surface_analysis.near_hazard_count),
            "road_risk_score": _round_float(surface_analysis.road_risk_score),
            "road_danger_level": int(surface_analysis.road_danger_level),
            "road_warning_text": str(surface_analysis.warning_text),
            "dominant_hazard_type": surface_analysis.dominant_hazard_type,
            "crack_severity": str(surface_analysis.crack_severity),
            "road_model_label": str(surface_analysis.model_label),
        },
    }


class StructuredOutputWriter:
    def __init__(self, output_dir: str | Path, filename: str = "frame_results.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / filename
        self._handle = self.output_path.open("w", encoding="utf-8")

    def write_frame(self, frame_record: Dict[str, Any]) -> None:
        self._handle.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.close()

