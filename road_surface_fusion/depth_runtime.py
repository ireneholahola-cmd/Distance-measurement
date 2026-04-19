from __future__ import annotations

import os
import socket
from typing import Dict

import numpy as np

from depth_model import DepthEstimator


class FallbackDepthEstimator:
    """
    Lightweight heuristic depth used when the transformer depth model is not
    available locally. It keeps the fusion pipeline runnable offline.
    """

    backend_name = "heuristic-gradient"

    def estimate_depth(self, image):
        height, width = image.shape[:2]
        vertical = np.linspace(1.0, 0.0, height, dtype=np.float32).reshape(height, 1)
        return np.repeat(vertical, width, axis=1)

    def get_depth_in_region(self, depth_map, bbox, method="median", scale=0.5):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        new_width = width * scale
        new_height = height * scale
        x1 = int(center_x - new_width / 2)
        y1 = int(center_y - new_height / 2)
        x2 = int(center_x + new_width / 2)
        y2 = int(center_y + new_height / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        if method == "mean":
            return float(np.mean(region))
        if method == "min":
            return float(np.min(region))
        return float(np.median(region))


class RobustDepthEstimator:
    def __init__(self, model_size="small", device="cpu", offline_first=False, backend="depth-anything"):
        self.backend_name = "depth-anything"
        self.impl = None

        if backend == "heuristic":
            self.impl = FallbackDepthEstimator()
            self.backend_name = self.impl.backend_name
            return

        if not self._can_reach_huggingface():
            print("Depth model network is unavailable, switching to fallback depth estimator.")
            self.impl = FallbackDepthEstimator()
            self.backend_name = self.impl.backend_name
            return

        env_backup: Dict[str, str] = {}
        forced_keys = ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"]
        try:
            if offline_first:
                for key in forced_keys:
                    env_backup[key] = os.environ.get(key, "")
                    os.environ[key] = "1"
            self.impl = DepthEstimator(model_size=model_size, device=device)
        except Exception as exc:
            print(f"Depth model unavailable, switching to fallback depth estimator: {exc}")
            self.impl = FallbackDepthEstimator()
            self.backend_name = self.impl.backend_name
        finally:
            if offline_first:
                for key in forced_keys:
                    original = env_backup.get(key, "")
                    if original:
                        os.environ[key] = original
                    elif key in os.environ:
                        del os.environ[key]

    def estimate_depth(self, image):
        return self.impl.estimate_depth(image)

    def get_depth_in_region(self, depth_map, bbox, method="median", scale=0.5):
        return self.impl.get_depth_in_region(depth_map, bbox, method=method, scale=scale)

    def _can_reach_huggingface(self) -> bool:
        try:
            with socket.create_connection(("huggingface.co", 443), timeout=2):
                return True
        except OSError:
            return False
