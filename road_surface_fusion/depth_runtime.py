from __future__ import annotations

from depth_model import DepthEstimator


class RobustDepthEstimator:
    def __init__(self, model_size="small", device="cpu", offline_first=False, backend="depth-anything"):
        if backend != "depth-anything":
            raise ValueError("Only 'depth-anything' is supported.")

        self.backend_name = "depth-anything"
        try:
            self.impl = DepthEstimator(model_size=model_size, device=device)
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything initialization failed. Ensure the model is available "
                "locally or that the runtime can access Hugging Face to download it."
            ) from exc

    def estimate_depth(self, image):
        return self.impl.estimate_depth(image)

    def get_depth_in_region(self, depth_map, bbox, method="median", scale=0.5):
        return self.impl.get_depth_in_region(depth_map, bbox, method=method, scale=scale)
