from __future__ import annotations

import builtins
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any

from .alerter import RiskSoundAlerter


_DISABLED_VALUES = {"0", "false", "no", "off"}
_original_import = builtins.__import__
_hook_installed = False
_patched = False
_alerter: RiskSoundAlerter | None = None


def maybe_install() -> None:
    """Install the write_frame patch only for detect_3d_with_surface.py runs."""

    if not _should_enable():
        return

    _try_patch_loaded_writer()
    if not _patched:
        _install_import_hook()


def _should_enable() -> bool:
    enabled = os.environ.get("DRIVESAFE_SOUND_ALERT", "1").strip().lower() not in _DISABLED_VALUES
    if not enabled:
        return False

    script_name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return script_name == "detect_3d_with_surface.py"


def _install_import_hook() -> None:
    global _hook_installed
    if _hook_installed:
        return

    builtins.__import__ = _import_hook
    _hook_installed = True


def _import_hook(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    module = _original_import(name, globals, locals, fromlist, level)
    if name == "road_surface_fusion" or name.startswith("road_surface_fusion."):
        _try_patch_loaded_writer()
    return module


def _try_patch_loaded_writer() -> None:
    module = sys.modules.get("road_surface_fusion.structured_output")
    writer_cls = getattr(module, "StructuredOutputWriter", None) if module is not None else None
    if writer_cls is not None:
        _patch_writer(writer_cls)


def _patch_writer(writer_cls: type) -> None:
    global _patched, _alerter
    if _patched or getattr(writer_cls.write_frame, "_risk_sound_alert_patched", False):
        _patched = True
        return

    original_write_frame = writer_cls.write_frame
    _alerter = _alerter or RiskSoundAlerter()

    @wraps(original_write_frame)
    def patched_write_frame(self: Any, frame_record: dict) -> Any:
        result = original_write_frame(self, frame_record)
        try:
            _alerter.handle_frame_record(frame_record)
        except Exception as exc:
            print(f"[risk-sound-alert] alert handling failed: {exc}")
        return result

    patched_write_frame._risk_sound_alert_patched = True
    writer_cls.write_frame = patched_write_frame
    _patched = True
    _restore_import_hook()


def _restore_import_hook() -> None:
    global _hook_installed
    if _hook_installed and builtins.__import__ is _import_hook:
        builtins.__import__ = _original_import
    _hook_installed = False
