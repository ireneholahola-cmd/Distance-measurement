from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional


_DISABLED_VALUES = {"0", "false", "no", "off"}
_LEVEL_PRIORITY = {
    "MEDIUM": 1,
    "HIGH": 2,
}


class RiskSoundAlerter:
    """Play warning sounds for MEDIUM and HIGH risk frame decisions."""

    def __init__(
        self,
        sounds_dir: Optional[str | Path] = None,
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env_sounds_dir = os.environ.get("DRIVESAFE_SOUND_DIR")
        self.sounds_dir = Path(sounds_dir or env_sounds_dir or repo_root / "sounds")
        self.cooldown_seconds = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else float(os.environ.get("DRIVESAFE_SOUND_ALERT_COOLDOWN", "2.0"))
        )
        self.enabled = os.environ.get("DRIVESAFE_SOUND_ALERT", "1").strip().lower() not in _DISABLED_VALUES
        self._last_play_time = 0.0
        self._last_level: Optional[str] = None
        self._warned_keys: set[str] = set()
        self._winsound = None

        if self.enabled and os.name == "nt":
            try:
                import winsound

                self._winsound = winsound
            except Exception as exc:
                self._warn_once("winsound", f"[risk-sound-alert] winsound unavailable: {exc}")
        elif self.enabled:
            self._warn_once("platform", "[risk-sound-alert] sound alerts require Windows winsound")

    def handle_frame_record(self, frame_record: dict) -> None:
        self.handle_status(frame_record.get("decision_status"))

    def handle_status(self, status: object) -> None:
        if not self.enabled:
            return

        level = str(status or "").upper()
        if level not in _LEVEL_PRIORITY:
            return

        now = time.monotonic()
        if not self._should_play(level, now):
            return

        if self._play(level):
            self._last_play_time = now
            self._last_level = level

    def _should_play(self, level: str, now: float) -> bool:
        elapsed = now - self._last_play_time
        if elapsed >= self.cooldown_seconds:
            return True

        last_priority = _LEVEL_PRIORITY.get(self._last_level or "", 0)
        current_priority = _LEVEL_PRIORITY[level]
        return current_priority > last_priority

    def _play(self, level: str) -> bool:
        if self._winsound is None:
            return False

        sound_path = self.sounds_dir / f"{level.lower()}_risk.wav"
        if not sound_path.is_file():
            self._warn_once(str(sound_path), f"[risk-sound-alert] missing sound file: {sound_path}")
            return False

        try:
            self._winsound.PlaySound(
                str(sound_path),
                self._winsound.SND_FILENAME | self._winsound.SND_ASYNC,
            )
            return True
        except Exception as exc:
            self._warn_once(f"play:{level}", f"[risk-sound-alert] failed to play {sound_path}: {exc}")
            return False

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(message)
