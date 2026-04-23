try:
    from risk_alerts.sound_processing.bootstrap import maybe_install

    maybe_install()
except Exception as exc:
    print(f"[risk-sound-alert] bootstrap failed: {exc}")
