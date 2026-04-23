"""
Logger setup — reads config/logging_config.yaml and configures loguru.

Import and call setup_logging() once at application startup:

    from src.utils.logger import setup_logging
    setup_logging()

After that, any module can use loguru directly:

    from loguru import logger
    logger.info("Ready")
"""
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


def setup_logging(
    config_path: Optional[str] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Configure loguru from config/logging_config.yaml.

    Args:
        config_path: Path to logging_config.yaml. Defaults to
                     <project_root>/config/logging_config.yaml.
        log_level:   Override the log level from config (useful for tests).
                     Also reads from LOG_LEVEL environment variable.
    """
    # ── Resolve config path ──────────────────────────────────────────────────
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = str(project_root / "config" / "logging_config.yaml")

    # ── Load YAML ────────────────────────────────────────────────────────────
    cfg: dict = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Logging config not found at {config_path}; using defaults.")

    # ── Determine effective log level ─────────────────────────────────────────
    effective_level = (
        log_level
        or os.getenv("LOG_LEVEL")
        or cfg.get("level", "INFO")
    ).upper()

    # ── Remove default loguru handler ─────────────────────────────────────────
    logger.remove()

    sinks = cfg.get("sinks", {})

    # ── Console sink ──────────────────────────────────────────────────────────
    console_cfg = sinks.get("console", {})
    if console_cfg.get("enabled", True):
        logger.add(
            sys.stderr,
            level=console_cfg.get("level", effective_level),
            format=console_cfg.get(
                "format",
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>",
            ),
            colorize=console_cfg.get("colorize", True),
            enqueue=cfg.get("enqueue", True),
            backtrace=cfg.get("backtrace", True),
            diagnose=cfg.get("diagnose", False),
        )

    # ── File sinks ───────────────────────────────────────────────────────────
    for sink_name, sink_cfg in sinks.items():
        if sink_name == "console" or not sink_cfg.get("enabled", False):
            continue

        log_path = sink_cfg.get("path")
        if not log_path:
            continue

        # Ensure log directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

        kwargs = dict(
            level=sink_cfg.get("level", effective_level),
            format=sink_cfg.get(
                "format",
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            ),
            rotation=sink_cfg.get("rotation", "00:00"),
            retention=sink_cfg.get("retention", "30 days"),
            compression=sink_cfg.get("compression", "zip"),
            encoding=sink_cfg.get("encoding", "utf-8"),
            enqueue=cfg.get("enqueue", True),
            backtrace=cfg.get("backtrace", True),
            diagnose=cfg.get("diagnose", False),
            serialize=sink_cfg.get("serialize", False),
        )

        logger.add(log_path, **kwargs)

    logger.info(
        f"Logging configured — level={effective_level}, "
        f"active sinks: {[k for k, v in sinks.items() if v.get('enabled', k == 'console')]}"
    )
