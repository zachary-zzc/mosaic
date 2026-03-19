"""Load config from mosaic config dir or env."""
from __future__ import annotations

import configparser
import os
from pathlib import Path


def _mosaic_root() -> Path:
    """Resolve mosaic project root (directory containing config/)."""
    # From src/config_loader.py -> mosaic/
    this_file = Path(__file__).resolve()
    src = this_file.parent
    root = src.parent
    return root


def load_api_config() -> configparser.ConfigParser:
    """Load API config from config/config.cfg (mosaic-relative or env CONFIG_PATH)."""
    config = configparser.ConfigParser()
    config_path = os.environ.get("MOSAIC_CONFIG_PATH")
    if config_path and os.path.isfile(config_path):
        config.read(config_path, encoding="utf-8")
        return config
    default = _mosaic_root() / "config" / "config.cfg"
    if default.is_file():
        config.read(str(default), encoding="utf-8")
    return config


def get_api_key_and_base_url() -> tuple[str, str]:
    """Return (ali_api_key, ali_base_url) from config, or empty strings."""
    config = load_api_config()
    if not config.has_section("API_KEYS"):
        return "", ""
    key = config.get("API_KEYS", "ali_api_key", fallback="")
    base = config.get("API_KEYS", "ali_base_url", fallback="")
    return key, base
