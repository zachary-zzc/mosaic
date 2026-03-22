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
    """
    Return (ali_api_key, ali_base_url).

    优先级（与 DashScope 文档一致，便于服务器用环境变量覆盖）：
    - Key: 环境变量 DASHSCOPE_API_KEY，否则 config [API_KEYS] ali_api_key
    - Base: DASHSCOPE_API_BASE 或 MOSAIC_ALI_BASE_URL，否则 config [API_KEYS] ali_base_url

    国内兼容模式通常为 https://dashscope.aliyuncs.com/compatible-mode/v1；
    国际站等需使用对应地域 endpoint，否则易出现鉴权或 Request 类错误。
    """
    config = load_api_config()
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    base = (
        os.environ.get("DASHSCOPE_API_BASE", "").strip()
        or os.environ.get("MOSAIC_ALI_BASE_URL", "").strip()
    )
    if config.has_section("API_KEYS"):
        if not key:
            key = config.get("API_KEYS", "ali_api_key", fallback="").strip()
        if not base:
            base = config.get("API_KEYS", "ali_base_url", fallback="").strip()
    base = base.rstrip("/")
    return key, base


def resolve_under_mosaic(path_value: str) -> Path:
    """
    将配置中的路径解析为绝对 Path。
    若 path_value 为绝对路径则直接 resolve；否则视为相对于 mosaic 项目根目录。
    """
    raw = (path_value or "").strip()
    if not raw:
        return _mosaic_root() / "embedding_models" / "all-MiniLM-L6-v2"
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (_mosaic_root() / p).resolve()


def get_embedding_model_path() -> str:
    """
    返回 SentenceTransformer 可用的本地模型目录绝对路径字符串。
    优先 [PATHS] embedding_model；可被环境变量 MOSAIC_EMBEDDING_MODEL 覆盖。
    """
    override = os.environ.get("MOSAIC_EMBEDDING_MODEL", "").strip()
    if override:
        p = Path(override)
        return str(p.resolve() if p.is_absolute() else (_mosaic_root() / p).resolve())
    config = load_api_config()
    if config.has_section("PATHS"):
        rel = config.get("PATHS", "embedding_model", fallback="embedding_models/all-MiniLM-L6-v2")
    else:
        rel = "embedding_models/all-MiniLM-L6-v2"
    return str(resolve_under_mosaic(rel))
