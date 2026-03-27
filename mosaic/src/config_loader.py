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


_DEFAULT_CHAT_PROVIDER = "ali_api"
_DEFAULT_CHAT_MODEL = "qwen3.5-plus"


def get_mosaic_build_mode() -> str:
    """
    构图模式（docs/optimization.md §5 B-1）：

    - ``hybrid``：``sense_classes`` 对新建类使用 LLM；实例更新非纯 hash（``save`` 主路径）。
    - ``hash_only``：与 ``save_hash`` 一致，不作构图 LLM 调用（基线）。

    优先级：环境变量 ``MOSAIC_BUILD_MODE``（``hybrid`` / ``hash_only``）→ ``[BUILD] mode`` → 默认 ``hybrid``。

    注意：CLI ``build --hash`` 在 ``cli.py`` 中强制 ``hash_only``，不经过本函数返回值单独判断。
    """
    env = os.environ.get("MOSAIC_BUILD_MODE", "").strip().lower()
    if env in ("hybrid", "hash_only"):
        return env
    config = load_api_config()
    if config.has_section("BUILD"):
        m = config.get("BUILD", "mode", fallback="hybrid").strip().lower()
        if m in ("hybrid", "hash_only"):
            return m
    return "hybrid"


def get_mosaic_chat_model_spec() -> str:
    """
    Mosaic 内所有 LLM 调用（构图 / query / judge）共用的「完全限定名」，供 load_chat_model 使用。

    格式：``provider|model``；自定义 HTTP 服务端为 ``custom|model_name|url``（与 load_chat_model 一致）。

    优先级：
    - 环境变量 ``MOSAIC_CHAT_MODEL_SPEC``（整条覆盖）
    - ``MOSAIC_CHAT_PROVIDER`` + ``MOSAIC_CHAT_MODEL``（分别覆盖）
    - ``[LLM]`` 段 ``provider``、``chat_model``（缺省为 ali_api + qwen3.5-plus，见 docs/optimization.md P-0）
    """
    env_full = os.environ.get("MOSAIC_CHAT_MODEL_SPEC", "").strip()
    if env_full:
        return env_full

    config = load_api_config()
    provider = _DEFAULT_CHAT_PROVIDER
    model = _DEFAULT_CHAT_MODEL
    if config.has_section("LLM"):
        provider = config.get("LLM", "provider", fallback=_DEFAULT_CHAT_PROVIDER).strip() or _DEFAULT_CHAT_PROVIDER
        model = config.get("LLM", "chat_model", fallback=_DEFAULT_CHAT_MODEL).strip() or _DEFAULT_CHAT_MODEL

    env_provider = os.environ.get("MOSAIC_CHAT_PROVIDER", "").strip()
    if env_provider:
        provider = env_provider
    env_model = os.environ.get("MOSAIC_CHAT_MODEL", "").strip()
    if env_model:
        model = env_model

    return f"{provider}|{model}"


def get_mosaic_chat_model_name() -> str:
    """当前配置下的对话模型标识（如 qwen3.5-plus），不含 provider。供需直接构造 QwenChatModel 的脚本使用。"""
    spec = get_mosaic_chat_model_spec()
    if "|" not in spec:
        return spec
    tokens = spec.split("|")
    if len(tokens) >= 2:
        return tokens[1]
    return spec


def get_query_neighbor_traversal_config() -> tuple[int, int, frozenset[str]]:
    """
    P-2：查询阶段在嵌入/TF-IDF/关键词种子之外，沿双图实例邻接做有限跳扩展。

    返回 (max_hops, max_extra_instances, allowed_edge_legs)。
    max_hops == 0 表示关闭（与 docs/optimization.md 中「可关」一致）。

    配置节 ``[QUERY]``：
    - neighbor_hops：最大跳数，默认 1
    - neighbor_max_extra：最多追加的邻域实例数，默认 16
    - neighbor_edge_legs：``ALL``、``P``、``A`` 或逗号分隔 ``P,A``（与 dual_graph 中 E_P/E_A 一致）

    环境变量（可选）：MOSAIC_QUERY_NEIGHBOR_HOPS、MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA、MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS
    """
    from src.data.dual_graph import ALL_EDGE_LEGS, EDGE_LEG_ASSOCIATIVE, EDGE_LEG_PRAGMATIC

    config = load_api_config()
    hops = 1
    max_extra = 16
    legs_raw = "ALL"
    if config.has_section("QUERY"):
        hops = config.getint("QUERY", "neighbor_hops", fallback=1)
        max_extra = config.getint("QUERY", "neighbor_max_extra", fallback=16)
        legs_raw = config.get("QUERY", "neighbor_edge_legs", fallback="ALL").strip() or "ALL"

    eh = os.environ.get("MOSAIC_QUERY_NEIGHBOR_HOPS", "").strip()
    if eh.isdigit():
        hops = int(eh)
    em = os.environ.get("MOSAIC_QUERY_NEIGHBOR_MAX_EXTRA", "").strip()
    if em.isdigit():
        max_extra = int(em)
    el = os.environ.get("MOSAIC_QUERY_NEIGHBOR_EDGE_LEGS", "").strip()
    if el:
        legs_raw = el

    u = legs_raw.upper().replace(" ", "")
    if u in ("ALL", "*", "BOTH"):
        legs: frozenset[str] = frozenset(ALL_EDGE_LEGS)
    else:
        parts = [p.strip().upper() for p in legs_raw.split(",") if p.strip()]
        chosen: set[str] = set()
        for p in parts:
            if p in ("P", "PRAGMATIC", "E_P"):
                chosen.add(EDGE_LEG_PRAGMATIC)
            elif p in ("A", "ASSOCIATIVE", "E_A"):
                chosen.add(EDGE_LEG_ASSOCIATIVE)
        legs = frozenset(chosen if chosen else ALL_EDGE_LEGS)

    return hops, max_extra, legs


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
