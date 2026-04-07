"""Load config from mosaic config dir or env."""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EdgeConstructionConfig:
    """
    构图结束后的双图增强（docs/optimization.md §5 B-2、§4）。

    环境变量覆盖（可选）：
    MOSAIC_EDGE_SEMANTIC_A (0/1), MOSAIC_EDGE_SEMANTIC_MIN_SIM, MOSAIC_EDGE_SEMANTIC_MAX_PAIRS,
    MOSAIC_EDGE_PREREQ_LLM (0/1), MOSAIC_EDGE_PREREQ_MAX_PAIRS, MOSAIC_EDGE_PREREQ_BATCH
    """

    semantic_a_enabled: bool
    semantic_min_similarity: float
    semantic_max_pairs: int
    semantic_min_text_len: int
    prerequisite_llm_enabled: bool
    prerequisite_min_similarity: float
    prerequisite_max_pairs: int
    prerequisite_batch_size: int

    def enabled_summary(self) -> str:
        parts: list[str] = []
        if self.semantic_a_enabled:
            parts.append("semantic_bge")
        if self.prerequisite_llm_enabled:
            parts.append("prerequisite_llm")
        return ",".join(parts) or "none"


def _cfg_bool(val: str, default: bool) -> bool:
    s = (val or "").strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def get_edge_construction_config() -> EdgeConstructionConfig:
    cp = load_api_config()
    d_sem_a, d_sem_sim, d_sem_max, d_sem_len = True, 0.55, 2000, 24
    d_pre_llm, d_pre_sim, d_pre_max, d_pre_batch = False, 0.22, 64, 6
    if cp.has_section("EDGE"):
        sec = cp["EDGE"]
        semantic_a = _cfg_bool(sec.get("semantic_a_enabled", "true"), d_sem_a)
        sem_sim = float(sec.get("semantic_min_similarity", str(d_sem_sim)))
        sem_max = int(sec.get("semantic_max_pairs", str(d_sem_max)))
        sem_len = int(sec.get("semantic_min_text_len", str(d_sem_len)))
        pre_llm = _cfg_bool(sec.get("prerequisite_llm_enabled", "false"), d_pre_llm)
        pre_sim = float(sec.get("prerequisite_min_similarity", str(d_pre_sim)))
        pre_max = int(sec.get("prerequisite_max_pairs", str(d_pre_max)))
        pre_batch = int(sec.get("prerequisite_batch_size", str(d_pre_batch)))
    else:
        semantic_a, sem_sim, sem_max, sem_len = d_sem_a, d_sem_sim, d_sem_max, d_sem_len
        pre_llm, pre_sim, pre_max, pre_batch = d_pre_llm, d_pre_sim, d_pre_max, d_pre_batch

    if os.environ.get("MOSAIC_EDGE_SEMANTIC_A", "").strip():
        semantic_a = _cfg_bool(os.environ["MOSAIC_EDGE_SEMANTIC_A"], semantic_a)
    if os.environ.get("MOSAIC_EDGE_SEMANTIC_MIN_SIM", "").strip():
        sem_sim = float(os.environ["MOSAIC_EDGE_SEMANTIC_MIN_SIM"])
    if os.environ.get("MOSAIC_EDGE_SEMANTIC_MAX_PAIRS", "").strip():
        sem_max = int(os.environ["MOSAIC_EDGE_SEMANTIC_MAX_PAIRS"])
    if os.environ.get("MOSAIC_EDGE_PREREQ_LLM", "").strip():
        pre_llm = _cfg_bool(os.environ["MOSAIC_EDGE_PREREQ_LLM"], pre_llm)
    if os.environ.get("MOSAIC_EDGE_PREREQ_MAX_PAIRS", "").strip():
        pre_max = int(os.environ["MOSAIC_EDGE_PREREQ_MAX_PAIRS"])
    if os.environ.get("MOSAIC_EDGE_PREREQ_BATCH", "").strip():
        pre_batch = int(os.environ["MOSAIC_EDGE_PREREQ_BATCH"])

    return EdgeConstructionConfig(
        semantic_a_enabled=semantic_a,
        semantic_min_similarity=sem_sim,
        semantic_max_pairs=max(0, sem_max),
        semantic_min_text_len=max(0, sem_len),
        prerequisite_llm_enabled=pre_llm,
        prerequisite_min_similarity=pre_sim,
        prerequisite_max_pairs=max(0, pre_max),
        prerequisite_batch_size=max(1, pre_batch),
    )


@dataclass(frozen=True)
class QueryRetrievalConfig:
    """轨 D：查询时 TF-IDF + BGE 融合（docs/optimization.md §7）。"""

    bge_lambda: float
    bge_max_encode_instances: int
    max_context_chars: int


def get_query_retrieval_config() -> QueryRetrievalConfig:
    lam, mx, mcc = 0.0, 600, 0
    cp = load_api_config()
    if cp.has_section("QUERY"):
        sec = cp["QUERY"]
        lam = float(sec.get("bge_lambda", "0"))
        mx = int(sec.get("bge_max_encode_instances", "600"))
        mcc = int(sec.get("max_context_chars", "0"))
    el = os.environ.get("MOSAIC_QUERY_BGE_LAMBDA", "").strip()
    if el:
        lam = float(el)
    em = os.environ.get("MOSAIC_QUERY_BGE_MAX_ENCODE", "").strip()
    if em.isdigit():
        mx = int(em)
    ec = os.environ.get("MOSAIC_QUERY_MAX_CONTEXT_CHARS", "").strip()
    if ec.isdigit():
        mcc = int(ec)
    lam = max(0.0, min(1.0, lam))
    mx = max(32, min(50000, mx))
    mcc = max(0, mcc)
    return QueryRetrievalConfig(bge_lambda=lam, bge_max_encode_instances=mx, max_context_chars=mcc)


def get_control_profile() -> str:
    """轨 C：``static`` | ``evolving`` | ``memory_only``（占位，供 Runner 读取）。"""
    env = os.environ.get("MOSAIC_CONTROL_PROFILE", "").strip().lower()
    if env in ("static", "evolving", "memory_only"):
        return env
    cp = load_api_config()
    if cp.has_section("CONTROL"):
        v = cp.get("CONTROL", "profile", fallback="static").strip().lower()
        if v in ("static", "evolving", "memory_only"):
            return v
    return "static"


def get_ncs_trace_path() -> str:
    """NCS JSONL 路径：``[NCS] trace_jsonl`` 或 ``MOSAIC_NCS_TRACE_JSONL``。"""
    p = os.environ.get("MOSAIC_NCS_TRACE_JSONL", "").strip()
    if p:
        return p
    cp = load_api_config()
    if cp.has_section("NCS"):
        return cp.get("NCS", "trace_jsonl", fallback="").strip()
    return ""
