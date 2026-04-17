from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from glob import glob

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.data.graph import ClassGraph
from src.assist import (
    conv_message_splitter,
    message_splitter,
    read_to_file_json,
)
from src.config_loader import get_control_profile
from src.logger import log_pipeline_event, setup_logger
from src.llm.telemetry import (
    dump_build_metrics_file,
    get_llm_counters,
    llm_phase_scope,
    reset_build_llm_counters,
)
from src.telemetry.ingest_log import append_ingest_record

_logger = setup_logger("save")

_CHECKPOINT_META_VERSION = 1


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_meta_path(checkpoint_pkl: str) -> str:
    base = checkpoint_pkl
    if base.lower().endswith(".pkl"):
        base = base[:-4]
    return base + ".meta.json"


def _atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def write_build_checkpoint(
    memory: ClassGraph,
    checkpoint_pkl: str,
    *,
    batches_done: int,
    total_batches: int,
    source_sha256: str,
    conv_name: str,
    build_mode: str,
) -> None:
    """每批完成后写入完整 ClassGraph + meta（与对话 JSON 指纹绑定，防串档）。"""
    meta = {
        "version": _CHECKPOINT_META_VERSION,
        "batches_done": batches_done,
        "total_batches": total_batches,
        "source_sha256": source_sha256,
        "conv_name": conv_name,
        "build_mode": build_mode,
    }
    tmp_pkl = checkpoint_pkl + ".tmp"
    d = os.path.dirname(os.path.abspath(checkpoint_pkl)) or "."
    os.makedirs(d, exist_ok=True)
    with open(tmp_pkl, "wb") as f:
        pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_pkl, checkpoint_pkl)
    _atomic_write_text(_checkpoint_meta_path(checkpoint_pkl), json.dumps(meta, ensure_ascii=False, indent=2))


def load_build_checkpoint(
    checkpoint_pkl: str,
    *,
    source_sha256: str,
    total_batches: int,
    conv_name: str,
    build_mode: str,
) -> tuple[ClassGraph, int]:
    meta_path = _checkpoint_meta_path(checkpoint_pkl)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("version", 0)) != _CHECKPOINT_META_VERSION:
        raise ValueError(f"断点 meta 版本不兼容: {meta.get('version')!r}，期望 {_CHECKPOINT_META_VERSION}")
    if meta.get("source_sha256") != source_sha256:
        raise ValueError(
            "断点与当前对话 JSON 指纹不一致（文件已变或路径不同），请删除断点文件后全量重跑，"
            f"checkpoint={checkpoint_pkl}"
        )
    if int(meta.get("total_batches", -1)) != total_batches:
        raise ValueError(
            f"断点总批次数 {meta.get('total_batches')} 与当前划分 {total_batches} 不一致，请删除断点后重跑"
        )
    if meta.get("conv_name") != conv_name:
        raise ValueError(f"断点 conv_name={meta.get('conv_name')!r} 与当前 {conv_name!r} 不一致")
    if meta.get("build_mode") != build_mode:
        raise ValueError(f"断点 build_mode={meta.get('build_mode')!r} 与当前 {build_mode!r} 不一致")
    batches_done = int(meta["batches_done"])
    if batches_done < 0 or batches_done > total_batches:
        raise ValueError(f"断点 batches_done 非法: {batches_done}")
    with open(checkpoint_pkl, "rb") as f:
        memory = pickle.load(f)
    if not isinstance(memory, ClassGraph):
        raise TypeError(f"断点 pickle 类型错误: {type(memory)}")
    if batches_done >= total_batches:
        return memory, total_batches
    return memory, batches_done


def remove_build_checkpoint(checkpoint_pkl: str | None) -> None:
    if not checkpoint_pkl:
        return
    for p in (checkpoint_pkl, _checkpoint_meta_path(checkpoint_pkl)):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _progress_bar(iterable, total: int, desc: str):
    """构图主 stdout：批次进度条（与主程序按批存储一致）。"""
    if tqdm is None or total <= 0:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="batch",
        smoothing=0.08,
        mininterval=0.3,
    )


def _twrite(msg: str) -> None:
    """在 tqdm 运行时安全输出一行摘要（不破坏进度条）。"""
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def _distinct_canonical_instance_names(memory: ClassGraph) -> int:
    """B-4：实例级 distinct canonical（instance_name 优先，否则类名），与 EntityGraph canonical_name 口径一致。"""
    names: set[str] = set()
    for n in memory.graph.nodes:
        cname = (getattr(n, "class_name", None) or "").strip()
        for inst in getattr(n, "_instances", None) or []:
            if not isinstance(inst, dict):
                continue
            raw = (inst.get("instance_name") or cname or "").strip()
            if raw:
                names.add(raw)
    return len(names)


def _build_metrics_path(memory: ClassGraph) -> str:
    """构图 LLM 统计 JSON：与 graph_snapshots 同级的 artifacts 目录优先。"""
    base = getattr(memory, "_graph_save_dir", None) or os.path.join(os.getcwd(), "results", "graph")
    ab = os.path.abspath(base)
    parent = os.path.dirname(ab)
    if os.path.basename(ab) == "graph_snapshots" and parent:
        return os.path.join(parent, "build_llm_metrics.json")
    return os.path.join(ab, "build_llm_metrics.json")


def _write_build_llm_metrics(memory: ClassGraph, conv_name: str, build_mode: str, wall_s: float) -> None:
    tel = getattr(memory, "sense_class_telemetry_cumulative", None) or {}
    dump_build_metrics_file(
        _build_metrics_path(memory),
        extra={
            "conversation_id": conv_name,
            "build_mode": build_mode,
            "wall_clock_build_s": round(wall_s, 3),
            "sense_class_telemetry_cumulative": dict(tel) if isinstance(tel, dict) else {},
        },
    )


def _log_construction_telemetry_summary(
    memory: ClassGraph, total_msgs: int, *, build_mode: str
) -> None:
    """B-3/B-4：构图结束汇总 sense_classes 遥测、类名数、实例 canonical 多样性。"""
    tel = getattr(memory, "sense_class_telemetry_cumulative", None) or {}
    distinct_class_names = len({(getattr(n, "class_name", None) or "").strip() for n in memory.graph.nodes})
    distinct_canon = _distinct_canonical_instance_names(memory)
    mode = (build_mode or "hybrid").strip().lower()
    _logger.info(
        "构图遥测(B): mode=%s matched_labels=%s unmatched_frags=%s llm_new_inv=%s llm_json_fail=%s "
        "distinct_class_names=%d distinct_instance_canonical=%d messages=%d",
        mode,
        tel.get("tfidf_matched_fragment_labels", 0),
        tel.get("tfidf_unmatched_fragments", 0),
        tel.get("llm_new_class_invocations", 0),
        tel.get("llm_new_class_json_failures", 0),
        distinct_class_names,
        distinct_canon,
        total_msgs,
    )
    _twrite(
        f"构图遥测(mode={mode}): TF-IDF 命中标签累计="
        f"{tel.get('tfidf_matched_fragment_labels', 0)}, 未匹配片段累计="
        f"{tel.get('tfidf_unmatched_fragments', 0)}, LLM 新类调用="
        f"{tel.get('llm_new_class_invocations', 0)}, JSON 解析失败="
        f"{tel.get('llm_new_class_json_failures', 0)}, 不同 class_name 数={distinct_class_names}, "
        f"实例 distinct canonical 名={distinct_canon}"
    )
    # B-4 验收（optimization.md §5 B-4）：hash_only 不得触发 sense_classes 新类 LLM
    if mode == "hash_only":
        inv = int(tel.get("llm_new_class_invocations", 0) or 0)
        if inv != 0:
            _logger.error(
                "B-4 失败: hash_only 下 llm_new_class_invocations 应为 0，实际=%d（构图 LLM 泄漏）",
                inv,
            )
            _twrite(f"错误 B-4: hash_only 下不应有 LLM 新类调用，实际 {inv} 次")
    elif mode == "hybrid":
        if distinct_canon < 3:
            _logger.warning(
                "B-4 提示: hybrid 下实例 distinct canonical 仅 %d，验收目标常为 ≥3（视对话与标定可调整）",
                distinct_canon,
            )
            _twrite(
                f"警告 B-4: hybrid 下实例 distinct canonical={distinct_canon}，若长期低于 3 请检查数据或路由"
            )


def _conversation_message_totals(result: list) -> tuple[int, list[int]]:
    """返回 (全部分组内对话消息总条数, 每批条数列表)。"""
    sizes = [len(batch) for batch, _ in result]
    return sum(sizes), sizes


def _write_construction_progress(
    current_1based: int,
    total_batches: int,
    *,
    messages_done: int | None = None,
    total_messages: int | None = None,
) -> None:
    """
    若设置环境变量 MOSAIC_PROGRESS_FILE，写入批次与（可选）消息级进度。
    messages_done: 已完成处理的对话消息条数（累计）
    """
    path = os.environ.get("MOSAIC_PROGRESS_FILE")
    if not path or total_batches <= 0:
        return
    try:
        pct_b = 100.0 * current_1based / total_batches
        lines = [f"batches {current_1based}/{total_batches} ({pct_b:.1f}%)"]
        if total_messages is not None and total_messages > 0 and messages_done is not None:
            pct_m = 100.0 * messages_done / total_messages
            rem = total_messages - messages_done
            lines.append(f"messages {messages_done}/{total_messages} ({pct_m:.1f}%) remaining {rem}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def run_build_batch(
    memory: ClassGraph,
    data: list,
    context: list,
    *,
    build_mode: str,
) -> ClassGraph:
    """
    B-3：单批构图入口；build_mode 为 ``hybrid`` 或 ``hash_only``（与 CLI / MOSAIC_BUILD_EFFECTIVE 一致）。
    """
    mode = (build_mode or "hybrid").strip().lower()
    if mode not in ("hybrid", "hash_only"):
        mode = "hybrid"
    if mode == "hash_only":
        relevant_class_messages, new_class_messages = memory.sense_classes(
            data, context, use_llm_for_new=False
        )
        processed_classes = memory.process_relevant_class_instances(
            relevant_class_messages, use_hash=True
        )
        added_class_nodes = memory.add_classnodes(new_class_messages, use_hash=True)
    else:
        # Hybrid: LLM only for new class creation; matched classes use hash
        # for instance processing to reduce LLM calls significantly.
        relevant_class_messages, new_class_messages = memory.sense_classes(data, context)
        processed_classes = memory.process_relevant_class_instances(
            relevant_class_messages, use_hash=True
        )
        added_class_nodes = memory.add_classnodes(new_class_messages)
    _logger.debug("relevant_class_messages: %s; new_class_messages: %s", relevant_class_messages, new_class_messages)
    _logger.debug("processed_classes: %s", processed_classes)
    _logger.debug("added_class_nodes: %s", added_class_nodes)
    memory.update_class_relationships(data, processed_classes, added_class_nodes)
    return memory


def _process_data_truncation(memory: ClassGraph, data: list, context: list) -> ClassGraph:
    eff = os.environ.get("MOSAIC_BUILD_EFFECTIVE", "hybrid")
    return run_build_batch(memory, data, context, build_mode=eff)


def _process_data_truncation_hash(memory: ClassGraph, data: list, context: list) -> ClassGraph:
    """仅用 TF-IDF/hash 构图，不调用 LLM（未匹配片段归入 Unclassified，实例用文本拼接）。"""
    return run_build_batch(memory, data, context, build_mode="hash_only")


#error compounding
def save_error(data):
    # 初始化memory
    memory = ClassGraph()

    # 调用message_splitter函数
    result = message_splitter(data)

    message_labels = []
    for i, (batch, context) in enumerate(result):
        message_labels.extend(batch)

    # 需要提前建立好message_labes数组，因为要记录下error涉及到哪些信息label,去查找具体的信息
    memory.message_labels = message_labels

    total = len(result)
    total_msgs, _ = _conversation_message_totals(result)
    done_before = 0
    pbar = _progress_bar(result, total, "构图(error)")
    for i, (batch, context) in enumerate(pbar):
        n = len(batch)
        k, pct_b = i + 1, 100.0 * (i + 1) / total
        done_after = done_before + n
        pct_m = 100.0 * done_after / total_msgs if total_msgs else 0.0
        rem = total_msgs - done_after
        _logger.debug(
            "构图进度: 批 [%d/%d] %.1f%% | 对话消息 本批 %d 条，此前已处理 %d/%d，本批完成后 %d/%d (%.1f%%)，尚余 %d 条",
            k, total, pct_b, n, done_before, total_msgs, done_after, total_msgs, pct_m, rem,
        )
        _logger.debug("当前消息: %s; 上文前三条: %s", batch, context[:3] if context else [])
        memory = _process_data_truncation(memory, batch, context)
        log_pipeline_event(f"build batch {k}/{total} messages_done={done_after}/{total_msgs} classes={len(memory.graph.nodes)}")
        _write_construction_progress(k, total, messages_done=done_after, total_messages=total_msgs)
        done_before = done_after
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                msg=f"{done_after}/{total_msgs}",
                n_cls=len(memory.graph.nodes),
            )
    if total:
        _write_construction_progress(total, total, messages_done=total_msgs, total_messages=total_msgs)
    try:
        memory.sweep_cross_class_cooccurrence_edges()
    except Exception as e:
        _logger.warning("Cross-class edge sweep (error) failed: %s", e)
    try:
        memory.sweep_uncovered_messages(result)
    except Exception as e:
        _logger.warning("Uncovered message sweep (error) failed: %s", e)
    try:
        memory.enrich_dual_graph_edges_post_build()
    except Exception as e:
        _logger.warning("构图(error) 后双图增强失败: %s", e)
    return memory



def save(
    data,
    conv_name,
    *,
    checkpoint_path: str | None = None,
    resume: bool = False,
    source_path: str | None = None,
):
    reset_build_llm_counters()
    t_build0 = time.perf_counter()
    with llm_phase_scope("build"):
        result = conv_message_splitter(data)

        total = len(result)
        total_msgs, batch_sizes = _conversation_message_totals(result)
        eff = os.environ.get("MOSAIC_BUILD_EFFECTIVE", "hybrid")
        source_sha256 = _sha256_file(source_path) if source_path and os.path.isfile(source_path) else ""

        start_batch = 0
        memory: ClassGraph | None = None
        if resume and checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                memory, start_batch = load_build_checkpoint(
                    checkpoint_path,
                    source_sha256=source_sha256,
                    total_batches=total,
                    conv_name=conv_name,
                    build_mode=eff,
                )
                prefix_msgs = sum(batch_sizes[:start_batch]) if start_batch > 0 else 0
                _logger.info(
                    "构图断点续跑: 已从批次 %d/%d 恢复（已完成对话消息约 %d/%d）",
                    start_batch,
                    total,
                    prefix_msgs,
                    total_msgs,
                )
                _twrite(
                    f"构图断点续跑: 从第 {start_batch + 1} 批继续（共 {total} 批），"
                    f"已处理消息约 {prefix_msgs}/{total_msgs}"
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
                _logger.warning("加载构图断点失败，将从头开始: %s", e)
                _twrite(f"构图断点无效或损坏，从头开始: {e}")
                memory = None
                start_batch = 0
        elif resume and checkpoint_path:
            _twrite("指定了 --resume 但断点文件不存在，从头构图")

        if memory is None:
            memory = ClassGraph()
        memory.filepath = conv_name

        _logger.info(
            "构图开始: BUILD_EFFECTIVE=%s, 共 %d 个批次，合计 %d 条对话消息",
            eff,
            total,
            total_msgs,
        )
        if start_batch == 0:
            _twrite(
                f"构图开始: 模式 {eff}，{total} 批次，合计 {total_msgs} 条对话消息"
                f"（每批最多 10 条，与主程序 conv_message_splitter 一致）"
            )
        log_pipeline_event(
            f"build start conv={conv_name} mode={eff} batches={total} messages={total_msgs}"
        )

        done_before = sum(batch_sizes[:start_batch]) if start_batch > 0 else 0
        tail = list(enumerate(result))[start_batch:]
        pbar = _progress_bar(tail, max(total - start_batch, 0), "构图")
        for i, (batch, context) in pbar:
            n = len(batch)
            k, pct_b = i + 1, 100.0 * (i + 1) / total
            done_after = done_before + n
            pct_m = 100.0 * done_after / total_msgs if total_msgs else 0.0
            rem = total_msgs - done_after
            _logger.debug(
                "构图进度: 批 [%d/%d] %.1f%% | 对话消息 本批 %d 条，此前已处理 %d/%d，本批完成后 %d/%d (%.1f%%)，尚余 %d 条 → 当前类数: %d",
                k, total, pct_b, n, done_before, total_msgs, done_after, total_msgs, pct_m, rem,
                len(memory.graph.nodes),
            )
            _logger.debug("当前消息: %s; 上文: %s", batch, context[:3] if context else [])
            memory = _process_data_truncation(memory, batch, context)
            log_pipeline_event(
                f"build batch {k}/{total} messages_done={done_after}/{total_msgs} classes={len(memory.graph.nodes)}"
            )
            _write_construction_progress(k, total, messages_done=done_after, total_messages=total_msgs)
            done_before = done_after
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(msgs=f"{done_after}/{total_msgs}", n_cls=len(memory.graph.nodes))
            if checkpoint_path and source_sha256:
                write_build_checkpoint(
                    memory,
                    checkpoint_path,
                    batches_done=k,
                    total_batches=total,
                    source_sha256=source_sha256,
                    conv_name=conv_name,
                    build_mode=eff,
                )
        if total:
            _write_construction_progress(total, total, messages_done=total_msgs, total_messages=total_msgs)
        try:
            sweep_stats = memory.sweep_cross_class_cooccurrence_edges()
            _logger.info("Cross-class edge sweep: %s", sweep_stats)
        except Exception as e:
            _logger.warning("Cross-class edge sweep failed: %s", e)
        try:
            uncov_stats = memory.sweep_uncovered_messages(result)
            _logger.info("Uncovered message sweep: %s", uncov_stats)
            if uncov_stats.get("instances_created", 0) > 0:
                _twrite(
                    f"Uncovered message sweep: {uncov_stats['instances_created']} hash instances "
                    f"created for {uncov_stats['uncovered']} dropped messages"
                )
        except Exception as e:
            _logger.warning("Uncovered message sweep failed: %s", e)
        try:
            memory.enrich_dual_graph_edges_post_build()
        except Exception as e:
            _logger.warning("构图后双图增强（E_A/E_P 后处理）失败: %s", e)
        _logger.info("构图完成: 共 %d 个类，累计处理对话消息 %d 条", len(memory.graph.nodes), total_msgs)
        _twrite(f"构图完成: {len(memory.graph.nodes)} 个类，累计 {total_msgs} 条对话消息")
        _log_construction_telemetry_summary(memory, total_msgs, build_mode=eff)
        _write_build_llm_metrics(memory, conv_name, eff, time.perf_counter() - t_build0)
        bc = int(get_llm_counters().get("build_calls", 0))
        log_pipeline_event(
            f"build done conv={conv_name} classes={len(memory.graph.nodes)} llm_http_calls_build={bc}"
        )
        try:
            append_ingest_record(
                conversation_id=conv_name,
                wall_s=time.perf_counter() - t_build0,
                memory=memory,
                llm_calls=int(get_llm_counters().get("build_calls", 0)),
                json_failures=int(
                    getattr(memory, "sense_class_telemetry_cumulative", {}).get(
                        "llm_new_class_json_failures", 0
                    )
                ),
                extra={"build_mode": eff, "control_profile": get_control_profile()},
            )
        except Exception as exc:
            _logger.debug("ingest_jsonl: %s", exc)
        return memory


def save_hash(
    data,
    conv_name,
    graph_save_dir=None,
    final_graph_path=None,
    final_tags_path=None,
    *,
    checkpoint_path: str | None = None,
    resume: bool = False,
    source_path: str | None = None,
):
    """
    仅用 TF-IDF/hash 构图，不调用 LLM。适合无 API 或基线实验。
    返回构建好的 ClassGraph。
    若提供 final_graph_path / final_tags_path，则最后将图 pickle 到该路径，并用 TF-IDF 生成 tags 写入 final_tags_path。
    """
    reset_build_llm_counters()
    t_build0 = time.perf_counter()
    with llm_phase_scope("build"):
        result = conv_message_splitter(data)
        total = len(result)
        total_msgs, batch_sizes = _conversation_message_totals(result)
        eff = os.environ.get("MOSAIC_BUILD_EFFECTIVE", "hash_only")
        source_sha256 = _sha256_file(source_path) if source_path and os.path.isfile(source_path) else ""

        start_batch = 0
        memory: ClassGraph | None = None
        if resume and checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                memory, start_batch = load_build_checkpoint(
                    checkpoint_path,
                    source_sha256=source_sha256,
                    total_batches=total,
                    conv_name=conv_name,
                    build_mode=eff,
                )
                prefix_msgs = sum(batch_sizes[:start_batch]) if start_batch > 0 else 0
                _logger.info(
                    "构图断点续跑(hash): 已从批次 %d/%d 恢复（已完成对话消息约 %d/%d）",
                    start_batch,
                    total,
                    prefix_msgs,
                    total_msgs,
                )
                _twrite(
                    f"构图断点续跑(hash): 从第 {start_batch + 1} 批继续（共 {total} 批），"
                    f"已处理消息约 {prefix_msgs}/{total_msgs}"
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
                _logger.warning("加载构图断点失败，将从头开始: %s", e)
                _twrite(f"构图断点无效或损坏，从头开始: {e}")
                memory = None
                start_batch = 0
        elif resume and checkpoint_path:
            _twrite("指定了 --resume 但断点文件不存在，从头构图(hash)")

        if memory is None:
            memory = ClassGraph()
        memory.filepath = conv_name
        if graph_save_dir is not None:
            memory._graph_save_dir = graph_save_dir

        _logger.info(
            "构图开始(hash): BUILD_EFFECTIVE=%s, 共 %d 个批次，合计 %d 条对话消息",
            eff,
            total,
            total_msgs,
        )
        if start_batch == 0:
            _twrite(f"构图开始(hash): 模式 {eff}，{total} 批次，合计 {total_msgs} 条对话消息")

        done_before = sum(batch_sizes[:start_batch]) if start_batch > 0 else 0
        tail = list(enumerate(result))[start_batch:]
        pbar = _progress_bar(tail, max(total - start_batch, 0), "构图(hash)")
        for i, (batch, context) in pbar:
            n = len(batch)
            k, pct_b = i + 1, 100.0 * (i + 1) / total
            done_after = done_before + n
            pct_m = 100.0 * done_after / total_msgs if total_msgs else 0.0
            rem = total_msgs - done_after
            _logger.debug(
                "构图进度: 批 [%d/%d] %.1f%% | 对话消息 本批 %d 条，此前已处理 %d/%d，本批完成后 %d/%d (%.1f%%)，尚余 %d 条 → 当前类数: %d",
                k, total, pct_b, n, done_before, total_msgs, done_after, total_msgs, pct_m, rem,
                len(memory.graph.nodes),
            )
            memory = _process_data_truncation_hash(memory, batch, context)
            _write_construction_progress(k, total, messages_done=done_after, total_messages=total_msgs)
            done_before = done_after
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(msgs=f"{done_after}/{total_msgs}", n_cls=len(memory.graph.nodes))
            if checkpoint_path and source_sha256:
                write_build_checkpoint(
                    memory,
                    checkpoint_path,
                    batches_done=k,
                    total_batches=total,
                    source_sha256=source_sha256,
                    conv_name=conv_name,
                    build_mode=eff,
                )
        if total:
            _write_construction_progress(total, total, messages_done=total_msgs, total_messages=total_msgs)
        try:
            sweep_stats = memory.sweep_cross_class_cooccurrence_edges()
            _logger.info("Cross-class edge sweep (hash): %s", sweep_stats)
        except Exception as e:
            _logger.warning("Cross-class edge sweep (hash) failed: %s", e)
        try:
            uncov_stats = memory.sweep_uncovered_messages(result)
            _logger.info("Uncovered message sweep (hash): %s", uncov_stats)
            if uncov_stats.get("instances_created", 0) > 0:
                _twrite(
                    f"Uncovered message sweep (hash): {uncov_stats['instances_created']} hash instances "
                    f"created for {uncov_stats['uncovered']} dropped messages"
                )
        except Exception as e:
            _logger.warning("Uncovered message sweep (hash) failed: %s", e)
        try:
            memory.enrich_dual_graph_edges_post_build()
        except Exception as e:
            _logger.warning("构图后双图增强（E_A/E_P 后处理）失败: %s", e)
        _logger.info("构图完成: 共 %d 个类，累计处理对话消息 %d 条", len(memory.graph.nodes), total_msgs)
        _twrite(f"构图完成: {len(memory.graph.nodes)} 个类，累计 {total_msgs} 条对话消息")
        _log_construction_telemetry_summary(memory, total_msgs, build_mode=eff)
        _write_build_llm_metrics(memory, conv_name, eff, time.perf_counter() - t_build0)
        if final_graph_path:
            os.makedirs(os.path.dirname(os.path.abspath(final_graph_path)) or ".", exist_ok=True)
            with open(final_graph_path, "wb") as f:
                pickle.dump(memory, f, protocol=pickle.HIGHEST_PROTOCOL)
            _logger.info("图已保存到 %s（完整 ClassGraph，含 edges/G_p/G_a）", final_graph_path)
        if final_tags_path and hasattr(memory, "generate_tags_tfidf"):
            memory.generate_tags_tfidf(final_tags_path)
        try:
            append_ingest_record(
                conversation_id=conv_name,
                wall_s=time.perf_counter() - t_build0,
                memory=memory,
                llm_calls=int(get_llm_counters().get("build_calls", 0)),
                json_failures=int(
                    getattr(memory, "sense_class_telemetry_cumulative", {}).get(
                        "llm_new_class_json_failures", 0
                    )
                ),
                extra={"build_mode": eff, "control_profile": get_control_profile()},
            )
        except Exception as exc:
            _logger.debug("ingest_jsonl: %s", exc)
        return memory


def process_single_conv(file_path):
    """处理单个conv文件"""
    _logger.info("处理文件: %s", file_path)

    # 提取conv名称（例如从"locomo_conv9.json"中提取"conv9"）
    file_name = os.path.basename(file_path)
    conv_name = file_name.replace("locomo_", "").replace(".json", "")

    # 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 保存处理结果
    return save(data, conv_name)


def process_all_convs():
    """处理目录下所有符合条件的conv文件"""
    base_dir = os.environ.get("MOSAIC_DATA_DIR", os.path.join(os.path.dirname(__file__), ".."))
    conv_dir = os.path.join(base_dir, "locomo results", "conv")
    file_pattern = os.path.join(conv_dir, "locomo_conv*.json")
    conv_files = sorted(glob(file_pattern))



    if not conv_files:
        _logger.info("未找到匹配的文件: %s", file_pattern)
        return

    _logger.info("找到 %d 个 conv 文件，开始构图", len(conv_files))

    # 处理每个文件
    results = {}
    for file_path in conv_files:

        memory = process_single_conv(file_path)
        file_name = os.path.basename(file_path)
        results[file_name] = memory

    return results
if __name__ == '__main__':
    #对话准确率数据的测试
    process_all_convs()
