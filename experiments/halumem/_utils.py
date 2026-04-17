# -*- coding: utf-8 -*-
"""
Shared utilities for HaluMem experiments: path resolution, mosaic environment
setup, JSONL helpers, and dataset access.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import re
import sys
import time
from string import Template

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MOSAIC_DIR = os.path.join(PROJECT_ROOT, "mosaic")
MOSAIC_SRC = os.path.join(MOSAIC_DIR, "src")
DATASET_HALUMEM_DIR = os.path.join(PROJECT_ROOT, "dataset", "halumem", "data")

# Paths relative to each sub-experiment
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_mosaic_path():
    """Add mosaic and mosaic/src to sys.path so ``from src.…`` works."""
    for p in (MOSAIC_DIR, MOSAIC_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# JSONL / JSON I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(path: str, data, indent: int = 2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# HaluMem-specific helpers
# ---------------------------------------------------------------------------
def extract_user_name(persona_info: str) -> str:
    m = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if m:
        return m.group(1).strip()
    raise ValueError("No user name found in persona_info")


def build_graph_for_user(user_data: dict, save_path: str, *, batch_size: int = 10):
    """
    Build a MOSAIC graph for one HaluMem user.

    Iterates sessions → dialogue, batches messages in groups of *batch_size*,
    calls the mosaic build pipeline, and tracks new instances per session
    (needed for integrity/accuracy evaluation).

    Returns ``(memory, new_user_data)`` where *new_user_data* is augmented
    with ``extracted_memories``, timing, and QA search results.
    """
    setup_mosaic_path()
    from src.data.graph import ClassGraph
    from src.save import run_build_batch
    from src.assist import load_graphs, serialize_instance_eval

    user_name = extract_user_name(user_data["persona_info"])
    sessions = user_data["sessions"]

    graph_dir = os.path.join(save_path, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    graph_file = os.path.join(graph_dir, f"{user_name}_graph.pkl")

    memory = ClassGraph()
    memory.filepath = graph_dir
    if os.path.exists(graph_file):
        try:
            memory.graph = load_graphs(graph_file)
        except Exception:
            pass

    new_user_data: dict = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": [],
    }

    global_message_count = 1
    labeled_dialogue: list[dict] = []

    for session in sessions:
        new_session: dict = {
            "memory_points": session["memory_points"],
            "dialogue": session["dialogue"],
        }

        message_buffer: list[dict] = []
        previous_batch_last: list[dict] = []
        instance_list: list = []
        build_time_ms = 0.0

        for turn in session["dialogue"]:
            msg_text = (
                f"Dialogue time:{turn['timestamp']}; role:{turn['role']}; "
                f"content:{turn['content']};"
            )
            labeled_turn = {"message": msg_text, "label": global_message_count}
            labeled_dialogue.append(labeled_turn)
            message_buffer.append(labeled_turn)

            if len(message_buffer) == batch_size:
                old_state = memory.get_all_instances_state()
                memory = run_build_batch(
                    memory, message_buffer, previous_batch_last, build_mode="hash_only"
                )
                memory.record_current_state()
                new_insts = memory.get_new_instances_since_state(old_state)
                instance_list.append(serialize_instance_eval(new_insts))
                build_time_ms += 0  # placeholder; real timing in full run
                previous_batch_last = [message_buffer[-1]]
                message_buffer = []
            global_message_count += 1

        # flush remaining
        if message_buffer:
            old_state = memory.get_all_instances_state()
            memory = run_build_batch(
                memory, message_buffer, previous_batch_last, build_mode="hash_only"
            )
            memory.record_current_state()
            new_insts = memory.get_new_instances_since_state(old_state)
            instance_list.append(serialize_instance_eval(new_insts))

        memory.message_labels = labeled_dialogue

        if session.get("is_generated_qa_session", False):
            new_session["is_generated_qa_session"] = True
            new_session.pop("dialogue", None)
            new_session.pop("memory_points", None)
            new_user_data["sessions"].append(new_session)
            continue

        new_session["extracted_memories"] = instance_list

        # search update memories
        for mp in new_session["memory_points"]:
            if mp["is_update"] == "False" or not mp.get("original_memories"):
                continue
            ctx, _ = _search_memory(memory, mp["memory_content"])
            mp["memories_from_system"] = ctx

        new_user_data["sessions"].append(new_session)

    # persist full ClassGraph (needed for _search_by_sub_hash with edges)
    with open(graph_file, "wb") as f:
        pickle.dump(memory, f)

    return memory, new_user_data


def query_qa_for_user(memory, new_user_data: dict):
    """
    For each QA question in the user's sessions, retrieve context and
    generate a system response.  Modifies *new_user_data* in-place.
    """
    setup_mosaic_path()
    from src.assist import llm_request_for_json
    from src.prompts_en import PROMPT_QUERY_TEMPLATE_EVAL

    for session in new_user_data["sessions"]:
        if session.get("is_generated_qa_session"):
            continue
        if "questions" not in session.get("dialogue", [{}])[0:0] and "questions" not in (
            session.get("memory_points", [{}])[0:0]
        ):
            # look in original session dict
            pass
        qs = session.get("questions")
        if not qs:
            # questions may be in the original data; caller should ensure
            continue

        new_questions = []
        for qa in qs:
            context_answer, dur_ms = _search_memory(memory, qa["question"])
            new_qa = copy.deepcopy(qa)
            new_qa["context"] = context_answer
            new_qa["search_duration_ms"] = dur_ms

            prompt = Template(PROMPT_QUERY_TEMPLATE_EVAL).substitute(
                INFORMATION=context_answer, QUESTION=qa["question"]
            )
            t0 = time.time()
            try:
                resp = llm_request_for_json(prompt)
                answer = resp.get("response", "")
            except Exception:
                answer = ""
            new_qa["system_response"] = answer
            new_qa["response_duration_ms"] = (time.time() - t0) * 1000
            new_questions.append(new_qa)
        session["questions"] = new_questions


def _search_memory(memory, query_str: str, top_k_class=10, top_k_instances=20):
    """Wrapper around ``memory._search_by_sub_hash``, returns (context_str, duration_ms)."""
    t0 = time.time()
    ret = memory._search_by_sub_hash(query_str, top_k_class, top_k_instances)
    dur = (time.time() - t0) * 1000
    # ret is (combined_str, trace_dict) in current mosaic
    if isinstance(ret, tuple):
        return ret[0], dur
    return ret, dur
