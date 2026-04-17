#!/usr/bin/env python3
"""Verify HaluMem compatibility additions to Mosaic codebase."""
import sys
import os
os.chdir(os.path.join(os.path.dirname(__file__), "mosaic"))
sys.path.insert(0, ".")

from src.data.graph import ClassGraph

g = ClassGraph()
assert hasattr(g, "get_all_instances_state"), "Missing get_all_instances_state"
assert hasattr(g, "record_current_state"), "Missing record_current_state"
assert hasattr(g, "get_new_instances_since_state"), "Missing get_new_instances_since_state"
assert hasattr(g, "last_session_instance_states"), "Missing last_session_instance_states"
assert hasattr(g, "all_instances"), "Missing all_instances"

state = g.get_all_instances_state()
assert isinstance(state, set)
g.record_current_state()
new = g.get_new_instances_since_state(state)
assert isinstance(new, list)
print("graph.py: All HaluMem methods OK")

from src.assist import serialize_instance_eval, llm_request_for_json
assert callable(serialize_instance_eval)
assert callable(llm_request_for_json)

result = serialize_instance_eval([])
assert result == []
result = serialize_instance_eval([{
    "instance_name": "test",
    "attributes": {
        "a": {"description": "d", "value": "v", "occurred": "o", "recorded_at": "r"}
    },
    "uninstance_field": "extra",
}])
assert len(result) == 1
assert "test" in result[0]
assert "extra" in result[0]
print("assist.py: serialize_instance_eval OK")

from src.prompts_en import PROMPT_QUERY_TEMPLATE_EVAL
assert "QUESTION" in PROMPT_QUERY_TEMPLATE_EVAL
assert "INFORMATION" in PROMPT_QUERY_TEMPLATE_EVAL
print("prompts_en.py: PROMPT_QUERY_TEMPLATE_EVAL OK")

print("\nALL CHECKS PASSED")
