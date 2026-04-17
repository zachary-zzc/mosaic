"""
手稿级双图边构造提示（docs/optimization.md §5 B-3）：LLM 判断实体对是否存在 **非对称先决** 关系。

返回严格 JSON；与 ``parse_llm_json_object`` / json_object 模式配合使用。
"""
from __future__ import annotations

from string import Template

# batch: list of { "index": int, "u_id": str, "v_id": str, "u_text": str, "v_text": str }
PROMPT_PREREQUISITE_BATCH = Template(
    """You are annotating prerequisite (hard dependency) edges for a task graph.
For EACH pair below, decide if one entity must be resolved before the other in information gathering.

Rules:
- Output ONLY valid JSON: an object with key "decisions" whose value is an array of objects.
- Each object must have: "index" (int), "relation": one of "u_before_v", "v_before_u", "none".
  Here u_before_v means entity u is a prerequisite of v (directed u -> v in the prerequisite DAG).
- If uncertain or symmetric association only, use "none".
- Avoid cycles in your head: prefer "none" when the dependency is weak.

Pairs:
${PAIRS_JSON}

JSON:"""
)
