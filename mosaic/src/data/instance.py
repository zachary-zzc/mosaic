import time
from string import Template
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from src.assist import fetch_default_llm_model
from src.prompts_en import PROMPT_CREATE_INSTANCE, PROMPT_UPDATE_INSTANCE
#from src.prompts import PROMPT_CREATE_INSTANCE,PROMPT_UPDATE_INSTANCE

from src.logger import setup_logger
from src.utils.io_utils import llm_response_text, parse_llm_json_value

_logger = setup_logger("instance cudr")
_llm = fetch_default_llm_model()


def merge_canonical_message_labels(instance: Dict[str, Any], messages: List) -> None:
    """
    将 ``messages`` 中的规范 ``label``（如 conv_message_splitter 的全局整数）并入实例的
    ``message_labels``。LLM 返回的实例常使用自造标签名，导致 update_class_relationships
    无法按对话标签共现建边；合并后可与本批 ``data`` 中的 label 对齐。
    """
    if not isinstance(instance, dict):
        return
    extra: List[Any] = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("label") is not None:
            extra.append(m["label"])
    if not extra:
        return
    cur = list(instance.get("message_labels") or [])
    norm = {str(x) for x in cur}
    for lab in extra:
        s = str(lab)
        if s not in norm:
            norm.add(s)
            cur.append(lab)
    instance["message_labels"] = cur

# DashScope 兼容模式 JSON Mode：https://help.aliyun.com/zh/model-studio/json-mode
_DASHSCOPE_JSON_OBJECT = {"type": "json_object"}


def _format_messages_for_prompt(messages: List[Any]) -> str:
    """将消息列表格式化为 prompt 行（兼容 dict 含 message / 纯 str）。"""
    lines: List[str] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            lines.append(f"- {msg.get('message', msg)}")
        else:
            lines.append(f"- {msg}")
    return "\n".join(lines)


def _invoke_json_object(prompt: str):
    """百炼 json_object 要求 messages 含「json」字样，且根节点须为 JSON 对象。"""
    from src.llm.telemetry import llm_call_scope

    with llm_call_scope("build.instance_json_object"):
        return _llm.invoke(prompt, response_format=_DASHSCOPE_JSON_OBJECT)


def _payload_from_json_object(content: str) -> Any:
    """解析 json_object 模式返回的文本；兼容围栏与非严格 JSON（与 mosaic 其它 LLM 解析一致）。"""
    raw = (content or "").strip()
    if not raw:
        raise ValueError("LLM 返回空内容，无法解析 JSON")
    parsed = parse_llm_json_value(raw)
    if parsed is not None:
        return parsed
    preview = raw[:240] + ("…" if len(raw) > 240 else "")
    raise ValueError(f"LLM 返回无法解析为 JSON 对象或数组，正文预览: {preview!r}")


# 首次请求 + 至少 1 次重试（共 2 轮）
_JSON_OBJECT_RETRIES = 2
_JSON_OBJECT_RETRY_DELAY_SEC = 1.5


def _invoke_resolve_json_payload(prompt: str) -> tuple[Optional[Any], str]:
    """
    调用 json_object 并解析；带重试。

    Returns:
        (payload, status): status 为 "ok" 时 payload 为解析结果；
        "empty" 表示各次提取正文均为空（视为无输出，由调用方跳过）；
        "bad_json" 表示出现过非空正文但始终无法解析（复杂/损坏 JSON，可走 hash 降级）。
    """
    saw_nonempty = False
    for attempt in range(1, _JSON_OBJECT_RETRIES + 1):
        response = _invoke_json_object(prompt)
        text = (llm_response_text(response) or "").strip()
        if not text:
            _logger.warning(
                "json_object 第 %d/%d 次: 提取正文为空",
                attempt,
                _JSON_OBJECT_RETRIES,
            )
            if attempt < _JSON_OBJECT_RETRIES:
                time.sleep(_JSON_OBJECT_RETRY_DELAY_SEC * attempt)
            continue
        saw_nonempty = True
        try:
            return _payload_from_json_object(text), "ok"
        except ValueError as e:
            _logger.warning(
                "json_object 第 %d/%d 次解析失败: %s；正文长度=%d",
                attempt,
                _JSON_OBJECT_RETRIES,
                e,
                len(text),
            )
            if attempt < _JSON_OBJECT_RETRIES:
                time.sleep(_JSON_OBJECT_RETRY_DELAY_SEC * attempt)
    if not saw_nonempty:
        return None, "empty"
    return None, "bad_json"


def _normalize_instances_list(parsed: Any) -> List[Dict[str, Any]]:
    """兼容 {\"instances\": [...]} 与历史上的顶层数组。"""
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return []
        if "instances" in parsed and isinstance(parsed["instances"], list):
            return [x for x in parsed["instances"] if isinstance(x, dict)]
        if "instance_name" in parsed or "attributes" in parsed:
            return [parsed]
    raise ValueError(f"无法从 LLM 结果解析实例列表: {type(parsed).__name__}")


class Instance:
    def __init__(self,
                 instance_name: str):
        self.instance_name = instance_name
        self.instance_id = None

    def __hash__(self):
        return hash(f"{self.instance_name}")

    def __eq__(self, other):
        return self.instance_id == other.instance_id

    #这个用不上，新增实例直接通过prompt完成了
    #而且互相引用报错
    @staticmethod
    def new_instance(instance_name: str, classnode):
        inst = Instance(instance_name=instance_name)
        built_in_types = ["attributes", "functions", "unclassified"]
        exclusive_attrs = built_in_types + []  # classnode built-in attributes should be specifically initialized as each built-in type provide a page storage
        for attr in dir(classnode):
            if attr in exclusive_attrs:
                continue
            if attr.startswith("_"):
                continue
            if hasattr(inst, attr):
                continue
            setattr(inst, attr, getattr(classnode, attr))
        for type_name in built_in_types:
            if not hasattr(inst, type_name):
                setattr(inst, type_name, {})
            for item in getattr(classnode,
                                type_name):  # for each built-in type, loop through the existing keys that occurred in previous instances
                if item not in getattr(inst, type_name):
                    getattr(inst, type_name)[item] = []
        return inst

def update_data_from_messages(instance,messages: List[str]):
    # update instance data according to messages
    align_update_prompt = Template(PROMPT_UPDATE_INSTANCE).substitute(
        update_message=_format_messages_for_prompt(messages),
        instance=instance
    )

    _logger.debug("ALIGN_UPDATE_PROMPT: %s", align_update_prompt)
    updated_instances, status = _invoke_resolve_json_payload(align_update_prompt)
    if status == "empty":
        _logger.warning("实例更新 LLM 返回空，跳过更新（保留原实例结构）")
        return dict(instance)
    if status == "bad_json":
        _logger.warning("实例更新 LLM JSON 无法解析，降级为 hash 合并消息")
        return update_data_from_messages_hash(instance, messages)
    assert updated_instances is not None
    _logger.debug("ALIGN_UPDATE_RESPONSE parsed keys: %s", list(updated_instances) if isinstance(updated_instances, dict) else type(updated_instances))
    if not isinstance(updated_instances, dict):
        raise ValueError("更新实例期望 JSON 对象（单条实例），得到: %s" % type(updated_instances).__name__)

    return updated_instances


def update_data_from_messages_hash(instance: Dict, messages: List) -> Dict:
    """仅用文本拼接更新实例，不调用 LLM。将新消息追加到 uninstance_field。"""
    msg_strs = [m.get("message", m) if isinstance(m, dict) else str(m) for m in messages]
    existing = instance.get("uninstance_field", "") or ""
    new_text = "\n".join(msg_strs)
    updated = dict(instance)
    updated["uninstance_field"] = (existing + "\n" + new_text).strip() if existing else new_text
    existing_labels = list(instance.get("message_labels", []))
    new_labels = [m.get("label") for m in messages if isinstance(m, dict) and m.get("label") is not None]
    updated["message_labels"] = existing_labels + new_labels
    return updated


def create_instances_from_messages_hash(
    messages: List,
    context_messages: List,
    class_node: Any,
) -> List[Dict]:
    """仅用文本拼接创建实例，不调用 LLM。每条消息或整批作为一个实例的 uninstance_field。"""
    if not messages:
        return []
    # 整批作为一个实例，便于 TF-IDF 检索
    msg_strs = [m.get("message", m) if isinstance(m, dict) else str(m) for m in messages]
    labels = [m.get("label") for m in messages if isinstance(m, dict)]
    text = "\n".join(msg_strs)
    if context_messages:
        ctx_strs = [c if isinstance(c, str) else str(c) for c in context_messages]
        text = "\n".join(ctx_strs) + "\n" + text
    instance = {
        "instance_id": None,  # 由 add_instances 填充
        "instance_name": getattr(class_node, "class_name", "Unclassified") + " instance",
        "attributes": {},
        "operations": {},
        "uninstance_field": text,
        "message_labels": labels or [],
    }
    return [instance]

# 构建新增实例的prompt，使用全部相关信息而非仅未使用部分
def create_instances_from_messages(messages: List[str],
                                   context_messages: List[str],
                                   class_node):
    # create instances from messages using LLM or other methods
    _logger.debug("Creating instances from messages: %s", class_node)
    # 从class_node提取信息构建class_info字典
    class_info = _extract_class_info_from_node(class_node)

    align_add_prompt = Template(PROMPT_CREATE_INSTANCE).substitute(
        class_node=class_info,
        related_messages=_format_messages_for_prompt(messages),
        context_messages=_format_messages_for_prompt(context_messages),
    )

    _logger.debug("ALIGN_ADD_PROMPT: %s", align_add_prompt)
    parsed, status = _invoke_resolve_json_payload(align_add_prompt)
    if status == "empty":
        _logger.warning("创建实例 LLM 返回空，跳过（不新增实例）")
        return []
    if status == "bad_json":
        _logger.warning("创建实例 LLM JSON 无法解析，降级为 hash 实例")
        return create_instances_from_messages_hash(messages, context_messages, class_node)

    assert parsed is not None
    _logger.debug("ALIGN_ADD_RESPONSE normalized from type=%s", type(parsed).__name__)
    try:
        return _normalize_instances_list(parsed)
    except ValueError as e:
        _logger.warning("LLM 返回 JSON 结构无法归一化为实例列表，本批降级 hash: %s", e)
        return create_instances_from_messages_hash(messages, context_messages, class_node)



def _extract_class_info_from_node(class_node) -> Dict[str, Any]:
    """
    从ClassNode对象提取需要的类信息，构建结构化字典

    Args:
        class_node: 类节点对象

    Returns:
        Dict[str, Any]: 包含类信息的结构化字典
    """
    class_info = {
        "class_name": getattr(class_node, 'class_name', 'Unknown')
    }
    #
    # # 添加attributes（如果存在且非空）
    # attributes = getattr(class_node, 'attributes', [])
    # if attributes and len(attributes) > 0:
    #     class_info["attributes"] = attributes
    #
    # # 添加operations（如果存在且非空）
    # operations = getattr(class_node, 'operations', [])
    # if operations and len(operations) > 0:
    #     class_info["operations"] = operations
    #
    # # 添加unclassified（如果存在且非空）
    # unclassified = getattr(class_node, 'unclassified', [])
    # if unclassified and len(unclassified) > 0:
    #     class_info["unclassified"] = unclassified
    #
    # # 添加实例数量信息
    # instances = getattr(class_node, '_instances', [])
    # if instances and len(instances) > 0:
    #     class_info["instance_count"] = len(instances)

    _logger.debug("Extracted class info: %s", class_info)
    return class_info



