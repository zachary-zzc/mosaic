from string import Template
from typing import List, Dict, Any, TYPE_CHECKING
from src.assist import fetch_default_llm_model
from src.prompts_en import PROMPT_CREATE_INSTANCE, PROMPT_UPDATE_INSTANCE
#from src.prompts import PROMPT_CREATE_INSTANCE,PROMPT_UPDATE_INSTANCE

from src.logger import setup_logger
from src.utils.io_utils import parse_llm_json_value

_logger = setup_logger("instance cudr")
_llm = fetch_default_llm_model()

# DashScope 兼容模式 JSON Mode：https://help.aliyun.com/zh/model-studio/json-mode
_DASHSCOPE_JSON_OBJECT = {"type": "json_object"}


def _invoke_json_object(prompt: str):
    """百炼 json_object 要求 messages 含「json」字样，且根节点须为 JSON 对象。"""
    return _llm.invoke(prompt, response_format=_DASHSCOPE_JSON_OBJECT)


def _payload_from_json_object(content: str) -> Any:
    """解析 json_object 模式返回的文本；兼容围栏与非严格 JSON（与 mosaic 其它 LLM 解析一致）。"""
    if not (content or "").strip():
        raise ValueError("LLM 返回空内容，无法解析 JSON")
    parsed = parse_llm_json_value(content)
    if parsed is not None:
        return parsed
    raise ValueError("LLM 返回无法解析为 JSON 对象或数组")


def _normalize_instances_list(parsed: Any) -> List[Dict[str, Any]]:
    """兼容 {\"instances\": [...]} 与历史上的顶层数组。"""
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
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
        update_message="\n".join([f"- {msg}" for msg in messages]),
        instance=instance
    )

    _logger.debug("ALIGN_UPDATE_PROMPT: %s", align_update_prompt)
    response = _invoke_json_object(align_update_prompt)
    _logger.debug("ALIGN_UPDATE_RESPONSE: %s", response.content)
    updated_instances = _payload_from_json_object(response.content)
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
        related_messages="\n".join([f"- {msg}" for msg in messages]),
        context_messages="\n".join([f"- {msg}" for msg in context_messages])
    )

    _logger.debug("ALIGN_ADD_PROMPT: %s", align_add_prompt)
    response = _invoke_json_object(align_add_prompt)
    _logger.debug("ALIGN_ADD_RESPONSE: %s", response.content)

    parsed = _payload_from_json_object(response.content)
    instances_data = _normalize_instances_list(parsed)
    return instances_data



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



