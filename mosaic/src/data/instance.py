from string import Template
from typing import List, Dict, Any, TYPE_CHECKING
from src.assist import fetch_default_llm_model
from src.prompts_en import PROMPT_CREATE_INSTANCE, PROMPT_UPDATE_INSTANCE
#from src.prompts import PROMPT_CREATE_INSTANCE,PROMPT_UPDATE_INSTANCE

from src.logger import setup_logger
import json

_logger = setup_logger("instance cudr")
_llm = fetch_default_llm_model()


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

    _logger.info(f"ALIGN_UPDATE_PROMPT: {align_update_prompt}")
    response = _llm.invoke(align_update_prompt)
    _logger.info(f"ALIGN_UPDATE_RESPONSE: {response.content}")
    updated_instances=json.loads(response.content)

    return updated_instances

# 构建新增实例的prompt，使用全部相关信息而非仅未使用部分
def create_instances_from_messages(messages: List[str],
                                   context_messages: List[str],
                                   class_node):
    # create instances from messages using LLM or other methods
    _logger.info(f"Creating instances from messages: {class_node}")
    # 从class_node提取信息构建class_info字典
    class_info = _extract_class_info_from_node(class_node)

    align_add_prompt = Template(PROMPT_CREATE_INSTANCE).substitute(
        class_node=class_info,
        related_messages="\n".join([f"- {msg}" for msg in messages]),
        context_messages="\n".join([f"- {msg}" for msg in context_messages])
    )

    _logger.info(f"ALIGN_ADD_PROMPT: {align_add_prompt}")
    response = _llm.invoke(align_add_prompt)
    _logger.info(f"ALIGN_ADD_RESPONSE: {response.content}")

    # 解析LLM返回的实例信息
    instances_data = json.loads(response.content)
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

    _logger.info(f"Extracted class info: {class_info}")
    return class_info



