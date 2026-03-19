# Mosaic 命名规范

代码库内变量与方法命名遵循以下约定，便于统一维护。

## 变量命名

- **相关/匹配**：使用 `relevant_*`（完整拼写），避免缩写 `relv_*`。
  - 例：`relevant_class_messages`（class_id → 消息）、`relevant_instances`、`relevant_instance_messages`。
- **新增/未匹配**：
  - 需新增的类：`new_class_messages`（class_name → 消息）。
  - 需新增的实例 / 未匹配消息：`new_instance_messages`、`unmatched_messages_by_node`、`unmatched_data`。
- **计数与索引**：使用完整词 `current_*`、`count_*`、`index`，避免 `curr_`、`cnt`、`idx`（循环内短变量除外）。
- **实例与类**：统一使用 `class_node`（snake_case），避免 `classnode`；实例列表用 `_instances`，单例用 `instance`。
- **消息**：单条用 `message` 或 `message_item`（含 `message`/`label` 的 dict），列表用 `messages`；避免单独使用 `msg` 作为公开变量名。
- **序列化结果**：字符串形式用 `*_str` 或 `*_serialized`，如 `relevant_instances_str`、`instances_in_classes_str`。

## 方法命名

- **处理相关类/实例**：`process_relevant_class_instances`、`update_relevant_instances`（替代原 `process_relvclass_instances`、`update_relv_instances`）。
- **关键词检索**：`find_keyword_relevant_instance_tags`、`find_keyword_coverage_instances_with_tfidf`（替代原 `find_kw_*`）。
- **一致性校验**：`consistency_valid_dynamic`（拼写修正：原 `consistency_vaild_dynamic`）。

## 类型与结构

- 类节点列表：`list[ClassNode]`，变量名如 `processed_classes`、`added_class_nodes`。
- 实例 ID → 消息：`Dict[str, List[str]]` 或 `Dict[str, List[dict]]`，变量名 `instance_id_to_messages`。
- 类节点 → 消息/上下文：`Dict[ClassNode, Dict[str, List]]`，变量名 `unmatched_messages_by_node`。
