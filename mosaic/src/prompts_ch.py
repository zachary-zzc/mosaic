PROMPT_NEW_CLASS_SENSE = """
# Task Objective: Create appropriate categories for the given information fragments.

## Input Information
- **Information Fragment** (`${DATA}`): The information content to be categorized.
- **Dialogue History** (`${CONTEXT}`): Relevant contextual information used to assist in categorization.


## Processing Principles
1. Ignore purely greeting messages and do not create new categories for them.
2. Ensure that the `label` field of each message fragment is fully preserved in the output.
3. Multiple semantically similar message fragments can be grouped into the same new category.
4. New category names should be clear and unambiguous for subsequent retrieval.
5. Information completeness: All message fragments, except purely greeting messages, must be included in the output.

## Output Format 
Please strictly adhere to the following JSON format. This JSON must be correctly parsed by Python's `json.loads()`:

{
  "new_classes": [
    {
      "class_name": "The specific name of the new category",
      "related_message": [
        {"message": "The context message content", "label": "The unique label ID corresponding to this context message"},
        {"message": "The context message content", "label": "The unique label ID corresponding to this context message"}
      ]
    }
  ]
}

If no new category needs to be created, return：{"new_classes": []}

OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to be parsed by json.loads() in Python, and should not include any headers like ```json or```python.
3. The return should ONLY be a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_TAGS = """
你是一个专门用于文本分析的AI模型，擅长从给定文本中精准识别和提取核心关键词/关键短语。你的任务是分析用户提供的文本，提取下面文本的tags。
文本：${TEXT}

# 输出格式
你必须返回一个 **JSON 对象**，格式如下：
{
  "keywords": ["关键词", "关键词", "关键词", ,,,, "关键词"]
}
1. All strings must be  English.
2. Please return the entities as a JSON object.
3. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
4. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_TAGS_QUERY = """
你是一个专业的信息检索与匹配引擎，专门用于根据用户问题在知识图谱实例中精准定位相关条目。

# 角色与任务
你的核心任务是根据用户问题的语义内容，与提供的实例关键词进行智能匹配，返回最相关的10个实例标识（按相关性从高到低排序）。

# 输入数据
1. **用户问题**：${QUESTION}
2. **实例关键词库**：${TAGS}（每个实例包含关键词列表、类ID和实例ID）

# 匹配逻辑与优先级
1. **语义匹配优先**：分析用户问题的核心意图，优先匹配语义相近的关键词（如"情感支持"匹配"emotional support"）
2. **关键词重叠度**：计算问题与实例关键词的重合程度（权重较高的关键词包括实体名称、核心概念）
3. **多维度评估**：综合考虑以下因素：
   - 关键词匹配数量和质量
   - 语义相关性（如同义词、上下文关联）
   - 实例的典型性和代表性

# 输出规范
你必须**仅返回一个标准JSON数组**，格式如下：
[
  {"class_id": "class_1", "instance_id": "instance_1"},
  {"class_id": "class_2", "instance_id": "instance_2"},
  ...
]
1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""
COMMON_QUERY_RESPONSE_PROMPT = """
你是一个信息整合与推理专家，擅长将特定信息与通用知识相结合以构建完整答案。

**#核心任务**
你的核心任务是基于“检索片段”提供的信息回答用户问题。当片段信息**不足以直接回答**问题时，你需要主动识别信息缺口，并恰当地引入**广泛认可的常识或世界事实**进行逻辑推理，以填补空白并提供合理的答案。

**#输入信息**
用户问题: ${QUESTION}
检索片段: ${INFORMATION}

**#处理规则**
1.  **信息充分性评估**：首先，严格评估“检索片段”是否包含直接回答用户问题所需的全部信息。
2.  **信息缺口识别与常识引入**：
    *   如果片段信息**充分**，则优先基于片段内容生成答案。
    *   如果片段信息**不充分、模糊或完全缺失**，则基于公认的常识或世界事实，对缺失部分进行合乎逻辑的推理和补充。
3.  **逻辑连贯性**：确保最终答案是一个将片段信息与常识推理**有机融合**的整体，逻辑通顺，直接回答问题。

# 输出格式
你必须返回一个 **JSON 对象**，且该对象必须包含以下字段：
{
  "response": "回答问题的答案"
}

1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.

"""

PROMPT_CONFLICT = """

请仔细分析下列文本信息，判断是否存在逻辑冲突。

**冲突包括：**
1.  **单条信息内部矛盾**：同一条信息内部的事实、观点或指令存在自我冲突。
2.  **多条信息之间矛盾**：不同信息对同一事实、概念或建议的描述相互冲突。

### 判断规则：
- 如果存在冲突，则判断为“有冲突”。
- 如果信息之间相互补充、一致，或从不同角度描述同一主题，则判断为“无冲突”。
- 请基于信息本身的内容进行判断，而不是格式或表述风格。

输入信息：${messages}

# 输出格式
你必须返回一个 **JSON 对象**，且该对象必须包含以下字段：
{
  "is_conflict": true 或 false,
  "conflicts": [
    {
      "conflict_reason": "简要描述冲突点，例如：'A 信息主张……，而 B 信息主张……'",
      "conflict_message_labels": ["涉及冲突信息的 label"]
    }
  ]
}

### 注意事项：
- 如果没有冲突，`conflicts` 数组应为空。
- 如果存在多处冲突，请在 `conflicts` 数组中逐一列出。

1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_CONFLICT_JUDGE = """
请判断以下冲突原因描述是否与error_data中的冲突表述一致：

冲突原因描述:
${conflict_reason}

error_data中的冲突表述
${error_data_str}

# 输出格式
你必须返回一个 **JSON 对象**，且该对象必须包含以下字段：
{
  "label": true 或 false
}

1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_QUERY_CLASSES="""
你是一个专业的图数据查询分析专家。请分析以下问题，并从提供的类集合中最能回答该问题的类。

问题：${question}

可选的类集合（每个类包含名称和描述信息）：
${classes}

任务要求：
1. 仔细分析问题的核心意图和关键信息点
2. 评估每个类与问题的语义相关性
3. 选择最能回答该问题的前${top_k}个最相关类
4. 按相关性从高到低排序

输出格式：
请严格按照以下JSON格式输出结果：
{
    "selected_classes": [
        {
            "class_id": "类ID",
            "class_name": "类名称"
        }
    ],
    "total_selected": ${top_k}
}

1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_QUERY_TEMPLATE = """
# 角色与任务
您是一位智能信息专家。您的核心任务是**严格遵循“用户问题”**，通过从提供的“检索片段”中筛选出**与问题直接相关的信息**来构建答案，并明确忽略所有不相关的内容。

# 核心指令
**精准匹配**：答案必须完全基于与当前问题直接相关的检索内容。检索片段中与问题无关的信息必须被**坚决忽略**。
**时间类问题的特殊处理**：当用户的提问涉及时间查询时，请务必提取并输出所有相关的时间点或时间段。

# 输入信息
用户问题：${QUESTION}
检索片段：${INFORMATION}
如果检索片段中不包含对该问题的直接答案，请输出最可能的信息作为答案，并尽可能多地输出信息，包括所有可能的答案。

# 操作步骤（模型的内部思考过程）
请按照以下步骤进行处理：
1. **分析问题的核心**：精准理解用户问题所询问的核心实体、属性或事件。
2. **筛选相关片段**：逐一审查每个检索片段，**仅保留那些内容与问题核心直接相关的片段**。对于时间类问题，请留意所有提及相关事件、实体或状态的时间信息。
3. **提取答案信息**：直接从筛选出的相关片段中提取能够回答问题的具体信息。
4. **构建最终答案**：仅使用上一步中提取的信息来构建一个简洁且准确的答案。

**注**
- `recorded_at`：表示该信息被记录或提及的时间点
- `occurred_at`：表示该信息所描述的事件实际发生的时间点
在处理时间相关信息时，请遵循以下准则：
1. **相对时间转换**：将“去年”、“两个月前”、“上周”等相对时间表达转换为具体的日期。
- 计算依据：以片段中的 `recorded_at` 时间戳作为参考点。
- 示例：如果 `recorded_at`若当前日期为“2023-05-04”，则“去年”将被转换为“2022”。
2. 务必将相对时间引用转换为具体的日期、月份或年份。例如：
根据内存中的时间戳，将“去年”转换为“2022”，或将“两个月前”转换为“2023年3月”。
3. 如果具体日期难以推断，请提供相对时间描述。例如：2023年5月的最后一周。

# 输出格式
你必须返回一个 **JSON 对象**，且该对象必须包含以下字段：
{
  "response": "回答问题的答案"
}

1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""



JUDGE_ANSWER = """
Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given
the following data: (1) a question (posed by one user to another user), (2) a ‘gold’
(ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.
The point of the question is to ask about something one user should know about the other
user based on their prior conversations. The gold answer will usually be a concise and
short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading
- as long as it touches on the same topic as the gold answer, it should be counted as
CORRECT.
For time related questions, the gold answer will be a specific date, month, year, etc. The
generated answer might be much longer or use relative time references (like ‘last Tuesday’
or ‘next month’), but you should be generous with your grading - as long as it refers to
the same date or time period as the gold answer, it should be counted as CORRECT. Even if
the format differs (e.g., ‘May 7th’ vs ‘7 May’;2023-05-20 vs The sunday before 25 May 2023), consider it CORRECT if it’s the same date.
Now it’s time for the real question:
Question: ${question}
Gold answer: ${gold_answer}
Generated answer: ${generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with
CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break
the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label".

# Output Format:
{
  "label": "CORRECT or WRONG"
}

# Output Format
1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_CLASS_SENSE = """
# 任务目标
# 角色与任务
你是一个精细化的信息感知与分类系统。核心任务是深度分析**当前输入信息** (`${DATA}`)，并结合**对话历史** (`${CONTEXT}`) 的完整上下文，执行细致的内容感知、提取与归类。


## 核心原则
- **颗粒度优先**: 追求细粒度的信息识别，倾向于创建小而精确的类别，而不是将信息强行归入宽泛的通用类别。
- **信息归类**：允许信息单元归入一个或多个现有类别，只要存在合理关联
- **信息保全**：任何有价值的信息都应被保留，包括看似不相关但可能重要的内容
- **问候过滤**：自动过滤掉纯问候语（如"你好"、"早上好"等不含实质内容的社交套话）,仅当问候语是包含实质性信息的消息的一部分时才保留。
- **标签完整性**：所有输出的信息片段必须包含其对应的标签ID，确保信息可追溯

## 分析流程
1. **内容筛选**：识别并过滤纯问候语，保留所有包含实质信息的语句
2. **信息提取**：识别实体、事件、属性、时间等关键信息单元
3. **关联分析**：将信息单元与现有类别进行多维匹配,对于仅在特定上下文中才能明确其含义的信息，准备存储相关的`dependent_context`。
4. **新类判断**：如信息单元满足以下条件，应创建新类：
   - 包含现有类别未覆盖的新概念
   - 具有新的属性组合或行为模式
   - 无法归类但包含有价值信息
5. **标签关联**：为每个信息单元维护其对应的标签ID，确保信息可追溯

**现有类别** (`${GRAPH_CLASSES}`)

## Output Format & Rules
你必须**严格**按照以下JSON格式输出结果。该JSON必须能被Python的 `json.loads()` 正确解析。

- **`related_classes`** (数组): 列出与输入信息相关的现有类别。
    - `class_id`: (字符串) 现有的类别ID。
    - `related_message`: (数组) **与此类明确相关的原始信息片段对象数组**。每个对象必须包含：
        - `message`: (字符串) 原始消息内容
        - `label`: (数字或字符串) 该消息对应的唯一标签ID
    - `dependent_context`: (数组) 来自对话历史，对于解释与此类别相关的消息至关重要的上下文信息对象数组。每个对象必须包含：
        - `message`: (字符串) 上下文消息内容
        - `label`: (数字或字符串) 该上下文消息对应的唯一标签ID

- **`new_classes`** (数组): 要创建的新类别。
    - `class_name`: (字符串) 新类别的名称，应明确具体。
    - `related_message`: (数组) **与此新类明确相关的原始信息片段对象数组**。每个对象必须包含：
        - `message`: (字符串) 原始消息内容
        - `label`: (数字或字符串) 该消息对应的唯一标签ID
    - `dependent_context`: (数组) 来自对话历史，对于解释与此新类别相关的消息至关重要的上下文信息对象数组。每个对象必须包含：
        - `message`: (字符串) 上下文消息内容
        - `label`: (数字或字符串) 该上下文消息对应的唯一标签ID


OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_CREATE_INSTANCE="""
您是一个具备深度时间感知能力的面向对象实例创建引擎。请根据提供的类别信息和相关消息片段，创建结构良好的实例，并特别关注与时间相关的信息
**关键：忽略非实质性内容**，例如问候语（例如，“你好”）、赞美语（例如，“谢谢”）和纯粹的社交客套话。仅关注事实信息、操作、事件和实体属性。

类别信息： ${class_node}

- 相关消息片段:
${related_messages}
- 上文信息:
${context_messages}

# 核心原则
- **清晰的属性**：定义明确的属性字段（例如，对于“人”对象，定义“姓名”和“年龄”；对于“事件”对象，定义“事件名称”和“参与者”）。
- **灵活的存储**：使用`uninstance_field` 用于无法明确归类到标准属性的实质性内容信息。
- **时间精度**：区分 `occurred`（事件实际发生的时间）和 `recorded_at`（事件在对话中被提及的时间）。
- **深度时间感知**：您必须额外遵循以下时间处理原则：
   主动识别并解析消息中所有明确提及（如“下周二下午三点”）和隐含（如“两天前”、“会议结束后”）的时间信息。
- **消息标签追踪 **：必须将直接用于创建该实例的所有消息标签完整记录到 `message_labels` 字段中，确保信息可追溯。

**示例说明**
假设使用label为3、5、7的消息创建了一个实例，则`message_labels`字段应为 [3, 5, 7]。这表示该实例是基于这三个消息片段的内容构建的

# 输出格式
严格遵循以下JSON数组格式：

[
  {
    "instance_id": "o_<递增数字>",
    "instance_name": "描述性实例名称",
    "attributes": {
      "attribute_key_1": {
        "description": "属性描述",
        "value": "提取的属性值或null",
        "occurred": "信息单元发生相对时间或null",
        "recorded_at": "对话日期或null"
      }
    },
    "operations": {
      "operation_name_1": {
        "description": "相关操作描述"
      }
    },
    "uninstance_field":"无法归入已定义属性的非结构化信息"
    "message_labels": 直接贡献于该实例属性的所有消息的标签ID数组
  }
]

OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_UPDATE_INSTANCE="""
# 角色与任务
您是一名精密的数据协调引擎，专门负责处理信息冲突。您的核心任务是：**仅当新信息与目标实例的当前属性明确指向同一事实并存在直接矛盾时，才执行更新**。您的决策必须极度审慎，以避免引入错误。

1. **待评估的新信息片段与上下文**：
${update_message}

2. 目标实例（当前状态）：
${instance}

*# 核心操作规则
请严格遵循以下决策流程：

## 1. 冲突判定（必须同时满足以下两点）：
- **同一事实检验**：新信息描述的事实点（例如，某人的“年龄”、某事件的“状态”）必须与实例中特定属性的现有值所描述的是**同一个客观事实的同一个方面**。
- **直接矛盾检验**：新信息提供的值必须与实例当前值**逻辑上无法共存**。
   - **示例**：实例中`attributes.status.value`为"已完成"，而新信息明确指出"任务尚未开始"。这构成直接冲突。
   - **非冲突示例**：实例中记录了爱好是"阅读"，新信息提及"喜欢运动"。这属于**补充信息**，而非冲突，应忽略本次更新意图，并将新信息酌情存入`uninstance_field`。

## 2. 更新执行逻辑：
- **满足冲突条件**：仅修改那些发生冲突的特定属性值。实例的其他部分**必须完全保留**。
- **不满足冲突条件**：如果新信息只是补充性内容，**禁止**覆盖现有值。应将新信息的实质性内容存入实例的 `"uninstance_field"` 中，作为附加信息。

## 3. 信息标签追踪与更新逻辑：
- **标签提取**：检查实例的 `message_labels` 字段
- **标签附加**：对于新出现的标签（不在现有 `message_labels` 中），必须将其添加到该字段
- **标签去重**：确保 `message_labels` 中不出现重复的标签ID
请输出更新后的实例节点，保持原结构不变。


OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""
