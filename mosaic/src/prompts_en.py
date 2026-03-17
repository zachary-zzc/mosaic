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
You must return a **JSON** The object must be a JSON object and must contain the following field:

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



PROMPT_QUERY_CLASSES="""
You are a professional graph data query and analysis expert. Please analyze the following question and select the classes from the provided set that best answer the question.

Question: ${question}

Available classes (each class includes a name and description):
${classes}

Task requirements:
1. Carefully analyze the core intent and key information points of the question.
2. Evaluate the relevance of each class to the question.
3. Select the top ${top_k} most relevant classes that best answer the question.
4. Sort the classes in descending order of relevance.

Output format:
Please strictly output the results in the following JSON format:
{
    "selected_classes": [
        {
            "class_id": "class_id",
            "class_name": "class_name"
        }
    ],
    "total_selected": ${top_k}
}
"""

PROMPT_QUERY_TEMPLATE = """
# Role and Task
You are an intelligent information expert. Your core task is to **strictly adhere to the "user question"** and construct an answer by filtering out **information directly relevant to the question** from the provided "retrieval snippets," explicitly ignoring all irrelevant content.

# Core Instructions
**Precise Matching**: The answer must be entirely based on the content of the nodes directly related to the current question.  Information in the retrieval snippets that is irrelevant to the question must be **firmly ignored**.
**Special handling for time-related questions:** When a user's question involves a time query, be sure to extract and output all relevant time points or time periods.

# Input Information
User Question: ${QUESTION}
Retrieval Snippets: ${INFORMATION}
If the retrieval snippets do not contain a direct answer to the question, output the most likely information as the answer, and output as much information as possible, including all possible answers.

# Operation Steps (Model's Internal Thinking Process)
Please process according to the following steps:
1. **Analyze the core of the question**: Precisely understand the core entity, attribute, or event the user question is asking about.
2. **Filter relevant snippets**: Examine each retrieval snippet one by one, **only retaining those snippets whose content is directly related to the core of the question**. For time-related questions, pay attention to all time information that mentions relevant events, entities, or states.
3. **Extract answer information**: Directly extract the specific information that answers the question from the filtered relevant snippets.
4. **Construct the final answer**: Use only the information extracted in the previous step to construct a concise and accurate answer.

**Note**
For time-related questions, the information in the `recorded_at` field indicates the time when this information was mentioned.
'occurred_at`: Indicates the specific point in time when the event described by this information actually took place.
When processing time-related information, please follow these guidelines:
1. **Relative time conversion**: Convert relative time expressions such as "last year," "two months ago," and "last week" into specific dates.
- Calculation basis: Use the `recorded_at` timestamp of the snippet as the reference point.
- Example: If `recorded_at` is "2023-05-04," "last year" is converted to "2022."
2. Always convert relative time references to specific dates, months, or years. For example,
Based on the timestamp in memory, convert "last year" to "2022," or "two months ago" to "March 2023."
3. If the date is difficult to deduce, provide the relative time. For example: last week of May 23, 2023.
# Output Format
You must return a **JSON** The object must be a JSON object and must contain the following field:
{
"response": "The answer to the question"
}

1. Please return the entities as a JSON object.
2. The JSON object should be parsable by `json.loads()` in Python, and should not include any headers like ```json or ```python.
3. The return should ONLY be a valid JSON object, and should not contain any extra text or explanation.
"""

JUDGE_ANSWER = """
Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data:
 (1) a question (posed by one user to another user), 
 (2) a ‘gold’(ground truth) answer, 
 (3) a generated answer which you will score as CORRECT/WRONG.
The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading- as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like ‘last Tuesday’ or ‘next month’), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. 
Even if the format differs (e.g., ‘May 7th’ vs ‘7 May’;2023-05-20 vs The sunday before 25 May 2023), consider it CORRECT if it’s the same date.
Now it’s time for the real question:
Question: ${question}
Gold answer: ${gold_answer}
Generated answer: ${generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
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
# Task Objective
# Role and Task
You are a sophisticated information perception and classification system. Your core task is to deeply analyze the **current input information** (`${DATA}`) and, in conjunction with the complete context of the **dialogue history** (`${CONTEXT}`), perform detailed content perception, extraction, and categorization.

## Core Principles
- **Granularity First**: Prioritize fine-grained information identification, tending to create small, precise categories rather than forcing information into broad, general categories.
- **Information Categorization**: Allow information units to be assigned to one or more existing categories, as long as a reasonable association exists.
- **Information Preservation**: Any valuable information should be retained, including seemingly irrelevant but potentially important content.
- **Greeting Filtering**: Automatically filter out pure greetings (e.g., "Hello," "Good morning," etc., social pleasantries without substantive content), retaining them only if the greeting is part of a message containing substantive information.
- **Label Integrity**: All output information fragments must include their corresponding label ID to ensure traceability.

## Analysis Process
1. **Content Filtering**: Identify and filter pure greetings, retaining all sentences containing substantive information.
2. **Information Extraction**: Identify key information units such as entities, events, attributes, and time.
3. **Association Analysis**: Perform multi-dimensional matching of information units with existing categories. For information whose meaning is only clear in a specific context, prepare to store the relevant `dependent_context`.
4. **New Category Judgment**: A new category should be created if the information unit meets the following conditions:
- Contains new concepts not covered by existing categories.
- Has a new combination of attributes or behavioral patterns.
- Cannot be categorized but contains valuable information.
5. **Label Association**: Maintain the corresponding label ID for each information unit to ensure traceability.

**Existing Categories** (`${GRAPH_CLASSES}`)

## Output Format & Rules
You must **strictly** output the results in the following JSON format. This JSON must be correctly parsed by Python's `json.loads()`.

- **`related_classes`** (array): List the existing categories related to the input information.
- `class_id`: (string) The existing category ID.
- `related_message`: (Array) **An array of original message segment objects explicitly related to this category.** Each object must contain:
- `message`: (string) The original message content
- `label`: (number or string) The unique label ID corresponding to this message
- `dependent_context`: (Array) An array of context information objects from the conversation history that are crucial for interpreting messages related to this category. Each object must contain:
- `message`: (string) The context message content
- `label`: (number or string) The unique label ID corresponding to this context message

- **`new_classes`** (Array): New categories to be created.
- `class_name`: (string) The name of the new category, which should be clear and specific. 
- `related_message`: (Array) **An array of original message segment objects explicitly related to this new category.** Each object must contain:
- `message`: (string) The original message content
- `label`: (number or string) The unique label ID corresponding to this message
- `dependent_context`: (Array) An array of context information objects from the conversation history that are crucial for interpreting messages related to this new category. Each object must contain:
- `message`: (string) The context message content
- `label`: (number or string) The unique label ID corresponding to this context message


OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to be parsed by json.loads() in Python, and should not include any headers like ```json or```python.
3. The return should ONLY be a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_CREATE_INSTANCE = """
You are an object-oriented instance creation engine with advanced temporal awareness.  Based on the provided class information and relevant message snippets, create well-structured instances, paying particular attention to time-related information.
**Key:** Ignore non-substantive content, such as greetings (e.g., "hello"), compliments (e.g., "thank you"), and purely social pleasantries. Focus only on factual information, actions, events, and entity attributes.

Class Information: ${class_node}

- Relevant Message Snippets:
${related_messages}
- Contextual Information:
${context_messages}

# Core Principles
- **Clear Attributes:** Define clearly defined attribute fields (e.g., for a "person" object, define "name" and "age"; for an "event" object, define "event name" and "participants").
- **Flexible Storage:** Use `uninstance_field` for substantive content information that cannot be clearly categorized into standard attributes.
- **Temporal Precision:** Distinguish between `occurred` (the actual time the event occurred) and `recorded_at` (the time the event was mentioned in the conversation).
- **Advanced Temporal Awareness:** You must additionally adhere to the following time processing principles:
  Actively identify and parse all explicitly mentioned (e.g., "next Tuesday at 3 PM") and implicitly mentioned (e.g."two days ago," "after the meeting") time information in the messages.
- **Message Label Tracking:**  All message labels directly used to create this instance must be fully recorded in the `message_labels` field to ensure traceability.

**Example Explanation**
If an instance is created using messages with labels 3, 5, and 7, the `message_labels` field should be [3, 5, 7]. This indicates that the instance was built based on the content of these three message snippets.

# Output Format
Strictly follow the following JSON array format:

[
  {
    "instance_id": "o_<incrementing number>",
    "instance_name": "Descriptive instance name",
    "attributes": {
      "attribute_key_1": {
        "description": "Attribute description",
        "value": "Extracted attribute value or null",
        "occurred": "Relative time of information unit occurrence or null",
        "recorded_at": "Conversation date or null"
      }
    },
    "operations": {
      "operation_name_1": {
        "description": "Description of the relevant operation"
      }
    },
    "uninstance_field":"Unstructured information that cannot be categorized into defined attributes"
    "message_labels": "Array of label IDs for all messages that directly contribute to this instance attribute"
  }
]

OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to be parsed by json.loads() in Python, and should not include any header like ```json or ```python.
3. The return should ONLY be a valid JSON object, and should not contain any extra text or explanation.
"""

PROMPT_UPDATE_INSTANCE = """
# Role and Task
You are a precise data reconciliation engine, specializing in handling information conflicts. 
Your core task is: **to perform updates only when the new information and the target instance's current attributes clearly point to the same fact and there is a direct contradiction**.
Your decisions must be extremely cautious to avoid introducing errors.

1. **New information snippet and context to be evaluated:**
${update_message}

2. Target instance (current state):
${instance}

*# Core Operating Rules
Please strictly follow the following decision-making process:

## 1. Conflict Determination (Both of the following must be met): 
- **Same Fact Verification**:
The fact described by the new information (e.g., someone's "age," the "status" of an event) must be the **same aspect of the same objective fact** as described by the existing value of the specific attribute in the instance.
- **Direct Contradiction Verification:** The value provided by the new information must be **logically incompatible** with the current value in the instance. 
- **Example:** The instance's `attributes.status.value` is "completed," while the new information explicitly states "task has not started." This constitutes a direct conflict. 
- **Non-conflict Example:** The instance records the hobby as "reading," and the new information mentions "likes sports." This is **supplementary information**, not a conflict, and the update intention should be ignored, and the new information should be stored in `uninstance_field` as appropriate.

## 2. Update Execution Logic: 
- **Conflict Condition Met:** Only modify the specific attribute values that are in conflict. The rest of the instance **must be completely preserved**. 
- **Conflict Condition Not Met:** If the new information is only supplementary content, **do not** overwrite existing values. The substantive content of the new information should be stored in the instance's `"uninstance_field"` as additional information.

## 3. Information Label Tracking and Update Logic:
- **Label Extraction:** Check the `message_labels` field of the instance.
- **Label Addition:** For new labels (not present in the existing `message_labels`), they must be added to this field.
- **Label Deduplication:** Ensure that there are no duplicate label IDs in `message_labels`.
Please output the updated instance node, maintaining the original structure.


OUTPUT FORMAT:
1. Please return the entities as a JSON object.
2. The JSON object should be able to parsed by json.loads() in Python, also do not include any header like ```json or ```python.
3. The return be ONLY a valid JSON object, and should not contain any extra text or explanation.
"""