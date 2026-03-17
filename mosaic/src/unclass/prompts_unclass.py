
PROMPT_CREATE_INSTANCE_UNCLASS = """
You are an object-oriented instance creation engine with advanced temporal awareness.  Based on the provided message fragments, please generate a well-structured instance, paying particular attention to time-related information.
**Key:** Ignore non-substantive content, such as greetings (e.g., "hello"), compliments (e.g., "thank you"), and purely social pleasantries. Focus only on factual information, actions, events, and entity attributes.

- Message Fragments
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