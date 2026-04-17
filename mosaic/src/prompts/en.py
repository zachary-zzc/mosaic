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

## Output
Return one JSON object with field `new_classes` (array). Example shape:

{
  "new_classes": [
    {
      "class_name": "The specific name of the new category",
      "related_message": [
        {"message": "The context message content", "label": 1},
        {"message": "The context message content", "label": 2}
      ]
    }
  ]
}

If no new category is needed: `{"new_classes": []}`. Preserve each fragment's `label` in outputs.
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

Output: one JSON object:
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
# Output
JSON object: `{"response": "<answer string>"}`.
"""

PROMPT_QUERY_TEMPLATE_EVAL = """
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
`occurred_at`: Indicates the specific point in time when the event described by this information actually took place.
When processing time-related information, please follow these guidelines:
1. **Relative time conversion**: Convert relative time expressions such as "last year," "two months ago," and "last week" into specific dates.
- Calculation basis: Use the `recorded_at` timestamp of the snippet as the reference point.
- Example: If `recorded_at` is "2023-05-04," "last year" is converted to "2022."
2. Always convert relative time references to specific dates, months, or years. For example,
Based on the timestamp in memory, convert "last year" to "2022," or "two months ago" to "March 2023."
3. If the date is difficult to deduce, provide the relative time. For example: last week of May 23, 2023.

# Output
JSON object: `{"response": "<answer string>"}`.
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
Even if the format differs (e.g., ‘May 7th’ vs ‘7 May’;2023-05-20 vs The sunday before 25 May 2023), consider it CORRECT if it’s the same date.Important: The generated answer may use synonyms, paraphrases, or different wording to express the same meaning as the gold answer. As long as the core factual content or sentiment is equivalent, it should be counted as CORRECT. For example, "an ongoing adventure of learning and growing" and "keep motivating and helping each other as we journey through life" convey the same idea and should both be CORRECT. Similarly, if the gold answer lists traits like "thoughtful, authentic, driven" and the generated answer mentions equivalent traits such as "caring, courageous, genuine", evaluate whether the overall characterization aligns rather than requiring exact word matches.
If the generated answer says it cannot find the information or does not know, label it WRONG.Now it’s time for the real question:
Question: ${question}
Gold answer: ${gold_answer}
Generated answer: ${generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
End with JSON: `{"label": "CORRECT"}` or `{"label": "WRONG"}`.
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

## Output
JSON object with:

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
"""

PROMPT_INFER_IMPLICIT_FACTS = """
You are a knowledge inference engine. Given conversation messages and the category they belong to, infer **implicit facts** that are strongly implied but never explicitly stated.

**JSON** (required keyword for API JSON mode.)

Category: ${class_name}
Messages:
${messages}

## Rules
1. Only infer facts that are **strongly supported** by the messages — do NOT hallucinate.
2. Focus on: medical conditions implied by symptoms, relationships implied by context, locations implied by events, time periods implied by references, causes/effects, professional roles implied by activities.
3. Each inferred fact should be a concise sentence.
4. If nothing can be reliably inferred, return an empty list.

Return:
{"inferred_facts": ["fact 1", "fact 2", ...]}
"""

PROMPT_CREATE_INSTANCE = """
You create structured instances from class context and messages (temporal awareness: parse explicit/implicit times). Ignore greetings; keep facts, events, entities.

**JSON** (required keyword for API JSON mode.)

Class: ${class_node}
Messages:
${related_messages}
Context:
${context_messages}

Use `attributes` / `operations` with nested fields as needed (`description`, `value`, `occurred`, `recorded_at` for attributes). Put overflow text in `uninstance_field`. List all message label IDs used in `message_labels`.

Return:
{"instances": [ { "instance_id": "o_1", "instance_name": "...", "attributes": {}, "operations": {}, "uninstance_field": "", "message_labels": [] } ]}
Use `"instances": []` if nothing substantive.
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
Output the full updated instance as one JSON object (same keys/structure as the input instance).
"""