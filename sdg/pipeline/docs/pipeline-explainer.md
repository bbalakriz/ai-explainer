# Telco Chat SDG Pipeline Explainer

This document explains how the synthetic data generation (SDG) pipeline works,
covering every function in `run.py` and the step by step flow defined in
`flow.yaml`.

---

## High Level Overview

The pipeline generates synthetic Thai language multi turn chat training data
for telecom intent classification. It takes a small set of real (seed) examples
per intent and asks an LLM to produce many more examples that follow the same
format, style and language patterns.

There are two execution modes:

| Mode | Backend | When to use |
|------|---------|-------------|
| `standalone` | litellm directly | lighter dependency, one intent at a time |
| `flow` | sdg_hub Flow engine | batch all intents together with concurrency |

Both modes share the same helper functions for loading data, building prompts,
parsing LLM output and validating the result.

---

## Input Files

### `samples/training_dataset.json`

A flat JSON array of real training examples. Each object looks like this:

```json
{
  "user_message": "งั้นช่วยเปิดไช้งานเบอร์รายเดือนนี้ให้เลยคับ ...",
  "intent": "MOBILE_SIM_ACTIVATION_POSTPAID",
  "session_history": [
    { "role": "assistant", "content": "สวัสดีค่ะ ติดต่อเรื่องเปิดใช้งานซิมรายเดือนใช่ไหมคะ" },
    { "role": "user", "content": "ใช่ครับ เพิ่งได้รับซิมใหม่มา ..." },
    { "role": "assistant", "content": "ได้ค่ะ รบกวนแจ้งหมายเลขที่ต้องการเปิดใช้งานก่อนนะคะ" }
  ]
}
```

The three fields are:

- `user_message` : the latest message from the customer (the one to classify)
- `intent` : the target label, e.g. `MOBILE_BILLING_CHECK_DUE_DATE`
- `session_history` : up to 5 previous messages providing conversational context

### `pipeline/intent_descriptions.json`

A dictionary mapping each intent name to a plain English description of what the
intent means. This description is injected into the prompt so the LLM
understands the intent it needs to generate for.

### `pipeline/prompts/generate_telco_chat.yaml`

A YAML file containing the chat messages (system + user) that form the prompt
template. Placeholder variables like `{{ intent }}` and `{{ seed_examples }}`
get substituted at runtime. See the "Prompt Template" section below for details.

---

## `run.py` Function Reference

### `load_seed_data(seed_file)`

Reads `training_dataset.json` and groups examples by intent.

Input: the path to the seed JSON file.

Output: a dictionary where each key is an intent name and the value is a list of
example objects belonging to that intent.

```json
{
  "MOBILE_BILLING_CHECK_DUE_DATE": [
    {
      "user_message": "บิลเดือนนี้ครบกำหนดวันไหนคะ",
      "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
      "session_history": []
    },
    {
      "user_message": "เช็คเดดไลน์จ่ายบิลหน่อยค้าบ",
      "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
      "session_history": [
        {"role": "assistant", "content": "สวัสดีค่ะ มีอะไรให้ช่วยคะ"},
        {"role": "user", "content": "อยากดูบิลค่ะ"}
      ]
    },
    {
      "user_message": "due date bill เมื่อไหร่อ่ะ",
      "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
      "session_history": [
        {"role": "assistant", "content": "สวัสดีค่ะ"},
        {"role": "user", "content": "สอบถามบิลหน่อย"},
        {"role": "assistant", "content": "ได้เลยค่ะ ต้องการทราบด้านไหนคะ"}
      ]
    }
  ],
  "MOBILE_SIM_ACTIVATION_POSTPAID": [
    {
      "user_message": "งั้นช่วยเปิดไช้งานเบอร์รายเดือนนี้ให้เลยคับ",
      "intent": "MOBILE_SIM_ACTIVATION_POSTPAID",
      "session_history": [
        {"role": "assistant", "content": "สวัสดีค่ะ ติดต่อเรื่องเปิดใช้งานซิมรายเดือนใช่ไหมคะ"},
        {"role": "user", "content": "ใช่ครับ เพิ่งได้รับซิมใหม่มา"}
      ]
    }
  ]
}
```

Each intent key maps to every example that carries that label. This grouped
dictionary is the starting point for every downstream step.

---

### `select_seed_examples(examples, max_seeds=5)`

Picks 5 examples with varied history lengths for one intent. This runs once
per intent, independently.

For example, MOBILE_BILLING_CHECK_DUE_DATE might have 12 examples in the seed
data with history lengths of 0, 2, 3 and 5 messages. The function picks one
from each history length group until it has 5 total, then trims any history
longer than 5 messages down to the last 5 (sliding window).

Input: a flat list of example objects for a single intent.

Output: up to 5 examples covering different conversation depths.

```json
[
  {"user_message": "บิลเดือนนี้ครบกำหนดวันไหนคะ",    "session_history": []},
  {"user_message": "เช็คเดดไลน์จ่ายบิลหน่อยค้าบ",     "session_history": ["..2 msgs.."]},
  {"user_message": "due date bill เมื่อไหร่อ่ะ",       "session_history": ["..3 msgs.."]},
  {"user_message": "รบกวนเช็คเดดไลน์บิลหน่อยนะ",      "session_history": ["..5 msgs.."]},
  {"user_message": "วันสุดท้ายจ่ายบิลวันไหน",          "session_history": []}
]
```

These 5 examples are all that go into the prompt for this intent. Each intent
gets its own 5. The LLM never sees seeds from other intents.

---

### `build_prompt(intent, seed_examples, num_examples=20)`

Assembles the final chat messages that will be sent to the LLM.

Steps:

1. Reads the YAML template at `prompts/generate_telco_chat.yaml`
2. Serialises the seed examples to a JSON string
3. Looks up the intent description from `intent_descriptions.json`
4. Replaces the template placeholders:

   | Placeholder | Replaced with |
   |-------------|---------------|
   | `{{ intent }}` | e.g. `MOBILE_BILLING_CHECK_DUE_DATE` |
   | `{{ intent_description }}` | the English description of the intent |
   | `{{ seed_examples }}` | JSON string of the selected seed examples |
   | `{{ num_examples }}` | number of examples to generate (default `20`) |
   | `{{ num_seed }}` | count of seed examples provided |

Output: a list of message dicts ready for an LLM chat completion call.

```python
[
  {"role": "system", "content": "You are an expert synthetic data generator ..."},
  {"role": "user",   "content": "Generate exactly 20 new training examples for the intent: MOBILE_BILLING_CHECK_DUE_DATE ..."}
]
```

---

### `parse_json_output(text)`

Extracts a JSON array from the raw LLM response string.

The function handles common edge cases:

- strips markdown code fences if the LLM wraps its output in triple backticks
- locates the first `[` and last `]` to isolate the JSON array
- falls back to an empty list with a warning if parsing fails

Input: raw text string from the LLM.

Output: a Python list of dicts (the generated examples) or an empty list on
failure.

Example: the LLM might return a raw string that looks like this (with markdown
fences and extra whitespace):

    ```json
    [
      {
        "user_message": "บิลเดือนนี้ครบกำหนดวันไหนคะ",
        "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
        "session_history": []
      },
      {
        "user_message": "เช็คเดดไลน์จ่ายบิลหน่อยค้าบ",
        "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
        "session_history": [
          {"role": "assistant", "content": "สวัสดีค่ะ มีอะไรให้ช่วยคะ"},
          {"role": "user", "content": "อยากดูบิลค่ะ"}
        ]
      }
    ]
    ```

The function strips the triple backtick fences, locates the outer `[` and `]`
and parses the content in between. The return value is a plain Python list:

```json
[
  {
    "user_message": "บิลเดือนนี้ครบกำหนดวันไหนคะ",
    "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
    "session_history": []
  },
  {
    "user_message": "เช็คเดดไลน์จ่ายบิลหน่อยค้าบ",
    "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
    "session_history": [
      {"role": "assistant", "content": "สวัสดีค่ะ มีอะไรให้ช่วยคะ"},
      {"role": "user", "content": "อยากดูบิลค่ะ"}
    ]
  }
]
```

If the JSON cannot be parsed at all, the function returns an empty list `[]`
with a warning printed to the console.

---

### `validate_examples(examples, intent)`

Cleans and validates every generated example to ensure it conforms to the
expected schema.

For each example it:

1. Skips anything that is not a dict
2. Skips anything missing a `user_message` field
3. Forces the `intent` field to the correct value (overrides whatever the LLM
   returned to prevent label contamination)
4. Validates `session_history` is a list of `{role, content}` objects where
   `role` is either `"user"` or `"assistant"`
5. Enforces the sliding window constraint: trims history to the last 5 messages

Output: a list of clean, validated example dicts.

```json
{
  "user_message": "เช็คยอดค้างจ่ายหน่อยค่ะ",
  "intent": "MOBILE_BILLING_PAYMENT_STATUS_POSTPAID",
  "session_history": [
    {"role": "assistant", "content": "สวัสดีค่ะ มีอะไรให้ช่วยไหมคะ"},
    {"role": "user", "content": "อยากดูบิลค่ะ"}
  ]
}
```

---

### `generate_with_litellm(intent, seed_examples, num_examples, model, ...)`

The standalone generation backend. Calls the LLM through litellm for a single
intent at a time.

Steps:

1. Calls `build_prompt()` to assemble the messages
2. Sends them to the model via `litellm.completion()` with `temperature=0.8`
   and `max_tokens=16384`
3. Passes the response text through `parse_json_output()`

Output: a list of raw (unvalidated) example dicts.

---

### `generate_with_flow(intents_data, model, ...)`

The batch generation backend using the sdg_hub Flow engine. Processes all
intents in a single call with concurrency.

Steps:

1. Loads `flow.yaml` into an sdg_hub `Flow` object
2. Applies model configuration (model name, api base, api key)
3. Converts the intents data into a pandas DataFrame
4. Calls `flow.generate(df, max_concurrency=5)` which runs the three block
   pipeline defined in `flow.yaml`
5. Reads the `extract_output_content` column from the result DataFrame

Output: a dictionary mapping intent name to the raw LLM output string.

```
{
  "MOBILE_BILLING_CHECK_DUE_DATE": "[{...}, {...}, ...]",
  "MOBILE_SIM_ACTIVATION_POSTPAID": "[{...}, {...}, ...]",
  ...
}
```

These raw strings then go through `parse_json_output()` and
`validate_examples()` in the main function.

---

### `main()`

The CLI entry point that ties everything together.

1. Parses command line arguments (mode, model, output dir, num examples, etc.)
2. Loads seed data from `samples/training_dataset.json`
3. Filters intents if `--intents` was specified
4. Depending on `--mode`:
   - `standalone` : loops over each intent one by one, calling
     `generate_with_litellm()` then `validate_examples()` for each
   - `flow` : prepares a DataFrame of all intents, calls
     `generate_with_flow()`, then post processes each intent's output
5. Saves each intent's results to a separate JSON file in the output directory
6. Prints a summary showing example counts per intent

---

## How `flow.yaml` Works

The `flow.yaml` file defines a three block pipeline that sdg_hub executes
sequentially for every row in the input DataFrame. Each row represents one
intent to generate data for.

### Input DataFrame (one row per intent)

Before the flow starts, `run.py` builds a DataFrame like this:

| intent | intent_description | seed_examples | num_examples | num_seed |
|--------|--------------------|---------------|--------------|----------|
| MOBILE_BILLING_CHECK_DUE_DATE | Customer wants to check the due date for their mobile bill... | [{"user_message": "...", ...}] | 20 | 5 |
| MOBILE_SIM_ACTIVATION_POSTPAID | Customer wants to activate a new... | [{"user_message": "...", ...}] | 20 | 5 |

Each column maps directly to a placeholder in the prompt template.

### Block 1: PromptBuilderBlock (`build_generation_prompt`)

```yaml
- block_type: "PromptBuilderBlock"
  block_config:
    block_name: "build_generation_prompt"
    input_cols: ["intent", "intent_description", "seed_examples",
                 "num_examples", "num_seed"]
    output_cols: "generation_prompt"
    prompt_config_path: "prompts/generate_telco_chat.yaml"
```

What it does: reads the prompt template from `generate_telco_chat.yaml` and
substitutes the five input columns into the template placeholders.

Input columns consumed: `intent`, `intent_description`, `seed_examples`,
`num_examples`, `num_seed`

Output column added: `generation_prompt`

The output is a structured list of chat messages (system + user) that is ready
to send to an LLM. This is conceptually identical to what `build_prompt()` does
in standalone mode.

Example output (the value stored in the `generation_prompt` column for one row):

```json
[
  {
    "role": "system",
    "content": "You are an expert synthetic data generator for Thai telecom customer service ..."
  },
  {
    "role": "user",
    "content": "Generate exactly 20 new training examples for the intent: MOBILE_BILLING_CHECK_DUE_DATE\n\nHere is a description of this intent:\nCustomer wants to check the due date for their mobile bill payment...\n\nHere are 5 existing examples for reference:\n[{...seed data...}]\n\nREQUIREMENTS:\n1. Output MUST be a valid JSON array\n..."
  }
]
```

### Block 2: LLMChatBlock (`generate_examples`)

```yaml
- block_type: "LLMChatBlock"
  block_config:
    block_name: "generate_examples"
    input_cols: "generation_prompt"
    output_cols: "raw_output"
    max_tokens: 16384
    temperature: 0.8
    async_mode: true
```

What it does: sends the `generation_prompt` messages to the configured LLM model
and captures the full response object.

Input column consumed: `generation_prompt`

Output column added: `raw_output`

Key settings:

- `max_tokens: 16384` gives the LLM plenty of room to output 20 full JSON
  examples with session histories
- `temperature: 0.8` adds enough randomness for diverse output while keeping
  it coherent
- `async_mode: true` allows sdg_hub to run multiple intents concurrently
  (controlled by `max_concurrency=5` in the `flow.generate()` call)

The `raw_output` column contains the full LLM response object (not just the
text). This is why the next block is needed.

### Block 3: LLMResponseExtractorBlock (`extract_output`)

```yaml
- block_type: "LLMResponseExtractorBlock"
  block_config:
    block_name: "extract_output"
    input_cols: "raw_output"
    extract_content: true
    expand_lists: true
```

What it does: extracts the text content from the LLM response object.

Input column consumed: `raw_output`

Output column added: `extract_output_content`

With `extract_content: true`, the block pulls out just the message text from
the response wrapper. The result is a plain string containing the JSON array
that the LLM generated.

Example output (the value in `extract_output_content` for one row):

```
[
  {
    "user_message": "บิลครบกำหนดวันไหนคะ",
    "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
    "session_history": []
  },
  {
    "user_message": "เช็คเดดไลน์จ่ายบิลหน่อยค้าบ 📱",
    "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
    "session_history": [
      {"role": "assistant", "content": "สวัสดีค่ะ มีอะไรให้ช่วยคะ"},
      {"role": "user", "content": "ดูยอดค่าโทรหน่อย"}
    ]
  },
  ...
]
```

This raw string is then read by `run.py` via `row["extract_output_content"]`,
passed through `parse_json_output()` and `validate_examples()`, and saved as a
JSON file.

### How the three blocks execute at runtime

The blocks run sequentially, not per row. Each block processes the entire
DataFrame before the next block starts. Parallelism only happens inside the
LLMChatBlock where `async_mode: true` fires multiple LLM calls at once.

```
Step 1: PromptBuilderBlock
  row 0 (BILLING_CHECK_DUE_DATE)     -> build prompt
  row 1 (SIM_ACTIVATION_POSTPAID)    -> build prompt
  row 2 (BILLING_PAYMENT_STATUS)     -> build prompt
  ...all 16 rows, sequentially (instant, pure string substitution)

  ALL prompts done. Move to step 2.

Step 2: LLMChatBlock (async_mode: true, max_concurrency: 5)
  row 0  ──┐
  row 1  ──┤
  row 2  ──┼── batch of 5, fired in parallel, wait for all to finish
  row 3  ──┤
  row 4  ──┘
  row 5  ──┐
  row 6  ──┤
  row 7  ──┼── next batch of 5
  row 8  ──┤
  row 9  ──┘
  ...and so on until all 16 rows are done.

  ALL LLM calls done. Move to step 3.

Step 3: LLMResponseExtractorBlock
  row 0  -> extract content
  row 1  -> extract content
  ...all 16 rows, sequentially (instant, just unwraps the response object)
```

Steps 1 and 3 are pure data transformations that finish in milliseconds. Step 2
is where all the time is spent. The pipeline does not stream individual rows
through all three steps; it completes each step for every row before moving on.

---

## End to End Data Flow

Below is how a single intent travels through the pipeline from seed data to
final output file.

```
training_dataset.json
        |
        v
  load_seed_data()
  groups examples by intent
        |
        v
  select_seed_examples()
  picks 5 diverse seeds for one intent
        |
        v
  build_prompt()  /  PromptBuilderBlock
  assembles system + user messages with seeds injected
        |
        v
  litellm.completion()  /  LLMChatBlock
  LLM generates ~20 new examples as a JSON array
        |
        v
  LLMResponseExtractorBlock  (flow mode only)
  extracts text content from the LLM response wrapper
        |
        v
  parse_json_output()
  strips markdown fences, isolates the JSON array, parses it
        |
        v
  validate_examples()
  enforces schema, forces correct intent label,
  trims history to 5 messages
        |
        v
  generated_intents/MOBILE_BILLING_CHECK_DUE_DATE.json
  one file per intent containing validated examples
```

---

## Output Format

Each output file (e.g. `generated_intents/MOBILE_BILLING_CHECK_DUE_DATE.json`)
is a JSON array of validated examples. Every example has exactly three fields:

```json
[
  {
    "user_message": "วันสุดท้ายจ่ายบิลคือวันไหนคะ",
    "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
    "session_history": [
      {"role": "assistant", "content": "สวัสดีค่ะ ยินดีให้บริการค่ะ"},
      {"role": "user", "content": "สอบถามเรื่องบิลหน่อยค่ะ"},
      {"role": "assistant", "content": "ได้เลยค่ะ ต้องการทราบข้อมูลบิลด้านไหนคะ"}
    ]
  },
  {
    "user_message": "due date bill เดือนนี้เมื่อไหร่อ่ะ",
    "intent": "MOBILE_BILLING_CHECK_DUE_DATE",
    "session_history": []
  }
]
```

These files can be directly concatenated or loaded as additional training data
for an intent classification model.

---

## CLI Quick Reference

```bash
# standalone mode, one intent at a time
python run.py --mode standalone --model gemini/gemma-4-31b-it

# flow mode, all intents batched with concurrency
python run.py --mode flow --model gemini/gemma-4-31b-it

# generate for specific intents only
python run.py --mode standalone --intents MOBILE_BILLING_CHECK_DUE_DATE MOBILE_SIM_ACTIVATION_POSTPAID

# custom output directory and example count
python run.py --mode flow --model anthropic/claude-sonnet-4-20250514 --output ../my_output --num-examples 50

# local model via Ollama
python run.py --mode standalone --model ollama/gemma2 --api-base http://localhost:11434
```
