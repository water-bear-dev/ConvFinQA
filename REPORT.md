# ConvFinQA Report

*The headers here are guidelines, you can structure your report however you like.*
## Method

### Current implementation (Blocks 1–2: core solver + eval harness)

I implemented a minimal, LLM-native prototype that answers one dialogue turn at a time using:

- **Full-document context**: each record’s `doc` (pre-text + table + post-text) is formatted into a markdown prompt via `src/format.py`.
- **Conversation state**: prior turns are carried forward as `(question, numeric_answer)` pairs to support coreference across turns (e.g. “what about in 2008?”, “what is the difference?”).
- **Structured LLM output**: the solver requests JSON output constrained by a Pydantic schema (`src/schemas.py`).
- **Deterministic arithmetic**: when the model returns an `expression`, it is executed using a restricted `ast` evaluator (`src/calculator.py`) to reduce arithmetic mistakes and prevent code execution.
- **Backend flexibility**: the same solver interface supports either:
  - **Local Ollama** (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`) to avoid API quotas and keep data local, or
  - **Gemini** (`GOOGLE_API_KEY`, `GEMINI_MODEL`) as a hosted alternative.

### Files added/updated

- `src/models.py`: Pydantic dataset models (matches `dataset.md`).
- `src/data.py`: dataset loader + O(1) record lookup by id.
- `src/format.py`: document formatter to markdown table.
- `src/prompts.py`: system prompt + user prompt builder (document + history + question).
- `src/schemas.py`: `SolverResponse` for JSON structured output.
- `src/calculator.py`: `safe_eval()` for simple arithmetic.
- `src/solver.py`: LLM backend call (Ollama) + parse + optional expression evaluation.
- `src/main.py`: CLI commands `show` and `chat` wired to the above.
- `src/eval.py`: evaluation harness for turn-level execution accuracy + breakdowns.
- `.env` (gitignored): local config for either Gemini or Ollama settings.

### Notes / constraints

- **Percentages**: the prompt instructs the model to return percentages as decimals (e.g. 14.1% → 0.141), aligning with `dialogue.executed_answers` formatting.
- **Evaluation harness**: added in Block 2 (`src/eval.py`) to compute turn-level execution accuracy + breakdowns (turn index, Type I vs Type II).
 - **Conversational realism**: in evaluation, we append predicted answers to history even when wrong, so later turns reflect error compounding (matches real chat usage).

## Error Analysis

### Metrics (Block 2)

Run:

```bash
uv run main eval --split dev --limit 50
```

Then copy/paste the printed summary here:

- **Turn-level Exe Acc (dev50)**: TBD
- **Turn 0–1 Exe Acc**: TBD
- **Turn 2+ Exe Acc**: TBD
- **Type I Exe Acc**: TBD
- **Type II Exe Acc**: TBD

Note: I did not run this benchmark inside chat because it requires using your API key. The key should stay only in your local `.env` (gitignored), not in commit history or terminal logs shared externally.

### Failure modes to capture (after running dev50)

Pick 3–5 examples from the “Sample Failures” printed by `main eval`, and for each include:

- **Record / turn / question**
- **Gold vs pred**
- **Why it failed** (table row/column mismatch, coreference, percentage scaling, expression formatting, or context truncation)
- **What fix to try** (prompt rule, post-processing rule, retry, etc.)

## Future Work 

- Expand eval beyond dev50 and add richer breakdowns (by turn position distribution, numeric vs non-numeric tables, duplicate columns, etc.).
- Improve robustness of the solver (retry logic, better coreference handling, stricter expression formatting).
- Add lightweight error analysis tooling (store a small set of failures for inspection).

## [may not apply] If & how you've used coding assistants or gen AI tools to help with this assignment 
Please be honest.

Used Cursor as an AI-assisted IDE for implementation planning and code generation/refactoring. All code was reviewed and adapted to match the dataset schema and the structured-output approach.
