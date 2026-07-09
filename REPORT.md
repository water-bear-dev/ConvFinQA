# ConvFinQA Report

*The headers here are guidelines, you can structure your report however you like.*
## Method

### Current implementation (Block 1: core solver)

I implemented a minimal, LLM-native prototype that answers one dialogue turn at a time using:

- **Full-document context**: each record’s `doc` (pre-text + table + post-text) is formatted into a markdown prompt via `src/format.py`.
- **Conversation state**: prior turns are carried forward as `(question, numeric_answer)` pairs to support coreference across turns (e.g. “what about in 2008?”, “what is the difference?”).
- **Structured LLM output**: the solver requests JSON output constrained by a Pydantic schema (`src/schemas.py`).
- **Deterministic arithmetic**: when the model returns an `expression`, it is executed using a restricted `ast` evaluator (`src/calculator.py`) to reduce arithmetic mistakes and prevent code execution.

### Files added/updated

- `src/models.py`: Pydantic dataset models (matches `dataset.md`).
- `src/data.py`: dataset loader + O(1) record lookup by id.
- `src/format.py`: document formatter to markdown table.
- `src/prompts.py`: system prompt + user prompt builder (document + history + question).
- `src/schemas.py`: `SolverResponse` for JSON structured output.
- `src/calculator.py`: `safe_eval()` for simple arithmetic.
- `src/solver.py`: Gemini Flash call + parse + optional expression evaluation.
- `src/main.py`: CLI commands `show` and `chat` wired to the above.
- `.env` (gitignored): local config for `GOOGLE_API_KEY` and `GEMINI_MODEL`.

### Notes / constraints

- **Percentages**: the prompt instructs the model to return percentages as decimals (e.g. 14.1% → 0.141), aligning with `dialogue.executed_answers` formatting.
- **No evaluation harness yet**: execution accuracy benchmarking (`eval.py`) and breakdowns are intentionally deferred to the next block.

## Error Analysis

Not run yet. Next step is to set `GOOGLE_API_KEY` and run `uv run main chat "Single_JKHY/2009/page_28.pdf-3"` to validate the 4-turn example (lookup → lookup → difference → percentage) and then implement the eval harness to measure turn-level Exe Acc on the dev split.

## Future Work 

- Add `eval.py` to compute execution accuracy against `executed_answers` on the dev split (plus turn-position and Type I/II breakdowns).
- Improve robustness of the solver (retry logic, better coreference handling, stricter expression formatting).
- Add lightweight error analysis tooling (store a small set of failures for inspection).

## [may not apply] If & how you've used coding assistants or gen AI tools to help with this assignment 
Please be honest.

Used Cursor as an AI-assisted IDE for implementation planning and code generation/refactoring. All code was reviewed and adapted to match the dataset schema and the structured-output approach.
