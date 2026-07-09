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

For the benchmark below, I used **Ollama** with `qwen2.5:7b-instruct` (see `.env`).

### Accuracy improvements (post-baseline)

After the initial 50.07% benchmark, I implemented a multi-phase accuracy pass:

- **Foundation fixes**: corrected numeric regex parsing, Ollama `temperature=0`, expression normalization before `safe_eval`.
- **Post-processing**: `src/postprocess.py` auto-corrects 100× percentage scale errors (e.g. `-21.05` → `-0.2105`).
- **Auto-routed two-step solver**: `src/router.py` sends turn-0 lookups and re-asks through retrieve-then-compute (`RetrievalResponse` → `SolverResponse`); math follow-ups stay on the faster single-call path.
- **Consistency checks**: on re-ask turns, retrieved cell values that disagree with history trigger a correction note and optional retry.
- **Richer eval**: per-turn `route`, `error_tag`, `retried` fields; `main eval-compare` for before/after deltas.

Re-benchmark with:

```bash
cp eval_results.json eval_results_baseline.json   # if not already saved
uv run main eval --split dev --limit 421 --output eval_results_v2.json --baseline eval_results_baseline.json
uv run main eval-compare eval_results_baseline.json eval_results_v2.json
```

### Files added/updated (accuracy pass)

- `src/postprocess.py`: percentage scale correction.
- `src/router.py`: auto routing (`SOLVER_ROUTER=auto|single|two_step`).
- `src/history.py`: `TurnState` with optional active metric for Type II.
- `src/prompts.py`: hardened prompts + retrieve/compute prompt builders.
- `src/schemas.py`: `CellRetrieval`, `RetrievalResponse`, optional `source` on `SolverResponse`.
- `tests/test_calculator.py`: unit tests for calculator, postprocess, router.
- `.env.example`: documented model and routing options.

### Files added/updated (initial)

- `src/models.py`: Pydantic dataset models (matches `dataset.md`).
- `src/data.py`: dataset loader + O(1) record lookup by id.
- `src/format.py`: document formatter to markdown table.
- `src/prompts.py`: system prompt + user prompt builder (document + history + question).
- `src/schemas.py`: `SolverResponse` for JSON structured output.
- `src/calculator.py`: `safe_eval()` for simple arithmetic.
- `src/solver.py`: LLM backend call (Ollama) + parse + optional expression evaluation.
- `src/main.py`: CLI commands `show`, `chat`, and `eval` wired to the above.
- `src/eval.py`: evaluation harness for turn-level execution accuracy + breakdowns.
- `.env` (gitignored): local config for either Gemini or Ollama settings.

### Notes / constraints

- **Percentages**: the prompt instructs the model to return percentages as decimals (e.g. 14.1% → 0.141), aligning with `dialogue.executed_answers` formatting.
- **Evaluation harness**: added in Block 2 (`src/eval.py`) to compute turn-level execution accuracy + breakdowns (turn index, Type I vs Type II).
- **Conversational realism**: in evaluation, we append predicted answers to history even when wrong, so later turns reflect error compounding (matches real chat usage).
- **Local-model robustness (Ollama)**: local LLMs sometimes emit non-numeric / formatted answers (e.g. `"$4 million"` or sentences) or omit fields. To keep benchmarking stable:
  - The prompt explicitly demands **JSON-only** output and **numeric-only** `answer`.
  - The schema is more tolerant (accepts `answer` as `float | str | None`, `reasoning` optional).
  - The solver normalizes common formats (currency symbols, commas, `million/billion/thousand`, parentheses negatives) into floats when possible.
  - If parsing still fails after retries, the solver returns a sentinel numeric value (`0.0`) with an error string in reasoning so eval can continue.

## Error Analysis

### Metrics (full dev split)

Run:

```bash
uv pip install . --reinstall
uv run main eval --split dev --limit 421 --output eval_results.json
```

**Turn-level execution accuracy (421 conversations, 1490 turns):**

| Metric | Accuracy |
|--------|----------|
| **Overall** | **50.07%** (746/1490) |
| Turn 0–1 | 58.08% |
| Turn 2+ | 39.66% |
| Type I (single table) | 54.18% |
| Type II (multi-table / context switch) | 40.18% |

**High-level takeaways:**

- Early turns are easier: the model can often answer a direct lookup on turn 0–1, but accuracy drops ~18 points on turn 2+ as errors compound through conversation history.
- Type II conversations are harder than Type I (~14 points), consistent with the need to track which table or metric the dialogue has switched to.
- Among incorrect turns, the dominant failure modes are **percentage/ratio scaling** (~211 turns), **arithmetic / derived answers** (~200), and **initial lookup errors** (~161). Roughly **135** incorrect percentage turns look like a 100× scale error (model returns `-21.05` when gold is `-0.21053`). **47** turns returned the sentinel `0.0` after parse/eval failures.

### Failure examples (from `eval_results.json`)

#### 1. Lookup — wrong column on turn 0

- **Record / turn**: `Single_MRO/2007/page_134.pdf-1`, turn 0
- **Question**: “what was the weighted average exercise price per share in 2007?”
- **Gold vs pred**: `60.94` vs `17.24`
- **Why it failed**: the table contains multiple “weighted average …” columns (exercise price vs grant-date fair value). The model read the wrong column on the first turn, poisoning all downstream math.
- **Fix to try**: prompt the model to quote the exact row + column header before answering; optionally add a lightweight column-disambiguation step for tables with similar headers.

#### 2. Coreference — follow-up year lookup drift

- **Record / turn**: `Single_MRO/2007/page_134.pdf-1`, turn 1
- **Question**: “and what was it in 2005?”
- **Gold vs pred**: `25.14` vs `6.15`
- **Why it failed**: even with prior context, the model did not reliably map “it” back to the same column for a different year row.
- **Fix to try**: require the model to restate the metric from turn 0 in follow-up turns; include explicit row/column labels in conversation history.

#### 3. Math — compounding from wrong prior answers

- **Record / turn**: `Single_MRO/2007/page_134.pdf-1`, turn 2
- **Question**: “what was, then, the change over the years?”
- **Gold vs pred**: `35.8` vs `11.09`
- **Why it failed**: the model subtracted its own incorrect 2007/2005 values (`17.24 − 6.15`) rather than the gold operands. Eval appends predicted answers to history, so early lookup errors cascade.
- **Fix to try**: prefer `expression` fields that reference prior turn indices; add validation that operands match history before `safe_eval`.

#### 4. Percentage — 100× scale error

- **Record / turn**: `Double_HII/2017/page_104.pdf`, turn 1
- **Question**: “so what was the percentage change during this time?”
- **Gold vs pred**: `-0.21053` vs `-21.05`
- **Why it failed**: arithmetic was correct (`−4/19`), but the model multiplied by 100 and returned a percentage value instead of the dataset’s decimal convention. Reasoning explicitly shows `* 100`.
- **Fix to try**: strengthen the decimal-percentage rule in the prompt; add a post-check that flags answers ~100× off gold magnitude on percentage questions.

#### 5. Type II — context switch to wrong metric

- **Record / turn**: `Double_HII/2017/page_104.pdf`, turn 2
- **Question**: “what was the change associated with credit facilities during that time?”
- **Gold vs pred**: `3.0` vs `1.0`
- **Why it failed**: after answering about senior-note issuance costs, the dialogue switched to credit facilities. The model instead differenced total long-term debt (`1279 − 1278`), reading the wrong row entirely.
- **Fix to try**: prompt the model to identify which table section and line item each turn refers to; track “active metric” explicitly across Type II switches.

#### 6. Sentinel fallback — wrong prior value locks in zero

- **Record / turn**: `Single_UNP/2008/page_77.pdf-2`, turn 2
- **Question**: “so what was the difference between the two years?”
- **Gold vs pred**: `-10.0` vs `0.0`
- **Why it failed**: turn 1 incorrectly returned `93.0` for 2007 (gold `103.0`). On turn 2 the model concluded both years were `93.0` and answered `0.0` — a compounding lookup error rather than a parse failure.
- **Fix to try**: on large discrepancies between re-asked values and history, re-fetch from the document instead of trusting prior turns.

## Future Work

- **Re-run full dev421 benchmark** with the improved solver and record v1 vs v2 metrics in this report.
- **Stronger local model or hosted API**: try `qwen2.5:14b-instruct` or Gemini if accuracy plateaus.
- **Finer-grained routing**: distinguish ratio questions that need two-step compute vs single-call.
- **Per-record error dashboards**: use `error_tag_counts` from eval JSON for targeted prompt tuning.

## [may not apply] If & how you've used coding assistants or gen AI tools to help with this assignment
Please be honest.

Used Cursor as an AI-assisted IDE for implementation planning and code generation/refactoring. All code was reviewed and adapted to match the dataset schema and the structured-output approach.
