# ConvFinQA Report

## Method

### Why this approach

ConvFinQA is a **multi-turn, table-grounded numerical QA** task: each turn may require (1) reading the right cell from a financial table, (2) resolving coreference to prior turns, and (3) doing arithmetic. I chose an **LLM-native, turn-by-turn solver** rather than training a bespoke model or hard-coding a FinQA DSL interpreter for three reasons:

1. **Time and data**: the assignment is a prototype on ~3.5k cleaned records. A general-purpose LLM with the full document in context is the fastest path to a working conversational system without finetuning.
2. **Conversational structure**: turns build on prior answers (“what about 2005?”, “what is the difference?”). Maintaining explicit `(question, answer)` history and feeding it back each turn mirrors how a chat product would behave.
3. **Separation of reading vs math**: the model is strong at language understanding but weak at reliable arithmetic and percentage scaling. I therefore combine **structured JSON output** (what to compute) with **`safe_eval`** (deterministic execution) so math is not left entirely to the LLM.

**Alternatives considered and not pursued:**

- **End-to-end program synthesis** (predict `turn_program` directly): more faithful to the original FinQA paper, but heavier to build and debug in a short assignment window.
- **Retrieval-only / no LLM**: insufficient for coreference and multi-step reasoning over dialogue.
- **Single-shot “answer all turns at once”**: harder to evaluate turn-by-turn and unrealistic for an interactive chat CLI.

**Backend choice:** I used **Ollama + `qwen2.5:7b-instruct`** locally to avoid API quota limits and keep iteration fast. The same code path supports **Gemini** as a hosted fallback when `GOOGLE_API_KEY` is set.

### Current implementation

I implemented a minimal, LLM-native prototype that answers one dialogue turn at a time using:

- **Full-document context**: each record’s `doc` (pre-text + table + post-text) is formatted into a markdown prompt via `src/format.py`.
- **Conversation state**: prior turns are carried forward as `(question, numeric_answer)` pairs to support coreference across turns (e.g. “what about in 2008?”, “what is the difference?”).
- **Structured LLM output**: the solver requests JSON output constrained by a Pydantic schema (`src/schemas.py`).
- **Deterministic arithmetic**: when the model returns an `expression`, it is executed using a restricted `ast` evaluator (`src/calculator.py`) to reduce arithmetic mistakes and prevent code execution.
- **Backend flexibility**: the same solver interface supports either:
  - **Local Ollama** (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`) to avoid API quotas and keep data local, or
  - **Gemini** (`GOOGLE_API_KEY`, `GEMINI_MODEL`) as a hosted alternative.

For the benchmark below, I used **Ollama** with `qwen2.5:7b-instruct` (see `.env`).

### Accuracy improvements (post-baseline, not fully re-benchmarked)

After the initial 50.07% benchmark, I implemented a multi-phase accuracy pass. A full dev421 re-run was started but **aborted partway through** (~4.5 hours) due to runtime; the two-step router roughly doubles LLM calls on lookup turns. The improvements below are implemented in code but **not reflected in the headline metrics** in this report, which remain from the baseline run.

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

## Strengths and limitations

### Strengths

- **End-to-end prototype**: load dataset → format document → multi-turn chat → turn-level eval with breakdowns (turn index, Type I/II).
- **Structured output + safe arithmetic**: JSON schema constrains the model; `safe_eval` handles `+ − × /` without arbitrary code execution.
- **Conversational eval fidelity**: predicted (including wrong) answers are appended to history, so metrics reflect real error compounding.
- **Backend flexibility**: Ollama for local iteration, Gemini as an alternative; routing and post-processing are modular.
- **Actionable error analysis**: failure examples map cleanly to fix categories (lookup, coreference, percentage scale, Type II context switch).

### Limitations (honest assessment)

- **50% turn accuracy is modest**: on a local 7B model, half of all turns are wrong. This is not production-ready without stronger models or more engineering.
- **Turn-0 lookup is the bottleneck**: ~161 conversations start with a wrong first answer; ~51% of later errors cascade from that. Column disambiguation in tables with similar headers remains hard.
- **Percentage convention drift**: despite prompt rules, the model often returns whole-number percentages (`-21.05`) instead of decimals (`-0.21053`). Post-processing helps but is heuristic.
- **Local model fragility**: Ollama sometimes returns invalid JSON, empty cell values in retrieval, or expressions with unsupported syntax (`min()`, variable names). These fall back to `0.0` sentinel answers, which hurts accuracy silently.
- **Two-step solver trade-off**: retrieve-then-compute is conceptually right but slower and introduces new failure modes (retrieval validation errors observed in logs). Not fully validated at scale.
- **No gold programs at runtime**: `turn_program` in the dataset is useful for analysis but unused; a hybrid LLM+program approach might be more accurate.
- **Incomplete v2 benchmark**: improvements were implemented after the baseline run; a fair before/after comparison on dev421 was not completed.

## Future work (if I had more time)

**Already implemented but not re-benchmarked:**

- Expression normalization, percentage `/100` post-check, auto-routed two-step solver, consistency retry on history mismatch, richer eval tags and `eval-compare`.

**What I would do next:**

1. **Finish a fair comparison**: run `--limit 50` (or full dev421 overnight) with `SOLVER_ROUTER=auto` vs `single` and report deltas using `eval-compare`.
2. **Stronger model**: try `qwen2.5:14b-instruct` or Gemini 2.0 Flash with native JSON schema — likely the highest ROI for table grounding.
3. **Robust retrieval parsing**: tolerate empty/non-numeric cell values in `RetrievalResponse`; skip bad cells and retry retrieval instead of failing the turn.
4. **Hybrid program synthesis**: use the LLM to emit a constrained DSL step per turn (lookup / subtract / divide) and execute deterministically — closer to the original ConvFinQA approach.
5. **Column disambiguation**: when multiple headers match a question, score columns by string similarity to the question and prior `active_metric`.
6. **Self-correction without gold**: on re-ask turns, always re-fetch from the table and prefer document values over history when they disagree by >5%.
7. **Error-driven prompt tuning**: use `error_tag_counts` from eval JSON to target the worst categories (percentage, lookup) with few-shot examples.

## AI tool disclosure

I used **Cursor** as an AI-assisted IDE throughout this assignment. Specifically:

- **Planning and architecture**: discussed approach trade-offs (LLM-native vs program synthesis, eval design, accuracy improvement phases).
- **Code generation and refactoring**: scaffolded modules (`solver.py`, `eval.py`, `router.py`, `postprocess.py`, tests) and iterated on bug fixes (e.g. numeric regex, percentage post-processing).
- **Report drafting**: initial structure and error-analysis write-up were drafted with assistance, then reviewed and edited manually.

All generated code and report text were reviewed, tested (unit tests + eval runs), and adapted to match the dataset schema and assignment requirements. Final decisions on method, metrics, and limitations are my own.
