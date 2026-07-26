# Session-Aware Task Assistant with Long-Term Memory

ReAct-style task assistant with dual-tier memory (short-term working context + long-term vector store), LangGraph orchestration, local/Langfuse traces, and a fixed eval suite.

**Team:** Ajibade Ayomide, Ali Salloum (Innopolis University)

**Deliverables:** LaTeX report (`report/report.pdf`), this repository, live demo video.

## Repository layout

```
docs/proposal/          # project proposal (md)
docs/                   # architecture, memory policy
kb/                     # scoped search corpus
eval/scenarios/         # benchmark scripts
eval/suites/            # offline full + live report subsets
eval/results/           # metrics output (gitignored; regenerate locally)
report/                 # LaTeX report + compiled PDF
src/assistant/          # agent package (+ eval harness)
scripts/                # smoke, chat, demos, run_eval, view_trace
tests/
traces/                 # run traces (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,observability]"
cp .env.example .env   # set LLM_API_KEY (OpenRouter)
```

Required: `LLM_API_KEY` (OpenRouter).

Optional Langfuse (cloud traces): set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and either `LANGFUSE_HOST` or `LANGFUSE_BASE_URL` (default `https://cloud.langfuse.com`). Local JSON traces under `traces/` are always written.

Default model: OpenRouter [`google/gemini-2.5-flash`](https://openrouter.ai/google/gemini-2.5-flash) via `https://openrouter.ai/api/v1`.

## Quick start (live run)

```bash
python scripts/smoke_llm.py
AGENT_MODE=memory python scripts/run_chat.py "What is 9*9? Use the calculator."
python scripts/view_trace.py --latest
```

If Langfuse keys are set, the chat script prints `langfuse: <trace_id>` after the run.

## Modes (baselines)

| Env `AGENT_MODE` | Alias | Behavior |
|------------------|-------|----------|
| `stateless` | B0 | no STM history, no LTM, no reflection |
| `full_history` | B1 | append-all STM, no LTM |
| `memory` | B2 | bounded STM + LTM + reflection |

Orchestration: `AGENT_BACKEND=graph` (default LangGraph + sqlite checkpointer) or `loop`.

## Features by phase

| Phase | What | Try |
|------|------|-----|
| 0 | LLM smoke | `python scripts/smoke_llm.py` |
| 1 | ReAct tools (`calculator`, scoped `search`) | `python scripts/run_chat.py` |
| 2 | Scratchpad PLAN/DONE/NEXT | included in `run_chat` |
| 3 | Context budget | `python scripts/demo_context_budget.py` |
| 4 | Cross-session LTM | `python scripts/demo_cross_session_memory.py --fresh` |
| 5 | Reflection / dedup | `python scripts/demo_memory_policy.py --fresh` |
| 6 | LangGraph | `AGENT_BACKEND=graph AGENT_MODE=memory python scripts/run_chat.py` |
| 7 | Traces + optional Langfuse | `python scripts/view_trace.py --latest` |
| 8 | Eval | see below |
| 9 | Report / video | `report/` |

Docs: `docs/architecture.md`, `docs/memory-policy.md`.

## Evaluation (Phase 8)

Two layers — **do not mix** percentages:

1. **Offline harness** — scripted tools + observation-only finals + negative controls. Checker / efficiency evidence.
2. **Live system** — frozen report subset via OpenRouter. Quality claims for the report.

```bash
# both → eval/results/REPORT.md (+ latest_harness.md, latest_live.md)
python scripts/run_eval.py --both

python scripts/run_eval.py --suite full           # offline harness
python scripts/run_eval.py --suite report --live  # live system metrics
```

```bash
pytest                      # offline (fakes); skips @pytest.mark.live
RUN_LIVE=1 pytest -m live   # optional API smoke
```

## Report PDF

```bash
cd report && latexmk -pdf report.tex
# or: pdflatex report.tex && pdflatex report.tex
```
