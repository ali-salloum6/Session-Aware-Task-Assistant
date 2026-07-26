# Project Proposal: Session-Aware Task Assistant with Long-Term Memory

## I. Administrative Details

**Project Title:** Session-Aware Task Assistant with Long-Term Memory

**Team Members:** Ajibade Ayomide (a.ajibade@innopolis.university), Ali Salloum (a.salloum@innopolis.university)

## II. Problem Statement and Motivation

LLM agents are stateless: each call sees only the current context. Multi-turn and cross-session tasks therefore lose user preferences, constraints, and prior outcomes unless the user repeats them. Chat UIs retain history within one session only; truncation or a new session erases salient state.

Static retrieval (search, RAG over documents) grounds answers in corpora but not in *interaction-derived* knowledge—e.g., "I prefer metric units" or "we rejected design A last week." The gap is a lightweight agent that executes tasks via tools while *managing* memory explicitly: what to write, what to recall, and how to stay within a bounded context—without relying on million-token windows.

## III. Proposed Methodology

A **ReAct agent** with **dual-tier memory** (short-term working context + long-term external store):

- **Short-term:** transcript, structured scratchpad (plan / done / next), selective recall of relevant turns; rolling summarization when approaching a token budget.
- **Long-term:** vector-indexed store with `remember` / `recall` tools; writes at end-of-task, on user request, or via a reflection step; episodic, semantic, and procedural entries with timestamps.

**Stack:** LLM (API or Ollama); LangGraph orchestration with checkpoints; Chroma/FAISS vector store; tools (`recall`, `remember`, `calculator`, scoped `search`); Langfuse tracing.

**Loop:** assemble context (profile + scratchpad + top-*k* recalled memories + recent turns) → ReAct until answer or step budget → optional reflection → persist durable facts.

## IV. Evaluation Plan

Fixed suite of 20–25 scripted multi-turn and multi-session scenarios. Metrics: **recall accuracy** (correct fact retrieved later; target >= 85%), **faithfulness** (no confabulated memories; >= 90%), **task completion** (>= 80%), **context efficiency** (>= 30% token reduction vs. append-all-history baseline at equal success). Compare against stateless ReAct and naive full-history baselines. Report latency and cost.

## V. Expected Deliverables

1. LangGraph agent with scratchpad, summarization, selective recall, and long-term memory tools.
2. Benchmark scenarios (single-session recall, cross-session persistence, deduplication, abstention).
3. Evaluation report with metrics and example traces.
4. Architecture documentation and reproduction instructions.
