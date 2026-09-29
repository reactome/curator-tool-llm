# Reactome Curator Tool — LLM Gene→Pathway Annotation Pipeline

A curator-assist tool that drafts a Reactome annotation for a gene: it finds the relevant
literature, extracts the biochemical reactions from the full text, and converts them into
draft Reactome data-model instances with a QA report telling a human curator exactly what to
fix. **It produces a reviewed draft, not a finished annotation** — see [How to read the
result](#how-to-read-the-result).

The whole pipeline runs from one command:

```bash
conda run -n paperqa python run_curator.py "GENE"
```

---

## Table of contents
- [What it does](#what-it-does)
- [Architecture: the 3 agents](#architecture-the-3-agents)
- [Key Details](#key-details) ← read this
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Configuration (.env)](#configuration-env)
- [Running it](#running-it)
- [How to read the result](#how-to-read-the-result)
- [How full-text extraction works (Tool 4)](#how-full-text-extraction-works-tool-4)
- [Repository layout](#repository-layout)
- [Known limitations](#known-limitations)
- [Experimental: dense-retrieval investigation](#experimental-dense-retrieval-investigation-dense_retrieval)

---

## What it does

Given a gene symbol, the pipeline:
1. **Places** the gene into a Reactome pathway (even if the gene has no existing annotation).
2. **Retrieves** the most relevant papers from PubMed (or uses papers you supply).
3. **Extracts** biochemical reactions from those papers' full text.
4. **Converts** the reactions into Reactome data-model instances (entities, complexes, reactions, a pathway).
5. **QA-checks** the draft and emits a per-instance report of what a curator must fix.

Output for each gene lands in `results/`:
- `results/<gene>_curator_instances.json` — the draft annotation
- `results/<gene>_qa_report.json` — the QA findings (overall score + per-instance verdicts)

---

## Architecture: the 3 agents

`run_curator.py` runs three agents **synchronously** (no framework, no event loop). The loop
between the Curator and the Reviewer, and the retry cap, live in `run_curator.py` itself.

```
          ┌─────────────── adjustment (Reviewer's proposed knobs) ───────────────┐
          ▼                                                                       │
  Curator.run(gene, adjustment)  ──▶  Reviewer.review(result)  ──sufficient?──────┤ no
   (deterministic: 4 tools)          (LLM: decide + adjust)                       │
                                              │ yes / give_up / retry-cap hit      │
                                              ▼                                    │
   QA.check: build_instances → schema + Neo4j checks + LLM review → verdict   ◀────┘
```

**Agent 1 — Curator** (`reactome_llm/ReactomeCurator.py`), deterministic, calls four tools:
| Tool | Module | Does |
|---|---|---|
| 1. Pathway placement | `ReactomePathwayPlacement` | Predicts a Reactome pathway from interaction-partner enrichment (LLM-free). |
| 2. Literature retrieval | `ReactomeLiteratureExtractor` | Union keyword query → cross-encoder rerank → LLM judge selects the top papers. |
| 3. Full-text resolver | `ReactomeFullTextResolver` | Maps each selected PMID to a full-text source (local PDF / PMC XML / miss). |
| 4. Full-text analysis | `ReactomeFullTextAnalyzer` | Extracts reactions from the resolved full text (see [Tool 4](#how-full-text-extraction-works-tool-4)). |

Placement is computed **once** and reused by retrieval (as the rerank target), by the Reviewer, and by the instance builder.

**Agent 2 — Reviewer** (`ReactomeReviewer.py`), LLM (or `--no-llm-review` for rules): looks at
the Curator's result and returns `sufficient` / `retry` (with a proposed adjustment, e.g. broaden
the query) / `give_up`. It can only *propose* a retry — `run_curator.py` owns the cap
(`--max-attempts`, default 3).

**Agent 3 — QA** (`ReactomeQA.py`): turns the approved reactions into instances and validates them:
1. `build_instances` (`reaction_to_instances.py`) — LLM converts reactions → Reactome data-model instances.
2. **Deterministic schema check** — required fields (`class`, `displayName`) + referential integrity (every reaction/complex reference resolves to a defined entity — no "dangling" arrows) + optional JSON-schema validation against `resources/reactome_domain_model.json`. Structure, not biology; an invalid schema is a hard fail.
3. **Deterministic Neo4j merge check** — if a generated pathway already exists in Reactome, it's a **merge target**, not a conflict: the check pulls the reactions already curated in that pathway so the extracted ones can be added as *new* reactions rather than duplicating the pathway.
4. **LLM expert review** — one call scoring the whole draft: an overall `qa_score` **plus a per-instance verdict** for every instance that isn't curator-ready (`needs_revision`/`bad`); anything unlisted is `good`. A repair loop re-converts with QA's corrections up to a cap, keeping the best attempt.

---

## Key Details

- **A QA verdict of FAIL / NEEDS_REVISION is normal.** This is a curator-*assist* tool: the
  deliverable is a draft plus a precise fix-list. A `FAIL` means "here's a draft and exactly
  what a human must fix," not "the pipeline broke." A genuinely clean draft is rare for a
  richly-studied gene.

- **The overall `qa_score` hides good instances — so read the per-instance breakdown.** One
  score of `0.41` can be dragged down by a few bad instances while dozens are fine. The
  terminal prints `per-inst : 87 good · 43 needs-revision · 13 bad (of 143)` and names every
  flagged one; `flagged_instances` in the QA report has the same data. **Unlisted = good.**

- **How a PMID is recovered from a PDF (there's no PMID printed on a paper).** For a local PDF
  we read the **DOI off page 1**, then resolve DOI → PMID via NCBI esearch
  (`FullTextResolver.build_index`). Chain: `PDF → page-1 DOI → PMID`. A PDF with no page-1 DOI
  can't be identified and is skipped (logged as "unresolvable").

- **"has-data" vs "cold-start", and the placement gate.** The final `RESULT` block classifies
  the gene: *has-data* = the gene already has curated Reactome pathways (annotated); *cold-start*
  = none yet. The *placement gate* is `pass` when interaction-partner enrichment gives a confident
  pathway, `fail` when it doesn't (reactions, if any, come back unplaced for manual placement).

- **Full-text extraction runs as subprocesses.** Tool 4 (`fulltext_extractor.py`) shells out to
  `run_extraction.py` → `run_merge.py` → `run_review.py` (the partner's tested CLIs). They *look*
  like standalone scripts but are a live part of the pipeline — don't delete them.

- **Two MongoDBs, one server.** MongoDB (`:27017`) holds **two** databases the pipeline uses:
  `PUBMED_MONGO_DB` (cached PubMed abstracts) and `FIS_MONGO_DB` (Reactome functional-interaction
  partners, used for placement). Both must be present.

- **`--papers-only` skips retrieval.** When a curator already has the papers they want, point the
  tool at a folder of PDFs and it annotates those directly (placement + extraction + QA), skipping
  Tool 2. It's a single attempt (nothing to retry without retrieval). See [Running it](#running-it).

---

## Prerequisites

- **Python 3.10** (conda env recommended; on this setup it's `paperqa`)
- **Neo4j** with a Reactome graph loaded (Bolt on `:7687`)
- **MongoDB** on `:27017` with two databases (abstracts + functional interactions)
- **API keys:** Anthropic (extraction/merge/convert/QA), OpenAI (`run_review.py` full-text review only), NCBI/PubMed (E-utilities)

---

## Install

Three steps — **all three are required**; the Python install alone will not run the pipeline.

**1. Python environment + dependencies:**
```bash
conda create -n paperqa python=3.10
conda activate paperqa
pip install -r requirements.txt
```
`requirements.txt` is the **single, authoritative** dependency list — install from it. (There is
deliberately no conda `environment.yaml`: a stale export from another machine caused more confusion
than it solved.) Core libraries: `anthropic`, `openai`, `langchain-*` / `langgraph`, `neo4j`,
`pymongo`, `sentence-transformers` + `flair` + `torch` (cross-encoder rerank), `PyMuPDF` (PDF
parsing), `pandas`/`numpy`/`scipy`, `pydantic`, `jsonschema`.

**2. Services + data** — a Neo4j Reactome graph and two MongoDB databases that live *outside* this
repo. **This is the step people miss.** See [Services & data you must set up](#services--data-you-must-set-up).

**3. `.env`** — API keys + connection strings. See [Configuration (.env)](#configuration-env).

---

## Configuration (.env)

Create a `.env` in the repo root (values loaded via `python-dotenv`; the Neo4j/Mongo vars can
alternatively live in your shell or `conda env config vars`):

```env
# --- API keys ---
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...              # used only by run_review.py (full-text cross-model review)
PUBMED_API_KEY=...                 # NCBI E-utilities (raises rate limit to 10/sec)
NCBI_EMAIL=you@example.com         # optional; NCBI politeness (default is hardcoded)
NCBI_TOOL=curator-tool-llm         # optional

# --- Neo4j (Reactome graph) ---
REACTOME_NEO4J_URI=bolt://localhost:7687
REACTOME_NEO4J_USER=neo4j
REACTOME_NEO4J_PWD=your_password
REACTOME_NEO4J_DATABASE=graph.db   # the Reactome graph DB name (NOT "reactome")

# --- MongoDB: PubMed abstract cache ---
PUBMED_MONGO_URI=mongodb://localhost:27017
PUBMED_MONGO_DB=pubmed_cache
PUBMED_MONGO_COLLECTION=abstracts

# --- MongoDB: Reactome functional-interaction partners (for placement) ---
FIS_MONGO_DB=idg_pairwise
FIS_MONGO_GENE_INDEX=...            # collection name for the gene index
FIS_MONGO_RELATIONSHIPS=...         # collection name for the pairwise relationships
```

`run_curator.py` sets `TOKEN_PROFILE=1` itself, so per-gene token/cost accounting is on by default.

### Services & data you must set up

The `pip install` is **not enough** — the pipeline reads a Reactome graph and two Mongo databases
that are not shipped in this repo. This is what silently breaks a fresh setup, so do each explicitly:

**1. Neo4j — the Reactome graph.** Install Neo4j and load a **Reactome graph database** into it
(obtain it from Reactome's data downloads at <https://reactome.org> — the "Reactome Graph Database";
it is *not* in this repo). Point `REACTOME_NEO4J_URI/USER/PWD/DATABASE` at it. On this setup the DB
name is `graph.db` (not `reactome`). Sanity check: `run_curator.py` exits at startup if Bolt `:7687`
is unreachable.

**2. MongoDB — two databases on `:27017`:**
- **`PUBMED_MONGO_DB` (PubMed abstract cache) — self-populating.** Abstracts/JATS are fetched from
  NCBI on demand and cached as you run (needs `PUBMED_API_KEY`). Nothing to preload.
- **`FIS_MONGO_DB` = `idg_pairwise` (functional-interaction partners, used for placement) — must be
  loaded separately.** This repo only *reads* it. The data is produced by the Reactome
  `org.reactome.idg.pairwise` Java project (`MainApp`), or copied from a machine that already has it
  (see migration below). **Without this DB, pathway placement finds no interaction partners** and
  cold-start genes can't be placed.

**3. Interaction flat files (`resources/interactions/`)** — BioGRID + IntAct, gitignored (large).
Download the exact versions listed in `resources/llm_interactions/README.txt`. These feed the
non-Mongo interaction path (`ProteinProteinInteractionsLoader.load_interactions`).

**Moving the Mongo DBs between machines** (e.g. from a dev box to `curator.reactome.org`):
```bash
mongodump    --db idg_pairwise --out /path/to/backup
mongorestore --db idg_pairwise /path/to/backup/idg_pairwise   # same for the abstract-cache DB
```

### Tracked resource files (shipped in the repo)
- `resources/reactome_domain_model.json` — the Reactome schema QA validates against.
- `resources/ReactomePathwayGenes_Ver_91.txt` — pathway↔gene map for enrichment.

### Local caches (gitignored, auto-created under `data/`)
`fulltext_pdf/` (curator PDFs you point `--papers-dir` at), `fulltext_cache/` (downloaded PMC XML),
`abstract_cache/`, and `user_config.json` (remembers your last `--papers-dir`).

---

## Running it

```bash
# One or more genes (full pipeline: retrieval → extraction → QA)
conda run -n paperqa python run_curator.py SHANK3
conda run -n paperqa python run_curator.py SHANK3 TANC1 CTTNBP2

# Point at a folder of curator PDFs (remembered for next time)
conda run -n paperqa python run_curator.py SHANK3 --papers-dir /path/to/pdfs

# Papers-only: skip retrieval, annotate ONLY the PDFs in the folder
conda run -n paperqa python run_curator.py SHANK3 --papers-only --papers-dir /path/to/pdfs

# Cheap smoke test (no full text, rule-based reviewer/QA — no LLM cost)
conda run -n paperqa python run_curator.py SHANK3 --no-full-text --no-llm-review --no-llm-qa
```

| Flag | Default | Meaning |
|---|---|---|
| `--papers-dir PATH` | prompt/last used | Folder of local full-text PDFs. |
| `--papers-only` | off | Skip retrieval; annotate only the PDFs in `--papers-dir`. Implies full text on; single attempt. |
| `--max-attempts N` | 3 | Curator↔Reviewer retries (full-text mode). |
| `--max-papers N` | 5 | Papers the retrieval judge selects. |
| `--no-full-text` | off | Retrieval + placement only, no extraction. |
| `--no-llm-review` | off | Rule-based Reviewer (free, deterministic) instead of the LLM. |
| `--no-llm-qa` | off | QA's deterministic checks only (skips the LLM expert review + per-instance verdicts). |

Prereqs are checked at startup — Neo4j (`:7687`) and MongoDB (`:27017`) must be reachable or the
run exits with a clear message.

---

## How to read the result

Each gene ends with a `RESULT` block and a batch summary:

```
======================================================================
RESULT — SHANK3
======================================================================
  Gene type : ANNOTATED (has-data) — already has 7 curated Reactome pathway(s).
  Placement : gate PASS — confident placement under "Neurexins and neuroligins".
  Outcome   : 48 reaction(s) drafted; QA verdict FAIL (qa_score 0.41).
  Status    : DRAFT -> needs curator review before acceptance (see the QA flags above).
  Cost/time : ~$6.45 est (list price) · 1250k in + 180k out tokens · 30.4 min
```

- **Gene type / Placement** — see [Key Details](#key-details).
- **Status** — always a draft; the QA flags above it are the fix-list.
- **Cost/time** — combined token spend (retrieval + convert + QA *and* the full-text extraction
  subprocesses) and wall-clock. It's a list-price estimate (Sonnet-4.6 $3/$15 per Mtok, ignores
  cache discounts), so treat it as an upper-ish bound.

The artifacts are `results/<gene>_curator_instances.json` (the draft) and
`results/<gene>_qa_report.json` (score + `flagged_instances`).

---

## How full-text extraction works (Tool 4)

Tool 4 turns resolved papers into reactions. For each paper, `fulltext_extractor.py` drives three
subprocesses in order:

1. **`run_extraction.py`** — isolates the paper's Results section (deterministic heading match, LLM
   fallback for PDFs; exact JATS tags for PMC XML), then extracts reactions **chunk by chunk** with
   a LangGraph memory window (2 previous + 1 next chunk) so reactions spanning chunk boundaries are
   still captured.
2. **`run_merge.py`** — a four-stage semantic merge (pairwise LLM judging → conflict-aware
   clustering → whole-cluster validation → subsection consolidation) that deduplicates reactions
   *within a paper*.
3. **`run_review.py`** — a second-opinion quality read from a **different model (OpenAI)**, with an
   automatic missed-reaction sweep when the score is below threshold.

Resolution sources (Tool 3): a curator's **local PDF** (matched by DOI→PMID), **PMC JATS XML**
(fetched + cached for any PMID PMC serves), or a **miss** (falls back to mining the abstract,
labeled so a curator can weight it below full text).

Supporting modules: `PubMedFetcher` (PMID/PMCID → JATS or local PDF), `FullTextPDFSections`
(Results-section isolation), `FullTextPDFPrompts` (prompt templates), `abstract_extractor`
(abstract fallback).

---

## Repository layout

```
run_curator.py                 # ⭐ entry point — the 3-agent pipeline
reaction_to_instances.py       # convert step: reactions → Reactome data-model instances
fulltext_extractor.py          # drives the full-text subprocess pipeline (Tool 4)
run_extraction.py              #   stage 1 — chunk-by-chunk reaction extraction
run_merge.py                   #   stage 2 — four-stage semantic merge/dedup
run_review.py                  #   stage 3 — cross-model (OpenAI) quality review
abstract_extractor.py          # abstract fallback when no full text is available
TextEmbedder.py                # cross-encoder embeddings for retrieval rerank

reactome_llm/
  ReactomeCurator.py           # Agent 1 — deterministic 4-tool orchestration
  ReactomeReviewer.py          # Agent 2 — sufficiency decision + proposed adjustment
  ReactomeQA.py                # Agent 3 — build instances, validate, per-instance verdicts
  ReactomePathwayPlacement.py  # Tool 1 — interaction-partner pathway placement
  ReactomeLiteratureExtractor.py # Tool 2 — retrieval (union query → rerank → LLM judge)
  ReactomeFullTextResolver.py  # Tool 3 — PMID → PDF / PMC XML / miss
  ReactomeFullTextAnalyzer.py  # Tool 4 — reactions from full text (wraps fulltext_extractor)
  GenePathwayAnnotator.py      # gene→accession + embedding/enrichment helpers
  QueryBuilder.py              # builds the retrieval queries + rerank targets
  CuratorRubric.py             # the LLM curator-judge rubric (paper selection)
  ProteinProteinInteractionsLoader.py # functional-interaction partners (Mongo + BioGRID/IntAct)
  ReactomeUtils.py             # placement enrichment (binomial test + FDR)
  FullTextResolver.py          # PDF→DOI→PMID index; PMC XML fetch/cache
  FullTextPDFSections.py       # Results-section isolation (heuristic + LLM fallback)
  FullTextPDFPrompts.py        # full-text extraction/merge prompt templates
  PubMedFetcher.py             # NCBI E-utilities: abstracts, JATS full text, ID conversion
  ReactomeModels.py            # pydantic schemas (incl. QAReport + per-instance verdicts)
  ReactomeNeo4jUtils.py        # Reactome graph queries
  ModelConfig.py               # model settings (claude-sonnet-4-6)
  ReactomePrompts.py, ReactomePubMed.py, ReactomeLLMErrors.py,
  token_profiler.py, logging_config.py   # infra

resources/                     # schema, pathway↔gene map, interaction data (see above)
results/                       # all run outputs (gitignored)
data/                          # local caches: PDFs, PMC XML, abstracts (gitignored)
```

---

## Known limitations

- **Very large annotations may still need batched conversion.** The reactions→instances step is
  one LLM call capped at 64K output tokens (Sonnet-4.6's max). That fits genes up to ~50 reactions;
  a larger gene would need the conversion batched (convert N reactions at a time and stitch). The
  cap was raised from 32K→64K after SHANK3 truncated at 32K.
- **No cross-paper reaction dedup.** Merge deduplicates *within* a paper; the same reaction reported
  in multiple papers can appear more than once in the pooled set.
- **Reaction-level duplicate detection vs. the graph is pathway-name-based.** The Neo4j merge check
  matches whole pathways by name; it does not yet detect that an individual extracted *reaction*
  already exists in Reactome.
- **A local PDF with no page-1 DOI can't be identified** and is skipped.

---

## Experimental: dense-retrieval investigation (`dense_retrieval/`)

Reference code only — **not part of the live pipeline** and not imported by it. These scripts are
the investigation into whether a dense, embedding-based approach could improve on the current
lexical retrieval and on matching extracted reactions to curated Reactome reactions:

- `cosine_retrieval_test.py`, `embedding_benchmark.py`, `cosine_similarity_score.py` — measured the
  ceiling of embedding-only similarity and showed *why* it falls short (two reactions differing only
  in substrate score within ~0.02 of each other, so a wrong match can outrank the right one).
- `pool_match.py` + `ReactionMatcher.py` — the resulting "filter-then-rank" idea: a deterministic
  structural candidate pool (shared input/output/catalyst) → cross-encoder ranking → an LLM that can
  also answer "none".
- `ReactomeFullCacheEmbeddingTest.py`, `check_sections.py` — a bi-encoder positive-control eval and a
  chunk-preview diagnostic.

The written findings (`dense_retrieval_investigation.md`) are kept with the run outputs, outside
version control. These scripts are preserved as a record, not a maintained entry point — running them
as-is would need import fixes (they expect repo-root modules like `TextEmbedder`).
