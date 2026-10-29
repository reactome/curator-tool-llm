# Speed and quality tuning: what was measured, and why the defaults are what they are

Measured 2026-10-28 to 2026-10-29 on the paper-annotation pipeline (`run_extraction.py` → `run_merge.py` → draft build).
Everything here is from real API runs. Numbers are single runs unless stated, so differences of one reaction or a few
seconds are noise (the duplicate judge is not deterministic).

The settings live in [`curator_llm/model_settings.py`](../curator_llm/model_settings.py). This document holds the evidence.

## 1. Defaults that came out of it

| Setting | Default | Override |
|---|---|---|
| Extraction model | `claude-sonnet-5`, thinking **off** | `LLM_MODEL_EXTRACTION`, `LLM_THINKING_EXTRACTION=on` |
| Extraction concurrency | 16 chunks at once (raw text of the previous 2 chunks as context) | `EXTRACT_WORKERS` or `--workers` (`1` = original serial mode) |
| Prompt caching | on (instruction block of every extraction and judge call) | – |
| Merge duplicate judge | `claude-sonnet-5`, thinking **off** | `LLM_MODEL_MERGE`, `LLM_THINKING_MERGE=on` |
| Merge pair pre-filter | on (skip pairs whose inputs or outputs share no entity; catalyst counts as an input) | `run_merge.py --no-prefilter` |
| Draft build | `claude-sonnet-5-5`, thinking **off** | `LLM_MODEL_DRAFT`, `LLM_THINKING_DRAFT` |
| Section-title checks | `claude-sonnet-5`, thinking always off (`max_tokens` is 100–300) | – |
| Paper with no Results section | analyse the **whole paper** (Abstract to References), no LLM boundary guess | – |
| Chat | `claude-sonnet-5-5` (unchanged) | `LLM_MODEL_CHAT` or `CHAT_MODEL` |
| Gene-first pipeline, reaction conversion | `claude-sonnet-4-6` (unchanged, untested) | `LLM_MODEL_GENE`, `LLM_MODEL_CONVERT` |
| Cross-model review | `gpt-5.6-luna` (unchanged) | `LLM_MODEL_REVIEW` |

Turning thinking off differs by model: `claude-sonnet-5` takes `{"type": "disabled"}`, `claude-sonnet-5-5` takes
`{"type": "between_tools"}` (the API rejects `disabled`). `thinking_off()` in `model_settings.py` picks the right one.
`claude-sonnet-5` also rejects `temperature`, so roles on it send none.

## 2. Where the time went (PINK1, 20 chunks)

| Stage | Before | Evidence |
|---|---|---|
| Extraction | serial, 20 calls, 142k tokens in / **61k out** | most of the output tokens were hidden thinking |
| Merge | 233 calls, 394k in / 22k out, 12 concurrent | calls are tiny, so merge was never the bottleneck |
| Draft build | one structured call, about 48 s, about 4k tokens out | one serial output |

The original whole annotation took about 10 minutes (reported by the user, not re-measured end to end).

**The key finding is that extended thinking was most of the latency.** One PINK1 chunk, same prompt, same 4 reactions found:

| | Time | Output tokens |
|---|---|---|
| default (thinking on) | 95 s | 10,754 |
| thinking disabled | 22 s | 2,984 |

## 3. PINK1 (Kane et al. 2014, PMID 24751536, `PINK1.pdf`)

### Extraction

| Setting | Time | Reactions | Output tokens |
|---|---|---|---|
| parallel, 8 workers, thinking on | 306 s | 26 | 80k |
| parallel, 20 workers, thinking on | 119 s | 26 | 75k |
| parallel, 16 workers, **thinking off** | **30–32 s** | 31–34 | 25–27k |
| Sonnet 5.5, thinking off | 17 s | 18 (9 regulators) | 16k |
| Sonnet 5.5, thinking on | 32 s | 23 (8 regulators) | – |

- Quotes found in the paper: thinking on 50 exact + 8 fuzzy of 58; thinking off 58 exact + 10 fuzzy of 68. None unverified in either.
- Thinking off found more: regulators 27 vs 16, catalysts 18 vs 14, reactions with a condition 31 vs 20.
- Sonnet 5.5 found about half the regulators of Sonnet 5 (9 vs 28).
- Prompt caching worked (115k tokens read from cache in one run). The first chunk runs alone so its call writes the cache.
- 20 workers ran with no failures or 429s; the default is 16 to leave headroom.

### Merge (same extraction file, 29 unique reactions)

| Judge | Time | Reactions out | Notes |
|---|---|---|---|
| Sonnet 5, thinking on | 77 s | 15 | baseline |
| Sonnet 5, thinking off | **44 s** | 14 | same 24 merged pairs, plus 1 more (a true duplicate) |
| Haiku 5.5, thinking off | 33 s | 17 | **missed 10 of the 24 merges**, e.g. "PINK1 phosphorylates free ubiquitin" vs "PINK1 phosphorylates ubiquitin". Not safe. |
| Sonnet 5.5, thinking off | 54 s | 15 | 24 of 25 merged pairs agree with Sonnet 5; no faster |

Earlier 8-worker run: pre-filter left 137 of 300 pairs to judge; merge took 64 s and gave 14 reactions.

### Pair pre-filter

- Judging every pair is O(n²). The filter skips a pair when the inputs, or the outputs, share no name token (a stoplist drops modification and cofactor words).
- On saved PINK1 runs: 1,291 pairs, 691 kept (46% skipped). **0 of 61 pairs that were really merged were blocked.**
- First version blocked 4 of 61, because its stoplist removed "ubiquitin". Fixed.

### Draft build (12 merged reactions)

| Model | Time | Participants | Notes |
|---|---|---|---|
| Sonnet 4.6 (the old default) | 46 s | 27 (21 protein, 3 complex, 2 set, 1 simple) | ubiquitin typed as a protein |
| Sonnet 5, default | 25 s | 29 | **ubiquitin typed as a small molecule, five duplicate "UB" entries** |
| Sonnet 5, thinking off | 27 s | 29 | same problem |
| Sonnet 5.5, thinking off | **17 s** | 30 (23 protein, 2 simple: CCCP, ATP) | ubiquitin correct, 0 notes |
| Sonnet 5.5, default | 36 s | 29 | fine, slower |

Sonnet 5 ignored an existing prompt rule here, so it was a model difference and not a prompt gap.
Through the real default code path: 22 s, 12/12 reactions.

## 4. Two clock papers (to check the PINK1 findings hold elsewhere)

`clock_nature.pdf` (Hirayama et al., CLOCK acetylation of BMAL1) and `clock_science.pdf` (Ramsey et al., NAMPT and
NAD+; an author manuscript with no section headings at all).

### Extraction

First row is the original serial pipeline. Times are the extraction step only.

| Setting | Nature: time, reactions, regulators | Science: time, reactions, regulators |
|---|---|---|
| original (serial, Sonnet 5, thinking on) | 380 s, 15, 9 | 319 s, 17, 21 |
| parallel, thinking on | 76 s, 16, 11 | 98 s, 19, 17 |
| **parallel, thinking off (default)** | **21 s, 22, 14** | **30 s, 21, 22** |
| Sonnet 5.5, thinking on | 19 s, 14, 4 | 22 s, 11, 6 |
| Sonnet 5.5, thinking off | 11 s, 10, 5 | 20 s, 8, 2 |

Quotes not found in the paper: default 0 of 50 (Nature) and 2 of 33 (Science); original 0 of 44 and 1 of 33.

### Merge

Thinking on vs off took 88 s vs 49 s on Nature and 36 s vs 24 s on Science. Groupings were identical on Nature (18
merged pairs). On Science they differed by 1–3 pairs, the same size as run-to-run noise.

### Draft

| | Nature (10 reactions) | Science (17 reactions) |
|---|---|---|
| Sonnet 4.6 | 27 s | 55 s, 54 participants |
| Sonnet 5 | **failed schema validation** | 38 s, 53 participants |
| **Sonnet 5.5, thinking off** | **14 s**, same structure as 4.6 | **22 s**, 47 participants |

## 5. Entity types for genes, mRNAs and promoters

**Problem.** On the Science paper the draft typed about 33 participants as simple entities: `Per2 mRNA`, `Bmal1 gene`,
`Per2 promoter E-box`, `Per2 oscillation`, `NAMPT-mediated NAD+ biosynthesis`. Sonnet 4.6 and 5.5 both did this.

**Fix** (code, not only prompt, because the draft model had no type for them): new draft kinds `gene` and `rna`.
They become `EwasSpec` with `molecule='dna'|'rna'`, emitted as EntityWithAccessionedSequence pointing at a
`ReferenceDNASequence` or `ReferenceRNASequence`, with no UniProt lookup and a curator warning to set the identifier.
The prompt says: `small_molecule` is for chemical compounds only; promoters, enhancers and E-boxes are regions of a
gene, not entities; "expression" is the mRNA entity; processes and phenotypes are not entities.

**Result on the Science paper (real model):** 9 simple entities, all real compounds (NAD+, nicotinamide, FK866, EX-527,
resveratrol); `NAMPT gene`, `PER2 gene`, `PER2 mRNA` as sequence entities; complexes such as `CLOCK:BMAL1:NAMPT gene`.
Not re-checked on PINK1 after the change.

## 6. Papers with no Results section

The old behaviour was to abort. Now the whole paper is analysed (`whole_paper` in
`reactome_llm/FullTextPDFSections.py`: from the Abstract heading to the References heading).

- The LLM boundary guess was dropped for this case because it was not repeatable: on the Science paper one run
  chose 1,266 of 1,669 words (14 reactions) and another chose none.
- Science paper: 1,669 words, 10 chunks, 21 reactions.
- Nature paper: 4,911 words, 30 chunks, 32 reactions (22 when the LLM chose a 2,397-word span). The last 10 chunks
  (Methods) returned no reactions: no junk, just idle calls.
- Review articles with no Results section also take this path (23,000+ words gives 76 chunks).

## 7. Accuracy against curated Reactome reactions (8 papers)

Method: [`scripts/ground_truth_eval.py`](../scripts/ground_truth_eval.py). For each PMID it fetches the curated human
reactions that cite it from the Reactome graph and scores each configuration. A curated reaction is **recalled** when
one extracted reaction names its catalyst (if any) and at least 60% of its protein/gene/RNA entities, by any alias.
Entity recall asks whether the extraction names the curated entities anywhere. "Matches curated" is only an upper
bound on precision, because curators record a subset of what a paper says.

Papers: six research papers with a tagged Results section (24443471, 19490898, 20966070, 23667505, 32448362,
24368734; 41 curated reactions) and two review articles (23256519, 25670789; 20 curated reactions).

| Setting | Research: recall | Research: entity recall | Review: recall |
|---|---|---|---|
| original (serial, thinking on, no pre-filter) | 28/41 (68%) | 66% | 12/20 |
| serial extraction + new merge | 28/41 | 66% | 12/20 |
| **new defaults** | **29/41 (71%)** | **80%** | **12/20** |
| defaults, merge thinking on | 29/41 | 80% | 12/20 |
| defaults, merge without pre-filter | 29/41 | 80% | 7/9 (one review only) |
| Sonnet 5.5 extraction (thinking on) | 29/41 | 67% | 11/20 |

Reactions extracted: original 52, new defaults 91, Sonnet 5.5 42 (research papers). "Matches curated": 93% for the
defaults, 92% for the original, 86% for Sonnet 5.5 (research); 38–43% on reviews, which describe far more than gets curated.

Reading it:
- The defaults lose nothing against the original (+1 reaction, within noise) and name more of the curated entities.
- The pre-filter and merge thinking made no difference to recall. On the big review the pre-filter skipped 13,292 of
  16,653 pairs.
- Sonnet 5.5 matched curated recall but extracted half as many reactions and named fewer curated entities (67% vs
  80%). Extraction stays on Sonnet 5, but this is closer than the PINK1 numbers alone suggested.
- **Recall tops out near 70% for every setting.** Paper 24443471 has seven curated "MITF-M-dependent <gene>
  expression" reactions whose target genes are never named in the text (likely a supplementary table). Paper 32448362
  needs SMAD complexes the extraction never saw. No setting change recovers these.

## 8. Bugs found along the way

- **Section lookup returned empty replies** for papers without a standard Results heading: thinking used up the tiny
  `max_tokens`. Fixed by always sending thinking off for those calls.
- **Pre-filter blocked true duplicates** twice: first through a stoplist that dropped "ubiquitin" (PINK1), then because
  the extractor sometimes lists the enzyme as an input and sometimes only the substrate (NAMPT paper). Fixed; there is a
  regression test.
- **Models were configured in about 10 places**, with mismatched ids. Now all in `model_settings.py`.
- Bare PDF filenames given to `run_extraction.py` are looked up in `data/fulltext_pdf/`, not `data/papers/`; pass an
  absolute path (the service already does).

## 9. Not tested, or weakly tested

- Chunk size (1,200 characters), re-vote threshold (0.9), cluster-validation size (3), merge workers (12), extraction
  `max_tokens` (16,000).
- Sonnet 5.5 for the chat model and section checks; Haiku for anything other than the duplicate judge.
- The gene-first pipeline's models (`gene`, `convert`, still Sonnet 4.6) and its prompts.
- The PMC XML route end to end through the REST service, and a paper longer than about 20 pages at 16 workers.
- Entity typing after the gene/rna change was checked on one paper only.
- Only 8 papers were scored against curated reactions; that is enough to show no regression, not to rank close options.
- Papers with a PMC record may have no full-text body (3 of my first 5 candidates), so check before choosing test papers.

## 10. How to reproduce

```bash
# curated-reaction comparison (needs Neo4j with the Reactome graph; resumable; slow for the serial baseline)
python scripts/ground_truth_eval.py run   24443471 19490898 20966070 23667505 32448362 24368734
python scripts/ground_truth_eval.py score 24443471 19490898 20966070 23667505 32448362 24368734

# one paper with a non-default setting
EXTRACT_WORKERS=1 LLM_THINKING_EXTRACTION=on python run_extraction.py /abs/path/paper.pdf --gene x --tag mytag --overwrite
LLM_MODEL_MERGE=claude-sonnet-5-5 python run_merge.py results/<stem>_extraction.json --no-prefilter
```

Scratch outputs of these runs are in `results/gt_*` (gitignored) and can be deleted.
