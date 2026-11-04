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

### 7a. Direct evidence, extraction exclusions and site-aware merging (2026-11)

Changes: research papers drop reactions whose every quote is `cited` (`services/direct_evidence.py`; reviews keep them,
`--review`); the extraction prompt excludes hypotheses/model predictions, cell-level outcomes, intra-complex structural
contacts and tool-only events, and gives a modification site only to the enzyme its sentence names; the merge judge
treats fewer named sites as less detail. Same 8 papers, the 2 reviews run with `--review` (`ground_truth_eval.py run ...
--only gtv2`), compared with the stored new-defaults run:

| Setting | Recall (all 8) | Research recall | Entity recall | Reactions (research) |
|---|---|---|---|---|
| new defaults (before) | 41/61 | 29/41 | 78% | 91 |
| **2026-11 changes** | **41/61** | **29/41** | 77% | 86 |

- Recall is the same on every paper. One curated entity fewer was named (24368734: 11/13 vs 12/13), where extraction
  went from 37 to 23 records and the merge folded them to 11 reactions (was 23).
- The cited-only filter left out ONE reaction across the six research papers: these are extracted from a tagged Results
  section, which cites little. It matters for papers read whole (no tagged Results, e.g. the Nature WEE1/CDH1 PDF):
  there it left out 7 of the reactions, all background stated with references.
- WEE1/CDH1 PDF (whole paper, 121 chunks): 42 reactions before, 31 after. Gone: the WEE1-at-Thr129 record (a site the
  text gives to mTOR), the model hypotheses, cell-level outcomes and the degron artefact; site lists no longer split
  the CDH1 phosphorylation. Still wrong: CDH1 phosphorylation is 3 reactions (the in-vitro N186 fragment and the
  Xenopus ortholog were kept as named and judged different participants); a quote carrying "(ref. 43)" was tagged
  `this_paper`; hydrogen-bond records, a hedged Discussion sentence and two figure-legend-only "results" (one the paper
  contradicts) were still extracted despite the prompt rules.

Then constructs and orthologs as the protein (extraction normalizes "CDH1 N186", "Xenopus FZR (Cdh1)" to CDH1; the
merge judge treats an assay construct or stand-in ortholog as the protein), same 8 papers (`--only gtv3`):

| Setting | Recall (all 8) | Entity recall | Reactions (all 8) |
|---|---|---|---|
| 2026-11 changes above | 41/61 | 77% | 274 |
| **+ constructs/orthologs as the protein** | **42/61** | **80%** | 258 |

Within noise on recall (one review gained one curated reaction); 16 fewer reactions. On the WEE1/CDH1 PDF the CDH1
phosphorylation became ONE reaction (six variants folded: no site, Tyr130, Tyr91+Tyr130, "tyrosine residues"), 30
reactions in all. Run-to-run variation of the extraction is visible there: a negative result ("WEE1 phosphorylates
CDC20 (negligible)") and a Methods rationale ("CDK1 phosphorylates WEE1") appeared that the previous run did not have.

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
- **Fixed 2026-11-02 (was: a cross-paper exact duplicate was dropped with its quotes):** an identical reaction in another
  paper is no longer dropped; it stays its own record with its own paper, and the cross-paper stage (below) merges it
  for free. With `--no-cross-paper` the two simply stay separate reactions. Every quote carries its own `source`.
- Papers with a PMC record may have no full-text body (3 of my first 5 candidates), so check before choosing test papers.

## 10. REST service: measured end to end, progress messages, parallel lookups

Measured over real HTTP with the production wiring (only the login check faked) on `PINK1.pdf`:

| | Before | After |
|---|---|---|
| Whole job | 119 s | **109 s** |
| Identifier lookups (UniProt, OLS, Neo4j) | 11.0 s, one at a time | **about 2 s**, 8 at a time |
| Quote check + draft build | 24 s | unchanged (one LLM call) |
| Every other endpoint (session, issues, export, network, existing, usage, search) | about 10 ms | unchanged |
| Chat turn / per-reaction QA | 5-8 s / 9 s | unchanged (not optimised) |

The job's `progress` text now follows the work instead of saying "several minutes" for the whole extract-and-merge
phase: `extracting reactions: 7/20 chunks`, `merging reactions: compared 200/217 pairs`, `re-checking uncertain
pairs`, `checking the groups`, `combining duplicate details`, then the existing `building the Reactome draft` and
`resolving identifiers`. It is read live from the scripts' output (`StepProgress` in `services/extraction.py`).
The longest silent step left is the draft build (about 20 s, one LLM call).

Lookups run in a thread pool (`RESOLVE_WORKERS`, default 8, `1` = one at a time). Results and the order of the notes
are identical to the serial version (a test compares the two). The service must be **restarted** to pick up changes
to in-process code (resolvers, pipeline); the extraction and merge scripts are re-read on every run.

### Per-reaction QA: does thinking matter?

35 reactions from 3 papers (PINK1, clock Nature, clock Science), each reviewed three times with `claude-sonnet-5-5`:
thinking on, thinking on again (to measure run-to-run noise), and thinking off (`between_tools`).

| | on | on again | off |
|---|---|---|---|
| Latency, mean | 7.4 s | 7.5 s | 7.1 s |
| Output tokens, mean | 847 | 855 | 837 |
| Verdict `needs_work` | 35/35 | 35/35 | 35/35 |
| Mean score | 0.39 | 0.38 | 0.38 |
| Findings per reaction | 5.3 | 5.3 | 5.2 |
| LLM review failed | 0 | 0 | 0 |

Same verdicts in all 35 cases. Score difference on vs off: 0.037 mean (max 0.10), against 0.023 between the two
identical "on" runs. Finding text overlap is about 0.37 for on vs off and 0.38 for on vs on again, so the wording
varies as much between repeats as between settings. **Thinking off does not change the output, and it does not make QA
faster**: 5.5 does not think by default (the API says so), and the 850 output tokens are the findings JSON itself.
The `qa` role (`model_settings.py`) is set to thinking off anyway, with its own model instance so chat is unchanged.
Chat's own `run_qa` tool still uses the chat model.

Every one of the 35 reactions came back `needs_work` (mean score about 0.38), in all three runs. Why:

- The verdict rule (`qa_reaction`) is: `needs_work` if any finding has severity `action`, or if the LLM's own verdict
  is `needs_work`. The deterministic rule checks never emit `action` (only warning/info), so all 35 verdicts come from
  the LLM reviewer: 28 reactions had at least one LLM `action` finding (22 had two), the other 7 had none but the LLM
  still answered `needs_work`.
- The findings are mostly substantive (a missing PPi output, a summation that overstates the quote, a null compartment,
  a mutant ubiquitin recorded as plain ubiquitin, a reagent such as calf intestinal phosphatase recorded as a
  regulator), so the drafts do have real problems.
- But `QA_SYSTEM` never says when to answer `ok`, what separates `action` from `warning`, or what a score means. The
  best score of all 35 was 0.60, and reactions with only warnings and notes were still `needs_work`.
- Test of a prompt that defines them (`action` = wrong or unsupported as written, `needs_work` only with an action, score
  bands; not adopted): 7 of 35 reactions became `ok`, scores spread to 0.15-0.80, and 6 of 6 deliberately broken
  controls (wrong catalyst, no quotes) were still `needs_work`. Even then 28 of 35 kept an `action` finding, so most of
  the pattern is the drafts, not the prompt. Scores of the broken controls (0.2-0.55) overlap the real reactions'
  (0.05-0.60), so the score alone does not separate them.

### Making QA output useful to a curator

Problems with the QA output: a binary verdict that was `needs_work` for every reaction, findings in the model's own
order, about 9 of 44 notes that only confirmed something was fine ("no catalyst is correct for a binding reaction"; each
became an issue to dismiss), no one-line takeaway, and no way to see which reaction to open first. Also, QA held the
session lock while it waited for the model.

Changes (`services/qa.py`, `session_service.py`, `api/app.py`; front-end issues panel and reaction detail):

- Prompt: defines action / warning / info, the verdict rule and score bands; "report problems only, never say what is
  correct"; each finding says what to change; at most 5 findings, most important first; a one-sentence `summary`.
- `priority` derived in code from the findings: `fix_first` (3 or more action findings), `fix` (1-2), `review`
  (warnings only), `ok`. The verdict is kept for older callers.
- Findings sorted most serious first; `POST /sessions/{id}/qa` checks all reactions concurrently and returns them
  ranked (priority, then action count, then warnings, then lower score); the front-end issues panel has a "Check all
  reactions" button and a "Look at these first" list.
- The model call now runs outside the session lock (compute, then apply under the lock).

Measured on the same 35 reactions plus 6 deliberately broken controls (wrong catalyst, no quotes):

| | Before | After |
|---|---|---|
| Findings that only praise | 9 of 44 notes | **0** |
| Findings per reaction (LLM) | 5.3, no cap | 4.7, max 5 |
| Reactions with a summary | 0 | **35/35** |
| Verdict `needs_work` | 35/35 | 35/35 (the reviewer still finds an action in every draft; 75 action findings, 2.1 per reaction) |
| `priority` | – | `fix_first` for the worst, `fix` for the rest |
| Broken controls | 6/6 `needs_work` | 6/6 `needs_work` (5 of 6 with 3-4 actions) |

Number of action findings per real reaction: 1 for 7, 2 for 17, 3 for 10, 4 for 1; the broken controls had 2-4.
The cutoff of 3 is a judgement fitted to these 35 reactions, not a validated threshold. Scores still do not separate
broken from real reactions (controls 0.2-0.5, real 0.2-0.6), so the list is ranked by action count first.
End to end on the saved PINK1 session: all 11 reactions checked in 14.8 s, ranked, each with a summary.

### Merging across papers (2026-11-02)

**Step 1, per-quote source.** `stamp_sources` labels every quote with its paper before records are combined; each quote
is verified against its own paper (`attach_evidence`); `Evidence.source` records it (a PDF has no PMID); the draft's
`pmids` is the union over a merged reaction's papers. Identical reactions fold only within a paper.

**Step 2, speed.** Clustering per connected component (`merge_support.conflict_free_clusters`): identical groups to the
old all-pairs scan (checked against a copy of the old algorithm on 300 random inputs with conflicts and ties), but 600
reactions cluster in well under a second instead of 13 s. A **verdict cache** (`data/merge_cache/judge.jsonl`,
`--no-cache` or `MERGE_CACHE=0` to turn off), keyed by the model, its thinking setting, a fingerprint of the judge's
rules and the two reactions' text; re-votes always ask the model; a failed call is never cached. On 64 reactions from
two papers: judging 47 s cold, 0 s warm (390 verdicts reused); whole run 89 s and 608 calls cold, 18 s and 22 calls warm.

**Step 3, the cross-paper stage** (`merge_support.merge_across_papers`, run by default when the input has more than one
paper; `--no-cross-paper` to turn off; give `run_merge.py` several extraction files with `--out`):

1. Within-paper merge, as before.
2. Papers are added one at a time. A new reaction is compared only with the groups found so far, not with every
   reaction of every paper: candidates must pass the pre-filter and share no paper with it, are asked best-match first
   (TF-IDF over the entities named), at most 10 (`--cross-max-candidates`, default raised from 5, see below), stopping at the first "same". A group that
   already holds a reaction with the same signature is joined with no model call.
3. A group takes at most one reaction per paper (the more confident one wins a clash).
4. A final reconcile pass compares the groups with each other (cheap: far fewer groups than reactions), because the same
   reaction can start in two groups before either can see the other.
5. Groups of 3 or more are checked for drift by the model, as within a paper (told the papers differ); merged groups are
   cleaned up by the model; confidence is recomputed; `merge_confidence` is the weakest verdict; `cross_paper_merge_confidence`
   is the weakest cross-paper one.

Measured with the real model on 5 extraction runs of the PINK1 paper treated as 5 papers (the worst case: every
reaction has true duplicates in the other four; 121 records, 63 after the per-paper merge), against `--cross-all-pairs`
(judge every candidate pair, exact but quadratic, kept as a reference mode):

| | All pairs (reference) | Incremental, first version | Incremental, as shipped |
|---|---|---|---|
| Judge calls across papers | 929 | 109 | **116** (5 identical reactions joined free) |
| Reactions after the cross-paper merge | 22 | 25 | 20 |
| Pairs merged that the reference also merged | – | 66% recall | **98% recall, 91% precision** |

The first version lost recall (66%) for two reasons, both fixed: it folded identical reactions across papers up front,
which gave a folded reaction several "papers" and wrongly barred a later reaction of one of them from matching it; and
groups were never compared with each other. The reference is a proxy, not the truth: its clique rule (every pair in a group
must have been judged the same) is stricter, and most of the 9 pairs only the incremental stage merged look like
reasonable merges (`PINK1-dependent phosphorylation of Parkin` vs `... at S65`); one is arguable, the C431S
trapping mutant merged with the wild-type reaction. An earlier comparison in this work was thrown away because the
reference run had crashed and the file compared was a stale one; the numbers above are from a completed run.

**Several papers in one REST session (built later the same day, see below):** the service now runs this stage for a
session with several papers; `run_merge.py` also takes several extraction files directly. Not tested: many papers on different topics with the real
model (the diverse set was only simulated: about 100 calls for 8 papers), papers with very large numbers of reactions,
and papers with very large numbers of reactions.

**The cap on candidates** (how many existing groups the model is asked about per new reaction, best match first). Same
five-run PINK1 data (62 per-paper reactions), against the all-pairs reference (22 reactions, 90 merged pairs, 897 judge
calls); one run each, so differences of a few points are within the model's run-to-run noise:

| Cap | Reactions | Judge calls | Cap left candidates unasked for | Recall | Precision |
|---|---|---|---|---|---|
| 5 | 20 | 113 | 6 reactions/groups | 93% | 87% |
| 10 | 20 | 136 | 2 | 96% | 86% |
| 20 | 20 | 133 | 0 | 93% | 87% |

Cap 20 never cut a candidate off and did no better than 5, so the cap is not what limits accuracy here. Raising it to 10
costs about 20% more calls and cuts the cases where it mattered from 6 to 2; the default is now 10 as cheap insurance
against the similarity ranking (a word match, so a true match can rank low), not because a gain was shown. The run
prints how often the cap left candidates unasked (`[cross] ... left candidates unasked for N reactions/groups`).

A bug the unit tests could not see (a loop variable overwrote the `_key` function in `run_merge.py`, crashing it after
clustering) was caught by a real run. `tests/test_run_merge_script.py` now runs the actual script offline with a stub
model, including the cross-paper stage, and fails if that mistake comes back.

### Several papers in one REST session (2026-11-02)

`POST /sessions` takes `pmids`, `POST /sessions/upload` takes `files` and `pmids`; `scripts/annotate_papers.sh` does not use it: that script runs the pipeline in-process (`curator_llm/cli.py`) with no service, login or paper cap.
The pipeline extracts each paper on its own (`PAPER_WORKERS`, default 2 at a time), runs ONE `run_merge.py` over all
the extraction files (within each paper, then across papers), attaches each quote to its own paper, and keeps every
paper's text in the session (search, quote checks for edits). A paper that cannot be read is skipped and reported.

Real model, production wiring over HTTP, a mixed session: two PDFs (the PINK1 paper and a renamed copy, so they overlap
completely) plus PMID 32448362 (an open-access research paper):

| | |
|---|---|
| Whole job | 238 s (paper 3 waited for a free slot: 2 papers at a time) |
| Reactions in the draft | 20 (11 of them merged across the two PINK1 copies, `sources` lists both; 9 from one paper) |
| Evidence by paper | 62 + 24 quotes from the two PDFs, 8 from the PMID paper, each verified against its own text |
| Search | returns passages from the papers, each with its `source` |
| Progress | per paper while extracting, then the merge stages, then the draft |

Quote-level attribution undercounts overlap for duplicate copies: evidence is keyed by the quote text, so a sentence
found in both copies is attributed to the first. The reaction's `sources` list is right. Not tested: papers on different
topics through the service; sessions near `MAX_PAPERS`. The front-end's start form now takes several PubMed IDs and PDFs (unit-tested and built, not looked at in a browser).

### Command-line runner and the front-end start form (2026-11-02)

**Command-line runner** (`curator_llm/cli.py`, started by `scripts/annotate_papers.sh`). It runs the pipeline in this process,
with no service, login, upload or polling, and has no cap on the number of papers (`MAX_PAPERS` is the service's).
It writes `<name>.staged_instances.json`, `.issues.json`, `.session.json` and `.evidence.json`. The evidence file exists
because a local run has no session to ask for the quotes afterwards, and the staged-instances file does not hold them
(the curator tool drops fields it does not know, so the emitter returns them separately as `evidence_links`). It is
not loadable in the front-end's evidence viewer, which reads quotes from a service session.

Real model and real services, run from another folder: the clock Science PDF plus PMID 32448362, merged into one draft.

| | |
|---|---|
| Wall time | 2 min 4 s |
| Reactions in the draft | 22, from 2 papers |
| Open issues | 88 (evidence 1, resolver 18, existing-reaction check 38, emitter 31) |
| Quotes | 35 from the PDF, 11 from the PubMed paper, each tagged with its paper |
| Staged instances | 169 new instances |
| Model use | 155 calls, 142k tokens in, 36k out (178k in total), 272k read from the cache |

`tests/test_cli.py` (10 tests, fake pipeline): one merged draft vs `--separate`; the result files; progress is not printed
for every chunk; a finished result is skipped unless `--force`; one failing session does not stop the others and the exit
code says so; a skipped paper is reported; 40 papers at once are accepted; bad input (no papers, a non-number, a missing
file, an empty folder, a file that is not a PDF, two PDFs with the same name) is refused before anything runs.
The earlier version of the script talked to the service over HTTP and was tested against a stand-in service (merged,
separate, PubMed IDs only, error paths, the old wrapper, macOS bash 3.2); that script was then removed, so those runs
no longer describe anything that exists. Not tested: the runner with more than two papers on the real model.

**Front-end** (`curator-tool-frontend`, Paper annotation page): one "Papers to annotate" form takes one or more PubMed IDs
and several PDFs (listed, each removable); the session list shows "3 papers: ..."; the workspace header shows the
number and names of the papers; each quote in the reaction detail and the evidence dialog says which paper it is from
(only when the annotation has several). `LlmApiService.startPapers` replaces `start` and `upload`.

| Front-end check | Result |
|---|---|
| Paper-annotation specs | 144 pass (132 before; new ones cover the form, PMID parsing, PDF list, service calls, paper labels, evidence chips, workspace title, evidence dialog) |
| Development build (templates compiled) | clean |
| Whole suite | 1,206 pass, **2 fail**: `NLQueryPanelComponent` "emits the dbId when a dbId cell is clicked" and "switches to the raw rows and back" |

The 2 failures are in code this work did not touch, they fail every time (also when only that spec is run), and they
concern the instance-list view that recent `nl-query` commits changed, so they are very probably older than this work;
no clean baseline run was made (it would have meant stashing uncommitted changes), so that is not proven. One earlier
full run also failed in `InstanceEffects` once, and did not recur. The form has not been looked at in a browser.

**Test suites at the end of the previous step:** service repo 296 passed, 1 skipped (the skipped test needs a real
curator-tool-ws); front-end as above. Nothing is deployed to the server and nothing is committed.

### Fixing what the output showed, and suggesting fixes (2026-11-02)

The real clock Science PDF + PMID 32448362 run (22 reactions, 88 open issues) showed problems; the ones that were bugs are
fixed, and the ones that need judgement are offered as **suggested fixes**. Nothing is fixed automatically: a suggestion
is a pending proposal (actor `suggestions`) that changes the draft only when a curator accepts it.

| Problem in the output | What was done | Measured (live Reactome graph, 25 gene/transcript entities and 22 reactions of that run) |
|---|---|---|
| 22 serious "Reactome already has this reaction" issues, mostly wrong | Genes and transcripts have no UniProt accession, so a reaction about them is no longer compared (an info note says so); a match on the enzyme alone is never "same" | "same" matches 22 -> 6; 16 reactions skipped as gene/transcript reactions |
| 24 "new reference must be set" warnings, new ReferenceDNASequences for genes Reactome has | The resolver looks the gene symbol up among Reactome's ReferenceDNA/RNASequence and reuses the HUMAN one (the species is a graph relationship, so the lookup reads it from there); several candidates: the one named exactly, with a "confirm it" note; none stands out: no guess | new DNA references 15 -> 0, RNA 10 -> 9, warnings 24 -> 10; 16 of 25 reuse a human reference, no non-human id |
| An enzyme listed as an input (and not an output); a reaction with nothing changing | Structural checks (`qa_no_inputs`/`outputs`/`change`, catalyst and regulator rules, and the new `qa_catalyst_is_consumed`) now run on every refresh and show as issues (source `checks`) without a QA run. The evidence rules stay on demand | on the saved run: 2 `checks` issues (1 consumed enzyme, 1 no-change) |
| Body text labelled "Abstract" in a PDF with no headings | A text whose only heading is Abstract is labelled "Main Text" after 2,500 characters or the next page | tests only |
| Suggested fixes | `services/suggestions.py`: remove a consumed enzyme from the inputs (not when that would leave no input); cite the PubMed papers a reaction's quotes come from when it has no PMID. Made when a session is created and after each accepted proposal; `POST /sessions/{id}/suggestions`; `<name>.suggestions.json` from the command line; a "Suggested fixes" tab in the workspace. A rejected one is not made again; a pending one that no longer applies is dropped | on the saved run: 0 suggestions (its one consumed enzyme is the reaction's only input, so removing it would be wrong) |

**Not fixable safely, so left to the curator:** invented states such as "(repressed)"; related-but-different reactions kept
separate (and duplicates the merge missed: a patch cannot move evidence between reactions); evidence from another species
(whether to infer a human reaction is a curation decision); ChEBI identifiers (still flagged). The suggestions skip the
usual "needs a quote" rule because the curator who accepts one is the one deciding; the quote requirement still holds for
anything the chat proposes.

**What was not run:** the clock PDF was not at hand, so the whole pipeline was not rerun on it; the live numbers come from
re-resolving the saved draft against the graph, and the suggestion count from the saved run. The Suggested fixes tab has
been tested in specs and compiled, not looked at in a browser.

Tests: service repo 317 passed, 1 skipped; paper-annotation front-end 145 pass.

## 11. How to reproduce

```bash
# curated-reaction comparison (needs Neo4j with the Reactome graph; resumable; slow for the serial baseline)
python scripts/ground_truth_eval.py run   24443471 19490898 20966070 23667505 32448362 24368734
python scripts/ground_truth_eval.py score 24443471 19490898 20966070 23667505 32448362 24368734

# one paper with a non-default setting
EXTRACT_WORKERS=1 LLM_THINKING_EXTRACTION=on python run_extraction.py /abs/path/paper.pdf --gene x --tag mytag --overwrite
LLM_MODEL_MERGE=claude-sonnet-5-5 python run_merge.py results/<stem>_extraction.json --no-prefilter
```

Scratch outputs of these runs are in `results/gt_*` (gitignored) and can be deleted.
