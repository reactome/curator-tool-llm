# Full-text extractor — integration interface

This is the **only** surface you need to build against. The pipeline handles
everything else: choosing which papers to read, locating the files, downloading
missing ones, and merging your output into the rest of the annotation.

## What the pipeline gives you

For each selected paper, the pipeline resolves a local file:

- a **PDF** from the curator's own folder, or
- a **PMC XML** the pipeline downloaded and cached.

It then calls your function once per resolved paper. Papers with no obtainable
full text (a "miss") are never passed to you.

## The function to implement

```python
def extract_fulltext(path: str, source: str, gene: str, model=None) -> dict:
    """Parse one full-text paper and return gene-related molecular evidence.

    Args:
        path:   Absolute path to a local file that is guaranteed to exist.
        source: "pdf" or "xml" — the file type at `path`.
        gene:   Target gene symbol, e.g. "SHANK3". Focus extraction on this gene.
        model:  A ChatAnthropic instance supplied by the pipeline. Use THIS for any
                LLM calls — do not construct your own Anthropic client (keeps token
                accounting + cost tracking correct). May be None in unit tests.

    Returns:
        dict with exactly the three lists below. Empty lists are fine.
    """
```

- **No CrewAI, no pipeline imports.** Just a plain function I can `import` and call.
- Handle **both** `source` types (PDF and XML) inside this one function.

## Required return schema

```json
{
  "interactions": [
    {
      "partner": "CTTNBP2",              // REQUIRED — interacting gene/protein symbol
      "interaction_type": "binding",     // binding / regulation / phosphorylation / ...  ("" ok)
      "evidence": "co-IP in HEK293",     // experimental method / description ("" ok)
      "confidence": "high",              // high / medium / low ("" ok)
      "evidence_strength_score": 0.8,    // REQUIRED type — float 0.0-1.0
      "pmid": "25391454",                // REQUIRED — the PMID of THIS paper
      "context": "..."                   // brief biological context ("" ok)
    }
  ],
  "pathways": [
    {
      "pathway_name": "...",             // REQUIRED
      "role": "regulator",               // catalyst / regulator / target / ... ("" ok)
      "evidence": "...",
      "confidence": "medium",
      "evidence_strength_score": 0.5,    // float 0.0-1.0
      "pmid": "..."                      // REQUIRED
    }
  ],
  "functions": [
    {
      "function": "...",                 // REQUIRED
      "evidence": "...",
      "confidence": "...",
      "evidence_strength_score": 0.0,    // float 0.0-1.0
      "pmid": "..."                      // REQUIRED
    }
  ]
}
```

### Two non-negotiables
1. **`pmid` on every item.** Provenance and the pipeline's `papers_processed` count
   depend on it. Use the PMID of the paper you were handed.
2. **`evidence_strength_score` is a float** in `[0.0, 1.0]`, not a string.

Field names must match exactly — they map straight onto the pipeline's Phase 1
schema (`LiteratureExtraction`). Anything extra you add will be ignored.

## Errors

If a file is unparseable or empty, return the empty result rather than raising:

```python
{"interactions": [], "pathways": [], "functions": []}
```

(Log the reason if useful. The pipeline degrades gracefully on empty output.)

## Dependencies

List every library + version you need to parse PDFs and XML (e.g. PyMuPDF,
pdfplumber, lxml). They'll be added to the project's environment and must import
cleanly in the `paperqa` conda env.

## What you do NOT handle

- Deciding which papers to read
- Locating files, PMID lookup, or downloading anything
- Config, the manifest, or any pipeline wiring

You receive a file path that exists and a source type. Parse it, return the
schema above. That's the whole job.
