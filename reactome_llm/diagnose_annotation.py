"""Reusable diagnostic report for a CrewAI annotation result.

Usage (from repo root, paperqa env):
    python reactome_llm/diagnose_annotation.py results/annotation_RAB6C_current.json
    python reactome_llm/diagnose_annotation.py RAB6C          # resolves results/annotation_RAB6C_current.json

Reports five dimensions instead of just the headline score:
  1. Pathway grounding      - output pathways that are real Reactome names vs curator-invented
  2. Evidence traceability  - cited PMIDs that don't trace to Phase-1 extraction (should be none)
  3. Literature specificity - of the papers actually used, how many mention the target gene
  4. Reviewer concerns      - flagged issues split into scientific vs technical
  5. Quality scores         - per-dimension breakdown, not just overall

Neo4j / Mongo / UniProt lookups are best-effort: if a source is unreachable the section
degrades to "n/a" rather than failing.
"""
import os, sys, re, json, glob

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("reactome_llm")


def _load(path_or_gene):
    if os.path.isfile(path_or_gene):
        return path_or_gene, json.load(open(path_or_gene))
    hits = sorted(glob.glob(f"results/annotation_{path_or_gene}*.json"))
    if not hits:
        raise SystemExit(f"No result file for '{path_or_gene}' (looked for results/annotation_{path_or_gene}*.json)")
    # prefer a *_current.json if present
    path = next((h for h in hits if h.endswith("_current.json")), hits[0])
    return path, json.load(open(path))


def _existing_pathway_names(names):
    """Which of `names` exist as Pathway nodes anywhere in the Reactome graph (leaf OR ancestor).

    Replaces the old leaf-only flat-file lookup (ReactomePathwayGenes_Ver_91.txt), which
    false-flagged real higher-level pathways as "invented" -- e.g. SHANK3's "Protein-protein
    interactions at synapses" (dbId 6794362), a genuine ancestor pathway absent from the leaf TSV.
    Queries (p:Pathway) by displayName (the graph holds leaves and their ancestors), so any real
    Reactome pathway is recognized regardless of hierarchy level.

    Returns the subset of `names` found in the graph; None if Neo4j is unreachable (caller
    degrades the section to "n/a" rather than false-flagging). Diagnostic-only -- does not touch
    the retrieval/placement pipeline.
    """
    names = [n for n in names if n]
    if not names:
        return set()
    try:
        import ReactomeNeo4jUtils as neo
        from neo4j import GraphDatabase
        import neo4j as _neo4j
        query = ("MATCH (p:Pathway) WHERE p.displayName IN $names "
                 "RETURN DISTINCT p.displayName AS name")
        with GraphDatabase.driver(neo.URI, auth=neo.AUTH) as driver:
            df = driver.execute_query(query, db=neo.DB, names=names,
                                      result_transformer_=_neo4j.Result.to_df)
        return set(df["name"].astype(str)) if df is not None and not df.empty else set()
    except Exception:
        return None  # signal "unavailable"


def _synonyms(gene):
    try:
        import QueryBuilder as qb, ReactomeNeo4jUtils as neo
        return qb.get_uniprot_synonyms(neo.query_accession_for_gene(gene)) or []
    except Exception:
        return []


def _mentions_gene_fraction(pmids, gene, syns):
    """Best-effort: fraction of the given PMIDs whose cached abstract mentions gene/synonyms."""
    try:
        from ReactomePubMed import ReactomePubMedRetriever
        r = ReactomePubMedRetriever()
    except Exception as e:
        return None, f"cache unavailable ({e})"
    terms = [t for t in [gene] + syns if t]
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
    checked = hit = 0
    for p in pmids:
        try:
            doc = r.get_abstract_from_mongodb(p)
        except Exception:
            doc = None
        if not doc:
            continue
        checked += 1
        if pat.search(doc.get("Summary", "") or ""):
            hit += 1
    return (hit, checked), None


TECH_KW = ["schema", "identifier", "stable id", "stableidentifier", "displayname",
           "display name", "format", "ptm", "hasmodifiedresidue", "modification annotation",
           "go term", "go molecular function", "referential", "integration", "already exists",
           "naming convention", "duplicat", "field", "catalystactivity"]
SCI_KW = ["evidence", "unsupported", "incorrect", "wrong", "biolog", "implausible",
          "contradict", "weak", "overstat", "mechanis", "not supported", "questionable",
          "single reference", "validation recommended", "indirect"]


def _classify(issue):
    t = issue.lower()
    if any(k in t for k in TECH_KW):
        return "technical"
    if any(k in t for k in SCI_KW):
        return "scientific"
    return "other"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: diagnose_annotation.py <result.json | GENE>")
    path, d = _load(sys.argv[1])
    gene = d.get("gene", "?")
    ri = d.get("reactome_instances", {}) or {}
    le = d.get("literature_evidence", {}) or {}
    vr = d.get("validation_report", {}) or {}
    fc = d.get("final_consensus", {}) or {}
    qs = d.get("quality_scores", {}) or {}

    print("=" * 72)
    print(f"DIAGNOSTIC REPORT — {gene}   ({os.path.basename(path)})")
    print("=" * 72)

    # 1. Pathway grounding
    out_paths = [p.get("displayName", "") for p in ri.get("pathways", [])]
    existing = _existing_pathway_names(out_paths)
    print("\n[1] PATHWAY GROUNDING")
    if existing is None:
        print("    n/a (Neo4j unavailable)")
        for p in out_paths:
            print(f"      ? {p}")
    else:
        real = [p for p in out_paths if p in existing]
        invented = [p for p in out_paths if p not in existing]
        print(f"    real Reactome: {len(real)}/{len(out_paths)}   invented: {len(invented)}/{len(out_paths)}")
        for p in out_paths:
            print(f"      {'REAL    ' if p in existing else 'INVENTED'}  {p}")

    # 2. Evidence traceability
    extract_pmids = {str(it.get("pmid")) for grp in ("interactions", "pathways", "functions")
                     for it in le.get(grp, []) if it.get("pmid")}
    cited = {str(x) for grp in ("complexes", "reactions", "pathways")
             for it in ri.get(grp, []) for x in it.get("literatureReference", [])}
    leaked = sorted(cited - extract_pmids)
    print("\n[2] EVIDENCE TRACEABILITY")
    print(f"    cited PMIDs: {len(cited)} | Phase-1 extraction PMIDs: {len(extract_pmids)}")
    print(f"    cited but NOT traceable to extraction: {leaked if leaked else 'none (clean)'}")

    # 3. Literature specificity
    used = sorted(extract_pmids)
    syns = _synonyms(gene)
    print("\n[3] LITERATURE SPECIFICITY (papers actually used)")
    res, err = _mentions_gene_fraction(used, gene, syns)
    if res is None:
        print(f"    n/a ({err})")
    else:
        hit, checked = res
        pct = (100 * hit / checked) if checked else 0
        print(f"    of {checked} used papers with a cached abstract, {hit} mention {gene} "
              f"(or synonyms) = {pct:.0f}% on-target")
        if checked and pct < 40:
            print("    ⚠ low specificity — papers may be about a broader topic than the gene itself")

    # 4. Reviewer concerns
    issues = []
    for r in vr.get("instance_reviews", []):
        issues += r.get("issues", []) or []
    issues += fc.get("blocking_issues", []) or []
    buckets = {"scientific": [], "technical": [], "other": []}
    for it in issues:
        buckets[_classify(it)].append(it)
    print("\n[4] REVIEWER CONCERNS")
    print(f"    scientific: {len(buckets['scientific'])} | technical: {len(buckets['technical'])} | other: {len(buckets['other'])}")
    for kind in ("scientific", "other", "technical"):
        for it in buckets[kind][:6]:
            print(f"      [{kind[:4]}] {it[:130]}")

    # 5. Quality scores
    print("\n[5] QUALITY SCORES")
    for k, v in qs.items():
        print(f"    {k}: {v}")
    print(f"    validation approval : {vr.get('approval_status')}  (overall {vr.get('overall_score')})")
    print(f"    consensus decision  : {fc.get('decision')}  (confidence {fc.get('confidence')})")
    print("=" * 72)


if __name__ == "__main__":
    main()
