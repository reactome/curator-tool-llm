"""Glue between the extraction JSON (reactions carrying quotes) and verified Evidence objects.

Through extraction and merge a reaction keeps `evidence` as a list of quote STRINGS (merge and
review already handle that shape) plus `evidence_details`, a {quote: metadata} map the LLM filled
in (supports / system / experimental_species / claim_origin ...). After merge, `attach_evidence`
verifies each quote against the paper and registers it in the EvidenceStore. `evidence` is narrowed to
the quotes that were kept (never truncated otherwise), `evidence_ids` runs parallel to it, and quotes
that could not be found are listed in `evidence_rejected`.
"""
from typing import Callable, Dict, List, Optional

from curator_llm.models.evidence import ClaimOrigin, Evidence
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.paper_text import PaperText
from curator_llm.services.quote_verifier import normalize_with_map
from curator_llm.services.sources import pmid_of

_SPECIES = {'human': 'Homo sapiens', 'humans': 'Homo sapiens', 'mouse': 'Mus musculus',
            'mice': 'Mus musculus', 'rat': 'Rattus norvegicus', 'fly': 'Drosophila melanogaster',
            'yeast': 'Saccharomyces cerevisiae', 'zebrafish': 'Danio rerio'}
_META_FIELDS = ('supports', 'system', 'experimental_species', 'claim_origin', 'cited_reference', 'strength')


def split_evidence(reaction: dict) -> dict:
    """Normalise one extracted reaction in place: evidence items may be strings (old prompt) or
    objects {quote, supports, ...}; result is `evidence` (strings) + `evidence_details` (by quote)."""
    items = reaction.get('evidence') or []
    quotes, details = [], dict(reaction.get('evidence_details') or {})
    for it in items:
        if isinstance(it, dict):
            q = (it.get('quote') or '').strip()
            if not q:
                continue
            meta = {k: it[k] for k in _META_FIELDS if it.get(k) not in (None, '', [])}
            if q in details:
                meta = {**details[q], **meta}
            details[q] = meta
        else:
            q = str(it).strip()
        if q and q not in quotes:
            quotes.append(q)
    reaction['evidence'] = quotes
    reaction['evidence_details'] = details
    return reaction


def merge_details(d1: Optional[dict], d2: Optional[dict]) -> dict:
    """Union two {quote: metadata} maps; `supports` lists are unioned, other fields keep the first value."""
    out = {q: dict(m) for q, m in (d1 or {}).items()}
    for q, m in (d2 or {}).items():
        cur = out.setdefault(q, {})
        for k, v in m.items():
            if k == 'supports':
                cur['supports'] = list(dict.fromkeys((cur.get('supports') or []) + list(v)))
            else:
                cur.setdefault(k, v)
    return out


def _species(name):
    return _SPECIES.get(str(name).strip().lower(), name) if name else name


def _draft(quote: str, meta: dict) -> Evidence:
    origin = meta.get('claim_origin') or 'this_paper'
    try:
        origin = ClaimOrigin(origin)
    except ValueError:
        origin = ClaimOrigin.THIS_PAPER
    sup = meta.get('supports') or []
    return Evidence(quote=quote, supports=[sup] if isinstance(sup, str) else list(sup),
                    system=meta.get('system'), experimental_species=_species(meta.get('experimental_species')),
                    claim_origin=origin, cited_reference=meta.get('cited_reference'),
                    strength=meta.get('strength'))


def attach_evidence(reactions: List[dict], store: EvidenceStore,
                    paper_for_source: Callable[[str], Optional[PaperText]]) -> Dict[str, int]:
    """Verify and register every reaction's quotes. `reactions` are the wrapped extractor dicts
    ({source, annotation_result}). Returns counts; a source with no paper text leaves its quotes
    unverified (kept, so nothing is silently dropped) and is reported."""
    stats = {'verified': 0, 'unverified': 0, 'rejected': 0}
    papers: Dict[str, Optional[PaperText]] = {}
    for r in reactions:
        a = r.get('annotation_result', r)
        split_evidence(a)
        src = r.get('source', '')
        if src not in papers:
            papers[src] = paper_for_source(src)
        paper = papers[src]
        details = a.get('evidence_details') or {}
        # the details map is keyed by the exact quote; consolidation may have re-cased/trimmed quotes
        norm_details = {normalize_with_map(q)[0]: m for q, m in details.items()}
        ids, kept_quotes, rejected = [], [], []
        for q in a.get('evidence', []):
            meta = details.get(q) or norm_details.get(normalize_with_map(q)[0]) or {}
            draft = _draft(q, meta)
            if paper is None:
                draft = draft.model_copy(update={'pmid': pmid_of(src)})
            ev, kept = store.add(draft, paper)
            if kept:
                if ev.id not in ids:           # two spellings of one quote share an id
                    ids.append(ev.id)
                    kept_quotes.append(q)
                stats['verified' if paper is not None else 'unverified'] += 1
            else:
                rejected.append(q)
                stats['rejected'] += 1
        # `evidence` and `evidence_ids` stay parallel: only verified quotes remain in `evidence`
        a['evidence'] = kept_quotes
        a['evidence_ids'] = ids
        a['evidence_rejected'] = rejected
    return stats




def attach_gene_evidence(reactions: List[dict], gene: str, results_dir: str,
                         paper_for_source: Optional[Callable[[str], Optional[PaperText]]] = None) -> Optional[EvidenceStore]:
    """Batch-pipeline entry: one EvidenceStore per gene run (ids unique across its papers),
    written to <results_dir>/<gene>_evidence.json. Best-effort: on any failure the reactions are
    left as they were (quote strings intact) and None is returned."""
    import json
    import logging
    import os
    log = logging.getLogger(__name__)
    try:
        if paper_for_source is None:
            from curator_llm.services.paper_loader import load_paper_text as paper_for_source
        store = EvidenceStore()
        stats = attach_evidence(reactions, store, paper_for_source)
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, f'{gene.lower()}_evidence.json')
        with open(path, 'w') as f:
            json.dump(store.to_json(), f, indent=2)
        log.info('evidence for %s: %s -> %s', gene, stats, path)
        if stats['rejected']:
            log.warning('evidence for %s: %d quote(s) not found in the paper were dropped',
                        gene, stats['rejected'])
        return store
    except Exception as e:
        log.warning('evidence verification failed for %s: %s -- reactions left unverified', gene, e)
        return None
