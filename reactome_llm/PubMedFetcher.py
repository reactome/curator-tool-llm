"""Fetch full text from PubMed Central by PMID, as JATS XML.

Why XML and not PDF: PubMed serves no PDFs, and PMC's JATS XML tags sections
explicitly (<sec sec-type="results">). That replaces the heuristic + LLM section
detection in FullTextPDFSections.py with an exact lookup for any paper PMC serves.

Route: PMID -> PMCID (ID Converter) -> JATS XML (efetch db=pmc) -> Results section.

Public entry point is load_source(), which accepts either a PMID or a local PDF
filename and returns (source_id, results_text, how) for both, so callers do not
branch on the source type.

Environment:
    PUBMED_API_KEY   raises the efetch rate limit from 3/sec to 10/sec
    NCBI_EMAIL       contact address NCBI asks for (defaults to NCBI_TOOL owner)
    NCBI_TOOL        tool name NCBI asks for
"""
import os, re, sys, time, threading
import xml.etree.ElementTree as et

import requests

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'pmc')
PAPERS_DIR = os.path.join(PROJECT_ROOT, 'data', 'papers')

IDCONV_URL = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'
EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

API_KEY = os.getenv('PUBMED_API_KEY')
TOOL = os.getenv('NCBI_TOOL', 'curator-tool-llm')
EMAIL = os.getenv('NCBI_EMAIL', 'zhenga@ohsu.edu')

# NCBI allows 3 req/sec without an API key, 10 with one. Stay just under.
_MIN_INTERVAL = 1.0 / (9 if API_KEY else 2.5)
_throttle_lock = threading.Lock()
_last_call = [0.0]

# A Results section shorter than this is treated as a parse failure, matching
# the guard in FullTextPDFSections._heuristic.
MIN_SECTION_WORDS = 100


def _throttle():
    """Block until enough time has passed since the previous NCBI request."""
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# PMID -> PMCID
# ─────────────────────────────────────────────────────────────────────────────
def resolve_pmcids(pmids):
    """Map PMIDs to PMCIDs. Returns {pmid: pmcid_or_None}.

    The ID Converter lives outside E-utilities, so it takes tool/email rather
    than an api_key. It accepts up to 200 ids per request.
    """
    pmids = [str(p).strip() for p in pmids if str(p).strip()]
    out = {}
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        _throttle()
        r = requests.get(IDCONV_URL, timeout=30, params={
            'ids': ','.join(batch), 'format': 'json', 'tool': TOOL, 'email': EMAIL,
        })
        r.raise_for_status()
        for rec in r.json().get('records', []):
            pmid = str(rec.get('pmid', ''))
            if not pmid:
                continue
            out[pmid] = rec.get('pmcid') or None
            if rec.get('errmsg'):
                print(f"    [idconv] {pmid}: {rec['errmsg']}", flush=True)
        for p in batch:                      # ids the converter didn't echo back
            out.setdefault(p, None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PMCID -> JATS XML
# ─────────────────────────────────────────────────────────────────────────────
def fetch_jats(pmcid, use_cache=True):
    """Return raw JATS XML for a PMCID, caching it under data/pmc/."""
    pmcid = pmcid if pmcid.upper().startswith('PMC') else f'PMC{pmcid}'
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{pmcid}.xml')
    if use_cache and os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, encoding='utf-8') as f:
            return f.read()

    params = {'db': 'pmc', 'id': pmcid, 'retmode': 'xml', 'tool': TOOL, 'email': EMAIL}
    if API_KEY:
        params['api_key'] = API_KEY
    _throttle()
    r = requests.get(EFETCH_URL, params=params, timeout=60)
    r.raise_for_status()
    xml = r.text
    if '<ERROR>' in xml and '<article' not in xml:
        msg = re.search(r'<ERROR>(.*?)</ERROR>', xml, re.S)
        raise ValueError(f'efetch error for {pmcid}: {msg.group(1).strip() if msg else "unknown"}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml)
    return xml


# ─────────────────────────────────────────────────────────────────────────────
# JATS -> text
# ─────────────────────────────────────────────────────────────────────────────
def _tag(elem):
    """Local tag name, with any {namespace} prefix stripped."""
    return elem.tag.rsplit('}', 1)[-1] if isinstance(elem.tag, str) else ''


# Table bodies flatten into ungrammatical runs of cell text — the same problem
# PDFs have. Captions are kept (they sit outside <table>); the grid is dropped.
_SKIP_TAGS = {'table', 'thead', 'tbody', 'tr', 'th', 'td', 'math', 'inline-formula',
              'disp-formula', 'tex-math'}
_BLOCK_TAGS = {'p', 'title', 'caption', 'sec', 'list-item', 'fig', 'table-wrap'}


def _text_of(elem):
    """Flatten an element to prose, dropping citation markers and table grids."""
    parts = []

    def walk(e):
        tag = _tag(e)
        # Citation markers ("[12]", "Smith et al.") are noise for extraction.
        drop = tag in _SKIP_TAGS or (tag == 'xref' and e.get('ref-type') == 'bibr')
        if not drop:
            if e.text:
                parts.append(e.text)
            for child in e:
                walk(child)
            if tag in _BLOCK_TAGS:
                parts.append('\n\n')
        if e.tail:
            parts.append(e.tail)

    walk(elem)
    text = ''.join(parts)
    # Dropping citation xrefs leaves their punctuation behind — "elusive (; )".
    # Remove brackets left holding nothing but separators.
    text = re.sub(r'[(\[]\s*[;,\s]*[)\]]', '', text)
    text = re.sub(r'\s+([,.;:)])', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _find_body(xml):
    """Return the <body> element, or raise if the article has no full text.

    A paywalled article still returns HTTP 200 and well-formed XML — front
    matter and abstract, with no <body>. Without this guard the pipeline would
    happily extract reactions from an abstract and report them as full text.
    """
    root = et.fromstring(xml)
    for article in root.iter():
        if _tag(article) == 'body':
            return article
    raise ValueError('PMC returned no <body> — full text is not open access')


def _results_sections(body):
    """Sections that look like Results. Prefers sec-type, falls back to <title>."""
    secs = [e for e in body.iter() if _tag(e) == 'sec']

    typed = [s for s in secs if 'results' in (s.get('sec-type') or '').lower()]
    if typed:
        return typed, 'jats:sec-type'

    titled = []
    for s in secs:
        title = next((c for c in s if _tag(c) == 'title'), None)
        if title is not None and re.match(r'\s*(iii?\.|\d\.)?\s*results\b',
                                          ''.join(title.itertext()), re.I):
            titled.append(s)
    if titled:
        return titled, 'jats:title'
    return [], None


def results_from_jats(xml, client=None, model=None):
    """Return (results_text, how) for a JATS document.

    Three tiers: sec-type match, <title> match, then the existing text-based
    detection in FullTextPDFSections as a last resort.
    """
    body = _find_body(xml)
    secs, how = _results_sections(body)
    if secs:
        # Drop sections nested inside another match so text isn't duplicated.
        top = [s for s in secs if not any(s is not o and s in list(o.iter()) for o in secs)]
        text = '\n\n'.join(_text_of(s) for s in top).strip()
        if len(text.split()) >= MIN_SECTION_WORDS:
            titles = [''.join(c.itertext()).strip()
                      for s in top for c in s if _tag(c) == 'title']
            if any(re.search(r'discussion', t, re.I) for t in titles):
                print(f"    [jats] section spans Results and Discussion: {titles}", flush=True)
            return text, how

    # No usable tagged section — fall back to the PDF-era text heuristics.
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
    from FullTextPDFSections import extract_results_section
    body_text = _text_of(body)
    text, sub = extract_results_section(body_text, client=client, model=model)
    return text, f'jats-body:{sub}'


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────
def is_pmid(spec):
    return bool(re.fullmatch(r'\d{4,9}', str(spec).strip()))


def is_pmcid(spec):
    return bool(re.fullmatch(r'PMC\d+', str(spec).strip(), re.I))


def load_source(spec, client=None, model=None):
    """Load one paper from a PMID, a PMCID, or a local PDF filename.

    Returns (source_id, results_text, how). Raises ValueError when the paper has
    no reachable full text or no Results section, which callers treat as a skip.
    """
    spec = str(spec).strip()

    if is_pmid(spec) or is_pmcid(spec):
        if is_pmid(spec):
            pmcid = resolve_pmcids([spec]).get(spec)
            if not pmcid:
                raise ValueError(f'PMID {spec} has no PMCID — not in PubMed Central')
            source_id = f'PMID:{spec}'
        else:
            pmcid = spec.upper()
            source_id = pmcid
        xml = fetch_jats(pmcid)
        text, how = results_from_jats(xml, client=client, model=model)
        return source_id, text, how

    # Local PDF — unchanged from the original pipeline.
    import fitz
    path = spec if os.path.isabs(spec) else os.path.join(PAPERS_DIR, spec)
    if not os.path.isfile(path):
        raise ValueError(f'no such PDF: {path}')
    doc = fitz.open(path)
    full_text = ''.join(p.get_text() for p in doc)
    doc.close()
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
    from FullTextPDFSections import extract_results_section
    text, how = extract_results_section(full_text, client=client, model=model)
    return os.path.basename(path), text, how


def output_stem(spec, gene=None):
    """Filename stem for results/<stem>_2prev1next_extraction.json.

    Gene-prefixed when known, so downstream scoring can recover the gene from
    the filename the way it does for PDFs (PINK1.pdf -> PINK1).
    """
    spec = str(spec).strip()
    if is_pmid(spec) or is_pmcid(spec):
        base = f'pmid{spec}' if is_pmid(spec) else spec.lower()
    else:
        base = os.path.splitext(os.path.basename(spec))[0]
    base = base.lower()
    return f'{gene.lower()}_{base}' if gene else base
