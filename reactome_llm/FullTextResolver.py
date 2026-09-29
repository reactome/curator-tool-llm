"""Full-text resolution layer for the Reactome annotation pipeline.

This module locates full-text files for a set of PMIDs and reports WHERE they are.
It does NOT parse or extract content -- that is handled by the downstream full-text
extractor. The deliverable is a manifest: PMID -> {source, path}.

Three responsibilities:

  1. Persistent config (data/user_config.json) -- remembers the curator's folder of
     full-text PDFs so the wizard asks for it once, not every run.

  2. build_index(papers_dir) -> {pmid: filepath} -- the "IndexDoc". Scans the curator's
     folder, pulls a DOI off page 1 of each PDF, and resolves it to a PMID via PubMed
     esearch. An incremental cache (data/.indexdoc_cache.json) keyed by path+mtime means a
     1000-PDF folder is parsed + resolved once, then near-instant on later runs.

  3. resolve_fulltext(pmids, index) -> manifest -- for each PMID: use the local PDF if the
     index has it; else use a cached PMC XML; else try to download the PMC XML (PMID->PMCID
     via NCBI idconv, then efetch db=pmc); else mark it a miss. Downloaded XMLs accumulate in
     data/fulltext_cache/ so they are never re-downloaded.

Network use is confined to build_index's DOI->PMID esearch and resolve_fulltext's
idconv/efetch. Everything is best-effort and guarded: a failure degrades to "unresolvable"
(indexing) or "miss" (resolution) rather than raising.
"""

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------
# Paths (all under data/, all gitignored)
# ---------------------------------------------------------------------------------------------
DATA_DIR = Path("data")
CONFIG_PATH = DATA_DIR / "user_config.json"
INDEX_CACHE_PATH = DATA_DIR / ".indexdoc_cache.json"
XML_CACHE_DIR = DATA_DIR / "fulltext_cache"

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
# NCBI allows 3 req/s without a key, 10/s with one. Stay comfortably under either.
_SLEEP = 0.12 if PUBMED_API_KEY else 0.34
_TIMEOUT = 30

# A DOI: "10." then a registrant code, "/", then the suffix. Stop at whitespace/quotes/brackets;
# strip trailing sentence punctuation that commonly abuts a DOI printed in running text.
_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>()\[\]]+', re.IGNORECASE)


# ---------------------------------------------------------------------------------------------
# 1. Persistent config
# ---------------------------------------------------------------------------------------------
def load_config() -> dict:
    """Return the saved user config ({} if none / unreadable)."""
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        logger.warning(f"Could not read {CONFIG_PATH}: {e}")
    return {}


def save_config(cfg: dict) -> None:
    """Persist the user config (creates data/ if needed)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------------------------
# 2. IndexDoc builder
# ---------------------------------------------------------------------------------------------
def _open_pdf_first_page_text(path: str) -> str:
    """First-page text of a PDF, or '' on any failure. PyMuPDF only."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        try:
            import pymupdf as fitz
        except ImportError:
            logger.error("PyMuPDF (fitz) is not installed; cannot index PDFs.")
            return ""
    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                return ""
            return doc.load_page(0).get_text() or ""
    except Exception as e:
        logger.warning(f"Could not read page 1 of {path}: {e}")
        return ""


def _extract_doi_from_page1(text: str) -> Optional[str]:
    """First DOI found in page-1 text, cleaned of trailing junk. None if absent.

    This is the single point where a paper's identity is recovered from its file. If the
    DOI-on-page-1 heuristic ever needs to change, it changes here and nowhere else.
    """
    m = _DOI_RE.search(text or "")
    if not m:
        return None
    doi = m.group(0)
    # A "doi:" label or trailing period/comma often rides along in running text.
    return doi.rstrip(".,;)")


def _doi_to_pmid(doi: str) -> Optional[str]:
    """Resolve a DOI to a PMID via PubMed esearch. None if unmapped / on error.

    Queries the [doi] field (term=<doi>[doi]); falls back to a bare-term search, which PubMed
    also resolves for well-formed DOIs.
    """
    for term in (f"{doi}[doi]", doi):
        try:
            url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                   "?db=pubmed&retmode=json&term=" + urllib.parse.quote(term)
                   + (f"&api_key={PUBMED_API_KEY}" if PUBMED_API_KEY else ""))
            time.sleep(_SLEEP)
            with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            idlist = data.get("esearchresult", {}).get("idlist", [])
            if idlist:
                return str(idlist[0])
        except Exception as e:
            logger.debug(f"esearch failed for DOI {doi} (term={term!r}): {e}")
    return None


def build_index(papers_dir: Optional[str], verbose: bool = True) -> Dict[str, str]:
    """Scan `papers_dir` for PDFs and build {pmid: filepath}.

    Incremental: files whose path+mtime match the cache are reused without re-opening the PDF
    or re-hitting the network. Only new/modified files are parsed (DOI off page 1) and resolved
    (DOI -> PMID via esearch). Files with no recoverable DOI/PMID are logged and skipped.

    Returns {} when no folder is configured.
    """
    if not papers_dir:
        return {}
    folder = Path(papers_dir).expanduser()
    if not folder.is_dir():
        logger.warning(f"Configured papers_dir does not exist: {papers_dir}")
        if verbose:
            print(f"  ! papers folder not found: {papers_dir} — skipping full-text index")
        return {}

    # Load prior cache: {abspath: {"mtime": float, "pmid": str|null}}
    old_cache: dict = {}
    try:
        if INDEX_CACHE_PATH.exists():
            old_cache = json.loads(INDEX_CACHE_PATH.read_text())
    except Exception as e:
        logger.warning(f"Could not read index cache {INDEX_CACHE_PATH}: {e}")

    pdfs = sorted(str(p) for p in folder.rglob("*.pdf"))
    new_cache: dict = {}
    index: Dict[str, str] = {}
    n_new = n_cached = 0

    if verbose:
        print(f"  Indexing papers... {len(pdfs)} found", flush=True)

    for path in pdfs:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        prior = old_cache.get(path)
        if prior and prior.get("mtime") == mtime:
            pmid = prior.get("pmid")
            n_cached += 1
        else:
            doi = _extract_doi_from_page1(_open_pdf_first_page_text(path))
            pmid = _doi_to_pmid(doi) if doi else None
            n_new += 1
            if not doi:
                logger.info(f"Unresolvable (no DOI on page 1): {path}")
            elif not pmid:
                logger.info(f"Unresolvable (DOI {doi} did not map to a PMID): {path}")
        new_cache[path] = {"mtime": mtime, "pmid": pmid}
        if pmid:
            # First file wins if two PDFs resolve to the same PMID (duplicate copies).
            index.setdefault(str(pmid), path)

    # Persist the pruned cache (stale entries for deleted/moved files are dropped).
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_CACHE_PATH.write_text(json.dumps(new_cache, indent=2))
    except Exception as e:
        logger.warning(f"Could not write index cache {INDEX_CACHE_PATH}: {e}")

    n_unresolvable = len(pdfs) - len(index)
    if verbose:
        print(f"  Indexing papers... {len(pdfs)} found, {n_new} new, {n_cached} cached", flush=True)
        print(f"  Index built: {len(index)} resolved, "
              f"{n_unresolvable} unresolvable (no PMID found)", flush=True)
    return index


# ---------------------------------------------------------------------------------------------
# 3. Full-text resolution
# ---------------------------------------------------------------------------------------------
def _pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Map a PMID to a PMCID via the NCBI ID Converter. None if not in PMC / on error."""
    try:
        url = ("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
               "?tool=reactome-curator&format=json&ids=" + urllib.parse.quote(str(pmid)))
        time.sleep(_SLEEP)
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for rec in data.get("records", []):
            pmcid = rec.get("pmcid")
            if pmcid:
                return pmcid
    except Exception as e:
        logger.debug(f"idconv failed for PMID {pmid}: {e}")
    return None


def _fetch_pmc_xml(pmcid: str, dest: Path) -> bool:
    """efetch the PMC full-text XML for `pmcid` into `dest`. True on a plausible full-text hit.

    Non-open-access PMC records return only front-matter (no <body>); those are treated as a
    miss and not cached, so a later run can retry if access changes.
    """
    numeric = pmcid.replace("PMC", "")
    try:
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
               "?db=pmc&retmode=xml&id=" + urllib.parse.quote(numeric)
               + (f"&api_key={PUBMED_API_KEY}" if PUBMED_API_KEY else ""))
        time.sleep(_SLEEP)
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"efetch failed for {pmcid}: {e}")
        return False
    # Require an actual article body -- front-matter-only records aren't usable full text.
    if "<article" not in xml or "<body" not in xml:
        logger.info(f"{pmcid}: PMC returned no full-text body (likely not open access).")
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(xml)
        return True
    except Exception as e:
        logger.warning(f"Could not write XML cache {dest}: {e}")
        return False


def resolve_fulltext(pmids: List[str], index: Dict[str, str],
                     gene: str = "", verbose: bool = True) -> Dict[str, dict]:
    """Resolve each PMID to a full-text source.

    For each PMID, in order:
      (a) local PDF in `index`            -> {"source": "pdf", "path": ...}
      (b) cached XML at data/fulltext_cache/<pmid>.xml -> {"source": "xml", "path": ...}
      (c) PMID->PMCID (idconv) -> efetch PMC XML, cache it -> {"source": "xml", "path": ...}
      (d) otherwise                       -> {"source": "miss"}

    Returns a flat manifest {pmid: {"source", "path"?}}.
    """
    manifest: Dict[str, dict] = {}
    n_pdf = n_xml_cached = n_xml_new = n_miss = 0

    for pmid in [str(p) for p in pmids if p]:
        # (a) curator's local PDF
        if pmid in index:
            manifest[pmid] = {"source": "pdf", "path": index[pmid]}
            n_pdf += 1
            continue
        # (b) previously downloaded XML
        cached = XML_CACHE_DIR / f"{pmid}.xml"
        if cached.exists():
            manifest[pmid] = {"source": "xml", "path": str(cached)}
            n_xml_cached += 1
            continue
        # (c) download PMC XML
        pmcid = _pmid_to_pmcid(pmid)
        if pmcid and _fetch_pmc_xml(pmcid, cached):
            manifest[pmid] = {"source": "xml", "path": str(cached)}
            n_xml_new += 1
            continue
        # (d) no full text obtainable
        manifest[pmid] = {"source": "miss"}
        n_miss += 1

    if verbose:
        xml_note = n_xml_cached + n_xml_new
        cached_frag = f" ({n_xml_cached} cached)" if n_xml_cached else ""
        who = f" for {gene}" if gene else ""
        print(f"  Resolving full text{who} ... {n_pdf} local PDF · "
              f"{xml_note} PMC XML{cached_frag} · {n_miss} miss", flush=True)
    return manifest


# ---------------------------------------------------------------------------------------------
# Standalone smoke test: `python reactome_llm/FullTextResolver.py <papers_dir>`
# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python FullTextResolver.py <papers_dir>")
        sys.exit(1)
    idx = build_index(sys.argv[1])
    print(f"\nIndex ({len(idx)} entries):")
    for pmid, path in list(idx.items())[:20]:
        print(f"  {pmid}  <-  {path}")
