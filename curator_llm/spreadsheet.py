"""Annotate the papers listed in a spreadsheet (.xlsx): a PDF where the sheet names one, else the PubMed ID.

    scripts/annotate_spreadsheet.sh papers.xlsx --list              # what would be done: every row, PDF or PMC
    scripts/annotate_spreadsheet.sh papers.xlsx --download-only     # fetch the PMC full text only
    scripts/annotate_spreadsheet.sh papers.xlsx --rows 1-3          # try a few papers first
    scripts/annotate_spreadsheet.sh papers.xlsx                     # everything; run it again to continue

For each row: when the PDF column names a file, that PDF is annotated (looked for as given, next to the spreadsheet, and
in its pdf/ or PDF/ folder). Otherwise the PMID is, from PubMed Central's full text: the JATS XML is downloaded into a
PMC/ folder next to the spreadsheet (<PMCID>.xml; the PMCID column is used when there is one, else it is looked up),
and the workflow reads it from there. By default every paper becomes its own annotation; --group-by COLUMN merges the
papers sharing a value of that column (for example "Drug(s)") into one annotation each.

Results go to an annotations/ folder next to the spreadsheet: the command line's result files per annotation
(<name>.annotation.json, .progress.json, ...), and summary.tsv, one line per spreadsheet row with how it went. An
interrupted run continues where it stopped when run again (complete papers are reused, complete annotations skipped).
--owner USER also adds each annotation to that curator's annotations, for the front-end.
"""
import argparse
import csv
import os
import re
import sys
import threading
import time
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
       'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
_PKG_REL = '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'


# ── reading the spreadsheet (standard library only: no openpyxl needed) ─────
def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path: str, sheet: Optional[str] = None) -> List[Dict[str, str]]:
    """The rows of one sheet (the first, or the one named `sheet`) as dicts keyed by the header row. Values are text as
    shown; a whole number stored as a number reads as "39478230", not "39478230.0"."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared = []
        if 'xl/sharedStrings.xml' in names:
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', _NS):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % _NS['m'])))
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        sheets = [(s.get('name'), s.get('{%s}id' % _NS['r'])) for s in wb.find('m:sheets', _NS)]
        if sheet is not None and sheet not in [n for n, _ in sheets]:
            raise ValueError(f'no sheet named {sheet!r}; the sheets are {[n for n, _ in sheets]}')
        rid = next(r for n, r in sheets if sheet is None or n == sheet)
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target = next(r.get('Target') for r in rels.iter(_PKG_REL) if r.get('Id') == rid)
        target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        rows: List[List[str]] = []
        for row in ET.fromstring(z.read(target)).find('m:sheetData', _NS):
            cells = {}
            for c in row.findall('m:c', _NS):
                v, t = c.find('m:v', _NS), c.get('t')
                if t == 'inlineStr':
                    val = ''.join(x.text or '' for x in c.iter('{%s}t' % _NS['m']))
                elif v is None or v.text is None:
                    val = ''
                elif t == 's':
                    val = shared[int(v.text)]
                else:
                    val = v.text
                    if re.fullmatch(r'-?\d+\.0+', val):
                        val = val.split('.')[0]
                cells[_col(c.get('r'))] = val.strip()
            rows.append([cells.get(i, '') for i in range(max(cells) + 1)] if cells else [])
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header = rows[0]
    return [{h: (r[i] if i < len(r) else '') for i, h in enumerate(header) if h} for r in rows[1:]]


# ── what each row becomes ────────────────────────────────────────────────────
@dataclass
class Paper:
    row: str                       # the row's number (the id column, else its position)
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    pdf: Optional[str] = None      # path of the PDF to annotate, when the sheet names one
    group: str = ''                # the --group-by column's value
    text: str = ''                 # PMC full text: downloaded | cached | no body | failed: ... | (empty for a PDF)
    problem: str = ''              # why the row cannot be annotated (a PDF that is not there, no PMID)
    info: Dict[str, str] = field(default_factory=dict)
    title: str = ''                # the paper's title (--title-column), for its pathway in --hierarchy
    doi: str = ''
    pdf_note: str = ''             # where an automatically found or downloaded PDF came from

    @property
    def source(self) -> str:
        return os.path.basename(self.pdf) if self.pdf else f'PMID:{self.pmid}'


def find_column(header: List[str], wanted: str) -> Optional[str]:
    for h in header:
        if h.strip().lower() == wanted.strip().lower():
            return h
    return None


def pdf_folders(sheet_dir: str, pdf_dir: Optional[str]) -> List[str]:
    if pdf_dir:
        return [pdf_dir]
    out = [sheet_dir]
    try:
        out += [os.path.join(sheet_dir, d) for d in sorted(os.listdir(sheet_dir))
                if d.lower() == 'pdf' and os.path.isdir(os.path.join(sheet_dir, d))]
    except OSError:
        pass
    return out


def plan(rows: List[Dict[str, str]], sheet_dir: str, pdf_dir: Optional[str] = None, pmid_col: str = 'PMID',
         pmcid_col: str = 'PMCID', pdf_col: str = 'PDF File Name', id_col: str = '#',
         group_by: Optional[str] = None, title_col: str = 'Title', doi_col: str = 'DOI') -> List[Paper]:
    header = list(rows[0]) if rows else []
    cols = {k: find_column(header, v) for k, v in
            (('pmid', pmid_col), ('pmcid', pmcid_col), ('pdf', pdf_col), ('id', id_col), ('title', title_col),
             ('doi', doi_col))}
    if cols['pmid'] is None and cols['pdf'] is None:
        raise ValueError(f'the sheet has neither a {pmid_col!r} nor a {pdf_col!r} column; its columns are {header}')
    gcol = find_column(header, group_by) if group_by else None
    if group_by and gcol is None:
        raise ValueError(f'no column named {group_by!r} to group by; the columns are {header}')
    folders = pdf_folders(sheet_dir, pdf_dir)
    out = []
    for n, r in enumerate(rows, 1):
        p = Paper(row=(r.get(cols['id']) if cols['id'] else '') or str(n), group=r.get(gcol, '') if gcol else '',
                  title=r.get(cols['title'], '') if cols['title'] else '',
                  doi=re.sub(r'^(https?://(dx\.)?doi\.org/|doi:\s*)', '', r.get(cols['doi'], '').strip(), flags=re.I)
                  if cols['doi'] else '')
        pmid = re.sub(r'\D', '', r.get(cols['pmid'], '')) if cols['pmid'] else ''
        pmcid = (r.get(cols['pmcid'], '') if cols['pmcid'] else '').strip().upper()
        p.pmid = pmid or None
        p.pmcid = (pmcid if pmcid.startswith('PMC') else f'PMC{pmcid}') if pmcid else None
        name = r.get(cols['pdf'], '').strip() if cols['pdf'] else ''
        if name:
            found = [c for c in ([name] if os.path.isabs(name) else []) + [os.path.join(f, name) for f in folders]
                     if os.path.isfile(c)]
            if found:
                p.pdf = os.path.abspath(found[0])
            else:
                p.problem = f'PDF {name!r} not found in {", ".join(folders)}'
        elif p.pmid or p.pmcid:
            # a PDF put in the folder as <PMID>.pdf (or <PMCID>.pdf) counts even when the sheet does not name it: that is
            # how a paper PMC has no text for gets annotated, by hand or by the download below
            names = [f'{x}.pdf' for x in (p.pmid, p.pmcid) if x]
            found = [os.path.join(f, n) for f in folders for n in names if os.path.isfile(os.path.join(f, n))]
            if found:
                p.pdf, p.pdf_note = os.path.abspath(found[0]), f'found {os.path.basename(found[0])}'
        else:
            p.problem = 'no PDF and no PMID'
        out.append(p)
    return out


def select(papers: List[Paper], rows: Optional[str]) -> List[Paper]:
    """--rows "1-10,15": by the row number (the id column)."""
    if not rows:
        return papers
    want = set()
    for part in rows.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            want |= {str(i) for i in range(int(a), int(b) + 1)}
        elif part:
            want.add(part)
    return [p for p in papers if p.row in want]


# ── PubMed Central full text ─────────────────────────────────────────────────
def download_pmc(papers: List[Paper], say: Callable[[str], None], fetcher=None) -> None:
    """Fetch the JATS XML of every paper annotated from its PMID into the PMC folder (fetcher's CACHE_DIR, set through
    PMC_CACHE_DIR). Already there: left alone. Sets each paper's `text`."""
    if fetcher is None:
        import PubMedFetcher as fetcher
    todo = [p for p in papers if not p.pdf and p.pmid and not p.problem]
    if hasattr(fetcher, 'remember_pmcids'):        # the extraction then needs no NCBI lookup for these
        fetcher.remember_pmcids({p.pmid: p.pmcid for p in papers if p.pmid and p.pmcid})
    missing = [p.pmid for p in todo if not p.pmcid]
    if missing:
        try:
            found = fetcher.resolve_pmcids(missing)
        except Exception as e:
            found = {}
            say(f'[pmc] could not look up PMCIDs ({type(e).__name__}: {e})')
        for p in todo:
            if not p.pmcid and found.get(p.pmid):
                p.pmcid = found[p.pmid]
    for k, p in enumerate(todo, 1):
        if not p.pmcid:
            p.text = 'no PMCID (not in PubMed Central)'
            continue
        path = os.path.join(fetcher.CACHE_DIR, f'{p.pmcid}.xml')
        had = os.path.isfile(path) and os.path.getsize(path) > 0
        try:
            xml = fetcher.fetch_jats(p.pmcid)
        except Exception as e:
            p.text = f'failed: {type(e).__name__}: {e}'
        else:
            if '<body' in (xml or ''):
                p.text = 'cached' if had else 'downloaded'
            elif 'does not allow downloading of the full text' in (xml or ''):
                p.text = 'no full text: the publisher does not allow PMC to give it out as XML; add the PDF (PDF column)'
            else:
                p.text = 'no full text in PMC (metadata only); add the PDF (PDF column)'
        if k % 10 == 0 or k == len(todo):
            say(f'[pmc] {k}/{len(todo)} papers checked')


# ── a PDF where PMC has no text ──────────────────────────────────────────────
def find_oa_pdf(doi: str, email: str, http=None) -> Optional[tuple]:
    """(pdf bytes, where from) of a legal open-access copy Unpaywall knows for this DOI, or None. Only what a plain
    request gets: publishers and PMC that ask a browser to prove itself first are left to a person."""
    import requests
    http = http or requests
    try:
        j = http.get(f'https://api.unpaywall.org/v2/{doi}', params={'email': email}, timeout=30).json()
    except Exception:
        return None
    for loc in sorted(j.get('oa_locations') or [], key=lambda l: {'publishedVersion': 0, 'acceptedVersion': 1}.get(
            l.get('version'), 2)):
        url = loc.get('url_for_pdf')
        if not url:
            continue
        try:
            r = http.get(url, timeout=60, headers={'User-Agent': 'curator-tool-llm (mailto:%s)' % email})
        except Exception:
            continue
        body = r.content if getattr(r, 'status_code', 200) == 200 else b''
        if body.startswith(b'%PDF') and len(body) > 10_000:
            host = re.sub(r'^www\.', '', re.sub(r'^https?://([^/]+).*', r'\1', url))
            return body, f'{host}, {loc.get("version") or "unknown version"}'
    return None


def fetch_missing_pdfs(papers: List[Paper], pdf_folder: str, say: Callable[[str], None], email: Optional[str],
                       http=None) -> List[Paper]:
    """For every paper PMC has no text for: an open-access PDF through Unpaywall (by DOI), saved as <PMID>.pdf in
    `pdf_folder` and used from now on. Returns the papers still without text."""
    lacking = [p for p in papers if not p.pdf and not p.problem and p.text not in ('downloaded', 'cached')]
    if not lacking:
        return []
    still = []
    for p in lacking:
        got = find_oa_pdf(p.doi, email, http) if (p.doi and email) else None
        if got is None:
            still.append(p)
            continue
        os.makedirs(pdf_folder, exist_ok=True)
        path = os.path.join(pdf_folder, f'{p.pmid or p.pmcid}.pdf')
        with open(path, 'wb') as f:
            f.write(got[0])
        p.pdf, p.pdf_note = path, f'downloaded from {got[1]}'
        p.text = f'no XML in PMC; PDF downloaded from {got[1]}'
        say(f'[pdf] row {p.row} (PMID {p.pmid}): {p.text} -> {path}')
    return still


def write_to_download(path: str, papers: List[Paper]) -> None:
    """The papers a person has to fetch: where to get each, and the name to save it under in the pdf folder."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['row', 'pmid', 'pmcid', 'save as', 'PMC page', 'DOI link', 'title', 'why'])
        for p in papers:
            w.writerow([p.row, p.pmid or '', p.pmcid or '', f'{p.pmid or p.pmcid}.pdf',
                        f'https://pmc.ncbi.nlm.nih.gov/articles/{p.pmcid}/' if p.pmcid else '',
                        f'https://doi.org/{p.doi}' if p.doi else '', p.title, p.text])


# ── running the workflow ─────────────────────────────────────────────────────
def jobs_of(papers: List[Paper], group_by: Optional[str]) -> List[tuple]:
    """(annotation name, papers) in sheet order: one per paper, or one per value of the --group-by column."""
    if not group_by:
        width = max(3, max((len(p.row) for p in papers), default=1))
        return [(f'{p.row.zfill(width)}_' + (f'pmid{p.pmid}' if p.pmid else re.sub(r'\W+', '_', os.path.splitext(p.source)[0])), [p])
                for p in papers]
    groups: Dict[str, List[Paper]] = {}
    for p in papers:
        groups.setdefault(p.group or '(none)', []).append(p)
    return [(re.sub(r'[^A-Za-z0-9]+', '_', g).strip('_') or 'none', ps) for g, ps in groups.items()]


USAGE_COLUMNS = ['calls', 'input tokens', 'output tokens', 'cache read tokens', 'cache write tokens', 'cost USD']


def write_summary(path: str, papers: List[Paper], name_of: Dict[str, str], out_dir: str, group_by: Optional[str],
                  extra: Optional[Dict[str, dict]] = None) -> Dict[str, object]:
    """summary.tsv: one line per row, with each annotation's model calls, tokens and estimated cost (on the first row of
    an annotation that holds several papers), then a TOTAL line. `extra`: usage not in any annotation file (the
    hierarchy's duplicate judge), {label: {'calls', 'input_tokens', ..., 'usd'}}. Rewritten after every annotation, so it
    shows the cost so far while a run goes on. Returns the totals."""
    import json
    from curator_llm.pricing import cost_of_steps
    prog: Dict[str, dict] = {}
    keys = ('calls', 'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens')
    total = {k: 0 for k in keys}
    total['usd'], total['annotations'], unpriced = 0.0, 0, set()
    seen = set()
    with open(path + '.tmp', 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['row', 'pmid', 'pmcid', 'input', 'full text', 'annotation', 'run', 'paper', 'reason', 'reactions']
                   + USAGE_COLUMNS + ([group_by] if group_by else []))
        for p in papers:
            name = name_of.get(p.row, '')
            if name and name not in prog:
                try:
                    prog[name] = json.load(open(os.path.join(out_dir, f'{name}.progress.json')))
                except (OSError, ValueError):
                    prog[name] = {}
                try:
                    sess = json.load(open(os.path.join(out_dir, f'{name}.session.json')))
                except (OSError, ValueError):
                    sess = {}
                prog[name]['_n'] = sess.get('n_reactions', '')
                prog[name]['_usage'] = cost_of_steps((sess.get('usage') or {}).get('steps') or []) if sess else None
            pr = prog.get(name, {})
            mine = next((x for x in pr.get('papers', []) if x.get('source') == p.source), {})
            u = pr.get('_usage') if name not in seen else None          # an annotation's usage counts once
            seen.add(name)
            if u:
                for k in keys:
                    total[k] += u[k]
                total['usd'] += u['usd']
                total['annotations'] += 1
                unpriced |= set(u['unpriced'])
            w.writerow([p.row, p.pmid or '', p.pmcid or '',
                        ('PDF ' + os.path.basename(p.pdf) + (f' ({p.pdf_note})' if p.pdf_note else '')) if p.pdf else 'PMC',
                        p.text, name, pr.get('status', 'not run' if not p.problem else 'skipped'),
                        mine.get('state', ''), p.problem or mine.get('reason', ''), pr.get('_n', '')]
                       + ([u['calls'], u['input_tokens'], u['output_tokens'], u['cache_read_tokens'],
                           u['cache_write_tokens'], f"{u['usd']:.4f}"] if u else [''] * len(USAGE_COLUMNS))
                       + ([p.group] if group_by else []))
        for label, u in (extra or {}).items():
            for k in keys:
                total[k] += int(u.get(k) or 0)
            total['usd'] += float(u.get('usd') or 0.0)
            w.writerow(['', '', '', '', '', label, '', '', '', '']
                       + [u.get('calls', 0), u.get('input_tokens', 0), u.get('output_tokens', 0), u.get('cache_read_tokens', 0),
                          u.get('cache_write_tokens', 0), f"{float(u.get('usd') or 0):.4f}"] + ([''] if group_by else []))
        note = 'estimate; the Anthropic Console has the bill' + (f'; no price for: {", ".join(sorted(unpriced))}' if unpriced else '')
        w.writerow(['TOTAL', '', '', '', '', f"{total['annotations']} annotation(s)", '', '', note, '']
                   + [total['calls'], total['input_tokens'], total['output_tokens'], total['cache_read_tokens'],
                      total['cache_write_tokens'], f"{total['usd']:.2f}"] + ([''] if group_by else []))
    os.replace(path + '.tmp', path)                  # never half a file for someone watching it
    return total


def hierarchy(a, papers: List[Paper], jobs: List[tuple], out_dir: str, store, say: Callable[[str], None],
              judge=None) -> Optional[dict]:
    """Combine the per-paper annotations into one, under top -> group -> paper pathways, and write it. Returns the
    duplicate judge's usage ({'calls', 'input_tokens', ..., 'usd'}; None when nothing was combined)."""
    import json
    from curator_llm.annotation_files import add_to, read
    from curator_llm.services import issues as iss
    from curator_llm.services.combine import Part, combine
    from curator_llm.services.session_service import SessionService
    title = a.title or os.path.splitext(os.path.basename(a.spreadsheet))[0].replace('_', ' ')
    parts, left_out = [], []
    for name, ps in jobs:
        p = ps[0]
        path = os.path.join(out_dir, f'{name}.annotation.json')
        try:
            prog = json.load(open(os.path.join(out_dir, f'{name}.progress.json')))
        except (OSError, ValueError):
            prog = {}
        if prog.get('status') != 'complete' or not os.path.isfile(path):
            left_out.append(f'{name} ({prog.get("status", "not run")})')
            continue
        label = p.title or (f'PMID {p.pmid}' if p.pmid else p.source)
        parts.append(Part(read(path), prefix=name.split('_')[0], title=f'{label} (row {p.row})', group=p.group))
    if not parts:
        say('[hierarchy] no finished paper annotation to combine')
        return None
    say(f'[hierarchy] combining {len(parts)} paper annotation(s)' + (f'; left out {len(left_out)}: '
        + ', '.join(left_out[:8]) + (', ...' if len(left_out) > 8 else '') if left_out else ''))
    spent = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cache_read_tokens': 0, 'cache_write_tokens': 0}
    lock = threading.Lock()
    judge_model = None

    def count(msg):                                 # every judge call, for summary.tsv and the annotation's usage
        u = getattr(msg, 'usage', None)
        with lock:
            spent['calls'] += 1
            if u is not None:
                spent['input_tokens'] += getattr(u, 'input_tokens', 0) or 0
                spent['output_tokens'] += getattr(u, 'output_tokens', 0) or 0
                spent['cache_read_tokens'] += getattr(u, 'cache_read_input_tokens', 0) or 0
                spent['cache_write_tokens'] += getattr(u, 'cache_creation_input_tokens', 0) or 0
    if judge is None and not a.no_shared:
        import anthropic
        from curator_llm.model_settings import anthropic_kwargs
        from curator_llm.services.merge_support import VerdictCache, make_judge
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kw = anthropic_kwargs('merge')
        judge_model = kw['model']
        judge = make_judge(anthropic.Anthropic(), kw, VerdictCache(os.path.join(root, 'data', 'merge_cache', 'judge.jsonl')),
                           on_usage=count)
    with ThreadPoolExecutor(max_workers=8) as pool:
        s = combine(parts, title, a.owner or 'local', judge=None if a.no_shared else judge, map_fn=pool.map, say=say)
    from curator_llm.models.usage import UsageEntry
    from curator_llm.pricing import cost
    if spent['calls']:
        s.usage.append(UsageEntry(step='combine', model=judge_model, detail='shared reactions across papers', **spent))
    usage = {**spent, 'usd': cost(spent, judge_model) or 0.0}
    SessionService.refresh(s)
    SessionService(None, None, None).refresh_suggestions(s)
    iss.organize(s.issues, s.draft, s.proposals)
    name = re.sub(r'[^A-Za-z0-9]+', '_', title).strip('_') or 'hierarchy'
    def write(path, data):
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    write(os.path.join(out_dir, f'{name}.annotation.json'), s.model_dump(mode='json'))
    write(os.path.join(out_dir, f'{name}.staged_instances.json'), s.user_instances)
    write(os.path.join(out_dir, f'{name}.issues.json'), [i.model_dump(mode='json') for i in s.issues if i.status == 'open'])
    groups = [sp for sp in s.draft.pathway.subpathways]
    say(f'[hierarchy] {name}: {len(s.draft.reactions)} reactions from {len(parts)} paper(s), pathway "{title}" with '
        f'{len(groups)} sub-pathway(s), {sum(1 for i in s.issues if i.status == "open")} open issues -> '
        f'{os.path.join(out_dir, name + ".annotation.json")}')
    if a.owner and store is not None:
        add_to(store, s, a.owner, replace=True)
        say(f'[hierarchy] added to {a.owner}\'s annotations')
    say(f'[cost] combining: {spent["calls"]} judge call(s), ~${usage["usd"]:.2f}')
    return usage


def main(argv: Optional[List[str]] = None, pipeline=None, qa_model=None, fetcher=None,
         say: Callable[[str], None] = None) -> int:
    say = say or (lambda m: print(m, flush=True))
    ap = argparse.ArgumentParser(prog='annotate_spreadsheet', description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('spreadsheet', help='the .xlsx file listing the papers')
    ap.add_argument('--sheet', help='the sheet to read (default: the first)')
    ap.add_argument('--pmid-column', default='PMID')
    ap.add_argument('--pmcid-column', default='PMCID')
    ap.add_argument('--pdf-column', default='PDF File Name')
    ap.add_argument('--doi-column', default='DOI', help='used to look for an open-access PDF when PMC has no text')
    ap.add_argument('--no-pdf-download', action='store_true', help='do not look for open-access PDFs (Unpaywall) of '
                    'papers PMC has no text for')
    ap.add_argument('--id-column', default='#', help='the row-number column (default "#"; else the row\'s position)')
    ap.add_argument('--pdf-dir', help='where the PDFs are (default: next to the spreadsheet, and its pdf/ or PDF/ folder)')
    ap.add_argument('--pmc-dir', help='where PubMed Central full text is kept (default: PMC/ next to the spreadsheet)')
    ap.add_argument('--out', help='where the results go (default: annotations/ next to the spreadsheet)')
    ap.add_argument('--rows', help='only these rows, by number: "1-10,15"')
    ap.add_argument('--group-by', metavar='COLUMN', help='one merged annotation per value of this column (e.g. "Drug(s)") '
                    'instead of one per paper')
    ap.add_argument('--hierarchy', action='store_true',
                    help='after annotating each paper, also make ONE annotation of them all: a top pathway (--title), a '
                         'sub-pathway per --group-by value, and under it a pathway per paper (named by --title-column); '
                         'a reaction several papers of a group report becomes one, listed under each of them')
    ap.add_argument('--title', default=None, help='with --hierarchy: the top pathway\'s name (default: the spreadsheet\'s '
                    'name)')
    ap.add_argument('--title-column', default='Title', help='with --hierarchy: the column naming each paper\'s pathway')
    ap.add_argument('--no-shared', action='store_true', help='with --hierarchy: keep each paper\'s copy of a reaction '
                    'other papers report too (no duplicate-judge calls)')
    ap.add_argument('--focus', metavar='GENE', help='focus gene or protein for every annotation (optional)')
    ap.add_argument('--owner', metavar='USER', help="also add each annotation to this curator's annotations (front-end)")
    ap.add_argument('--session-dir', metavar='DIR', help='with --owner: the service\'s session folder')
    ap.add_argument('--jobs', type=int, default=1, help='annotations run at the same time (default 1; each already runs '
                    'its chunks and papers in parallel, so 2-3 is plenty)')
    ap.add_argument('--no-qa', action='store_true', help='skip the model QA review of each reaction')
    ap.add_argument('--force', action='store_true', help='redo everything, even what finished before')
    ap.add_argument('--list', action='store_true', help='show what would be done, change nothing')
    ap.add_argument('--download-only', action='store_true', help='fetch the PMC full text, do not annotate')
    a = ap.parse_args(argv)

    sheet_dir = os.path.dirname(os.path.abspath(a.spreadsheet))
    pmc_dir = os.path.abspath(a.pmc_dir or os.path.join(sheet_dir, 'PMC'))
    out_dir = os.path.abspath(a.out or os.path.join(sheet_dir, 'annotations'))
    os.environ['PMC_CACHE_DIR'] = pmc_dir          # before PubMedFetcher is imported; the subprocesses inherit it
    try:
        rows = read_xlsx(a.spreadsheet, a.sheet)
        papers = select(plan(rows, sheet_dir, a.pdf_dir, a.pmid_column, a.pmcid_column, a.pdf_column, a.id_column,
                             a.group_by, a.title_column, a.doi_column), a.rows)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    n_pdf = sum(1 for p in papers if p.pdf)
    bad = [p for p in papers if p.problem]
    say(f'{len(papers)} paper(s): {n_pdf} from a PDF, {len(papers) - n_pdf - len(bad)} from PubMed Central'
        + (f', {len(bad)} cannot be annotated' if bad else ''))
    for p in bad:
        say(f'  row {p.row}: {p.problem}')
    jobs = jobs_of([p for p in papers if not p.problem], None if a.hierarchy else a.group_by)
    if a.list:
        for name, ps in jobs:
            say(f'  {name}: ' + ', '.join((f'PDF {os.path.basename(p.pdf)}' if p.pdf else f'PMID {p.pmid}'
                                           + (f' ({p.pmcid})' if p.pmcid else '')) for p in ps))
        say(f'{len(jobs)} annotation(s); PMC full text -> {pmc_dir}; results -> {out_dir}')
        return 0

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reactome_llm'))
    os.makedirs(pmc_dir, exist_ok=True)
    say(f'[pmc] full text -> {pmc_dir}')
    download_pmc(papers, say, fetcher)
    texts = [p.text for p in papers if not p.pdf and not p.problem]
    say('[pmc] ' + ', '.join(f'{sum(t == k for t in texts)} {k}' for k in ('downloaded', 'cached') if any(t == k for t in texts))
        + ''.join(f'; row {p.row} (PMID {p.pmid}): {p.text}' for p in papers
                  if p.text and p.text not in ('downloaded', 'cached')))
    folders = pdf_folders(sheet_dir, a.pdf_dir)
    pdf_folder = folders[-1] if a.pdf_dir or len(folders) > 1 else os.path.join(sheet_dir, 'pdf')
    if not a.no_pdf_download:
        import PubMedFetcher as pf
        email = os.getenv('UNPAYWALL_EMAIL') or os.getenv('NCBI_EMAIL') or getattr(pf, 'EMAIL', None)
        still = fetch_missing_pdfs(papers, pdf_folder, say, email)
    else:
        still = [p for p in papers if not p.pdf and not p.problem and p.text not in ('downloaded', 'cached')]
    todo_list = os.path.join(sheet_dir, 'pdf_to_download.tsv')
    if still:
        write_to_download(todo_list, still)
        say(f'[pdf] {len(still)} paper(s) have no text PMC or an open-access copy will give a script (PMC and most '
            f'publishers make a browser prove itself first). Download them by hand into {pdf_folder}, named as listed in '
            f'{todo_list}, and run again: they are then picked up.')
    elif os.path.exists(todo_list):
        os.remove(todo_list)
    jobs = jobs_of([p for p in papers if not p.problem and (p.pdf or p.text in ('downloaded', 'cached'))],
                   None if a.hierarchy else a.group_by)
    os.makedirs(out_dir, exist_ok=True)
    name_of = {p.row: name for name, ps in jobs for p in ps}
    if a.download_only:
        write_summary(os.path.join(out_dir, 'summary.tsv'), papers, name_of, out_dir, a.group_by)
        return 0

    from curator_llm.annotation_files import default_session_dir
    from curator_llm.cli import COMPLETE, EMPTY, PARTIAL, annotate
    from curator_llm.ports.pipeline import PaperRef
    if pipeline is None:
        from curator_llm.wiring import build_pipeline, load_env
        load_env()
        os.environ['PMC_CACHE_DIR'] = pmc_dir      # load_env reads .env with override: keep this run's folder
        pipeline, inner = build_pipeline()
        if hasattr(getattr(inner, 'extractor', None), 'resume'):
            inner.extractor.resume = not a.force
        if qa_model is None and not a.no_qa:
            from curator_llm.adapters.chat_anthropic import AnthropicChatModel
            qa_model = AnthropicChatModel(role='qa')
    if a.no_qa:
        qa_model = None
    store = None
    if a.owner:
        from curator_llm.adapters.sessions_file import FileSessionStore
        store = FileSessionStore(a.session_dir or default_session_dir())

    lock = threading.Lock()
    results: Dict[str, str] = {}

    def run(job):
        name, ps = job
        refs = [PaperRef(pdf_path=p.pdf, label=os.path.basename(p.pdf)) if p.pdf else PaperRef(pmid=p.pmid) for p in ps]
        state = annotate(name, refs, a.focus, out_dir, pipeline, say, a.force, qa_model=qa_model, owner=a.owner, store=store)
        with lock:
            results[name] = state
            t = write_summary(os.path.join(out_dir, 'summary.tsv'), papers, name_of, out_dir, a.group_by)
            say(f'[cost] {len(results)} of {len(jobs)} annotation(s) finished; so far ~${t["usd"]:.2f} '
                f'({t["calls"]} model calls; estimate, see summary.tsv)')

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        list(pool.map(run, jobs))
    write_summary(os.path.join(out_dir, 'summary.tsv'), papers, name_of, out_dir, a.group_by)
    done = sum(s == COMPLETE for s in results.values())
    empty = sum(s == EMPTY for s in results.values())
    partial = sorted(n for n, s in results.items() if s == PARTIAL)
    failed = sorted(n for n, s in results.items() if s not in (COMPLETE, EMPTY, PARTIAL))
    extra = None
    if a.hierarchy:
        used = hierarchy(a, papers, jobs, out_dir, store, say)
        if used and used.get('calls'):
            extra = {'combining papers (hierarchy)': used}
    t = write_summary(os.path.join(out_dir, 'summary.tsv'), papers, name_of, out_dir, a.group_by, extra)
    say(f'[cost] total ~${t["usd"]:.2f} for {t["calls"]} model calls (estimate; per paper in summary.tsv)')
    say(f'Finished in {(time.time() - t0) / 60:.0f} min: {done} of {len(jobs)} annotation(s) complete, '
        f'{empty} with no reactions, {len(partial)} partial, {len(failed)} failed' + (f' ({", ".join(failed[:10])}{", ..." if len(failed) > 10 else ""})' if failed else '')
        + f'. Per row: {os.path.join(out_dir, "summary.tsv")}'
        + ('. Run the same command again to finish the rest.' if partial or failed else ''))
    return 1 if partial or failed or bad else 0


if __name__ == '__main__':
    sys.exit(main())
