"""Annotate one or more papers from the command line. No service, login or upload: this runs the same pipeline the REST
service runs, in this process, and writes the results as files.

    python -m curator_llm.cli paper1.pdf paper2.pdf --pmid 24751536 --focus PINK1       (or scripts/annotate_papers.sh)

By default ALL the papers go into ONE draft: their reactions are merged within each paper and then across papers, and a
reaction that several papers describe cites all of them. With --separate each paper is annotated on its own. There is no
limit on the number of papers here (MAX_PAPERS belongs to the REST service); PAPER_WORKERS sets how many are read at once.
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Callable, List, Optional, Tuple

from curator_llm.annotation_files import add_to, default_session_dir
from curator_llm.models.session import PaperInfo, Session
from curator_llm.models.usage import summarize
from curator_llm.ports.pipeline import PaperRef, PipelineInput
from curator_llm.services.session_service import SessionService

DESCRIPTION = __doc__.split('\n\n')[0]
EPILOG = """\
results, per session, in --out:
  <name>.staged_instances.json   the draft, as the curator tool's "load staged instances" takes it
  <name>.issues.json             what needs a curator's attention (open issues)
  <name>.suggestions.json        fixes suggested for some of those issues, as edits (RFC 6902 patches) with the reason;
                                 none is applied: a curator accepts them in the curator tool's Paper2Path page
  <name>.session.json            the papers, number of reactions and issues, and the tokens used
  <name>.evidence.json           every quote with the paper, page and section it is from, and which instance it supports
  <name>.progress.json           how the run ended (complete / no_reactions / partial / failed) and each paper's extraction (complete,
                                 incomplete: worth another try, no_text: not open access or no usable text; reused)
  <name>.annotation.json         the whole annotation, as the service keeps it: with --owner it is also added to that
                                 curator's annotations, or add it later with scripts/import_annotation.sh, and open it
                                 in the front-end (Paper2Path, "Your annotations")
A paper that cannot be read (not open access, or nothing reaction-like in it) is skipped and listed in the issues; the
other papers still go through. Interrupted, or a paper incomplete because the model or PubMed Central could not be
reached? Run the same command again: each paper extracted completely before is reused (same paper, focus, model and
prompt), only the rest is extracted, and a run already complete is skipped. --force starts afresh. Needs the same .env as the service (ANTHROPIC_API_KEY, REACTOME_NEO4J_*, ...).
"""


class UsageError(Exception):
    pass


def parse_ids(text: str) -> List[str]:
    out = []
    for tok in re.split(r'[\s,;]+', text.strip()):
        if tok:
            if not tok.isdigit():
                raise UsageError(f'not a PubMed ID: {tok}')
            out.append(tok)
    return out


def collect_papers(paths: List[str], pmids: List[str], pmid_files: List[str]) -> Tuple[List[str], List[str]]:
    """(PDF paths, PubMed IDs), without repeats. A folder contributes its *.pdf files (not its subfolders)."""
    pdfs: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(p, f)))
            if not found:
                raise UsageError(f'no PDFs in folder {p}')
            pdfs += found
        elif os.path.isfile(p):
            pdfs.append(p)
        else:
            raise UsageError(f'no such file or folder: {p}')
    for f in pmid_files:
        if not os.path.isfile(f):
            raise UsageError(f'no such file: {f}')
        with open(f) as fh:
            pmids = pmids + [re.sub(r'#.*', '', line) for line in fh]
    ids: List[str] = []
    for chunk in pmids:
        for i in parse_ids(chunk):
            if i not in ids:
                ids.append(i)
    pdfs = list(dict.fromkeys(os.path.abspath(p) for p in pdfs))
    for p in pdfs:
        with open(p, 'rb') as fh:
            if not fh.read(5).startswith(b'%PDF-'):
                raise UsageError(f'{os.path.basename(p)}: the file is not a PDF')
    names = [os.path.basename(p) for p in pdfs]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:                                    # a paper's reactions and quotes are tagged with its file name
        raise UsageError(f'PDFs with the same file name cannot be told apart: {", ".join(clash)} (rename one)')
    return pdfs, ids


def _write(path: str, data) -> None:
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)                        # a reader never sees half a file


def _progress(name: str, say: Callable[[str], None]) -> Callable[[str], None]:
    """Print a progress line when the stage changes, and otherwise at most every 10 seconds (chunk counts change fast)."""
    state = {'stage': None, 'at': 0.0}

    def report(msg: str) -> None:
        stage = msg.split(':')[0]
        now = time.time()
        if stage != state['stage'] or now - state['at'] >= 10:
            state['stage'], state['at'] = stage, now
            say(f'[{name}] {msg}')
    return report


COMPLETE, PARTIAL, FAILED = 'complete', 'partial', 'failed'
# every paper was read completely (or has no usable text), but no reaction came out: finished, not worth another run
EMPTY = 'no_reactions'


def _paper_rows(refs: List[PaperRef], status: dict, default: str, reason: str = '') -> List[dict]:
    rows = []
    for r in refs:
        st = status.get(r.source) or {'state': default, 'reason': reason}
        rows.append({'paper': r.label or (f'PMID {r.pmid}' if r.pmid else r.source), 'source': r.source,
                     'state': st.get('state', default), 'reason': st.get('reason', ''), 'reused': bool(st.get('reused'))})
    return rows


def write_progress(out_dir: str, name: str, status: str, papers: List[dict], note: str = '') -> None:
    """<name>.progress.json: how this draft's last run ended and each paper's extraction. A run is skipped next time
    only when it is complete; otherwise running the same command again finishes it, reusing the complete papers."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {s: sum(p['state'] == s for p in papers) for s in ('complete', 'incomplete', 'no_text')}
    _write(os.path.join(out_dir, f'{name}.progress.json'),
           {'name': name, 'status': status, 'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'note': note,
            'counts': counts, 'reused': sum(p['reused'] for p in papers), 'papers': papers})


def read_progress(out_dir: str, name: str) -> Optional[dict]:
    try:
        with open(os.path.join(out_dir, f'{name}.progress.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def annotate(name: str, refs: List[PaperRef], focus: Optional[str], out_dir: str, pipeline, say: Callable[[str], None],
             force: bool = False, qa_model=None, owner: Optional[str] = None, store=None) -> str:
    """One draft from `refs`; writes the result files. Returns COMPLETE, PARTIAL (written, but some paper could not be
    extracted completely or the merge failed: running again finishes it) or FAILED (nothing written; the reason is
    printed). With `owner` and `store` (a SessionStore) the annotation is also saved as one of that curator's."""
    staged = os.path.join(out_dir, f'{name}.staged_instances.json')
    progress = read_progress(out_dir, name)
    finished = (progress or {}).get('status') in (COMPLETE, EMPTY) or (
        progress is None and os.path.isfile(staged) and os.path.getsize(staged) > 0)   # from before progress files
    if finished and not force:
        say(f'[{name}] already done (use --force to run again), skipping')
        return COMPLETE
    if progress and progress.get('status') not in (COMPLETE, EMPTY):
        say(f'[{name}] resuming: papers extracted completely before are reused')
    say(f'[{name}] annotating {len(refs)} paper(s)')
    spec = PipelineInput(papers=refs, focus=focus or None)
    try:
        result = pipeline.run(spec, _progress(name, say))
    except Exception as e:
        status = getattr(e, 'paper_status', {}) or {}
        if status and len(status) >= len(refs) and all(v.get('state') in ('complete', 'no_text') for v in status.values()):
            say(f'[{name}] no reactions: the paper(s) were read completely, but nothing in them is a molecular reaction '
                f'the workflow extracts (or there is no usable text); not tried again')
            write_progress(out_dir, name, EMPTY, _paper_rows(refs, status, 'complete'), f'{type(e).__name__}: {e}')
            return EMPTY
        say(f'[{name}] FAILED: {type(e).__name__}: {e}')
        write_progress(out_dir, name, FAILED, _paper_rows(refs, status, 'incomplete', str(e)), f'{type(e).__name__}: {e}')
        return FAILED
    papers = [PaperInfo(source=r.source, kind='pmid' if r.pmid else 'pdf', pmid=r.pmid,
                        name=r.label or (r.pmid if r.pmid else r.source),
                        review=result.reviews.get(r.source, r.review)) for r in refs]
    s = Session(owner=owner or 'local', pmid=next((r.pmid for r in refs if r.pmid), None), focus=focus, papers=papers,
                source=SessionService._source_label(papers), draft=result.draft, evidence=list(result.evidence),
                issues=list(result.issues), existing=list(result.existing), usage=list(result.usage),
                paper=result.paper, paper_texts=dict(result.papers), status='ready')
    SessionService.refresh(s)                    # the same step the service runs: instances and issues from the draft
    if qa_model is not None and s.draft.reactions:
        say(f'[{name}] checking each reaction ({len(s.draft.reactions)})')
        SessionService.qa_into(s, qa_model)      # the model's QA review, as issues (source 'qa')
    SessionService(None, None, pipeline).refresh_suggestions(s)   # suggested fixes, as pending proposals; never applied here
    suggestions = [p for p in s.proposals if p.status == 'pending']
    open_issues = [i for i in s.issues if i.status == 'open']
    os.makedirs(out_dir, exist_ok=True)
    _write(staged, s.user_instances)
    _write(os.path.join(out_dir, f'{name}.issues.json'), [i.model_dump(mode='json') for i in open_issues])
    _write(os.path.join(out_dir, f'{name}.evidence.json'),
           {'evidence': [e.model_dump(mode='json') for e in s.evidence], 'links': s.evidence_links})
    _write(os.path.join(out_dir, f'{name}.suggestions.json'), [p.model_dump(mode='json') for p in suggestions])
    usage = summarize(s.usage, saved_pipeline=result.replayed)
    _write(os.path.join(out_dir, f'{name}.session.json'),
           {'source': s.source, 'focus': focus, 'n_reactions': len(s.draft.reactions), 'n_open_issues': len(open_issues), 'n_suggested_fixes': len(suggestions),
            'papers': [p.model_dump() for p in papers], 'usage': usage})
    s.progress = 'done'
    _write(os.path.join(out_dir, f'{name}.annotation.json'), s.model_dump(mode='json'))   # the whole annotation
    say(f'[{name}] done: {len(s.draft.reactions)} reactions from {len(refs)} paper(s), {len(open_issues)} open issues, {len(suggestions)} suggested fixes'
        + (f' ({sum(1 for v in (getattr(result, "paper_status", {}) or {}).values() if v.get("reused"))} paper(s) reused from an earlier run)'
           if any(v.get('reused') for v in (getattr(result, 'paper_status', {}) or {}).values()) else ''))
    if owner and store is not None:
        add_to(store, s, owner, replace=True)
        say(f'[{name}] added to {owner}\'s annotations ({os.path.join(store.directory, owner, s.id + ".json")}): '
            f'open it in the front-end, Paper2Path, "Your annotations"')
    rows = _paper_rows(refs, getattr(result, 'paper_status', {}) or {}, 'complete')
    state = COMPLETE if getattr(result, 'complete', True) else PARTIAL
    write_progress(out_dir, name, state, rows)
    if state == PARTIAL:
        todo = [r for r in rows if r['state'] == 'incomplete']
        say(f'[{name}] PARTIAL: ' + (f'{len(todo)} paper(s) could not be extracted completely ('
             + '; '.join(f"{r['paper']}: {r['reason']}" for r in todo[:5]) + ('; ...' if len(todo) > 5 else '') + '). '
             if todo else 'the merge failed. ') + 'Run the same command again to finish; complete papers are reused.')
    notes = [i.message for i in open_issues if i.source == 'pipeline']
    if notes:
        say(f'[{name}] note: ' + '; '.join(notes))
    return state


def main(argv: Optional[List[str]] = None, pipeline=None, say: Callable[[str], None] = None, qa_model=None) -> int:
    say = say or (lambda m: print(m, flush=True))
    ap = argparse.ArgumentParser(prog='annotate_papers', description=DESCRIPTION, epilog=EPILOG,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', metavar='PATH', help='a PDF file, or a folder (every *.pdf in it, not its subfolders)')
    ap.add_argument('--pmid', action='append', default=[], metavar='ID', help='a PubMed ID (open access in PubMed Central); repeat for several')
    ap.add_argument('--pmids', action='append', default=[], metavar='"A B,C"', help='several PubMed IDs, separated by spaces or commas')
    ap.add_argument('--pmid-file', action='append', default=[], metavar='FILE', help='a file with one PubMed ID per line (# comments ignored)')
    ap.add_argument('--focus', metavar='GENE', help='gene or protein the curator cares about (optional)')
    ap.add_argument('--out', default='annotations', metavar='DIR', help='where the results go (default ./annotations)')
    ap.add_argument('--name', metavar='NAME', help='file name prefix of the merged result (default papers-<date and time>)')
    ap.add_argument('--separate', action='store_true', help='annotate each paper on its own instead of one merged draft')
    ap.add_argument('--force', action='store_true', help='run again even if the results already exist, and extract '
                    'every paper afresh (without it, papers extracted completely before are reused, so an interrupted '
                    'run continues where it stopped)')
    ap.add_argument('--owner', metavar='USER',
                    help="also add each annotation to this curator's annotations (their login name), so the front-end "
                         'shows it in Paper2Path, "Your annotations"; without it, add <name>.annotation.json later with '
                         'scripts/import_annotation.sh')
    ap.add_argument('--session-dir', metavar='DIR', help='with --owner: the service\'s session folder (default: SESSION_DIR '
                    'from .env, else data/sessions of this checkout)')
    ap.add_argument('--no-qa', action='store_true',
                    help='skip the model QA review of each reaction (one call per reaction); the automatic checks still run')
    ap.add_argument('--review', action='store_true',
                    help='the papers are review articles: keep reactions they only cite from earlier work. Without it '
                         'a PubMed ID is a review when PubMed says so, and a PDF is research: only reactions its own '
                         'results show are kept (Reactome needs direct evidence; what is left out is in the issues)')
    args = ap.parse_args(argv)
    try:
        pdfs, ids = collect_papers(args.paths, args.pmid + args.pmids, args.pmid_file)
        if not pdfs and not ids:
            raise UsageError('give at least one PDF, folder or --pmid')
    except UsageError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    review = True if args.review else None           # None: a PMID's kind comes from PubMed, a PDF is research
    refs = ([PaperRef(pdf_path=p, label=os.path.basename(p), review=review) for p in pdfs]
            + [PaperRef(pmid=i, review=review) for i in ids])
    if pipeline is None:
        from curator_llm.wiring import build_pipeline, load_env
        load_env()
        pipeline, inner = build_pipeline()
        if hasattr(getattr(inner, 'extractor', None), 'resume'):
            inner.extractor.resume = not args.force   # reuse papers extracted completely before (see --force)
        if qa_model is None and not args.no_qa:
            from curator_llm.adapters.chat_anthropic import AnthropicChatModel
            qa_model = AnthropicChatModel(role='qa')
    if args.no_qa:
        qa_model = None
    store = None
    if args.owner:
        from curator_llm.adapters.sessions_file import FileSessionStore
        store = FileSessionStore(args.session_dir or default_session_dir())
    done, partial, failed = 0, [], []
    if len(refs) == 1 and args.name:
        jobs = [(args.name, refs)]
    elif args.separate or len(refs) == 1:
        jobs = [((os.path.splitext(r.label)[0] if r.pdf_path else f'pmid-{r.pmid}'), [r]) for r in refs]
    else:
        jobs = [(args.name or f'papers-{time.strftime("%Y%m%d-%H%M%S")}', refs)]
    for name, group in jobs:
        state = annotate(name, group, args.focus, args.out, pipeline, say, args.force, qa_model=qa_model,
                         owner=args.owner, store=store)
        if state in (COMPLETE, EMPTY):
            done += 1
        elif state == PARTIAL:
            partial.append(name)
        else:
            failed.append(name)
    say(f'Finished: {done} session(s) done, {len(partial)} partial, {len(failed)} failed'
        + (f'; partial: {", ".join(partial)}' if partial else '') + (f'; failed: {", ".join(failed)}' if failed else '')
        + f'. Results in {args.out}' + ('. Run the same command again to finish the rest (complete papers are reused).'
                                        if partial or failed else ''))
    return 1 if partial or failed else 0


if __name__ == '__main__':
    sys.exit(main())
