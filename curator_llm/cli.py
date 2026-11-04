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
A paper that cannot be read (not open access, or nothing reaction-like in it) is skipped and listed in the issues; the
other papers still go through. Needs the same .env as the service (ANTHROPIC_API_KEY, REACTOME_NEO4J_*, ...).
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


def annotate(name: str, refs: List[PaperRef], focus: Optional[str], out_dir: str, pipeline, say: Callable[[str], None],
             force: bool = False, qa_model=None) -> bool:
    """One draft from `refs`; writes the result files. False if it failed (the reason is printed)."""
    staged = os.path.join(out_dir, f'{name}.staged_instances.json')
    if os.path.isfile(staged) and os.path.getsize(staged) > 0 and not force:
        say(f'[{name}] already done (use --force to run again), skipping')
        return True
    say(f'[{name}] annotating {len(refs)} paper(s)')
    spec = PipelineInput(papers=refs, focus=focus or None)
    try:
        result = pipeline.run(spec, _progress(name, say))
    except Exception as e:
        say(f'[{name}] FAILED: {type(e).__name__}: {e}')
        return False
    papers = [PaperInfo(source=r.source, kind='pmid' if r.pmid else 'pdf', pmid=r.pmid,
                        name=r.label or (r.pmid if r.pmid else r.source),
                        review=result.reviews.get(r.source, r.review)) for r in refs]
    s = Session(owner='local', pmid=next((r.pmid for r in refs if r.pmid), None), focus=focus, papers=papers,
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
    say(f'[{name}] done: {len(s.draft.reactions)} reactions from {len(refs)} paper(s), {len(open_issues)} open issues, {len(suggestions)} suggested fixes')
    notes = [i.message for i in open_issues if i.source == 'pipeline']
    if notes:
        say(f'[{name}] note: ' + '; '.join(notes))
    return True


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
    ap.add_argument('--force', action='store_true', help='run again even if the results already exist')
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
        pipeline, _ = build_pipeline()
        if qa_model is None and not args.no_qa:
            from curator_llm.adapters.chat_anthropic import AnthropicChatModel
            qa_model = AnthropicChatModel(role='qa')
    if args.no_qa:
        qa_model = None
    done, failed = 0, []
    if len(refs) == 1 and args.name:
        jobs = [(args.name, refs)]
    elif args.separate or len(refs) == 1:
        jobs = [((os.path.splitext(r.label)[0] if r.pdf_path else f'pmid-{r.pmid}'), [r]) for r in refs]
    else:
        jobs = [(args.name or f'papers-{time.strftime("%Y%m%d-%H%M%S")}', refs)]
    for name, group in jobs:
        if annotate(name, group, args.focus, args.out, pipeline, say, args.force, qa_model=qa_model):
            done += 1
        else:
            failed.append(name)
    say(f'Finished: {done} session(s) done, {len(failed)} failed' + (f' ({", ".join(failed)})' if failed else '') +
        f'. Results in {args.out}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
