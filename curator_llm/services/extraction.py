"""Extract -> merge -> (optional) review for ONE paper, behind a typed function.

The steps are still the existing scripts (run_extraction.py, run_merge.py, run_review.py), each in its
own subprocess. That is deliberate: run_merge/run_review keep their state in module globals and run
their driver at import, so two sessions merging in one process would corrupt each other. The process
boundary is the isolation. What callers get is a plain object, not CLIs and files to know about.
"""
import json
import logging
import os
import re
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional

from curator_llm.ports.extractor import ExtractionResult

logger = logging.getLogger(__name__)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_NUM = r'\d{1,3}(?:,\d{3})*'
_USAGE = re.compile(rf'({_NUM}) call\(s\), tokens in ({_NUM}) / out ({_NUM})(?:, cache r({_NUM})/w({_NUM}))?')
_REVIEW_USAGE = re.compile(rf'\[usage\] tokens in ({_NUM}) / out ({_NUM})')
_SCORE = re.compile(r'SCORE:\s*([\d.]+)')

_locks_guard = threading.Lock()
_locks: Dict[str, threading.Lock] = {}


def _lock_for(key: str) -> threading.Lock:
    """Two sessions on the same paper would write the same result files; make the second wait."""
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _n(s: str) -> int:
    return int(s.replace(',', '')) if s else 0


def parse_usage(stdout: str) -> Dict[str, int]:
    """Final token counts from a step's stdout (the scripts reprint a running total, so take the last)."""
    u = {'calls': 0, 'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}
    m = _USAGE.findall(stdout or '')
    if m:
        c, i, o, cr, cw = m[-1]
        u['calls'], u['input'], u['output'], u['cache_read'], u['cache_write'] = _n(c), _n(i), _n(o), _n(cr), _n(cw)
    r = _REVIEW_USAGE.findall(stdout or '')
    if r:
        u['calls'] += 1
        u['input'] += _n(r[-1][0])
        u['output'] += _n(r[-1][1])
    return u


def _add(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _subprocess_run(cmd: List[str], cwd: str, timeout: int):
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True)


class SubprocessExtractor:
    def __init__(self, root: str = ROOT, results_dir: Optional[str] = None, timeout: int = 1800,
                 review: bool = False, run: Callable = _subprocess_run, stem: Optional[Callable] = None):
        self.root, self.timeout, self.review, self._run = root, timeout, review, run
        self.results_dir = results_dir or os.path.join(root, 'results')
        self._stem = stem

    def _output_stem(self, spec: str, gene: str) -> str:
        if self._stem:
            return self._stem(spec, gene)
        sys.path.insert(0, os.path.join(self.root, 'reactome_llm'))
        from PubMedFetcher import output_stem
        return output_stem(spec, gene=gene)

    @staticmethod
    def _read(path: str) -> list:
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return []

    def _step(self, name: str, cmd: List[str], res: ExtractionResult) -> Optional[subprocess.CompletedProcess]:
        try:
            proc = self._run(cmd, self.root, self.timeout)
        except subprocess.TimeoutExpired:
            res.error = f'{name} timed out after {self.timeout}s'
            return None
        res.usage = _add(res.usage, parse_usage(proc.stdout))
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '').strip()[-600:]
            res.error = f'{name} failed (exit {proc.returncode}): {tail}'
            return None
        return proc

    def extract(self, pmid: Optional[str], pdf_path: Optional[str], gene: str) -> ExtractionResult:
        res = ExtractionResult()
        if not pmid and not pdf_path:
            res.error = 'need a pmid or a pdf'
            return res
        spec = str(pmid) if pmid else pdf_path
        tag = None if pmid else 'localpdf'
        stem = self._output_stem(spec, gene)
        label = f'{stem}_{tag}' if tag else stem
        res.extraction_path = os.path.join(self.results_dir, f'{label}_extraction.json')
        py = sys.executable
        with _lock_for(label):
            cmd = [py, 'run_extraction.py', spec, '--gene', gene, '--overwrite'] + (['--tag', tag] if tag else [])
            if self._step('extraction', cmd, res) is None:
                return res
            extracted = self._read(res.extraction_path)
            res.n_extracted = len(extracted)
            if not extracted:
                res.error = 'extraction found no reactions'
                return res
            res.reactions = extracted                      # fallback if merge fails
            res.merged_path = os.path.join(self.results_dir, f'{label}_merged.json')   # what run_merge writes
            err_before = res.error
            if self._step('merge', [py, 'run_merge.py', res.extraction_path], res) is None:
                res.warnings.append(f'{res.error}; using unmerged reactions')
                res.error, res.merged_path = err_before, None
            else:
                merged = self._read(res.merged_path)
                if merged:
                    res.reactions, res.n_merged = merged, len(merged)
                else:
                    res.warnings.append('merge wrote no reactions; using unmerged reactions')
                    res.merged_path = None
            if self.review and res.merged_path:
                if self._step('review', [py, 'run_review.py', res.merged_path], res) is None:
                    res.warnings.append(res.error)
                    res.error = err_before
                else:
                    rp = os.path.join(self.results_dir, f'{label}_review.md')
                    if os.path.isfile(rp):
                        res.review_path = rp
                        m = _SCORE.search(open(rp).read())
                        res.review_score = float(m.group(1)) if m else None
        res.ok = True
        return res
