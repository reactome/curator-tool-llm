"""Extract -> merge -> (optional) review for ONE paper, behind a typed function.

The steps are still the existing scripts (run_extraction.py, run_merge.py, run_review.py), each in its
own subprocess. That is deliberate: run_merge/run_review keep their state in module globals and run
their driver at import, so two sessions merging in one process would corrupt each other. The process
boundary is the isolation. What callers get is a plain object, not CLIs and files to know about.
"""
import json
import logging
import os
import os
import re
import subprocess
import sys
import threading
import time
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


def _subprocess_run(cmd: List[str], cwd: str, timeout: int, on_line: Optional[Callable[[str], None]] = None):
    """Run a step, handing each output line to `on_line` as it appears (so progress can be reported while the step
    runs). stderr is merged into stdout. Raises TimeoutExpired after `timeout` seconds, like subprocess.run."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONUNBUFFERED='1'))
    timed_out = []

    def kill():
        timed_out.append(True)
        proc.kill()
    timer = threading.Timer(timeout, kill)
    timer.start()
    lines: List[str] = []
    try:
        for line in proc.stdout:
            lines.append(line)
            if on_line:
                try:
                    on_line(line.rstrip('\n'))
                except Exception:                      # progress is cosmetic: never let it break a step
                    logger.debug('progress callback failed', exc_info=True)
        proc.wait()
    finally:
        timer.cancel()
        proc.stdout.close()
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)
    return subprocess.CompletedProcess(cmd, proc.returncode, ''.join(lines), '')


_CHUNKS = re.compile(r'->\s*(\d+) chunks')
_CHUNK_DONE = re.compile(r'^\s*chunk \d+/(\d+):')
_UNIQUE = re.compile(r'->\s*(\d+) unique signatures')
_PREFILTER = re.compile(r'pre-filter:\s*(\d+)/(\d+) pairs')
_JUDGED = re.compile(r'judged ~(\d+)/(\d+) pairs')


class StepProgress:
    """Turns the lines a step prints into short messages for the user ("extracting reactions: 7/20 chunks").
    Returns the new message from feed(), or None when the line changes nothing."""

    def __init__(self, step: str):
        self.step, self.done, self.total, self.last = step, 0, 0, None

    def _say(self, msg: str) -> Optional[str]:
        if msg == self.last:
            return None
        self.last = msg
        return msg

    def feed(self, line: str) -> Optional[str]:
        if self.step == 'extraction':
            if m := _CHUNKS.search(line):
                self.total = int(m.group(1))
                return self._say(f'extracting reactions: 0/{self.total} chunks')
            if _CHUNK_DONE.match(line):
                self.done += 1
                return self._say(f'extracting reactions: {self.done}/{self.total or "?"} chunks')
        elif self.step == 'merge':
            if m := _UNIQUE.search(line):
                return self._say(f'merging reactions: {m.group(1)} distinct reactions to compare')
            if m := _PREFILTER.search(line):
                self.total = int(m.group(1))
                return self._say(f'merging reactions: comparing {self.total} candidate pairs')
            if m := _JUDGED.search(line):
                return self._say(f'merging reactions: compared {m.group(1)}/{m.group(2)} pairs')
            for marker, msg in (('[revote]', 're-checking uncertain pairs'), ('[cluster]', 'grouping duplicates'),
                                ('[validate]', 'checking the groups'), ('[consolidate]', 'combining duplicate details')):
                if line.startswith(marker):
                    return self._say(f'merging reactions: {msg}')
        elif self.step == 'review' and self.last is None:
            return self._say('reviewing the reactions with a second model')
        return None


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

    def _step(self, name: str, cmd: List[str], res: ExtractionResult,
              report: Optional[Callable[[str], None]] = None) -> Optional[subprocess.CompletedProcess]:
        t0 = time.time()
        try:
            if report:
                progress = StepProgress(name)

                def on_line(line: str):
                    if msg := progress.feed(line):
                        report(msg)
                proc = self._run(cmd, self.root, self.timeout, on_line=on_line)
            else:
                proc = self._run(cmd, self.root, self.timeout)
            logger.info('%s step took %.0fs', name, time.time() - t0)
        except subprocess.TimeoutExpired:
            res.error = f'{name} timed out after {self.timeout}s'
            return None
        used = parse_usage(proc.stdout)
        res.usage = _add(res.usage, used)
        res.step_usage[name] = used
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '').strip()[-600:]
            res.error = f'{name} failed (exit {proc.returncode}): {tail}'
            return None
        return proc

    def extract(self, pmid: Optional[str], pdf_path: Optional[str], gene: str,
                report: Optional[Callable[[str], None]] = None) -> ExtractionResult:
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
            if self._step('extraction', cmd, res, report) is None:
                return res
            extracted = self._read(res.extraction_path)
            res.n_extracted = len(extracted)
            if not extracted:
                res.error = 'extraction found no reactions'
                return res
            res.reactions = extracted                      # fallback if merge fails
            res.merged_path = os.path.join(self.results_dir, f'{label}_merged.json')   # what run_merge writes
            err_before = res.error
            if self._step('merge', [py, 'run_merge.py', res.extraction_path], res, report) is None:
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
                if self._step('review', [py, 'run_review.py', res.merged_path], res, report) is None:
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
