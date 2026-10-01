"""Saved results, so a paper that was annotated once need not be extracted again.

Extraction and merging take 10-15 minutes of LLM calls. While developing the interface (or demonstrating it)
that wait is the whole problem, so every successful run is saved, and in replay mode a saved result is used
instead of calling the models.

  SNAPSHOT_MODE=record   (default) save each successful run; always run the pipeline
  SNAPSHOT_MODE=replay   use the saved result when there is one, otherwise run the pipeline and save it
  SNAPSHOT_MODE=off      neither save nor replay

A snapshot is keyed by the paper (PMID, or the PDF's content) and the focus gene. It holds what the pipeline
returned, before the session turns it into instances and issues, so replaying exercises the same code a real run
does after extraction. Snapshots are NOT refreshed when prompts or code improve: delete one to re-extract.
"""
import hashlib
import json
import logging
import os
import time
from typing import Callable, Optional

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import ReactomeDraft
from curator_llm.models.session import ExistingMatch, Issue
from curator_llm.ports.pipeline import Pipeline, PipelineInput, PipelineResult

logger = logging.getLogger(__name__)
VERSION = 1
MODES = ('record', 'replay', 'off')


def snapshot_key(spec: PipelineInput) -> str:
    """pmid-24751536__PINK1, or pdf-<hash of the file>__PINK1. The focus changes what the draft is about."""
    if spec.pmid:
        base = f'pmid-{spec.pmid.strip()}'
    elif spec.pdf_path:
        h = hashlib.sha256()
        with open(spec.pdf_path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        base = f'pdf-{h.hexdigest()[:16]}'
    else:
        raise ValueError('need a pmid or a pdf')
    focus = ''.join(c for c in (spec.focus or '').strip().upper() if c.isalnum() or c in '-_')
    return f'{base}__{focus}' if focus else base


class SnapshotStore:
    def __init__(self, directory: str):
        self.directory = directory

    def path(self, key: str) -> str:
        return os.path.join(self.directory, f'{key}.json')

    def save(self, key: str, result: PipelineResult) -> str:
        os.makedirs(self.directory, exist_ok=True)
        doc = {'version': VERSION, 'key': key, 'saved_at': time.time(),
               'draft': result.draft.model_dump(mode='json'),
               'evidence': [e.model_dump(mode='json') for e in result.evidence],
               'issues': [i.model_dump(mode='json') for i in result.issues],
               'paper': result.paper,
               'existing': [m.model_dump(mode='json') for m in result.existing]}
        path = self.path(key)
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(doc, f)
        os.replace(tmp, path)               # a reader never sees half a file
        return path

    def load(self, key: str) -> Optional[PipelineResult]:
        """The saved result, or None when there is none or it cannot be read (then the caller just runs the pipeline)."""
        path = self.path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                doc = json.load(f)
            if doc.get('version') != VERSION:
                logger.warning('snapshot %s has version %s, expected %s; ignoring it', key, doc.get('version'), VERSION)
                return None
            return PipelineResult(
                draft=ReactomeDraft.model_validate(doc['draft']),
                evidence=[Evidence.model_validate(e) for e in doc['evidence']],
                issues=[Issue.model_validate(i) for i in doc['issues']],
                paper=doc.get('paper'),
                existing=[ExistingMatch.model_validate(m) for m in doc.get('existing', [])])
        except Exception as e:
            logger.warning('snapshot %s could not be read (%s: %s); ignoring it', key, type(e).__name__, e)
            return None


class SnapshotPipeline:
    """Wraps a Pipeline: records its successful results and, in replay mode, serves them back instead of running it."""

    def __init__(self, inner: Pipeline, store: SnapshotStore, mode: str = 'record'):
        if mode not in MODES:
            raise ValueError(f'SNAPSHOT_MODE must be one of {MODES}, not {mode!r}')
        self.inner, self.store, self.mode = inner, store, mode

    def run(self, spec: PipelineInput, report: Callable[[str], None]) -> PipelineResult:
        if self.mode == 'off':
            return self.inner.run(spec, report)
        key = snapshot_key(spec)
        if self.mode == 'replay':
            saved = self.store.load(key)
            if saved is not None:
                report(f'using the saved result for {key} (replay mode, no models called)')
                return saved
        result = self.inner.run(spec, report)
        try:
            self.store.save(key, result)
        except Exception as e:               # failing to save must never fail the annotation
            logger.warning('could not save snapshot %s: %s', key, e)
        return result
