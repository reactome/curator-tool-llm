"""Simplest JobRunner: a daemon thread per job, state held in memory (Mongo persistence comes
with the session store in Phase 4). Jobs are lost on restart, which is the accepted trade-off."""
import threading
import uuid
from typing import Callable, Dict, Optional

from curator_llm.ports.jobs import Job


class InProcessJobRunner:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, owner: str, fn: Callable) -> str:
        job = Job(id=uuid.uuid4().hex, owner=owner)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, fn), daemon=True).start()
        return job.id

    def _run(self, job: Job, fn: Callable):
        job.status = 'running'
        try:
            job.result = fn(lambda msg: setattr(job, 'progress', msg))
            job.status = 'done'
        except Exception as e:                      # surfaced to the user as a failed job
            job.error, job.status = f'{type(e).__name__}: {e}', 'failed'

    def get(self, job_id: str, owner: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        return job if job and job.owner == owner else None
