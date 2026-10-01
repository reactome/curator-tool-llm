from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


@dataclass
class Job:
    id: str
    owner: str
    status: str = 'queued'        # queued | running | done | failed
    progress: str = ''
    result: Any = None
    error: Optional[str] = None


class JobRunner(Protocol):
    def submit(self, owner: str, fn: Callable[[Callable[[str], None]], Any]) -> str:
        """Run fn(report_progress) in the background and return a job id."""
        ...

    def get(self, job_id: str, owner: str) -> Optional[Job]:
        """The job, or None if it does not exist or belongs to someone else."""
        ...
