"""The real Pipeline: paper -> (extract, merge, optional review) -> verified evidence -> typed draft
-> resolved identifiers. Extraction/merge/review still run as the existing scripts (subprocesses);
everything after them is in-process and returns objects."""
import logging
import os
from typing import Callable, Dict, List, Optional

from curator_llm.models.session import Issue
from curator_llm.ports.external import OntologyClient, UniProtClient
from curator_llm.ports.lookup import InstanceLookup
from curator_llm.ports.events import EventLookup
from curator_llm.ports.extractor import Extractor
from curator_llm.ports.pipeline import PipelineInput, PipelineResult
from curator_llm.services import issues as iss
from curator_llm.services.draft_builder import build_draft
from curator_llm.services.evidence_attach import attach_evidence
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.existing_events import find_existing
from curator_llm.services.extraction import SubprocessExtractor
from curator_llm.services.paper_loader import load_paper_text
from curator_llm.services.paper_text import PaperText
from curator_llm.services.resolvers import Resolver

logger = logging.getLogger(__name__)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class DefaultPipeline:
    def __init__(self, lookup: InstanceLookup, uniprot: Optional[UniProtClient] = None,
                 ontology: Optional[OntologyClient] = None, model=None, review: bool = False,
                 timeout: int = 1800, extractor: Optional[Extractor] = None, events: Optional[EventLookup] = None):
        self.lookup, self.uniprot, self.ontology = lookup, uniprot, ontology
        self.model, self.events = model, events
        self.extractor = extractor or SubprocessExtractor(timeout=timeout, review=review)

    def run(self, spec: PipelineInput, report: Callable[[str], None]) -> PipelineResult:
        if not spec.pmid and not spec.pdf_path:
            raise ValueError('need a pmid or a pdf')
        gene = spec.focus or 'paper'
        report('extracting and merging reactions (several minutes)')
        res = self.extractor.extract(spec.pmid, spec.pdf_path, gene)
        reactions: List[dict] = res.reactions
        if not res.ok or not reactions:
            raise RuntimeError('no reactions could be extracted from this paper '
                               f'(no open-access full text, or nothing reaction-like in it): {res.error or ""}'.strip())

        report('verifying quotes against the paper')
        papers: Dict[str, Optional[PaperText]] = {}
        if spec.pdf_path:
            pdf = PaperText.from_pdf(None, spec.pdf_path)
            paper_for = lambda src: pdf
            papers['main'] = pdf
        else:
            def paper_for(src):
                if src not in papers:
                    papers[src] = load_paper_text(src)
                papers.setdefault('main', papers[src])
                return papers[src]
        store = EvidenceStore()
        attach_evidence(reactions, store, paper_for)
        issues: List[Issue] = iss.rejected_quote_issues({
            f'r{i}': (r.get('annotation_result', r).get('evidence_rejected') or [])
            for i, r in enumerate(reactions)})

        report('building the Reactome draft')
        accession = None
        if spec.focus and self.uniprot:
            try:
                accession = self.uniprot.search_gene(spec.focus)
            except Exception as e:
                logger.warning('UniProt lookup for focus %s failed: %s', spec.focus, e)
        draft, notes = build_draft(spec.focus or 'the paper', reactions, accession, self.model)
        issues += iss.issues_from_notes('builder', notes)

        report('resolving identifiers')
        issues += iss.issues_from_notes('resolver', Resolver(self.lookup, self.uniprot, self.ontology).resolve(draft))
        existing = []
        if self.events is not None:
            report('checking Reactome for existing reactions')
            existing, ex_issues = find_existing(draft, spec.pmid, self.events)
            issues += ex_issues
        main = papers.get('main')
        return PipelineResult(draft, store.all(), issues, main.to_dict() if main else None, existing)
