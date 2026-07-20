"""
CrewAI-based Multi-Agent Framework for Reactome Literature Annotation

This module implements a multi-agent framework using CrewAI to annotate literature
into Reactome pathway model instances. The framework consists of 4 specialized agents:

1. ReactomeCurator: Converts structured outputs into Reactome data model instances
2. LiteratureExtractor: Extracts relevant information from scientific papers  
3. Reviewer: Domain expert validation of generated instances
4. QualityChecker: Ensures consistency and QA compliance

Author: GitHub Copilot & Reactome Team
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

from crewai import Crew, Process
from crewai.agent import Agent
from crewai.task import Task
from zmq import log

from ReactomeAgents import ReactomeAgents
from CrewAIEventLogger import emit_agent_event, emit_job_event
from ReactomeTasks import ReactomeTasks  
from ReactomeTools import ReactomeToolkit, LiteratureSearchTool
from GenePathwayAnnotator import GenePathwayAnnotator
import ReactomeUtils as utils
from ReactomeLLMErrors import *
from ModelConfig import get_crewai_model_settings
from ReactomeModels import ReactomeEntity
from QueryBuilder import (build_query_and_search_terms, build_gene_specific_pathway_descriptions,
                          build_judge_context)
from CuratorRubric import judge_select
import token_profiler
import logging_config

# Set up logging
logging_config.setup_logging()

logger = logging.getLogger(__name__)


@dataclass
class AnnotationRequest:
    """Input data structure for literature annotation requests"""
    gene: Optional[str] = None
    papers: List[str] = field(default_factory=list)  # List of PMIDs mapped to local PDFs at data/papers/<pmid>.pdf
    pathways: Optional[List[str]] = None  # Target pathways for focused annotation
    schema_path: Optional[str] = 'resources/reactome_domain_model.json'  # Optional JSON schema file used during QA validation
    max_papers: int = 8
    quality_threshold: float = 0.7
    enable_full_text: bool = True
    enable_literature_search: bool = False
    enabled_phases: Optional[List[str]] = None
    enabled_agents: Optional[List[str]] = None
    enabled_tools: Optional[Dict[str, List[str]]] = None


@dataclass  
class AnnotationResult:
    """Output data structure for annotation results"""
    gene: str
    reactome_instances: Dict[str, Any]  # ReactomeDataModel dump: {gene, entities, complexes, reactions, pathways}
    literature_evidence: Dict[str, Any]  # LiteratureExtraction dump: {gene, interactions, pathways, functions, summary}
    quality_scores: Dict[str, float]
    validation_report: Dict[str, Any]
    consistency_check: Dict[str, Any]
    final_consensus: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class CrewAILiteratureAnnotator:
    """
    Main orchestrator for the multi-agent literature annotation framework.
    
    This class coordinates the 4 specialized agents to process literature and generate
    high-quality Reactome pathway annotations with comprehensive validation.
    """

    DEFAULT_PHASES = [
        "phase_1_literature_extraction",
        "phase_2_data_model_creation",
        "phase_3_expert_review",
        "phase_4_quality_assurance",
        "phase_5_final_consensus",
    ]
    DEFAULT_AGENTS = ["extractor", "curator", "reviewer", "qa_checker"]
    
    def __init__(self, 
                 gene_annotator: GenePathwayAnnotator,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_iter: int = 3,
                 verbose: bool = True):
        """
        Initialize the CrewAI framework with all agents and tools.
        
        Args:
            gene_annotator: Existing GenePathwayAnnotator instance for data access
            model: LLM model to use for agents (falls back to environment config)
            temperature: Temperature setting for creativity vs consistency (falls back to environment config)
            max_iter: Maximum iterations for complex tasks
            verbose: Enable detailed logging
        """
        default_model, default_temperature = get_crewai_model_settings()
        self.gene_annotator = gene_annotator
        self.model = model or default_model
        self.temperature = default_temperature if temperature is None else temperature
        self.max_iter = max_iter
        self.verbose = verbose
        # Verified UniProt accession for the gene, resolved once per run (see annotate_literature)
        self.resolved_accession = None
        # Deterministic pathway-placement suggestion (FI-partner enrichment), computed once per
        # run when no explicit target pathways are given (see annotate_literature)
        self.resolved_placement = None
        # Lightweight gene background (LLM prior knowledge), generated once per run ONLY in the
        # cold-start + gated-out case, to orient Phase 2 when it has no placement directive.
        self.resolved_description = None
        # Per-pathway gene-specific re-rank descriptions, generated once per run ONLY for has-data
        # genes, to replace the raw pathway summaries as the Stage-2 target (see annotate_literature).
        self.resolved_pathway_descriptions = None

        # Initialize components
        self.toolkit = ReactomeToolkit(gene_annotator)
        self.agents = ReactomeAgents(self.model, self.temperature, max_iter=self.max_iter)
        self.tasks = ReactomeTasks()
        
        # Create the crew
        self.crew = self._create_crew()
        
        logger.info(f"CrewAI Literature Annotator initialized with model: {model}")
    
    def _create_crew(self) -> Crew:
        """Create and configure the CrewAI crew with all agents"""
        return self._create_runtime_crew(self.DEFAULT_AGENTS, {})

    def _create_runtime_crew(self, enabled_agents: List[str], enabled_tools: Dict[str, List[str]]) -> Crew:
        enabled_agent_set = set(enabled_agents or self.DEFAULT_AGENTS)

        extractor_agent = None
        if "extractor" in enabled_agent_set:
            extractor_tools = self.toolkit.get_extractor_tools(enabled_tools.get("extractor"))
            extractor_agent = self.agents.create_literature_extractor(extractor_tools)

        curator_agent = None
        if "curator" in enabled_agent_set:
            curator_tools = self.toolkit.get_curator_tools(enabled_tools.get("curator"))
            curator_agent = self.agents.create_reactome_curator(curator_tools)

        reviewer_agent = None
        if "reviewer" in enabled_agent_set:
            reviewer_tools = self.toolkit.get_reviewer_tools(enabled_tools.get("reviewer"))
            reviewer_agent = self.agents.create_reviewer(reviewer_tools)

        qa_agent = None
        if "qa_checker" in enabled_agent_set:
            qa_tools = self.toolkit.get_qa_tools(enabled_tools.get("qa_checker"))
            qa_agent = self.agents.create_quality_checker(qa_tools)

        agents = [agent for agent in [curator_agent, extractor_agent, reviewer_agent, qa_agent] if agent is not None]
        if not agents:
            raise CrewAIAnnotationError("No agents enabled for this run")

        # Keep explicit references so each phase can bind a task to the intended specialist.
        self.curator_agent = curator_agent
        self.extractor_agent = extractor_agent
        self.reviewer_agent = reviewer_agent
        self.qa_agent = qa_agent

        return Crew(
            agents=agents,
            tasks=[],
            process=Process.sequential,  # Phases are explicitly orchestrated in code
            verbose=self.verbose,
            tracing=True,
            memory=False,  # Disabled: requires OpenAI embedder for ChromaDB
            max_iter=self.max_iter
        )

    def _configure_runtime(self, request: AnnotationRequest) -> None:
        self.enabled_phase_set = set(request.enabled_phases or self.DEFAULT_PHASES)
        self.enabled_agent_set = set(request.enabled_agents or self.DEFAULT_AGENTS)
        self.enabled_tools_map = request.enabled_tools or {}
        self.crew = self._create_runtime_crew(list(self.enabled_agent_set), self.enabled_tools_map)

    def _is_phase_enabled(self, phase_id: str) -> bool:
        return phase_id in self.enabled_phase_set

    def _phase_skip_context(self, phase_id: str, gene: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        emit_job_event("skip", phase=phase_id, gene=gene, reason="disabled")
        return payload
    
    async def annotate_literature(self, request: AnnotationRequest) -> AnnotationResult:
        """
        Main entry point for literature annotation using the multi-agent framework.
        
        Args:
            request: AnnotationRequest containing gene, papers, and parameters
            
        Returns:
            AnnotationResult with Reactome instances and validation reports
            
        Raises:
            CrewAIAnnotationError: When multi-agent annotation workflow fails
        """
        request.gene = (request.gene or "").strip() or "UNSPECIFIED_GENE"
        logger.info(f"Starting multi-agent annotation for gene: {request.gene}")

        try:
            # Clear the token-usage registry per run so a reused annotator can't bleed one gene's
            # counts into the next (no-op unless TOKEN_PROFILE is set).
            token_profiler.reset()
            self._configure_runtime(request)
            # Reset per-run so a reused annotator can't carry a prior gene's background into
            # either the Phase-2 GENE BACKGROUND or the Stage-2 rerank target (see below).
            self.resolved_description = None
            self.resolved_pathway_descriptions = None
            # Final judged paper selection, published for the Stage-1 tool to consume (see the
            # upfront retrieve+judge below). Reset per-run so a reused annotator can't return a
            # prior gene's papers if this run skips or fails retrieval.
            self.gene_annotator.judged_papers = {}
            # Resolve the gene's UniProt accession once, deterministically (Reactome graph,
            # then UniProt), so every phase uses the same verified identifier instead of each
            # agent recalling its own guess. Resolving here (not in an agent) is the fix for
            # the cross-phase accession inconsistency.
            self.resolved_accession = self.gene_annotator.resolve_uniprot_accession(request.gene)
            logger.info(f"Resolved UniProt accession for {request.gene}: {self.resolved_accession}")

            # Deterministic pathway placement from FI-partner enrichment: computed once here
            # (not inside an agent) and injected into the Phase-2 curation task for the curator
            # to VERIFY, not re-derive. Only when the caller gave no explicit target pathways —
            # if they did, respect that intent and skip. The directive is injected ONLY when the
            # placement is confident (gate); a weak placement (e.g. TANC1) would mislead the
            # curator and degrade the annotation, so in that case resolved_placement stays None
            # and no directive is injected.
            if not request.pathways:
                placement = utils.suggest_pathway_placement(request.gene)
                confident = utils.is_confident_placement(placement)
                self.resolved_placement = placement if confident else None
                logger.info(
                    f"Suggested pathway placement for {request.gene}: "
                    f"{placement['status']} (confident={confident}; "
                    f"{'injecting directive' if confident else 'gated out, no directive'})"
                )
                # Gate failed -> Phase 2 would otherwise see only the raw Phase-1 extraction.
                # Generate a lightweight gene background as orientation. build_query_and_search_terms
                # makes a BLOCKING LLM .invoke(); running it directly on the event loop collides with
                # the loop/LLM client CrewAI's flow uses and poisons the next agent call (empty-response
                # abort -- the failure get_reranking_target's comment warned about). Run it in a worker
                # thread so it can't touch the loop. Guarded so a failed/empty call degrades to None.
                if not confident:
                    try:
                        with token_profiler.label("desc_gate_fail_background"):
                            _, self.resolved_description = await asyncio.to_thread(
                                build_query_and_search_terms, request.gene
                            )
                    except Exception as e:
                        logger.warning(f"Gene description generation failed for {request.gene}: {e}")

            # Publish the gate-fail background to the SHARED gene_annotator so LiteratureSearchTool's
            # Stage-2 rerank can use it as the target (via get_reranking_target's description_override)
            # instead of the bare gene+synonyms identity string -- validated to rerank markedly better
            # on cold-start genes. Keyed by gene so a reused annotator can't leak across genes; None
            # for non-gate-fail genes (they rerank against pathway summaries, ignoring this). This
            # reuses the ONE precomputed description (also the Phase-2 GENE BACKGROUND) -- no extra LLM
            # call, and it stays LLM-free inside the async rerank path.
            self.gene_annotator.rerank_target_descriptions = {request.gene: self.resolved_description}

            # Has-data genes: precompute per-pathway gene-SPECIFIC descriptions and publish them the
            # same way, so Stage-2 re-ranks against "how {gene} functions within {pathway}" instead of
            # the raw pathway summary (which is generic and pulls broad reviews). Validated A/B:
            # SHANK3 curator usefulness 1.0->7.2, BRCA1 2.0->3.6. Runs in a worker thread -- it makes a
            # BLOCKING LLM .invoke() that would collide with CrewAI's event loop on the main thread
            # (same reason as the DESC precompute above). Self-gates: returns {} with NO LLM call for
            # cold-start genes (no released human pathways), so gate-fail/gate-pass genes are unaffected
            # and fall back to their existing targets. Guarded so a failure degrades to raw summaries.
            try:
                with token_profiler.label("desc_per_pathway"):
                    self.resolved_pathway_descriptions = await asyncio.to_thread(
                        build_gene_specific_pathway_descriptions, request.gene)
            except Exception as e:
                logger.warning(f"Gene-specific pathway descriptions failed for {request.gene}: {e}")
                self.resolved_pathway_descriptions = {}
            self.gene_annotator.rerank_pathway_descriptions = {
                request.gene: self.resolved_pathway_descriptions}

            # Upfront retrieval + LLM curator-judge. This is the piece that CAN'T follow the
            # precompute-and-cache pattern used for the rerank targets above: those depend only on
            # gene-level info, but the judge must read the already-fetched, cross-encoder-ranked
            # pool. So we run the WHOLE retrieval (Stage-1 merge + cross-encoder re-rank + judge)
            # here, off the event loop via asyncio.to_thread -- the judge makes a blocking LLM
            # .invoke() that would poison the next agent call if run on CrewAI's loop (same reason
            # the DESC/pathway precomputes use to_thread). Its LLM-free RESULT (the final papers,
            # with abstracts) is cached on the shared annotator; Phase-1's live literature_search
            # call returns it directly, so abstracts still reach the extractor the normal way and
            # the pool is fetched only once. On failure/empty, the cache stays {} and the tool
            # falls back to live merge + cross-encoder (no judge).
            if request.enable_literature_search:
                try:
                    with token_profiler.label("literature_retrieve_judge"):
                        cached = await asyncio.to_thread(
                            self._retrieve_and_judge, request.gene, request.max_papers)
                    if cached and cached["papers"]:
                        self.gene_annotator.judged_papers = {request.gene: cached}
                        logger.info(
                            f"Literature judge selected {len(cached['papers'])} papers for "
                            f"{request.gene} from {cached['candidate_pool']} candidates "
                            f"(pool {cached['pool_size']}; dropped "
                            f"{cached['dropped_below_threshold']} below the rubric floor)")
                    else:
                        logger.warning(
                            f"Upfront retrieve+judge produced no papers for {request.gene}; "
                            f"Phase-1 tool will fall back to live cross-encoder retrieval")
                except Exception as e:
                    logger.warning(
                        f"Upfront retrieve+judge failed for {request.gene}: {e}; "
                        f"Phase-1 tool will fall back to live cross-encoder retrieval")

            # Phase 1: Literature Extraction and Preprocessing
            extraction_context = await self._phase_1_literature_extraction(request)

            # if test:
            #     return extraction_context
            
            # Phase 2: Reactome Data Model Creation  
            curation_context = await self._phase_2_data_model_creation(
                request, extraction_context
            )
            
            # Phase 3: Domain Expert Review
            review_context = await self._phase_3_expert_review(
                request, extraction_context, curation_context
            )
            
            # Phase 4: Quality Assurance and Consistency Check
            qa_context = await self._phase_4_quality_assurance(
                request, extraction_context, curation_context, review_context
            )

            # Phase 5: Virtual meeting for final multi-agent consensus
            consensus_context = await self._phase_5_final_consensus_meeting(
                request,
                extraction_context,
                curation_context,
                review_context,
                qa_context
            )

            final_result = AnnotationResult(
                gene=request.gene,
                reactome_instances=curation_context["reactome_instances"],
                literature_evidence=extraction_context["structured_information"],
                quality_scores=review_context["quality_scores"],
                validation_report=review_context["validation_report"],
                consistency_check=qa_context["consistency_check"],
                final_consensus=consensus_context["final_consensus"],
                processing_metadata={
                    "model_used": self.model,
                    "temperature": self.temperature,
                    "papers_processed": extraction_context["papers_processed"],
                    "pathways_created": curation_context["pathways_created"],
                    "quality_threshold": request.quality_threshold,
                    "full_text_enabled": request.enable_full_text,
                    "literature_search_enabled": request.enable_literature_search,
                    "enabled_phases": sorted(list(self.enabled_phase_set)),
                    "enabled_agents": sorted(list(self.enabled_agent_set)),
                    "enabled_tools": self.enabled_tools_map,
                    "final_decision": consensus_context["final_consensus"].get("decision", "unknown")
                }
            )
            
            logger.info(f"Multi-agent annotation completed for gene: {request.gene}")
            # Write the token-usage CSV + ranked summary (no-op unless TOKEN_PROFILE is set).
            token_profiler.emit_report(request.gene)
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-agent annotation failed for {request.gene}: {str(e)}")
            raise CrewAIAnnotationError(f"CrewAI annotation failed: {str(e)}")
    
    # Size of the cross-encoder candidate pool the LLM curator-judge chooses the final papers from,
    # and the rubric floor below which a candidate is dropped (1-2 = "not usable for a specific
    # annotation"; 3+ = at least weak background). See CuratorRubric.judge_select.
    _JUDGE_CANDIDATE_POOL = 20
    _JUDGE_MIN_SCORE = 3

    def _retrieve_and_judge(self, gene: str, max_papers: int) -> Optional[Dict[str, Any]]:
        """Stage-1 merge + cross-encoder re-rank -> candidate pool, then the LLM curator-judge picks
        the final papers (score-threshold + top-N). SYNCHRONOUS and self-contained so it can run in
        a worker thread (asyncio.to_thread) -- the judge's blocking LLM .invoke() must stay off
        CrewAI's event loop. Returns the cache dict for gene_annotator.judged_papers, or None if
        retrieval yields no candidates.

        Reads the rerank targets already published on the shared annotator, so the cross-encoder
        re-ranks against the same gene-specific pathway/description text the live tool would use."""
        tool = LiteratureSearchTool(gene_annotator=self.gene_annotator)
        cand = tool.retrieve_candidates(gene, candidate_pool=self._JUDGE_CANDIDATE_POOL)
        candidates = cand["papers"]
        if not candidates:
            return None

        # Rich rubric context so the judge scores gene-absent pathway/partner papers correctly.
        # Reuse an already-precomputed prose lead (no extra LLM call): the gate-fail gene background,
        # else the gene-specific pathway descriptions; build_judge_context then appends the gene's
        # explicit Reactome pathway + FI-partner names (Neo4j/FI, LLM-free).
        if self.resolved_description:
            prose = self.resolved_description
        elif self.resolved_pathway_descriptions:
            prose = " ".join(str(v) for v in self.resolved_pathway_descriptions.values())
        else:
            prose = None
        judge_description = build_judge_context(gene, description=prose)

        judged = judge_select(gene, judge_description, candidates,
                              max_papers=max_papers, min_score=self._JUDGE_MIN_SCORE)
        return {
            "name_query": cand.get("name_query", ""),
            "context_query": cand.get("context_query", ""),
            "pool_size": cand.get("pool_size", 0),
            "candidate_pool": len(candidates),
            "papers": judged["selected"],
            "dropped_below_threshold": judged["dropped_below_threshold"],
        }

    async def _phase_1_literature_extraction(self, request: AnnotationRequest) -> Dict[str, Any]:
        """Phase 1: Extract and structure information from literature"""
        phase_id = "phase_1_literature_extraction"
        if not self._is_phase_enabled(phase_id) or self.extractor_agent is None:
            return self._phase_skip_context(phase_id, request.gene, {
                "raw_result": {},
                "structured_information": [],
                "papers_processed": 0,
                "gene": request.gene,
            })

        logger.info(f"Phase 1: Literature extraction for {request.gene}")
        emit_agent_event("LiteratureExtractor", "start", phase=phase_id, gene=request.gene)
        
        # Create dynamic task for literature extraction
        extraction_task = self.tasks.create_literature_extraction_task(
            gene=request.gene,
            papers=request.papers,
            max_papers=request.max_papers,
            enable_full_text=request.enable_full_text,
            enable_literature_search=request.enable_literature_search,
            accession=self.resolved_accession
        )
        extraction_task.agent = self.extractor_agent
        
        # Update crew with this task
        self.crew.tasks = [extraction_task]
        
        # Execute extraction
        with token_profiler.profile_kickoff("phase_1_literature_extraction", self.crew):
            extraction_result = await self.crew.kickoff_async({
                "gene": request.gene,
                "papers": request.papers,
                "max_papers": request.max_papers
            })
        emit_agent_event("LiteratureExtractor", "end", phase="phase_1_literature_extraction", gene=request.gene)

        extraction = self._structured(extraction_result, "LiteratureExtraction")
        # papers_processed = unique PMIDs cited across all extracted evidence — read straight
        # off the validated model instead of regex-scraping the raw text.
        evidence_items = extraction.interactions + extraction.pathways + extraction.functions
        pmids = {item.pmid.strip() for item in evidence_items if item.pmid and item.pmid.strip()}

        return {
            "raw_result": extraction_result.raw,
            "structured_information": extraction.model_dump(),
            "papers_processed": len(pmids),
            "gene": request.gene
        }
    
    async def _phase_2_data_model_creation(self, 
                                          request: AnnotationRequest,
                                          extraction_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Convert structured information to Reactome data model instances"""
        phase_id = "phase_2_data_model_creation"
        if not self._is_phase_enabled(phase_id) or self.curator_agent is None:
            return self._phase_skip_context(phase_id, request.gene, {
                "raw_result": {},
                "reactome_instances": [],
                "pathways_created": 0,
                "gene": request.gene,
            })

        logger.info(f"Phase 2: Data model creation for {request.gene}")
        emit_agent_event("ReactomeCurator", "start", phase=phase_id, gene=request.gene)
        
        # Create curation task
        curation_task = self.tasks.create_reactome_curation_task(
            gene=request.gene,
            structured_info=extraction_context["structured_information"],
            target_pathways=request.pathways,
            schema_path=request.schema_path,
            accession=self.resolved_accession,
            placement=self.resolved_placement,
            description=self.resolved_description
        )
        curation_task.agent = self.curator_agent
        
        self.crew.tasks = [curation_task]
        
        # Execute curation
        with token_profiler.profile_kickoff("phase_2_data_model_creation", self.crew):
            curation_result = await self.crew.kickoff_async({
                "gene": request.gene,
                "target_pathways": str(request.pathways or [])
            })
        emit_agent_event("ReactomeCurator", "end", phase="phase_2_data_model_creation", gene=request.gene)

        curation = self._structured(curation_result, "ReactomeDataModel")
        curation = self._repair_dangling_references(curation)
        return {
            "raw_result": curation_result.raw,
            # by_alias=True so the Reactome "class" key is emitted (the model field is `cls`).
            "reactome_instances": curation.model_dump(by_alias=True),
            "pathways_created": len(curation.pathways),
            "gene": request.gene
        }

    def _repair_dangling_references(self, data: Any) -> Any:
        """Guarantee referential integrity of the Phase 2 model.

        The curator sometimes references a product species inline in a reaction
        (e.g. 'STIM1 degradation products', 'NFE2L2 phospho-Ser40 (nuclear)') that it
        never defined as an entity, producing a dangling reference that violates
        Reactome integrity. For each simple (non-complex) reference in a reaction's
        input/output/catalystActivity or a complex's components that resolves to
        neither a defined entity nor a defined complex, create a minimal entity stub —
        inheriting the UniProt identifier/species from a base-name match (stripping a
        trailing '[compartment]' or '(qualifier)') when one exists — and log it.
        Colon-notation references ('A:B') are complexes, not entities, so they are left
        for QA to flag rather than materialised as bogus entities.
        """
        def base_name(name: str) -> str:
            n = re.sub(r"\s*\[[^\]]*\]\s*$", "", name or "").strip()
            n = re.sub(r"\s*\([^)]*\)\s*$", "", n).strip()
            return n

        entity_names = {e.displayName for e in data.entities}
        entities_by_base = {base_name(e.displayName): e for e in data.entities}
        complex_names = {c.displayName for c in data.complexes}

        referenced: List[str] = []
        for c in data.complexes:
            referenced.extend(c.components)
        for r in data.reactions:
            referenced.extend(r.input)
            referenced.extend(r.output)
            referenced.extend(r.catalystActivity)

        created: Dict[str, Any] = {}
        for ref in referenced:
            if (not ref) or ":" in ref or ref in entity_names or ref in complex_names or ref in created:
                continue
            src = entities_by_base.get(base_name(ref))
            created[ref] = ReactomeEntity(
                displayName=ref,
                identifier=(src.identifier if src else ""),
                species=(src.species if src else "Homo sapiens"),
                referenceEntity=(src.referenceEntity if src else ""),
            )

        if created:
            data.entities.extend(created.values())
            logger.warning(
                "Phase 2 integrity repair for %s: created %d stub entity(ies) for "
                "dangling reference(s): %s",
                data.gene, len(created), ", ".join(sorted(created)),
            )
        return data
    
    async def _phase_3_expert_review(self,
                                    request: AnnotationRequest,
                                    extraction_context: Dict[str, Any],
                                    curation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Domain expert validation of generated instances"""
        phase_id = "phase_3_expert_review"
        if not self._is_phase_enabled(phase_id) or self.reviewer_agent is None:
            return self._phase_skip_context(phase_id, request.gene, {
                "raw_result": {},
                "validation_report": {},
                "quality_scores": {},
                "recommendations": [],
                "gene": request.gene,
            })

        logger.info(f"Phase 3: Expert review for {request.gene}")
        emit_agent_event("Reviewer", "start", phase=phase_id, gene=request.gene)
        
        # Create review task
        review_task = self.tasks.create_expert_review_task(
            gene=request.gene,
            reactome_instances=curation_context["reactome_instances"],
            original_papers=extraction_context["structured_information"],
            quality_threshold=request.quality_threshold,
            accession=self.resolved_accession
        )
        review_task.agent = self.reviewer_agent
        
        self.crew.tasks = [review_task]
        
        # Execute review
        with token_profiler.profile_kickoff("phase_3_expert_review", self.crew):
            review_result = await self.crew.kickoff_async({
                "gene": request.gene,
                "quality_threshold": request.quality_threshold
            })
        emit_agent_event("Reviewer", "end", phase="phase_3_expert_review", gene=request.gene)

        review = self._structured(review_result, "ExpertReview")
        return {
            "raw_result": review_result.raw,
            "validation_report": review.model_dump(),
            # Flatten the reviewer's scores into the {name: float} shape the result expects.
            "quality_scores": {
                "overall_quality": review.overall_score,
                **review.criterion_scores.model_dump(),
            },
            "gene": request.gene
        }
    
    async def _phase_4_quality_assurance(self,
                                        request: AnnotationRequest,
                                        extraction_context: Dict[str, Any],
                                        curation_context: Dict[str, Any],
                                        review_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Final QA check and consistency validation"""
        phase_id = "phase_4_quality_assurance"
        if not self._is_phase_enabled(phase_id) or self.qa_agent is None:
            return self._phase_skip_context(phase_id, request.gene, {
                "raw_result": {},
                "consistency_check": {},
                "gene": request.gene,
            })

        logger.info(f"Phase 4: Quality assurance for {request.gene}")
        emit_agent_event("QualityChecker", "start", phase=phase_id, gene=request.gene)
        
        # Create QA task
        qa_task = self.tasks.create_quality_assurance_task(
            gene=request.gene,
            reactome_instances=curation_context["reactome_instances"],
            validation_report=review_context["validation_report"],
            schema_path=request.schema_path,
            quality_threshold=request.quality_threshold,
            accession=self.resolved_accession
        )
        qa_task.agent = self.qa_agent
        
        self.crew.tasks = [qa_task]
        
        # Execute QA
        with token_profiler.profile_kickoff("phase_4_quality_assurance", self.crew):
            qa_result = await self.crew.kickoff_async({
                "gene": request.gene,
                "quality_threshold": request.quality_threshold
            })
        emit_agent_event("QualityChecker", "end", phase="phase_4_quality_assurance", gene=request.gene)

        qa = self._structured(qa_result, "QAReport")
        return {
            "raw_result": qa_result.raw,
            "consistency_check": qa.model_dump(),
            "gene": request.gene
        }

    async def _phase_5_final_consensus_meeting(self,
                                               request: AnnotationRequest,
                                               extraction_context: Dict[str, Any],
                                               curation_context: Dict[str, Any],
                                               review_context: Dict[str, Any],
                                               qa_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Virtual meeting among all agents for a final consensus decision"""
        phase_id = "phase_5_final_consensus"
        if not self._is_phase_enabled(phase_id):
            return self._phase_skip_context(phase_id, request.gene, {
                "individual_votes": {},
                "final_consensus": {"decision": "skipped", "summary": "Phase disabled"},
                "gene": request.gene,
            })

        logger.info(f"Phase 5: Final consensus meeting for {request.gene}")
        emit_agent_event("ConsensusMeeting", "start", phase=phase_id, gene=request.gene)

        role_to_agent = {
            "reactome_curator": self.curator_agent,
            "literature_extractor": self.extractor_agent,
            "reviewer": self.reviewer_agent,
            "quality_checker": self.qa_agent,
        }

        # Only enabled specialists participate in the vote.
        participating = [(role, agent) for role, agent in role_to_agent.items() if agent is not None]

        async def _cast_vote(role_name: str):
            """Run one specialist's vote on its own crew + lightweight tool-less agent.
            Independent crews are what let the votes run concurrently — they don't share
            self.crew, so there's no task-list race."""
            vote_agent = self.agents.create_vote_agent(role_name)
            vote_task = self.tasks.create_final_vote_task(
                gene=request.gene,
                agent_role=role_name,
                extraction_context=extraction_context,
                curation_context=curation_context,
                review_context=review_context,
                qa_context=qa_context,
                quality_threshold=request.quality_threshold,
                accession=self.resolved_accession
            )
            vote_task.agent = vote_agent
            vote_crew = Crew(
                agents=[vote_agent],
                tasks=[vote_task],
                process=Process.sequential,
                verbose=self.verbose,
                memory=False,
            )
            emit_agent_event(role_name, "start", phase="phase_5_final_vote", gene=request.gene)
            # Fresh per-vote crew -> its usage_metrics is this kickoff's cost outright (no delta).
            with token_profiler.profile_kickoff(f"phase_5_vote:{role_name}", vote_crew, phase="phase_5"):
                vote_result = await vote_crew.kickoff_async({
                    "gene": request.gene,
                    "agent_role": role_name,
                    "quality_threshold": str(request.quality_threshold)
                })
            emit_agent_event(role_name, "end", phase="phase_5_final_vote", gene=request.gene)
            return role_name, self._structured(vote_result, "AgentVote").model_dump()

        # Cast all votes concurrently — they're independent, so there's no reason to serialize.
        vote_pairs = await asyncio.gather(*[_cast_vote(role) for role, _ in participating])
        votes: Dict[str, Any] = dict(vote_pairs)

        if self.reviewer_agent is None:
            emit_job_event("skip", phase="phase_5_consensus_synthesis", gene=request.gene, reason="reviewer_disabled")
            emit_agent_event("ConsensusMeeting", "end", phase=phase_id, gene=request.gene)
            return {
                "individual_votes": votes,
                "final_consensus": {"decision": "skipped", "summary": "Reviewer agent disabled"},
                "gene": request.gene
            }

        # Synthesis is mechanical (apply decision rules over the votes) -> lightweight agent.
        consensus_agent = self.agents.create_consensus_agent()
        consensus_task = self.tasks.create_final_consensus_task(
            gene=request.gene,
            individual_votes=votes,
            quality_threshold=request.quality_threshold,
            accession=self.resolved_accession
        )
        consensus_task.agent = consensus_agent
        consensus_crew = Crew(
            agents=[consensus_agent],
            tasks=[consensus_task],
            process=Process.sequential,
            verbose=self.verbose,
            memory=False,
        )
        emit_agent_event("ConsensusChair", "start", phase="phase_5_consensus_synthesis", gene=request.gene)
        with token_profiler.profile_kickoff("phase_5_consensus_synthesis", consensus_crew, phase="phase_5"):
            consensus_result = await consensus_crew.kickoff_async({
                "gene": request.gene,
                "quality_threshold": str(request.quality_threshold)
            })
        emit_agent_event("ConsensusChair", "end", phase="phase_5_consensus_synthesis", gene=request.gene)
        emit_agent_event("ConsensusMeeting", "end", phase=phase_id, gene=request.gene)

        return {
            "individual_votes": votes,
            "final_consensus": self._structured(consensus_result, "ConsensusDecision").model_dump(),
            "gene": request.gene
        }
    
    def _structured(self, result: Any, model_name: str) -> Any:
        """Return the validated Pydantic model CrewAI attached to a task result.

        With output_pydantic set on every Task, CrewAI validates the agent's answer
        against the schema and re-prompts the LLM until it matches, exposing the typed
        object on `result.pydantic`. This just unwraps it — there is no JSON scraping or
        regex fallback anywhere downstream. If `pydantic` is None, structured output
        genuinely failed after CrewAI's own retries, so we fail loudly rather than
        silently degrade to a placeholder."""
        model = getattr(result, "pydantic", None)
        if model is None:
            raw = str(getattr(result, "raw", result))[:500]
            raise CrewAIAnnotationError(
                f"{model_name}: no validated structured output (result.pydantic is None) "
                f"after CrewAI retries. Raw output began: {raw!r}"
            )
        return model

    def export_results(self, result: AnnotationResult, output_path: str) -> None:
        """Export annotation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Results exported to: {output_file}")


def create_crewai_annotator(gene_annotator: GenePathwayAnnotator) -> CrewAILiteratureAnnotator:
    """Factory function to create a CrewAI annotator instance"""
    return CrewAILiteratureAnnotator(gene_annotator)