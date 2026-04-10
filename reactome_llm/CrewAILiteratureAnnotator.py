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

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

from crewai import Crew, Process
from crewai.agent import Agent
from crewai.task import Task
from zmq import log

from ReactomeAgents import ReactomeAgents
from CrewAIEventLogger import emit_agent_event
from ReactomeTasks import ReactomeTasks  
from ReactomeTools import ReactomeToolkit
from GenePathwayAnnotator import GenePathwayAnnotator 
from ReactomeLLMErrors import *
from ModelConfig import get_crewai_model_settings
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


@dataclass  
class AnnotationResult:
    """Output data structure for annotation results"""
    gene: str
    reactome_instances: List[Dict[str, Any]]
    literature_evidence: List[Dict[str, Any]]
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
        
        # Initialize components
        self.toolkit = ReactomeToolkit(gene_annotator)
        self.agents = ReactomeAgents(self.model, self.temperature)
        self.tasks = ReactomeTasks()
        
        # Create the crew
        self.crew = self._create_crew()
        
        logger.info(f"CrewAI Literature Annotator initialized with model: {model}")
    
    def _create_crew(self) -> Crew:
        """Create and configure the CrewAI crew with all agents"""
        
        # Get agent instances
        extractor_agent = self.agents.create_literature_extractor(self.toolkit.get_extractor_tools())
        curator_agent = self.agents.create_reactome_curator(self.toolkit.get_curator_tools())
        reviewer_agent = self.agents.create_reviewer(self.toolkit.get_reviewer_tools())
        qa_agent = self.agents.create_quality_checker(self.toolkit.get_qa_tools())
        
        agents = [curator_agent, extractor_agent, reviewer_agent, qa_agent]

        # Keep explicit references so each phase can bind a task to the intended specialist.
        self.curator_agent = curator_agent
        self.extractor_agent = extractor_agent
        self.reviewer_agent = reviewer_agent
        self.qa_agent = qa_agent
        
        # Define task workflow - tasks will be created dynamically during annotation
        tasks = []
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,  # Phases are explicitly orchestrated in code
            verbose=self.verbose,
            tracing=True,
            memory=True,  # Enable memory for context between tasks
            max_iter=self.max_iter
        )
    
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
            # test = True
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
                    "final_decision": consensus_context["final_consensus"].get("decision", "unknown")
                }
            )
            
            logger.info(f"Multi-agent annotation completed for gene: {request.gene}")
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-agent annotation failed for {request.gene}: {str(e)}")
            raise CrewAIAnnotationError(f"CrewAI annotation failed: {str(e)}")
    
    async def _phase_1_literature_extraction(self, request: AnnotationRequest) -> Dict[str, Any]:
        """Phase 1: Extract and structure information from literature"""
        logger.info(f"Phase 1: Literature extraction for {request.gene}")
        emit_agent_event("LiteratureExtractor", "start", phase="phase_1_literature_extraction", gene=request.gene)
        
        # Create dynamic task for literature extraction
        extraction_task = self.tasks.create_literature_extraction_task(
            gene=request.gene,
            papers=request.papers,
            max_papers=request.max_papers,
            enable_full_text=request.enable_full_text,
            enable_literature_search=request.enable_literature_search
        )
        extraction_task.agent = self.extractor_agent
        
        # Update crew with this task
        self.crew.tasks = [extraction_task]
        
        # Execute extraction
        extraction_result = self.crew.kickoff({
            "gene": request.gene,
            "papers": request.papers,
            "max_papers": request.max_papers
        })
        emit_agent_event("LiteratureExtractor", "end", phase="phase_1_literature_extraction", gene=request.gene)
        
        return {
            "raw_result": extraction_result,
            "structured_information": self._parse_extraction_result(extraction_result),
            "papers_processed": len(request.papers),
            "gene": request.gene
        }
    
    async def _phase_2_data_model_creation(self, 
                                          request: AnnotationRequest,
                                          extraction_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Convert structured information to Reactome data model instances"""
        logger.info(f"Phase 2: Data model creation for {request.gene}")
        emit_agent_event("ReactomeCurator", "start", phase="phase_2_data_model_creation", gene=request.gene)
        
        # Create curation task
        curation_task = self.tasks.create_reactome_curation_task(
            gene=request.gene,
            structured_info=extraction_context["structured_information"],
            target_pathways=request.pathways,
            schema_path=request.schema_path
        )
        curation_task.agent = self.curator_agent
        
        self.crew.tasks = [curation_task]
        
        # Execute curation
        curation_result = self.crew.kickoff({
            "gene": request.gene,
            "target_pathways": str(request.pathways or [])
        })
        emit_agent_event("ReactomeCurator", "end", phase="phase_2_data_model_creation", gene=request.gene)
        
        return {
            "raw_result": curation_result,
            "reactome_instances": self._parse_curation_result(curation_result),
            "pathways_created": self._count_pathways_created(curation_result),
            "gene": request.gene
        }
    
    async def _phase_3_expert_review(self,
                                    request: AnnotationRequest,
                                    extraction_context: Dict[str, Any],
                                    curation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Domain expert validation of generated instances"""
        logger.info(f"Phase 3: Expert review for {request.gene}")
        emit_agent_event("Reviewer", "start", phase="phase_3_expert_review", gene=request.gene)
        
        # Create review task
        review_task = self.tasks.create_expert_review_task(
            gene=request.gene,
            reactome_instances=curation_context["reactome_instances"],
            original_papers=extraction_context["structured_information"],
            quality_threshold=request.quality_threshold
        )
        review_task.agent = self.reviewer_agent
        
        self.crew.tasks = [review_task]
        
        # Execute review
        review_result = self.crew.kickoff({
            "gene": request.gene,
            "quality_threshold": request.quality_threshold
        })
        emit_agent_event("Reviewer", "end", phase="phase_3_expert_review", gene=request.gene)
        
        return {
            "raw_result": review_result,
            "validation_report": self._parse_review_result(review_result),
            "quality_scores": self._extract_quality_scores(review_result),
            "recommendations": self._extract_recommendations(review_result),
            "gene": request.gene
        }
    
    async def _phase_4_quality_assurance(self,
                                        request: AnnotationRequest,
                                        extraction_context: Dict[str, Any],
                                        curation_context: Dict[str, Any],
                                        review_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Final QA check and consistency validation"""
        logger.info(f"Phase 4: Quality assurance for {request.gene}")
        emit_agent_event("QualityChecker", "start", phase="phase_4_quality_assurance", gene=request.gene)
        
        # Create QA task
        qa_task = self.tasks.create_quality_assurance_task(
            gene=request.gene,
            reactome_instances=curation_context["reactome_instances"],
            validation_report=review_context["validation_report"],
            schema_path=request.schema_path,
            quality_threshold=request.quality_threshold
        )
        qa_task.agent = self.qa_agent
        
        self.crew.tasks = [qa_task]
        
        # Execute QA
        qa_result = self.crew.kickoff({
            "gene": request.gene,
            "quality_threshold": request.quality_threshold
        })
        emit_agent_event("QualityChecker", "end", phase="phase_4_quality_assurance", gene=request.gene)
        
        return {
            "raw_result": qa_result,
            "consistency_check": self._parse_qa_result(qa_result),
            "gene": request.gene
        }

    async def _phase_5_final_consensus_meeting(self,
                                               request: AnnotationRequest,
                                               extraction_context: Dict[str, Any],
                                               curation_context: Dict[str, Any],
                                               review_context: Dict[str, Any],
                                               qa_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Virtual meeting among all agents for a final consensus decision"""
        logger.info(f"Phase 5: Final consensus meeting for {request.gene}")
        emit_agent_event("ConsensusMeeting", "start", phase="phase_5_final_consensus", gene=request.gene)

        role_to_agent = {
            "reactome_curator": self.curator_agent,
            "literature_extractor": self.extractor_agent,
            "reviewer": self.reviewer_agent,
            "quality_checker": self.qa_agent,
        }

        votes: Dict[str, Any] = {}
        for role_name, agent in role_to_agent.items():
            emit_agent_event(role_name, "start", phase="phase_5_final_vote", gene=request.gene)
            vote_task = self.tasks.create_final_vote_task(
                gene=request.gene,
                agent_role=role_name,
                extraction_context=extraction_context,
                curation_context=curation_context,
                review_context=review_context,
                qa_context=qa_context,
                quality_threshold=request.quality_threshold
            )
            vote_task.agent = agent
            self.crew.tasks = [vote_task]
            vote_result = self.crew.kickoff({
                "gene": request.gene,
                "agent_role": role_name,
                "quality_threshold": str(request.quality_threshold)
            })
            votes[role_name] = self._parse_vote_result(vote_result)
            emit_agent_event(role_name, "end", phase="phase_5_final_vote", gene=request.gene)

        consensus_task = self.tasks.create_final_consensus_task(
            gene=request.gene,
            individual_votes=votes,
            quality_threshold=request.quality_threshold
        )
        consensus_task.agent = self.reviewer_agent
        self.crew.tasks = [consensus_task]
        emit_agent_event("Reviewer", "start", phase="phase_5_consensus_synthesis", gene=request.gene)
        consensus_result = self.crew.kickoff({
            "gene": request.gene,
            "quality_threshold": str(request.quality_threshold)
        })
        emit_agent_event("Reviewer", "end", phase="phase_5_consensus_synthesis", gene=request.gene)
        emit_agent_event("ConsensusMeeting", "end", phase="phase_5_final_consensus", gene=request.gene)

        return {
            "individual_votes": votes,
            "final_consensus": self._parse_consensus_result(consensus_result),
            "gene": request.gene
        }
    
    def _parse_extraction_result(self, result: Any) -> List[Dict[str, Any]]:
        """Parse literature extraction results into structured format"""
        try:
            if isinstance(result, str):
                # Try to parse as JSON
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    # Fall back to text parsing
                    return [{"content": result, "type": "raw_text"}]
            elif isinstance(result, dict):
                return [result] 
            elif isinstance(result, list):
                return result
            else:
                return [{"content": str(result), "type": "unknown"}]
        except Exception as e:
            logger.warning(f"Failed to parse extraction result: {e}")
            return [{"content": str(result), "type": "parse_error", "error": str(e)}]
    
    def _parse_curation_result(self, result: Any) -> List[Dict[str, Any]]:
        """Parse curation results into Reactome instances"""
        # Implementation depends on the specific format returned by curator agent
        return self._parse_extraction_result(result)  # Placeholder
    
    def _parse_review_result(self, result: Any) -> Dict[str, Any]:
        """Parse expert review results"""
        return {"review": str(result)}  # Placeholder
    
    def _parse_qa_result(self, result: Any) -> Dict[str, Any]:
        """Parse QA results into consistency check format"""
        return {"qa_check": str(result)}  # Placeholder

    def _parse_vote_result(self, result: Any) -> Dict[str, Any]:
        """Parse one agent's vote in the final consensus meeting"""
        parsed_list = self._parse_extraction_result(result)
        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            first_item = parsed_list[0]
            if isinstance(first_item, dict):
                return first_item
        return {
            "decision": "unknown",
            "confidence": 0.0,
            "blocking_issues": [],
            "summary": str(result)
        }

    def _parse_consensus_result(self, result: Any) -> Dict[str, Any]:
        """Parse final synthesis result from the virtual meeting"""
        parsed_list = self._parse_extraction_result(result)
        if isinstance(parsed_list, list) and len(parsed_list) > 0 and isinstance(parsed_list[0], dict):
            return parsed_list[0]
        return {
            "decision": "requires_revision",
            "confidence": 0.5,
            "required_revisions": ["Consensus output was not parseable JSON"],
            "summary": str(result)
        }
    
    def _count_pathways_created(self, result: Any) -> int:
        """Count number of pathways created during curation"""
        # Placeholder implementation
        return 1
    
    def _extract_quality_scores(self, result: Any) -> Dict[str, float]:
        """Extract quality scores from review results"""
        return {"overall_quality": 0.8}  # Placeholder
    
    def _extract_recommendations(self, result: Any) -> List[str]:
        """Extract recommendations from review results"""
        return ["No specific recommendations"]  # Placeholder
    
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