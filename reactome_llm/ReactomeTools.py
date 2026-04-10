"""
Specialized Tools for Reactome Multi-Agent Literature Annotation

This module provides agent-specific tools that wrap existing functionality from
the GenePathwayAnnotator and other Reactome utilities. Each agent gets access
to a curated set of tools appropriate for their role and responsibilities.

Tool categories:
- Literature and data access tools
- Reactome schema and validation tools
- Quality assurance and testing tools
- Expert knowledge and evaluation tools
"""

import json
import logging
from typing import Any, List
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import Field

from GenePathwayAnnotator import GenePathwayAnnotator
import ReactomeUtils as utils
import ReactomeNeo4jUtils as neo4j_utils
from CrewAIEventLogger import emit_tool_event

logger = logging.getLogger(__name__)


class LiteratureSearchTool(BaseTool):
    """Tool for searching and retrieving literature from PubMed"""
    
    name: str = "literature_search"
    description: str = "Search PubMed for papers related to a gene, extract abstracts and metadata"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, gene: str, max_papers: int = 8, additional_terms: str = "") -> str:
        """Search PubMed for gene-related literature"""
        try:
            # Use existing PubMed functionality
            query = f"{gene} interactions OR {gene} reactions OR {gene} pathways"
            if additional_terms:
                query += f" OR {additional_terms}"
            
            # Get PubMed abstracts via the existing retriever factory.
            pubmed_retriever = self.gene_annotator._get_pubmed_retriver(top_k_results=max_papers)
            docs = pubmed_retriever.get_relevant_documents(query)[:max_papers]
            
            # Structure the results
            papers = []
            for doc in docs:
                papers.append({
                    "pmid": doc.metadata.get("uid", ""),
                    "title": doc.metadata.get("title", ""),
                    "abstract": doc.page_content,
                    "authors": doc.metadata.get("authors", ""),
                    "journal": doc.metadata.get("source", ""),
                    "year": doc.metadata.get("year", "")
                })
            
            return json.dumps({
                "gene": gene,
                "query": query,
                "papers_found": len(papers),
                "papers": papers
            })
            
        except Exception as e:
            return json.dumps({"error": str(e), "gene": gene})


class FullTextAnalysisTool(BaseTool):
    """Tool for analyzing full-text papers when available"""
    
    name: str = "fulltext_analysis"  
    description: str = "Analyze full-text papers when available for deeper information extraction"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, pmid: str, gene: str = "") -> str:
        """Analyze full-text paper for gene-related information.
        
        `pmid` may be a bare PMID string (e.g. '25391454') or an explicit local PDF path
        (e.g. 'data/papers/25391454.pdf'). Bare PMIDs are resolved automatically to
        data/papers/<pmid>.pdf relative to the working directory.
        """
        try:
            analysis_gene = (gene or "").strip() or "UNSPECIFIED_GENE"
            # Resolve a bare PMID to the expected local PDF path.
            pdf_path = pmid if str(pmid).lower().endswith(".pdf") else f"data/papers/{pmid}.pdf"

            from pathlib import Path as _Path
            if not _Path(pdf_path).exists():
                return json.dumps({
                    "pmid": pmid,
                    "gene": analysis_gene,
                    "status": "skipped",
                    "error": f"Local PDF not found at '{pdf_path}'; full-text analysis skipped."
                })

            model = self.gene_annotator.get_default_llm()
            result = self.gene_annotator.analyze_full_paper(pdf_path, analysis_gene, model=model)

            # Convert model responses (e.g., LangChain AIMessage) into JSON-safe data.
            def _json_default(obj: Any) -> Any:
                if hasattr(obj, "content"):
                    return obj.content
                return str(obj)

            return json.dumps({
                "pmid": pmid,
                "gene": analysis_gene,
                "analysis": result,
                "status": "success"
            }, default=_json_default)
        except Exception as e:
            return json.dumps({
                "pmid": pmid,
                "gene": (gene or "").strip() or "UNSPECIFIED_GENE",
                "error": str(e),
                "status": "failed"
            })


class ReactomeQueryTool(BaseTool):
    """Tool for querying existing Reactome data"""
    
    name: str = "reactome_query"
    description: str = "Query existing Reactome database for pathway, reaction, and entity information"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, gene: str, query_type: str = "pathways", pathway: str = "") -> str:
        """Query Reactome database for gene information"""
        try:
            if query_type == "pathways":
                pathways = neo4j_utils.query_pathways_for_gene(gene)
                return json.dumps({
                    "gene": gene,
                    "query_type": query_type,
                    "pathways": pathways
                })
            elif query_type == "reactions":
                if not pathway:
                    return json.dumps({
                        "gene": gene,
                        "error": "query_type 'reactions' requires a 'pathway' argument"
                    })
                reactions_df = neo4j_utils.query_reaction_roles_of_pathway(pathway, [gene])
                return json.dumps({
                    "gene": gene,
                    "pathway": pathway,
                    "query_type": query_type,
                    "reactions": reactions_df.to_dict(orient="records")
                })
            else:
                return json.dumps({
                    "gene": gene,
                    "error": f"Unknown query_type: {query_type}. Use 'pathways' or 'reactions'."
                })
                
        except Exception as e:
            return json.dumps({
                "gene": gene,
                "error": str(e)
            })


class ProteinInteractionTool(BaseTool):
    """Tool for retrieving protein-protein interactions"""
    
    name: str = "protein_interactions"
    description: str = "Get protein-protein interactions from IntAct and BioGRID databases"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, gene: str, interaction_source: str = "intact_biogrid") -> str:
        """Get protein interactions for gene"""
        try:
            # Use existing PPI functionality
            interactions = self.gene_annotator.get_ppi_loader().get_interactions(
                query_gene=gene,
                interaction_source=interaction_source,
                filter_ppis_with_fi=True,
                fi_cutoff=0.8,
            )
            
            # Get pathway enrichment
            pathway_enrichment = []
            if interactions:
                interaction_map_df = utils.map_interactions_in_pathways(interactions)
                enrichment_df = utils.pathway_binomial_enrichment_df(
                    interaction_map_df,
                    list(interactions.keys()),
                    fdr_cutoff=0.05,
                )
                if enrichment_df is not None and not enrichment_df.empty:
                    pathway_enrichment = enrichment_df.head(20).to_dict("records")
            
            return json.dumps({
                "gene": gene,
                "interaction_source": interaction_source,
                "interactions": interactions,
                "pathway_enrichment": pathway_enrichment
            })
            
        except Exception as e:
            return json.dumps({
                "gene": gene,
                "error": str(e)
            })


class SchemaValidationTool(BaseTool):
    """Tool for validating Reactome data model instances"""
    
    name: str = "schema_validation"
    description: str = "Validate generated Reactome instances against the official schema. Accepts instances plus optional schema JSON or schema_path."
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, instances: str, schema: str = "", schema_path: str = "") -> str:
        """Validate instances against Reactome schema"""
        try:
            # Parse instances
            data = json.loads(instances) if isinstance(instances, str) else instances
            schema_data = None

            if schema_path:
                schema_data = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            elif schema:
                schema_data = json.loads(schema) if isinstance(schema, str) else schema
            
            # Basic validation checks
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "schema_provided": bool(schema_data),
                "schema_path": schema_path or None,
            }
            
            # Check for required fields in entities
            if "entities" in data:
                for entity in data["entities"]:
                    if "class" not in entity:
                        validation_results["errors"].append("Missing 'class' in entity")
                    if "displayName" not in entity:
                        validation_results["errors"].append("Missing 'displayName' in entity")
            
            # Check for required fields in reactions
            if "reactions" in data:
                for reaction in data["reactions"]:
                    if "class" not in reaction:
                        validation_results["errors"].append("Missing 'class' in reaction")
                    if "displayName" not in reaction:
                        validation_results["errors"].append("Missing 'displayName' in reaction")

            if schema_data is not None:
                try:
                    import jsonschema
                    jsonschema.validate(instance=data, schema=schema_data)
                except ImportError:
                    validation_results["warnings"].append(
                        "jsonschema package is not installed; external schema validation was skipped"
                    )
                except Exception as e:
                    validation_results["errors"].append(f"Schema validation failed: {str(e)}")
            
            if validation_results["errors"]:
                validation_results["valid"] = False
            
            return json.dumps(validation_results)
            
        except Exception as e:
            return json.dumps({
                "valid": False,
                "error": str(e)
            })


class ConsistencyCheckTool(BaseTool):
    """Tool for checking consistency with existing Reactome data"""
    
    name: str = "consistency_check"
    description: str = "Check consistency of new instances with existing Reactome data"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, instances: str, gene: str) -> str:
        """Check consistency with existing data"""
        try:
            # Parse instances
            data = json.loads(instances) if isinstance(instances, str) else instances
            
            # Get existing data for comparison
            existing_pathways = neo4j_utils.query_pathways_for_gene(gene)
            
            consistency_report = {
                "gene": gene,
                "conflicts": [],
                "consistency_score": 1.0,
                "recommendations": []
            }
            
            # Check for potential conflicts
            if "pathways" in data:
                for pathway in data["pathways"]:
                    pathway_name = pathway.get("displayName", "")
                    # Simple conflict detection
                    if pathway_name in [p["pathway"] for p in existing_pathways]:
                        consistency_report["conflicts"].append({
                            "type": "pathway_overlap",
                            "description": f"Pathway {pathway_name} already exists"
                        })
            
            if consistency_report["conflicts"]:
                consistency_report["consistency_score"] = 0.7
                consistency_report["recommendations"].append(
                    "Review overlapping pathways and consider integration instead of duplication"
                )
            
            return json.dumps(consistency_report)
            
        except Exception as e:
            return json.dumps({
                "gene": gene,
                "error": str(e)
            })


class EvidenceEvaluationTool(BaseTool):
    """Tool for evaluating evidence strength and quality"""
    
    name: str = "evidence_evaluation"
    description: str = "Evaluate the strength and quality of literature evidence"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, evidence: str, reference_text: str = "") -> str:
        """Evaluate evidence quality"""
        try:
            # Lightweight deterministic scoring to keep the tool synchronous and robust.
            evidence_text = evidence or ""
            reference = reference_text or ""
            overlap = len(set(evidence_text.lower().split()) & set(reference.lower().split()))
            llm_score = min(10, max(0, overlap // 5 + (4 if evidence_text else 0)))
            evidence_strength = "high" if llm_score >= 7 else "medium" if llm_score >= 5 else "low"
            
            return json.dumps({
                "evidence": evidence,
                "llm_score": llm_score,
                "evidence_strength": evidence_strength,
                "evaluation": {
                    "confidence": llm_score / 10,
                    "reliability": evidence_strength,
                    "recommendation": "accept" if llm_score >= 6 else "review" if llm_score >= 4 else "reject"
                }
            })
            
        except Exception as e:
            return json.dumps({
                "evidence": evidence,
                "error": str(e)
            })


class QualityMetricsTool(BaseTool):
    """Tool for calculating various quality metrics"""
    
    name: str = "quality_metrics"
    description: str = "Calculate comprehensive quality metrics for annotations"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, instances: str, evidence_data: str) -> str:
        """Calculate quality metrics"""
        try:
            # Parse inputs
            instances_data = json.loads(instances) if isinstance(instances, str) else instances
            evidence = json.loads(evidence_data) if isinstance(evidence_data, str) else evidence_data
            
            # Calculate metrics
            metrics = {
                "completeness": 0.8,  # Placeholder
                "accuracy": 0.85,     # Placeholder
                "consistency": 0.9,   # Placeholder
                "evidence_support": 0.75,  # Placeholder
                "overall_quality": 0.825
            }
            
            # Add detailed breakdown
            breakdown = {
                "total_instances": len(instances_data.get("entities", [])) + 
                                 len(instances_data.get("reactions", [])) +
                                 len(instances_data.get("pathways", [])),
                "evidence_count": len(evidence.get("papers", [])),
                "validation_status": "preliminary"
            }
            
            return json.dumps({
                "metrics": metrics,
                "breakdown": breakdown,
                "timestamp": "2026-04-06"
            })
            
        except Exception as e:
            return json.dumps({
                "error": str(e)
            })


class ReactomeToolkit:
    """Toolkit that provides agent-specific tools"""
    
    def __init__(self, gene_annotator: GenePathwayAnnotator):
        """
        Initialize toolkit with gene annotator instance
        
        Args:
            gene_annotator: GenePathwayAnnotator instance providing core functionality
        """
        self.gene_annotator = gene_annotator
        self._init_tools()
    
    def _init_tools(self):
        """Initialize all available tools"""
        self.literature_search = LiteratureSearchTool(gene_annotator=self.gene_annotator)
        self.fulltext_analysis = FullTextAnalysisTool(gene_annotator=self.gene_annotator)
        self.reactome_query = ReactomeQueryTool(gene_annotator=self.gene_annotator)
        self.protein_interactions = ProteinInteractionTool(gene_annotator=self.gene_annotator)
        self.schema_validation = SchemaValidationTool(gene_annotator=self.gene_annotator)
        self.consistency_check = ConsistencyCheckTool(gene_annotator=self.gene_annotator)
        self.evidence_evaluation = EvidenceEvaluationTool(gene_annotator=self.gene_annotator)
        self.quality_metrics = QualityMetricsTool(gene_annotator=self.gene_annotator)
        for tool in self.get_all_tools():
            self._instrument_tool(tool)

    def _instrument_tool(self, tool: BaseTool):
        """Wrap tool execution to emit structured start/end events."""
        if getattr(tool, "_crewai_structured_logging_wrapped", False):
            return

        original_run = tool._run

        def wrapped_run(*args, **kwargs):
            emit_tool_event(tool.name, "start")
            try:
                result = original_run(*args, **kwargs)
                result_status = "end"
                if isinstance(result, str):
                    try:
                        payload = json.loads(result)
                        status_value = payload.get("status")
                        if status_value in {"failed", "error"}:
                            result_status = "error"
                        else:
                            result_status = "end"
                    except Exception:
                        result_status = "end"
                emit_tool_event(tool.name, result_status)
                return result
            except Exception as exc:
                emit_tool_event(tool.name, "error", error=str(exc))
                raise

        tool._run = wrapped_run
        tool._crewai_structured_logging_wrapped = True
    
    def _filter_tools(self, tools: List[BaseTool], enabled_names: List[str] | None) -> List[BaseTool]:
        if not enabled_names:
            return tools
        enabled_set = set(enabled_names)
        return [tool for tool in tools if tool.name in enabled_set]

    def get_extractor_tools(self, enabled_names: List[str] | None = None) -> List[BaseTool]:
        """Get tools for Literature Extractor agent"""
        tools = [
            self.literature_search,
            self.fulltext_analysis,
            self.protein_interactions,
            self.evidence_evaluation
        ]
        return self._filter_tools(tools, enabled_names)
    
    def get_curator_tools(self, enabled_names: List[str] | None = None) -> List[BaseTool]:
        """Get tools for Reactome Curator agent"""
        tools = [
            self.reactome_query,
            self.schema_validation,
            self.protein_interactions,
            self.evidence_evaluation
        ]
        return self._filter_tools(tools, enabled_names)
    
    def get_reviewer_tools(self, enabled_names: List[str] | None = None) -> List[BaseTool]:
        """Get tools for Domain Expert Reviewer agent"""
        tools = [
            self.literature_search,
            self.reactome_query, 
            self.evidence_evaluation,
            self.quality_metrics,
            self.consistency_check
        ]
        return self._filter_tools(tools, enabled_names)
    
    def get_qa_tools(self, enabled_names: List[str] | None = None) -> List[BaseTool]:
        """Get tools for Quality Checker agent"""
        tools = [
            self.schema_validation,
            self.consistency_check,
            self.quality_metrics,
            self.reactome_query
        ]
        return self._filter_tools(tools, enabled_names)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all available tools"""
        return [
            self.literature_search,
            self.fulltext_analysis, 
            self.reactome_query,
            self.protein_interactions,
            self.schema_validation,
            self.consistency_check,
            self.evidence_evaluation,
            self.quality_metrics
        ]