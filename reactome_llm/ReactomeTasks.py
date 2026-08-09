"""
Task Definitions for Reactome Multi-Agent Literature Annotation

This module defines the specific tasks that each agent performs during the 
literature annotation workflow. Tasks are designed to be executed in sequence
with proper context passing between agents.

Tasks correspond to the 4-phase workflow:
1. Literature Extraction and Processing
2. Reactome Data Model Creation 
3. Domain Expert Review and Validation
4. Quality Assurance and Consistency Checking
"""

import json
import logging
from typing import List, Dict, Any, Optional

from crewai import Task

from ReactomeModels import (
    LiteratureExtraction,
    ReactomeDataModel,
    ExpertReview,
    QAReport,
    AgentVote,
    ConsensusDecision,
)

logger = logging.getLogger(__name__)


class ReactomeTasks:
    """Factory class for creating tasks for the Reactome annotation agents"""
    
    def __init__(self):
        """Initialize the task factory"""
        pass
    
    def create_literature_extraction_task(self, 
                                        gene: Optional[str],
                                        papers: List[str], 
                                        max_papers: int = 8,
                                        enable_full_text: bool = False,
                                        enable_literature_search: bool = False,
                                        accession: Optional[str] = None,
                                        prefetched_fulltext: Optional[Dict[str, Any]] = None) -> Task:
        """
        Create a task for the Literature Extractor agent.
        
        This task involves processing scientific papers to extract molecular information
        relevant to the target gene, including interactions, pathways, and functions.
        
        Args:
            gene: Target gene symbol
            papers: List of PMIDs or paper content
            max_papers: Maximum number of papers to process
            enable_full_text: Whether to process full text or just abstracts
            enable_literature_search: Whether to search PubMed first instead of using the provided PMID list directly
            
        Returns:
            Configured Task instance
        """
        query_gene = (gene or "").strip()
        has_papers = len(papers) > 0
        tool_gene = query_gene if query_gene else "UNSPECIFIED_GENE"
        gene_label = query_gene if query_gene else "gene-agnostic full-text corpus"
        accession_directive = (
            f"\n        **VERIFIED UniProt ACCESSION:** The canonical UniProt accession for {tool_gene} is "
            f"{accession}. Whenever you record a UniProt identifier for {tool_gene}, use this exact value; "
            f"do not infer, recall, or guess a different accession.\n"
            if accession else ""
        )

        # When full text has already been parsed for this gene by the external extractor, its
        # structured evidence is injected here as authoritative input. The agent must fold it into
        # its output and NOT call `fulltext_analysis` (that path is being retired). Empty/absent ->
        # no directive, so behaviour is unchanged for runs without a full-text extractor.
        has_prefetched = bool(prefetched_fulltext and any(prefetched_fulltext.values()))
        prefetched_directive = (
            "\n        **PRE-EXTRACTED FULL-TEXT EVIDENCE (authoritative — use directly):** Full-text "
            f"analysis for {tool_gene} has ALREADY been performed by an external extractor. Do NOT call "
            "`fulltext_analysis`. Treat the JSON below as verified full-text evidence and MERGE it with any "
            "abstract-derived evidence into your structured output, de-duplicating by (partner/pathway/function "
            "+ pmid). Preserve each item's `pmid` and `evidence_strength_score`.\n        ```json\n"
            f"{json.dumps(prefetched_fulltext, indent=2, default=str)}\n        ```\n"
            if has_prefetched else ""
        )

        if enable_literature_search and query_gene:
            if enable_full_text:
                workflow_instructions = f"""
        Step 1 — Call `literature_search` with gene="{query_gene}" to retrieve abstracts and metadata.
               The JSON result will contain a `papers` list, each entry with a `pmid` field.
        Step 2 — For each PMID in the search results, call `fulltext_analysis` with pmid=<pmid_value> and gene="{tool_gene}" to analyze the local PDF if available. If the file is not found, skip that PMID and continue.
        Step 3 — Combine the available abstract and full-text evidence into the structured JSON output below.
                """
            else:
                workflow_instructions = f"""
        Step 1 — Call `literature_search` with gene="{query_gene}" to retrieve abstracts and metadata.
               The JSON result will contain a `papers` list, each entry with a `pmid` field.
        Step 2 — Full-text analysis is disabled for this run; use the search results only.
        Step 3 — Combine the available evidence into the structured JSON output below.
                """
        elif has_papers:
            if enable_literature_search and not query_gene:
                if enable_full_text:
                    workflow_instructions = f"""
        Step 1 — A query gene was not provided, so literature search is unavailable for this run.
        Step 2 — Use only the provided PMID list and call `fulltext_analysis` with gene="{tool_gene}" for each paper.
        Step 3 — Extract candidate genes, interactions, and pathways directly from the full text and aggregate into the output JSON.
                    """
                else:
                    workflow_instructions = """
        Step 1 — A query gene was not provided, so literature search is unavailable for this run.
        Step 2 — Full-text analysis is disabled; use only the provided PMID metadata/context.
        Step 3 — Return a conservative structured output and record that evidence is limited without full-text parsing.
                    """
            elif enable_full_text:
                workflow_instructions = f"""
        Step 1 — Use only the provided PMID list as the paper set for this run. Do not call `literature_search`.
        Step 2 — For each provided PMID, call `fulltext_analysis` with pmid=<pmid_value> and gene="{tool_gene}". The tool resolves the PMID to `data/papers/<pmid>.pdf`. If a PDF is missing, skip that PMID and continue.
        Step 3 — Combine findings from the provided PMID set into the structured JSON output below.
                """
            else:
                workflow_instructions = """
        Step 1 — Use only the provided PMID list as the paper set for this run. Do not call `literature_search`.
        Step 2 — Full-text analysis is disabled for this run; do not expand beyond the provided PMID list.
        Step 3 — Combine findings from the provided PMID set into the structured JSON output below.
                """
        else:
            workflow_instructions = """
        Step 1 — No papers were provided and literature search is unavailable in the current settings.
        Step 2 — Return an empty, well-formed structured output with a note that no evidence sources were available.
        Step 3 — Do not fabricate interactions/pathways without evidence.
            """

        description = f"""
        Extract and structure molecular information about {gene_label} from scientific literature.
        
        **Primary Objectives:**
        1. Process up to {max_papers} scientific papers (PMIDs provided: {papers[:5]}...)
        2. Extract molecular interactions, pathways, and functional annotations
        3. Identify evidence strength and experimental methods
        4. Structure findings in machine-readable format
        
        **Specific Information to Extract:**
        - Protein-protein interactions and binding partners
        - Pathway involvement and functional roles
        - Subcellular localization and tissue expression
        - Regulatory relationships (upstream/downstream)
        - Disease associations and phenotypes
        - Experimental evidence and confidence levels
        
        {accession_directive}
        {prefetched_directive}
        **Output Requirements:**
        Provide a structured JSON output with the following format:
        ```json
        {{
            "gene": "{tool_gene}",
            "interactions": [
                {{
                    "partner": "GENE_SYMBOL",
                    "interaction_type": "binding/regulation/etc",
                    "evidence": "experimental_method",
                    "confidence": "high/medium/low",
                    "evidence_strength_score": 0.0,
                    "pmid": "paper_id",
                    "context": "brief_description"
                }}
            ],
            "pathways": [
                {{
                    "pathway_name": "pathway_description",
                    "role": "catalyst/regulator/target",
                    "evidence": "supporting_evidence",
                    "confidence": "high/medium/low",
                    "evidence_strength_score": 0.0,
                    "pmid": "paper_id"
                }}
            ],
            "functions": [
                {{
                    "function": "molecular_function_description",
                    "evidence": "experimental_support",
                    "confidence": "high/medium/low",
                    "evidence_strength_score": 0.0,
                    "pmid": "paper_id"
                }}
            ],
            "summary": "concise_functional_summary"
        }}
        ```
        
        **Quality Standards:**
        - Only include information with clear experimental support
        - Distinguish between direct and indirect evidence
        - Prioritize recent, high-quality publications
        - Include confidence assessments for all claims

        **Tool Usage Workflow:**
          {workflow_instructions}
        """
        
        expected_output = f"""
        Structured molecular evidence for {gene_label}: interactions, pathway roles, and
        functions with confidence and PMIDs. Return ONLY the structured fields defined above
        — do not write a long prose report or narrative outside the schema.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=None,  # Will be assigned when creating crew
            output_pydantic=LiteratureExtraction,
        )

    def _build_description_context(self, gene: str, description: Optional[str]) -> str:
        """Render the gene's background description as ORIENTATION, not evidence.

        This is the model's prior knowledge of the gene (build_query_and_search_terms), NOT
        literature from Phase 1. It is injected only in the cold-start + gated-out case, where
        there is no placement directive, to give the curator a biological frame of reference.
        It must never be cited: every instance still has to trace to the Phase-1 extraction.
        Returns '' when no description is available.
        """
        if not description:
            return ""
        return (
            f"\n        **GENE BACKGROUND (orientation only — NOT evidence, do NOT cite):** "
            f"General reference on {gene}'s known biology: {description}\n        "
            f"Use this ONLY to interpret the literature evidence above. Every entity, reaction, "
            f"and pathway you create must still trace to the Phase-1 extraction (via pmid or "
            f"interaction partner), never to this background.\n"
        )

    def _build_placement_directive(self, gene: str,
                                   placement: Optional[Dict[str, Any]]) -> str:
        """Render the deterministic pathway-placement suggestion as a curator directive.

        The suggestion comes from suggest_pathway_placement() (FI-partner enrichment). The
        curator's job is to VERIFY the primary against the literature evidence and fall back
        to a secondary candidate if it doesn't fit -- not to re-derive placement. Returns ''
        when there is no usable suggestion (no placement computed, or no significant pathway).
        """
        if not placement or placement.get("status") != "ok" or not placement.get("primary"):
            return ""

        primary = placement["primary"]
        partners = ", ".join(primary.get("mapped_genes", [])) or "n/a"
        secondary = placement.get("secondary", [])
        if secondary:
            alts = "; ".join(
                f"{c['pathway_name']} (FDR {c['fdr']:.1e})" for c in secondary
            )
            secondary_line = f"Close alternatives (statistical ties): {alts}.\n        "
        else:
            secondary_line = "No close alternatives; the primary is the clear candidate.\n        "

        return (
            f"\n        **SUGGESTED PATHWAY PLACEMENT (verify against the evidence; do NOT re-derive):** "
            f"Deterministic interaction-partner enrichment suggests {gene} most likely belongs to "
            f"the Reactome pathway **{primary['pathway_name']}** (FDR {primary['fdr']:.1e}), "
            f"supported by these enriched interaction partners: {partners}.\n        "
            f"{secondary_line}"
            f"Confirm the PRIMARY pathway is consistent with the literature evidence above and "
            f"anchor the pathway/reaction instances you create to it. If the evidence clearly "
            f"contradicts the primary, fall back to one of the close alternatives. Do not invent "
            f"a placement outside these candidates unless the evidence contradicts all of them.\n"
        )

    def create_reactome_curation_task(self,
                                    gene: str,
                                    structured_info: Dict[str, Any],
                                    target_pathways: Optional[List[str]] = None,
                                    schema_path: Optional[str] = None,
                                    accession: Optional[str] = None,
                                    placement: Optional[Dict[str, Any]] = None,
                                    description: Optional[str] = None) -> Task:
        """
        Create a task for the Reactome Curator agent.
        
        This task converts structured literature information into valid Reactome
        data model instances following proper schema and relationship constraints.
        
        Args:
            gene: Target gene symbol
            structured_info: Output from literature extraction phase
            target_pathways: Specific pathways to focus on (optional)
            
        Returns:
            Configured Task instance
        """
        structured_info_json = json.dumps(structured_info, indent=2, default=str) if structured_info else "No structured information provided."
        accession_directive = (
            f"\n        **VERIFIED IDENTIFIER (use exactly):** The canonical UniProt accession for {gene} is "
            f"{accession}. Use this exact value for the `identifier` and `referenceEntity.identifier` of the "
            f"{gene} protein entity. Do NOT generate, infer, or substitute any other accession. If {gene} already "
            f"exists in Reactome under this accession, reuse that entity rather than creating a duplicate.\n"
            if accession else ""
        )

        placement_directive = self._build_placement_directive(gene, placement)
        description_context = self._build_description_context(gene, description)

        description = f"""
        Convert structured literature information about gene {gene} into valid Reactome 
        pathway model instances following the official Reactome data schema.
        
        **Input Context (Literature Extraction Output):**
        ```json
        {structured_info_json}
        ```
        Use the above structured data as the primary source of evidence for all entities,
        reactions, and pathways you create. Each annotation must be traceable to an entry
        in this input (via pmid or interaction partner).
        
        {accession_directive}
        {placement_directive}
        {description_context}
        **Primary Objectives:**
        1. Create valid Reactome Entity instances for {gene} and interaction partners
        2. Model biochemical reactions and regulatory relationships
        3. Integrate with existing pathway structures where appropriate
        4. Ensure proper hierarchy and relationship modeling
        
        **Reactome Data Model Requirements:**
        - Create EntityWithAccessionedSequence for proteins
        - Model Reaction instances for biochemical processes
        - Use proper ReactionLikeEvent for pathway integration
        - Include evidence attribution and literature references
        - Follow Reactome naming and identifier conventions
        
        **Target Pathways:** {target_pathways if target_pathways else "All relevant pathways"}
        
        **Output Format:**
        Provide structured Reactome instances in JSON format:
        ```json
        {{
            "entities": [
                {{
                    "class": "EntityWithAccessionedSequence",
                    "displayName": "{gene}",
                    "identifier": "UniProt_ID",
                    "species": "Homo sapiens",
                    "referenceEntity": "reference_details"
                }}
            ],
            "complexes": [
                {{
                    "class": "Complex",
                    "displayName": "complex_name",
                    "components": ["member_entities"],
                    "literatureReference": ["supporting_pmids"]
                }}
            ],
            "reactions": [
                {{
                    "class": "Reaction", 
                    "displayName": "reaction_description",
                    "input": ["input_entities"],
                    "output": ["output_entities"], 
                    "catalystActivity": ["catalyst_entities"],
                    "inferredFrom": ["orthologous_event"],
                    "literatureReference": ["pmid_references"]
                }}
            ],
            "pathways": [
                {{
                    "class": "Pathway",
                    "displayName": "pathway_name",
                    "hasEvent": ["contained_reactions"],
                    "summation": "pathway_description",
                    "literatureReference": ["supporting_pmids"]
                }}
            ]
        }}
        ```
        
        **Quality Requirements:**
        - All instances must be valid according to Reactome schema
        - Maintain referential integrity between related entities
        - Include proper evidence attribution
        - Use consistent naming conventions
        - Avoid duplicate instances for existing entities

        **Schema Validation:**
        After generating each batch of instances, call the `schema_validation` tool with:
        - `instances`: your generated JSON
        - `schema_path`: {schema_path if schema_path else 'not provided'}
        Fix any reported errors before finalising the output.
        If no schema_path is provided, perform structural field validation only.

        Use available tools to query existing Reactome data, validate against
        the schema, and ensure proper integration with current pathway models.
        """
        
        expected_output = f"""
        Valid Reactome data model instances for gene {gene} (entities, complexes, reactions,
        pathways). Return ONLY the structured fields defined above — do not write a long prose
        report or narrative outside the schema.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=None,
            output_pydantic=ReactomeDataModel,
        )

    def create_expert_review_task(self,
                                gene: str,
                                reactome_instances: List[Dict[str, Any]],
                                original_papers: List[Dict[str, Any]],
                                quality_threshold: float = 0.7,
                                accession: Optional[str] = None,
                                consistency_check_result: Optional[str] = None) -> Task:
        """
        Create a task for the Domain Expert Reviewer agent.
        
        This task performs expert validation of generated Reactome instances
        against biological knowledge and literature evidence.
        
        Args:
            gene: Target gene symbol
            reactome_instances: Generated instances from curation
            original_papers: Original literature information
            quality_threshold: Minimum quality score required
            
        Returns:
            Configured Task instance
        """
        accession_directive = (
            f"\n        **VERIFIED IDENTIFIER:** The canonical UniProt accession for {gene} is {accession}. "
            f"Treat this as ground truth when judging identifier correctness; do not substitute another accession.\n"
            if accession else ""
        )
        instances_json = json.dumps(reactome_instances, indent=2, default=str) if reactome_instances else "No instances were provided."
        papers_json = json.dumps(original_papers, indent=2, default=str) if original_papers else "No literature evidence was provided."
        # Deterministic overlap check is pre-run in Phase 3 and injected here, so the reviewer never
        # re-serializes the full instance JSON back through the LLM to call consistency_check as a
        # tool (that re-emission was both wasted output and, in practice, malformed -> the tool call
        # errored). Empty when not supplied -> falls back to the reviewer's own assessment.
        prerun_consistency_block = ""
        if consistency_check_result is not None:
            prerun_consistency_block = f"""
        **PRE-COMPUTED CONSISTENCY CHECK** (deterministic overlap check already run on the exact
        instances above vs. existing Reactome data — treat as ground truth; do NOT re-run it and do
        NOT reproduce the instance JSON yourself):
        ```json
        {consistency_check_result}
        ```
        Use this result when assessing Integration Quality (redundancies / conflicts with existing
        pathways) below.
"""
        description = f"""
        {accession_directive}
        Perform expert domain validation of generated Reactome instances for gene {gene}.

        **Reactome instances to review (curator output):**
        ```json
        {instances_json}
        ```

        **Original literature evidence (extractor output):**
        ```json
        {papers_json}
        ```
        Review the instances above. They ARE present — evaluate their biological accuracy and
        quality against the literature evidence and your domain knowledge. Do not report that
        instances are missing.
        {prerun_consistency_block}

        **Review Scope:**
        You will evaluate the biological accuracy and quality of generated Reactome
        instances by comparing them against:
        - Original literature evidence
        - Established biological knowledge
        - Existing Reactome annotations
        - Known molecular mechanisms
        
        **Validation Criteria:**
        
        1. **Biological Accuracy** (25% weight):
           - Are the molecular interactions biologically plausible?
           - Do pathway assignments match known gene functions?
           - Are regulatory relationships correctly modeled?
           
        2. **Evidence Support** (25% weight):
           - Is each annotation supported by experimental evidence?
           - Are confidence levels appropriately assigned?
           - Are conflicting studies appropriately handled?
           
        3. **Mechanistic Consistency** (25% weight):
           - Do biochemical reactions follow known mechanisms?
           - Are subcellular localizations consistent with function?
           - Do temporal and spatial constraints make sense?
           
        4. **Integration Quality** (25% weight):
           - How well do new instances integrate with existing pathways?
           - Are naming conventions and hierarchies respected?
           - Are redundancies and conflicts avoided?
           
        **Review Process:**
        1. Cross-reference each instance against original literature
        2. Validate biological plausibility using domain knowledge
        3. Check for consistency with existing Reactome data
        4. Assign quality scores (0-1) for each validation criterion
        5. Provide specific recommendations for improvements
        
        **Quality Threshold:** {quality_threshold}
        
        **Output Format:**
        Provide a comprehensive review report:
        ```json
        {{
            "gene": "{gene}",
            "overall_score": 0.85,
            "criterion_scores": {{
                "biological_accuracy": 0.9,
                "evidence_support": 0.8,
                "mechanistic_consistency": 0.85,
                "integration_quality": 0.9
            }},
            "instance_reviews": [
                {{
                    "instance_id": "entity_or_reaction_id",
                    "instance_type": "Entity/Reaction/Pathway",
                    "score": 0.8,
                    "issues": ["list_of_concerns"],
                    "recommendations": ["suggested_improvements"],
                    "evidence_assessment": "strong/moderate/weak"
                }}
            ],
            "summary": "overall_assessment_summary",
            "recommendations": [
                "high_level_improvement_suggestions"
            ],
            "approval_status": "approved/requires_revision/rejected"
        }}
        ```
        
        **Decision Guidelines:**
        - Approve: Overall score >= {quality_threshold} and no critical issues
        - Requires Revision: Score >= 0.5 but < {quality_threshold} or minor issues
        - Reject: Score < 0.5 or major biological inaccuracies
        
        Use available tools to query literature, validate against databases, 
        and cross-reference with expert knowledge sources.
        """
        
        expected_output = f"""
        A domain expert review of Reactome instances for gene {gene}: overall_score,
        criterion_scores, instance_reviews, and approval_status. Return ONLY the structured
        fields defined above — do not write a long prose report or narrative outside the schema.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=None,
            output_pydantic=ExpertReview,
        )

    def create_quality_assurance_task(self,
                                    gene: str,
                                    reactome_instances: List[Dict[str, Any]],
                                    validation_report: Dict[str, Any],
                                    schema_path: Optional[str] = None,
                                    quality_threshold: float = 0.7,
                                    accession: Optional[str] = None,
                                    schema_validation_result: Optional[str] = None,
                                    consistency_check_result: Optional[str] = None) -> Task:
        """
        Create a task for the Quality Checker agent.
        
        This task performs technical QA, consistency checking, and integration
        validation for the generated Reactome instances.
        
        Args:
            gene: Target gene symbol
            reactome_instances: Generated instances from curation
            validation_report: Domain expert review results
            quality_threshold: Quality threshold for acceptance
            
        Returns:
            Configured Task instance
        """
        accession_directive = (
            f"\n        **VERIFIED IDENTIFIER (ground truth):** The canonical UniProt accession for {gene} is {accession}. "
            f"Validate that every instance uses exactly this accession; do NOT introduce, recall, or substitute any "
            f"other accession in your report, and flag any deviation from {accession} as an error.\n"
            if accession else ""
        )
        instances_json = json.dumps(reactome_instances, indent=2, default=str) if reactome_instances else "No instances were provided."
        # The deterministic checks (schema_validation, consistency_check) are pre-run in Phase 4 and
        # injected here, so the agent never has to re-serialize the full instance JSON back through
        # the LLM to invoke them as tools -- that re-serialization was the ~166s output-heavy call
        # that dominated Phase-4 wall-clock. When results aren't supplied the block is empty and the
        # task falls back to the original agent-tool wording below.
        prerun_block = ""
        if schema_validation_result is not None or consistency_check_result is not None:
            prerun_block = f"""
        **PRE-COMPUTED AUTOMATED CHECK RESULTS** (already run deterministically on the exact
        instances above — treat these as ground truth; do NOT try to re-run them and do NOT
        reproduce the instance JSON yourself):
        - `schema_validation` result:
        ```json
        {schema_validation_result if schema_validation_result is not None else 'not run'}
        ```
        - `consistency_check` result:
        ```json
        {consistency_check_result if consistency_check_result is not None else 'not run'}
        ```
        Base your Schema Compliance, Referential Integrity, Data Consistency, and Integration
        findings on these results together with your own inspection of the instances above.
"""
        # Switch the tool-referencing wording depending on whether the deterministic checks were
        # pre-run (Phase-4 fix path) or the agent still has them as tools (legacy fallback).
        if prerun_block:
            automated_tests_intro = ("The following automated checks have ALREADY been run for you "
                                     "(see PRE-COMPUTED RESULTS above) — incorporate their findings; "
                                     "do NOT re-run them:")
            schema_input_block = (
                "**Schema Input:**\n"
                f"        - schema_validation has already been run against schema_path: "
                f"{schema_path if schema_path else 'not provided'} (see PRE-COMPUTED RESULTS above).\n"
                "        - If no schema path was provided, that result reflects structural validation "
                "only; report that full schema validation was not run.")
            closing_tools_line = ("The automated schema and consistency checks have already been run for "
                                  "you (see PRE-COMPUTED RESULTS above); synthesize their findings into "
                                  "your QA report rather than re-running them.")
        else:
            automated_tests_intro = "Run the following automated checks:"
            schema_input_block = (
                "**Schema Input:**\n"
                "        - If an official JSON schema file is available, use the `schema_validation` tool with:\n"
                "            - `instances`: the generated instance JSON\n"
                f"            - `schema_path`: {schema_path if schema_path else 'not provided'}\n"
                "        - If no schema path is provided, fall back to structural validation and report "
                "that full schema validation was not run.")
            closing_tools_line = ("Use available tools to run automated tests, check database consistency, "
                                  "and validate against production standards.")
        description = f"""
        {accession_directive}
        Perform comprehensive quality assurance and consistency checking for
        Reactome instances related to gene {gene}.

        **Reactome instances to QA (curator output):**
        ```json
        {instances_json}
        ```
        QA the instances above. They ARE present — assess schema compliance, referential
        integrity, consistency, and integration for them. Do not report that instances are missing.
        {prerun_block}

        **QA Scope:**
        1. Technical compliance with Reactome schema and standards
        2. Data consistency and referential integrity
        3. Integration compatibility with existing database
        4. Performance and indexing considerations
        
        **Technical Validation Checks:**
        
        1. **Schema Compliance** (30% weight):
           - Validate all instances against Reactome JSON schema
           - Check required fields and data types
           - Verify identifier formats and conventions
           - Ensure proper class inheritance
           
        2. **Referential Integrity** (25% weight):
           - Validate all cross-references between entities
           - Check for orphaned references
           - Verify proper relationship modeling
           - Ensure bidirectional consistency
           
        3. **Data Consistency** (25% weight):
           - Check for naming convention compliance
           - Validate identifier uniqueness
           - Ensure consistent nomenclature
           - Verify proper hierarchical relationships
           
        4. **Integration Testing** (20% weight):
           - Test compatibility with existing pathways
           - Check for conflicts with current annotations
           - Validate query performance implications
           - Ensure proper indexing compatibility
           
        **Automated QA Tests:**
        {automated_tests_intro}
        - Schema validation against official XSD/JSON schemas
        - Duplicate detection and conflict resolution
        - Cross-reference validation
        - Performance impact assessment
        - Ontology consistency checking

        {schema_input_block}
        
        **Expert Review Integration:**
        Review the domain expert's assessment (overall score: {validation_report.get('overall_score', 'N/A')}):
        - Address any technical concerns raised
        - Implement recommended technical improvements
        - Resolve integration conflicts
        - Ensure compliance with review requirements
        
        **Output Format:**
        Provide detailed QA report and final recommendations:
        ```json
        {{
            "gene": "{gene}",
            "qa_score": 0.91,
            "technical_issues": [
                {{
                    "severity": "high/medium/low",
                    "category": "schema/integrity/consistency/integration",
                    "description": "issue_description",
                    "location": "specific_instance_or_field",
                    "resolution": "recommended_fix"
                }}
            ],
            "integration_assessment": {{
                "conflicts_detected": false,
                "performance_impact": "minimal/moderate/significant",
                "compatibility_score": 0.9
            }}
        }}
        ```
        `qa_score` is your single overall QA score (0.0-1.0) combining schema compliance,
        referential integrity, data consistency, and integration compatibility.
        
        **Decision Criteria:**
        - Approve: QA score >= {quality_threshold}, no high-severity issues, expert approved
        - Conditional: QA score >= 0.6, only low/medium issues, expert approved with conditions  
        - Reject: QA score < 0.6, high-severity issues, or expert rejection
        
        {closing_tools_line}
        """
        
        expected_output = f"""
        A technical QA report for gene {gene}: qa_score, technical_issues, and
        integration_assessment. Return ONLY the structured fields defined above — do not write
        a long prose report or narrative outside the schema.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=None,
            output_pydantic=QAReport,
        )

    def create_final_vote_task(self,
                              gene: str,
                              agent_role: str,
                              extraction_context: Dict[str, Any],
                              curation_context: Dict[str, Any],
                              review_context: Dict[str, Any],
                              qa_context: Dict[str, Any],
                              quality_threshold: float = 0.7,
                              accession: Optional[str] = None) -> Task:
        """
        Create an individual vote task for one specialist in the final virtual meeting.

        Each agent casts a structured vote based on all previous phase outputs.
        """
        accession_directive = (
            f"\n        Note: the verified UniProt accession for {gene} is {accession}; reference only this value if you mention an identifier.\n"
            if accession else ""
        )
        description = f"""
        {accession_directive}
        You are acting as {agent_role} in the final virtual meeting for gene {gene}.

        Review all outputs from phases 1-4 and cast your vote.

        Inputs for your vote:
        - Literature extraction context: {json.dumps(extraction_context, default=str)[:4000]}
        - Curation context: {json.dumps(curation_context, default=str)[:4000]}
        - Expert review context: {json.dumps(review_context, default=str)[:4000]}
        - QA context: {json.dumps(qa_context, default=str)[:4000]}

        Voting rules:
        - Approve only if there are no high-severity blockers from your specialty.
        - Reject if you find high-severity blockers.
        - Otherwise choose requires_revision.
        - Focus on 1 to 3 top blocking issues.

        Output strict JSON only:
        {{
            "agent_role": "{agent_role}",
            "decision": "approve|requires_revision|reject",
            "confidence": 0.0,
            "blocking_issues": ["..."],
            "required_revisions": ["..."],
            "summary": "short rationale"
        }}
        """

        expected_output = f"""
        A strict JSON vote from {agent_role} with a decision, confidence, blockers,
        required revisions, and concise rationale for gene {gene}.
        """

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None,
            output_pydantic=AgentVote,
        )

    def create_final_consensus_task(self,
                                   gene: str,
                                   individual_votes: Dict[str, Any],
                                   quality_threshold: float = 0.7,
                                   accession: Optional[str] = None) -> Task:
        """
        Create the final synthesis task that consolidates all specialist votes.
        """
        accession_directive = (
            f"\n        Note: the verified UniProt accession for {gene} is {accession}; reference only this value.\n"
            if accession else ""
        )
        description = f"""
        {accession_directive}
        Chair the final virtual meeting for gene {gene} by synthesizing all specialist votes.

        Individual votes:
        {json.dumps(individual_votes, default=str)}

        Apply deterministic decision rules:
        1. If any vote is reject with high-confidence blocker, final decision is reject.
        2. Else if any vote is requires_revision, final decision is requires_revision.
        3. Else decision is approve.
        4. Include consolidated list of unique required revisions.
        5. Report overall confidence as the average vote confidence.

        Output strict JSON only:
        {{
            "decision": "approve|requires_revision|reject",
            "confidence": 0.0,
            "vote_tally": {{"approve": 0, "requires_revision": 0, "reject": 0}},
            "required_revisions": ["..."],
            "blocking_issues": ["..."],
            "summary": "short final rationale",
            "quality_threshold": {quality_threshold}
        }}
        """

        expected_output = f"""
        A final strict JSON consensus decision for gene {gene}, including decision,
        confidence, vote tally, blockers, required revisions, and concise rationale.
        """

        return Task(
            description=description,
            expected_output=expected_output,
            agent=None,
            output_pydantic=ConsensusDecision,
        )