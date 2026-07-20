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
import re
from typing import Any, List
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import Field

from GenePathwayAnnotator import GenePathwayAnnotator
import ReactomeUtils as utils
import ReactomeNeo4jUtils as neo4j_utils
from CrewAIEventLogger import emit_tool_event
from QueryBuilder import build_retrieval_query_pair, get_reranking_target
from TextEmbedder import (create_sentence_transformer, sentence_embed, cosine_similarity,
                          _get_cross_encoder)

# Lazily-constructed sentence-transformer for Stage-2 re-ranking; built once on first use
# (loading the model is expensive) and reused across tool calls.
_rerank_model = None


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = create_sentence_transformer()
    return _rerank_model

logger = logging.getLogger(__name__)


def parse_agent_json(text: Any) -> Any:
    """Parse JSON an agent passed as a tool argument.

    Agents rarely send clean JSON: they wrap it in a ```json ... ``` fence, surround
    it with prose, or send a single-element list where a dict is expected. Bare
    json.loads() then raises 'Expecting value...' and the tool returns an error blob
    even though the data was recoverable. This strips a fenced block (or the largest
    {...}/[...] span) before parsing, so the tools stop choking on well-formed output
    that merely had markdown around it.

    Returns the parsed object (dict/list passed through untouched), or raises
    ValueError with a readable message the agent can act on.
    """
    if isinstance(text, (dict, list)):
        return text
    s = str(text or "").strip()
    if not s:
        raise ValueError("no JSON provided (empty input)")
    candidate = s
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        span = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if span:
            candidate = span.group(1).strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"could not parse JSON from input: {e}")


def as_dict(parsed: Any) -> dict:
    """Coerce parsed agent JSON to a dict. A single-element list wrapping a dict is
    unwrapped; anything else non-dict becomes {} so downstream .get() calls never
    raise 'list' object has no attribute 'get'."""
    if isinstance(parsed, list):
        parsed = parsed[0] if len(parsed) == 1 else {}
    return parsed if isinstance(parsed, dict) else {}


class LiteratureSearchTool(BaseTool):
    """Tool for searching and retrieving literature from PubMed"""
    
    name: str = "literature_search"
    description: str = "Search PubMed for papers related to a gene, extract abstracts and metadata"
    
    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")
    
    def _run(self, gene: str, max_papers: int = 5, additional_terms: str = "",
             fetch_papers: int = 200) -> str:
        """Search PubMed for gene-related literature using the validated retrieve-and-rerank
        pipeline.

        Stage 1: build_retrieval_query_pair returns two INDEPENDENT queries by annotation status --
        a gene-name+synonyms query and a broad pathway/partner query -- and we run each as its own
        PubMed E-Search and UNION the PMID sets (see _merge_search). This is the RAB6C fix: a single
        combined OR query lets broad pathway terms dominate relevance ranking and bury the
        gene-specific papers below the fetch cap; two searches each get their own ranking, so the
        gene-specific set is guaranteed into the pool. Stage 2: re-rank the merged pool against
        pathway-level text (get_reranking_target, scored by MAX cosine over the gene's per-pathway
        summaries) and keep the top max_papers -- this stops the re-ranker discarding the
        pathway/mechanism-level papers that dominate curator citations (raised final-recall
        retention from 36% to 54% on the validation set vs the old gene-specific description).
        """
        try:
            # Clamp to sane ceilings: the agent tends to over-request (e.g. max_papers=50),
            # ballooning the returned abstracts into a ~79k-char blob that (a) blows past the
            # intended paper cap and (b) can trip Anthropic's safety classifier -> stop_reason
            # "refusal" -> empty completion -> pipeline abort (observed on TANC1). Keeping the
            # top-N reranked papers honors the configured cap and keeps the payload small.
            max_papers = min(max_papers, 5)
            fetch_papers = min(fetch_papers, 200)

            # If the upfront pipeline already ran the FULL retrieval for this gene (Stage-1 merge +
            # cross-encoder re-rank + LLM curator-judge) and cached the final selection on the shared
            # annotator, return it directly. The judge makes a blocking LLM .invoke() that would
            # poison the next agent call if made here on CrewAI's event loop -- so it runs off-loop in
            # annotate_literature's precompute (asyncio.to_thread) and only its LLM-free RESULT is
            # consumed here (same precompute-and-cache contract as the rerank targets). No re-fetch.
            cached = getattr(self.gene_annotator, "judged_papers", {}).get(gene)
            if cached is not None:
                return json.dumps({
                    "gene": gene,
                    "name_query": cached.get("name_query", ""),
                    "context_query": cached.get("context_query", ""),
                    "pool_size": cached.get("pool_size", 0),
                    "papers_found": len(cached.get("papers", [])),
                    "papers": cached.get("papers", []),
                })

            # Fallback (standalone tool use / judge disabled, e.g. retrieval_eval): Stage-1 merge +
            # cross-encoder re-rank, top max_papers, NO LLM judge.
            cand = self.retrieve_candidates(gene, candidate_pool=max_papers,
                                            fetch_papers=fetch_papers,
                                            additional_terms=additional_terms)
            return json.dumps({
                "gene": gene,
                "name_query": cand["name_query"],
                "context_query": cand["context_query"],
                "pool_size": cand["pool_size"],
                "papers_found": len(cand["papers"]),
                "papers": cand["papers"],
            })

        except Exception as e:
            return json.dumps({"error": str(e), "gene": gene})

    @staticmethod
    def _format_papers(docs: List[dict]) -> List[dict]:
        """Project raw pool/cache docs (uid/Summary/...) into the tool's public paper shape."""
        return [{
            "pmid": d.get("uid", ""),
            "title": d.get("title", ""),
            "abstract": d.get("Summary", ""),
            "authors": d.get("authors", ""),
            "journal": d.get("journal", ""),
            "year": d.get("year", ""),
            "cross_score": d.get("cross_score"),
        } for d in docs]

    def retrieve_candidates(self, gene: str, candidate_pool: int = 20, fetch_papers: int = 200,
                            additional_terms: str = "") -> dict:
        """Stage-1 merge + cross-encoder re-rank -> top-`candidate_pool` formatted paper dicts (with
        abstracts). LLM-FREE: safe to call live on the async path OR from a worker thread. This is
        the candidate set the upfront LLM curator-judge selects the final papers from."""
        name_query, context_query = build_retrieval_query_pair(gene)
        # Fold agent-supplied extra terms into the broad search (mirrors the old combined-query
        # behavior); if there's no broad search (cold-start gate-fail), fold into the name search.
        if additional_terms:
            if context_query:
                context_query = f"({context_query}) OR {additional_terms}"
            else:
                name_query = f"({name_query}) OR {additional_terms}"

        # Result-set MERGE. The gene-name search is small for the genes this fix targets, so half
        # the budget captures its full specific set while the broad search keeps the full budget
        # (the primary recall driver for have-data genes). Union dedupes overlap.
        pool = self._merge_search(name_query, context_query,
                                  name_fetch=max(1, fetch_papers // 2),
                                  context_fetch=fetch_papers)
        ranked = self._rerank(gene, pool, candidate_pool)
        return {
            "name_query": name_query,
            "context_query": context_query,
            "pool_size": len(pool),
            "papers": self._format_papers(ranked),
        }

    def _merge_search(self, name_query: str, context_query: str,
                      name_fetch: int, context_fetch: int) -> List[dict]:
        """Run the name and context queries as SEPARATE E-Searches and union the results, deduped
        by PMID. Each search gets its own independent relevance ranking, so the gene-specific
        (name) papers are guaranteed into the pool rather than buried below the fetch cap by the
        broad pathway/partner results (the RAB6C dilution fix). Name-search results are placed
        first so gene-specific papers win the degenerate no-rerank-target fallback in _rerank.
        lazy_load yields cache dicts with 'uid' and 'Summary'; maxdate defaults to the frozen
        cache boundary."""
        seen, pool = set(), []
        for query, cap in ((name_query, name_fetch), (context_query, context_fetch)):
            if not query:
                continue
            retriever = self.gene_annotator._get_pubmed_retriver(top_k_results=cap)
            for d in retriever.lazy_load(query=query):
                if d is None:
                    continue
                uid = d.get("uid")
                if uid in seen:
                    continue
                seen.add(uid)
                pool.append(d)
        return pool

    def _rerank(self, gene: str, pool: List[dict], top_k: int) -> List[dict]:
        """Re-rank pool docs by MAX cross-encoder score against the gene's re-ranking targets,
        returning the top `top_k`. Falls back to pool order (first top_k) if there are no embeddable
        abstracts or no re-ranking targets.

        Replaces the former bi-encoder (sentence_embed + cosine) scoring: the cross-encoder scores
        each (target, abstract) pair jointly for higher-fidelity relevance. Cheap enough to run on
        the whole pool on CPU."""
        embeddable = [d for d in pool if d.get("Summary")]
        if not embeddable:
            return pool[:top_k]

        # Precomputed re-rank targets, stashed on the shared gene_annotator by
        # CrewAILiteratureAnnotator (keyed by gene) so this stays LLM-free inside the async flow:
        #  - description_override : gate-fail cold-start genes -> LLM gene description vs identity string
        #  - pathway_descriptions : has-data genes -> per-pathway gene-specific descriptions vs raw
        #    pathway summaries (get_reranking_target falls back to the raw summary per missing pathway)
        # Both absent (e.g. standalone tool use) -> get_reranking_target's built-in fallbacks.
        override = getattr(self.gene_annotator, "rerank_target_descriptions", {}).get(gene)
        pathway_descs = getattr(self.gene_annotator, "rerank_pathway_descriptions", {}).get(gene)
        targets = get_reranking_target(gene, description_override=override,
                                       pathway_descriptions=pathway_descs)
        if not targets:
            return pool[:top_k]

        # Score every (target, abstract) pair in ONE batched predict call, then take the MAX score
        # per abstract across targets -- preserves the old MAX-over-pathway-targets semantics (the
        # per-pathway target is what lifted SHANK3 usefulness) while staying a single model pass.
        cross_encoder = _get_cross_encoder()
        pairs = [(t, d["Summary"]) for d in embeddable for t in targets]
        scores = cross_encoder.predict(pairs)
        n_t = len(targets)
        for i, d in enumerate(embeddable):
            d["cross_score"] = max(float(s) for s in scores[i * n_t:(i + 1) * n_t])
        embeddable.sort(key=lambda d: d["cross_score"], reverse=True)
        return embeddable[:top_k]


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
    description: str = (
        "Query existing Reactome data for a gene. query_type must be 'pathways' (default, "
        "returns the pathways the gene participates in) or 'reactions' (requires a 'pathway' "
        "argument, returns that pathway's reactions). Any other query_type falls back to 'pathways'."
    )

    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")

    def _run(self, gene: str, query_type: str = "pathways", pathway: str = "") -> str:
        """Query Reactome database for gene information"""
        try:
            # 'reactions' is the only branch that needs an extra argument (pathway), so it can't
            # be a safe fallback. Anything that isn't an explicit 'reactions' request — including
            # invented types the agent makes up (entity, comprehensive, ...) — routes to
            # 'pathways', which only needs the gene and so always returns useful data.
            if query_type == "reactions":
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

            if query_type not in ("pathways", "pathway"):
                logger.info(
                    f"reactome_query: unrecognized query_type {query_type!r}; defaulting to 'pathways'."
                )
            pathways = neo4j_utils.query_pathways_for_gene(gene)
            return json.dumps({
                "gene": gene,
                "query_type": "pathways",
                "requested_query_type": query_type,
                "pathways": pathways
            })

        except Exception as e:
            return json.dumps({
                "gene": gene,
                "error": str(e)
            })


class ProteinInteractionTool(BaseTool):
    """Tool for retrieving protein-protein interactions"""
    
    name: str = "protein_interactions"
    description: str = (
        "Get protein-protein interactions. interaction_source must be "
        "'intact_biogrid' (default, combines IntAct + BioGRID) or 'reactome_fis'."
    )

    gene_annotator: GenePathwayAnnotator = Field(..., description="Gene annotator instance")

    def _run(self, gene: str, interaction_source: str = "intact_biogrid") -> str:
        """Get protein interactions for gene"""
        try:
            # Normalize loose values the agent may pass (e.g. 'IntAct', 'BioGRID')
            # to the exact value the loader accepts; otherwise it raises ValueError.
            if interaction_source.lower() in ("intact", "biogrid", "intact_biogrid"):
                interaction_source = "intact_biogrid"
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
            
            # default=list so the set-valued interaction PMIDs serialize to JSON arrays
            # (interactions is {partner: set(pmids)}; sets aren't JSON-serializable).
            return json.dumps({
                "gene": gene,
                "interaction_source": interaction_source,
                "interactions": interactions,
                "pathway_enrichment": pathway_enrichment
            }, default=list)
            
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
    
    def _run(self, instances: str = "", schema: str = "", schema_path: str = "") -> str:
        """Validate instances against Reactome schema"""
        try:
            # Guard against the agent calling this tool before it has assembled the
            # instance JSON. Returning a readable result (instead of letting pydantic
            # raise a hard "field required" error) lets the agent recover instead of
            # looping on the same malformed call.
            if not instances:
                return json.dumps({
                    "valid": False,
                    "error": "No instances provided. Re-call schema_validation with `instances` set to the Reactome instance JSON you generated."
                })
            # Parse instances (tolerates markdown-wrapped / prose-wrapped JSON; coerce
            # to a dict so the "entities"/"reactions" field checks below stay safe).
            data = as_dict(parse_agent_json(instances))
            schema_data = None

            if schema_path:
                schema_data = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            elif schema:
                schema_data = parse_agent_json(schema)
            
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
            # Parse instances (tolerates markdown-wrapped / prose-wrapped JSON, then
            # coerce to a dict so the "pathways" lookup below can't crash on a list).
            data = as_dict(parse_agent_json(instances))

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
            self.consistency_check
        ]
        return self._filter_tools(tools, enabled_names)
    
    def get_qa_tools(self, enabled_names: List[str] | None = None) -> List[BaseTool]:
        """Get tools for Quality Checker agent"""
        tools = [
            self.schema_validation,
            self.consistency_check,
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
            self.evidence_evaluation
        ]