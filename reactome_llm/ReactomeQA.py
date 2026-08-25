"""ReactomeQA — data-model quality assurance (agent 3 of 3).

Receives the approved (or loop-exhausted) Curator reactions and turns them into a
validated Reactome data model. It is a DETERMINISTIC VALIDATOR with an LLM EXPERT-REVIEW
layer on top:

    1. build_instances(reactions, placement)     -> ReactomeDataModel   [deterministic]
    2. schema + referential-integrity checks       -> structural findings [deterministic]
    3. existing-data consistency (Neo4j overlap)    -> conflict findings   [deterministic]
    4. LLM Reactome-expert review over 1-3          -> QAReport            [agentic]
    5. derive a verdict: pass / needs_revision / fail

Why the split: schema and referential integrity are pure checks — an LLM adds nothing.
The LLM earns its place on the SEMANTIC / convention judgment a schema can't encode
(a reaction whose input == output isn't a real transition; a one-member "Complex" should
be an entity; an implausible compartment) and on reasoning about conflicts with existing
curated content. Set use_llm=False to run checks-only (fast, free) — same verdict shape.

Terminal in v1: emits validated instances + a QAReport for a human curator. A QA->Curator
feedback loop can be added later without changing this class's `check()` contract.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from reaction_to_instances import build_instances
import ReactomeNeo4jUtils as neo4j_utils
from ModelConfig import create_reactome_chat_model
from ReactomeModels import QAReport
import token_profiler
import logging_config

logging_config.setup_logging()
logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = "resources/reactome_domain_model.json"
QA_PASS_SCORE = 0.7   # qa_score at/above this (with no hard errors) -> pass
# Max chars of the instance JSON to show the QA reviewer. Big enough for a rich multi-paper gene
# (a 55-entity model is ~40-50k chars); if we must truncate we say so, so the reviewer doesn't
# mistake OUR truncation for a data defect (it did: "JSON payload truncated mid-entity").
_QA_REVIEW_MAX_CHARS = 60000

QA_REVIEW_PROMPT = """You are a Reactome quality-assurance curator: an expert in the Reactome \
data model and curation conventions. Assess the generated instances for gene {gene}.

The deterministic checks below have ALREADY been run — do NOT re-derive them. Focus your \
judgment on what they cannot catch: biological/semantic validity and Reactome conventions.

Generated instances (JSON):
{instances}

Deterministic schema / referential-integrity findings:
{schema_findings}

Existing-data consistency findings (overlap with curated Reactome pathways):
{consistency_findings}

Assess, and flag technical_issues for any of:
- a reaction whose inputs and outputs are identical (no molecular transformation)
- a reactionType that doesn't match the input/output pattern (e.g. 'binding' with no association)
- a Complex with fewer than two components (should be an entity)
- a compartment implausible for the event, or a species mismatch
- an entity/reaction/pathway that likely duplicates existing curated Reactome content
- a summation that overstates or misstates what the evidence supports

Give an overall qa_score in [0,1] (1 = clean, ready for a curator to accept as-is), the \
technical_issues (each with severity high/medium/low, category, description, location, \
resolution), and an integration_assessment. Be strict but fair; reserve 'high' severity for \
issues that make an instance wrong or unusable."""


@dataclass
class QAResult:
    gene: str
    instances: Dict[str, Any]           # ReactomeDataModel dump (by_alias=True)
    schema_check: Dict[str, Any]
    consistency_check: Dict[str, Any]
    report: Dict[str, Any]              # QAReport dump
    verdict: str                        # "pass" | "needs_revision" | "fail"
    passed: bool
    provenance: str = "approved"        # "approved" | "loop_exhausted" — how the Curator result got here
    repair_history: List[Dict[str, Any]] = field(default_factory=list)  # per repair iteration

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class ReactomeQA:
    def __init__(self, use_llm: bool = True, model: Any = None,
                 schema_path: str = DEFAULT_SCHEMA_PATH) -> None:
        self.use_llm = use_llm
        self.model = model
        self.schema_path = schema_path

    def check(self, gene: str, reactions: List[dict], accession: Optional[str] = None,
              placement: Optional[dict] = None, placement_status: Optional[dict] = None,
              target_pathways: Optional[List[str]] = None,
              provenance: str = "approved", max_fixes: int = 2) -> QAResult:
        """Build the data model, validate, and — if QA fails — RE-CONVERT with QA's corrections
        injected as fix_notes, up to `max_fixes` times. Stops early on pass or no improvement.

        The repair loop targets the representation/model issues QA finds (wrong UniProt IDs,
        EWAS-vs-Complex, missing literatureReference/compartments, off-target reactions) — the
        same reactions, converted better each pass. It cannot invent missing evidence.
        """
        gene = (gene or "").strip().upper()
        fix_notes = None
        repair_history: List[Dict[str, Any]] = []
        best = None  # (rank_key, instances, schema_check, consistency_check, report, verdict, passed)

        for i in range(max_fixes + 1):
            label = "qa_build_instances" if i == 0 else "qa_repair_build"
            with token_profiler.label(label):
                dm = build_instances(gene, reactions, accession=accession, placement=placement,
                                     placement_status=placement_status, target_pathways=target_pathways,
                                     fix_notes=fix_notes)
            instances = dm.model_dump(by_alias=True)
            schema_check = self._schema_check(instances)
            consistency_check = self._consistency_check(gene, instances)
            report = (self._llm_review(gene, instances, schema_check, consistency_check)
                      if self.use_llm else self._rule_report(schema_check, consistency_check))
            verdict, passed = self._verdict(report, schema_check)
            n_issues = len(report.get("technical_issues", [])) + len(schema_check.get("errors", []))
            repair_history.append({"iteration": i, "verdict": verdict,
                                   "qa_score": report.get("qa_score"), "n_issues": n_issues})
            logger.info(f"QA for {gene} iter {i}: verdict={verdict} qa_score={report.get('qa_score')} "
                        f"issues={n_issues} (schema_valid={schema_check['valid']})")

            # Keep the BEST iteration (passed > not; then higher qa_score; then fewer issues) — a
            # regressing repair must not overwrite an earlier, better conversion.
            rank = (1 if passed else 0, report.get("qa_score") or 0.0, -n_issues)
            if best is None or rank > best[0]:
                best = (rank, instances, schema_check, consistency_check, report, verdict, passed)

            if passed or i == max_fixes:
                break
            # Stop if the last repair didn't reduce the issue count (no point burning the cap).
            if i > 0 and n_issues >= repair_history[-2]["n_issues"]:
                logger.info(f"QA for {gene}: repair not improving ({n_issues} issues); stopping.")
                break
            new_notes = self._render_fix_notes(schema_check, report)
            if not new_notes:
                break  # nothing actionable to hand back to the converter
            fix_notes = new_notes

        _, instances, schema_check, consistency_check, report, verdict, passed = best
        return QAResult(gene=gene, instances=instances, schema_check=schema_check,
                        consistency_check=consistency_check, report=report,
                        verdict=verdict, passed=passed, provenance=provenance,
                        repair_history=repair_history)

    @staticmethod
    def _render_fix_notes(schema_check: Dict[str, Any], report: Dict[str, Any]) -> str:
        """Turn the actionable QA findings (schema errors + high/medium issues) into corrective
        bullet points for build_instances. Low-severity nits are skipped to avoid churn."""
        lines = [f"- {e}" for e in schema_check.get("errors", [])]
        for i in report.get("technical_issues", []):
            if i.get("severity") in ("high", "medium") and i.get("description"):
                res = f" -> {i['resolution']}" if i.get("resolution") else ""
                lines.append(f"- [{i.get('severity')}] {i['description']}{res}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ deterministic checks
    def _schema_check(self, instances: Dict[str, Any]) -> Dict[str, Any]:
        """Structural + referential-integrity validation (no LLM). Catches missing required
        fields, and — the valuable part — reaction/complex references that resolve to no defined
        entity or complex (dangling references). Colon-notation 'A:B' refs are complex shorthand,
        not entity names, so they're reported as warnings, not hard errors."""
        errors: List[str] = []
        warnings: List[str] = []

        for kind in ("entities", "reactions", "complexes", "pathways"):
            for item in instances.get(kind, []):
                if "class" not in item:
                    errors.append(f"Missing 'class' in {kind[:-1]}: {item.get('displayName', '?')}")
                if not item.get("displayName"):
                    errors.append(f"Missing 'displayName' in a {kind[:-1]}")

        defined = {e.get("displayName") for e in instances.get("entities", [])}
        defined |= {c.get("displayName") for c in instances.get("complexes", [])}
        referenced: List[str] = []
        for c in instances.get("complexes", []):
            referenced += c.get("components", [])
        for r in instances.get("reactions", []):
            referenced += r.get("input", []) + r.get("output", []) + r.get("catalystActivity", [])
        for ref in referenced:
            if not ref or ref in defined:
                continue
            if ":" in ref:
                warnings.append(f"Complex-notation reference not defined as its own instance: {ref}")
            else:
                errors.append(f"Dangling reference (no matching entity/complex): {ref}")

        # Optional external JSON-schema validation when the schema file is present.
        if self.schema_path and os.path.isfile(self.schema_path):
            try:
                import jsonschema
                schema = json.loads(open(self.schema_path, encoding="utf-8").read())
                jsonschema.validate(instance=instances, schema=schema)
            except ImportError:
                warnings.append("jsonschema not installed; external schema check skipped")
            except Exception as e:
                errors.append(f"External schema validation failed: {e}")

        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def _consistency_check(self, gene: str, instances: Dict[str, Any]) -> Dict[str, Any]:
        """Flag generated pathways that overlap existing curated Reactome pathways for the gene
        (integrate vs. duplicate). LLM-free; degrades gracefully if Neo4j is unreachable."""
        report = {"gene": gene, "conflicts": [], "recommendations": []}
        try:
            existing = {p["pathway"] for p in (neo4j_utils.query_pathways_for_gene(gene) or [])}
        except Exception as e:
            report["recommendations"].append(f"Could not query existing pathways ({e}); skipped overlap check.")
            return report
        for pw in instances.get("pathways", []):
            name = pw.get("displayName", "")
            if name in existing:
                report["conflicts"].append(
                    {"type": "pathway_overlap", "description": f"Pathway '{name}' already exists in Reactome"})
        if report["conflicts"]:
            report["recommendations"].append("Integrate into the existing pathway rather than duplicating.")
        return report

    # ------------------------------------------------------------------ agentic review
    def _llm_review(self, gene, instances, schema_check, consistency_check) -> Dict[str, Any]:
        """LLM Reactome-expert review -> QAReport (structured). Falls back to the rule report on error."""
        try:
            model = self.model or create_reactome_chat_model()
            inst_json = json.dumps(instances, indent=2)
            if len(inst_json) > _QA_REVIEW_MAX_CHARS:
                inst_json = (inst_json[:_QA_REVIEW_MAX_CHARS]
                             + "\n... [instances truncated here for review length only — "
                               "this is NOT a data defect; do not flag truncation]")
            prompt = QA_REVIEW_PROMPT.format(
                gene=gene,
                instances=inst_json,
                schema_findings=json.dumps(schema_check),
                consistency_findings=json.dumps(consistency_check))
            with token_profiler.label("qa_llm_review"):
                report: QAReport = model.with_structured_output(QAReport).invoke(prompt)
            data = report.model_dump()
            data.setdefault("gene", gene)
            return data
        except Exception as e:
            logger.warning(f"QA LLM review failed for {gene}: {e}; falling back to rule report.")
            return self._rule_report(schema_check, consistency_check)

    @staticmethod
    def _rule_report(schema_check, consistency_check) -> Dict[str, Any]:
        """Deterministic QAReport-shaped fallback (no LLM): score off the structural findings."""
        issues = []
        for err in schema_check.get("errors", []):
            issues.append({"severity": "high", "category": "schema", "description": err,
                           "location": "", "resolution": "Fix before acceptance."})
        for warn in schema_check.get("warnings", []):
            issues.append({"severity": "low", "category": "schema", "description": warn,
                           "location": "", "resolution": ""})
        conflicts = consistency_check.get("conflicts", [])
        for c in conflicts:
            issues.append({"severity": "medium", "category": "consistency",
                           "description": c.get("description", ""), "location": "",
                           "resolution": "Integrate rather than duplicate."})
        qa_score = 1.0 if not issues else (0.4 if not schema_check.get("valid") else 0.7)
        return {"qa_score": qa_score, "technical_issues": issues,
                "integration_assessment": {"conflicts_detected": bool(conflicts),
                                           "performance_impact": "minimal",
                                           "compatibility_score": 0.7 if conflicts else 1.0}}

    @staticmethod
    def _verdict(report: Dict[str, Any], schema_check: Dict[str, Any]) -> tuple[str, bool]:
        qa_score = report.get("qa_score", 0.0) or 0.0
        issues = report.get("technical_issues", []) or []
        has_high = any((i.get("severity") == "high") for i in issues)
        if not schema_check.get("valid") or has_high or qa_score < 0.5:
            return "fail", False
        if qa_score >= QA_PASS_SCORE and not any(i.get("severity") == "medium" for i in issues):
            return "pass", True
        return "needs_revision", False
