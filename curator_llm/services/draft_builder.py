"""Extracted reactions -> typed ReactomeDraft (one structured LLM call, then code).

The LLM only does what needs judgement: classify participants (protein / small molecule / complex /
set), split modified forms into base entity + residue modification, and wire reactions to those
participants. Everything checkable is taken from the input or done in code afterwards: PMIDs,
summation text and evidence ids come from the source reaction (matched by `source_index`, never by
name), identifiers come from the resolvers.
"""
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from curator_llm.models.reactome import (CatalystSpec, ComplexSpec, DefinedSetSpec, EwasSpec, GoActivity,
                                         ModifiedResidueSpec, PathwaySpec, ReactionSpec, ReactomeDraft,
                                         RegulationSpec, SimpleEntitySpec)
from curator_llm.services.resolvers import PSI_MOD
from curator_llm.services.sources import pmid_of as _pmid_of

logger = logging.getLogger(__name__)


# ── what the LLM returns ───────────────────────────────────────────────────
class LlmModification(BaseModel):
    psi_mod: str = Field(..., description="PSI-MOD id from the allowed list, e.g. MOD:00046")
    residue: Optional[str] = Field(None, description="One-letter amino acid, e.g. S")
    coordinate: Optional[int] = Field(None, description="Residue position in the protein, e.g. 65")


class LlmEntity(BaseModel):
    key: str = Field(..., description="Short unique handle, e.g. pink1, ub, p_ub_s65")
    kind: Literal['protein', 'small_molecule', 'complex', 'set']
    name: str = Field(..., description="Plain entity name (gene symbol for proteins), no tags, no modification text")
    uniprot: Optional[str] = Field(None, description="UniProt accession ONLY if it was given to you; otherwise null")
    chebi: Optional[str] = Field(None, description="ChEBI id ONLY if certain; otherwise null")
    compartment: Optional[str] = Field(None, description="Compartment name ONLY if the reaction states it; otherwise null")
    modifications: List[LlmModification] = Field(default_factory=list)
    components: List[str] = Field(default_factory=list, description="Entity keys, for complexes")
    members: List[str] = Field(default_factory=list, description="Entity keys, for sets")


class LlmCatalyst(BaseModel):
    entity: str = Field(..., description="Entity key of the enzyme that directly performs the reaction")
    go_function: Optional[str] = Field(None, description="GO molecular function label, e.g. protein kinase activity")


class LlmRegulation(BaseModel):
    kind: Literal['positive', 'negative', 'requirement']
    regulator: str = Field(..., description="Entity key")
    note: Optional[str] = Field(None, description="Copy the input's condition note, if any")


class LlmReaction(BaseModel):
    source_index: int = Field(..., description="Index of the input reaction this converts (0-based)")
    name: str
    reaction_type: Literal['transition', 'binding', 'dissociation', 'omitted', 'blackBoxEvent'] = 'transition'
    inputs: List[str] = Field(default_factory=list, description="Entity keys")
    outputs: List[str] = Field(default_factory=list, description="Entity keys")
    catalyst: Optional[LlmCatalyst] = None
    regulations: List[LlmRegulation] = Field(default_factory=list)
    preceding_source_indexes: List[int] = Field(default_factory=list,
                                                description="Input indexes of reactions this one directly follows, only if the input says so")


class DraftExtraction(BaseModel):
    entities: List[LlmEntity] = Field(default_factory=list)
    reactions: List[LlmReaction] = Field(default_factory=list)


def _items(reactions: List[dict]) -> List[Dict[str, Any]]:
    """What the model sees: no quotes (they are looked up by evidence id later)."""
    out = []
    for i, r in enumerate(reactions):
        a = r.get('annotation_result', r)
        keep = {k: a.get(k) for k in ('name', 'reactionType', 'input', 'output', 'catalystActivity',
                                      'regulatedBy', 'compartment', 'condition', 'relationships')
                if a.get(k) not in (None, '', [], {})}
        summ = a.get('summation')
        keep['summation'] = (summ[0].get('text') if isinstance(summ, list) and summ and isinstance(summ[0], dict)
                             else summ.get('text') if isinstance(summ, dict) else summ) or ''
        out.append({'index': i, 'pmid': _pmid_of(r.get('source', '')), 'reaction': keep})
    return out


def build_prompt(gene: str, reactions: List[dict], accession: Optional[str]) -> str:
    mods = '\n'.join(f'  {k}: {v[0]}' for k, v in PSI_MOD.items())
    acc = (f"The UniProt accession of {gene} is {accession}; set it on the {gene} protein entity. "
           f"Leave uniprot null for every other protein (code looks those up).") if accession else \
          "Leave uniprot null for every protein (code looks them up)."
    return f"""You are a Reactome biocurator. Below are biochemical reactions already extracted from the
literature for {gene}. Convert them into typed participants and reactions. Do NOT add, drop, merge or
re-derive reactions: return exactly one output reaction per input reaction, with its source_index.

{acc}

PARTICIPANTS (entities)
- kind: protein | small_molecule | complex | set. One entity per DISTINCT molecular species, reused across reactions.
- A MODIFIED protein is the base protein plus modifications, not a new name: "phospho-Ub (Ser65)" is
  entity name "UB" with modification MOD:00046, residue S, coordinate 65. The unmodified form is a separate
  entity (same name, no modifications). Use only these PSI-MOD ids, and only when the input states the
  residue and kind of modification:
{mods}
  If the modification is not in this list, keep the unmodified name and mention nothing invented.
- Names are plain: strip tags and fusion partners (HA-, His-, MBP-, GFP-); proteins use the gene symbol.
- Complexes list their component entity keys. Sets list member keys. Do not create entities for phenotypes
  or measurements.
- compartment: only when the reaction states where it happens or where the species is; else null.

REACTIONS
- inputs/outputs are entity keys. ATP/ADP and other cofactors are small_molecule entities when they are
  consumed or produced. A phosphorylation consumes the unmodified substrate and produces the modified one.
- catalyst: only the enzyme that directly performs the reaction (copy the input's catalyst; null if none).
  go_function is the GO molecular function label if you are sure (e.g. protein kinase activity), else null.
- regulations: copy the input's regulators (positive | negative | requirement) with their notes.
- preceding_source_indexes: only if the input explicitly says one reaction follows another.

INPUT REACTIONS (JSON):
```json
{json.dumps(_items(reactions), indent=2, default=str)}
```"""


# ── assembly ───────────────────────────────────────────────────────────────
_KIND = {'protein': EwasSpec, 'small_molecule': SimpleEntitySpec, 'complex': ComplexSpec, 'set': DefinedSetSpec}


def assemble(gene: str, reactions: List[dict], ex: DraftExtraction,
             pathway_name: Optional[str] = None) -> Tuple[ReactomeDraft, List[str]]:
    notes: List[str] = []
    d = ReactomeDraft()
    for e in ex.entities:
        if e.key in d.participants:
            notes.append(f'duplicate entity key {e.key}: the later one was ignored')
            continue
        common = {'key': e.key, 'name': e.name, 'compartment_name': e.compartment}
        if e.kind == 'protein':
            mods = []
            for m in e.modifications:
                if m.psi_mod not in PSI_MOD:
                    notes.append(f'{e.name}: modification {m.psi_mod} is not an allowed PSI-MOD id; dropped')
                    continue
                mods.append(ModifiedResidueSpec(psi_mod=m.psi_mod, residue=m.residue, coordinate=m.coordinate))
            d.participants[e.key] = EwasSpec(**common, uniprot=e.uniprot, modifications=mods)
        elif e.kind == 'small_molecule':
            d.participants[e.key] = SimpleEntitySpec(**common, chebi=e.chebi)
        elif e.kind == 'complex':
            d.participants[e.key] = ComplexSpec(**common, components=list(e.components))
        else:
            d.participants[e.key] = DefinedSetSpec(**common, members=list(e.members))
    # components/members must point at real entities
    for p in d.participants.values():
        refs = p.components if isinstance(p, ComplexSpec) else p.members if isinstance(p, DefinedSetSpec) else []
        bad = [k for k in refs if k not in d.participants]
        if bad:
            notes.append(f'{p.name}: unknown component keys {bad} removed')
            refs[:] = [k for k in refs if k in d.participants]

    sigs: Dict[tuple, List[str]] = {}
    for k, p in d.participants.items():
        mods = tuple((m.psi_mod, m.coordinate) for m in getattr(p, 'modifications', []))
        sigs.setdefault((p.kind, p.name.lower(), (p.compartment_name or '').lower(), mods), []).append(k)
    for (kind, name, _c, _m), keys in sigs.items():
        if len(keys) > 1:
            notes.append(f'{len(keys)} entities are defined identically as "{name}" ({", ".join(keys)}): they are '
                         f'probably mutants or condition variants the model could not express; curator to review')

    covered, key_of_index = set(), {}
    for lr in ex.reactions:
        if not (0 <= lr.source_index < len(reactions)) or lr.source_index in covered:
            notes.append(f'reaction "{lr.name}": bad or repeated source_index {lr.source_index}; dropped')
            continue
        src = reactions[lr.source_index]
        a = src.get('annotation_result', src)
        used = set(lr.inputs + lr.outputs + [g.regulator for g in lr.regulations]
                   + ([lr.catalyst.entity] if lr.catalyst else []))
        missing = used - set(d.participants)
        if missing:
            notes.append(f'reaction "{lr.name}": unknown entity keys {sorted(missing)}; reaction dropped')
            continue
        summ = a.get('summation')
        text = (summ[0].get('text') if isinstance(summ, list) and summ and isinstance(summ[0], dict)
                else summ.get('text') if isinstance(summ, dict) else summ) or ''
        pmid = _pmid_of(src.get('source', ''))
        key = f'r{lr.source_index}'
        covered.add(lr.source_index)
        key_of_index[lr.source_index] = key
        d.reactions.append(ReactionSpec(
            key=key, name=lr.name or a.get('name', ''), reaction_type=lr.reaction_type,
            inputs=lr.inputs, outputs=lr.outputs,
            catalyst=CatalystSpec(entity=lr.catalyst.entity,
                                  activity=GoActivity(name=lr.catalyst.go_function) if lr.catalyst.go_function else None)
            if lr.catalyst else None,
            regulations=[RegulationSpec(kind=g.kind, regulator=g.regulator, note=g.note) for g in lr.regulations],
            summation=text, pmids=[pmid] if pmid else [],
            evidence_ids=list(a.get('evidence_ids') or [])))
    for lr in ex.reactions:           # second pass: preceding links, now that every key exists
        if lr.source_index in key_of_index:
            spec = next(r for r in d.reactions if r.key == key_of_index[lr.source_index])
            spec.preceding = [key_of_index[i] for i in lr.preceding_source_indexes if i in key_of_index]
    for i, r in enumerate(reactions):
        if i not in covered:
            nm = r.get('annotation_result', r).get('name', '')
            notes.append(f'input reaction {i} "{nm}" is not in the draft (missing or invalid in the model output)')
    if d.reactions:
        d.pathway = PathwaySpec(name=pathway_name or f'{gene} candidate pathway (proposed - no confident placement)',
                                reactions=[r.key for r in d.reactions],
                                pmids=sorted({p for r in d.reactions for p in r.pmids}))
    return d, notes


def build_draft(gene: str, reactions: List[dict], accession: Optional[str] = None, model: Any = None,
                pathway_name: Optional[str] = None) -> Tuple[ReactomeDraft, List[str]]:
    """One structured LLM call, then assembly. `model` is any LangChain chat model (injected in tests)."""
    if not reactions:
        return ReactomeDraft(), ['no reactions']
    if model is None:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'reactome_llm'))
        from ModelConfig import create_reactome_chat_model
        model = create_reactome_chat_model()
    ex = model.with_structured_output(DraftExtraction).invoke(build_prompt(gene, reactions, accession))
    if not isinstance(ex, DraftExtraction):
        raise ValueError(f'unexpected model output: {type(ex)}')
    return assemble(gene, reactions, ex, pathway_name)
