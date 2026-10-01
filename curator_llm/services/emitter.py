"""Typed ReactomeDraft -> curator-tool-frontend `UserInstances` JSON. Pure code, no LLM.

Shape follows the frontend (Instance in reactome-instance.model.ts, UserInstances, and how
DataService.hydrateUserInstances reads them):
  * every NEW instance is its own entry in newInstances with a negative dbId;
  * references between instances are shells {dbId, displayName, schemaClassName} (the frontend
    swaps them for its cached full instances), so nothing is nested;
  * anything that already exists in gk_central is only ever a shell, never re-emitted.

Evidence is NOT written into the instances. The frontend rebuilds instances on persist and commit
and drops fields it does not know, so evidence is returned separately as `evidence_links`
(evidence id -> instance dbId + field) for the session store to serve by dbId.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from curator_llm.models.evidence import Evidence
from curator_llm.models.reactome import (ComplexSpec, DefinedSetSpec, EwasSpec, ExistingRef,
                                         ModifiedResidueSpec, ReactomeDraft, ReactionSpec,
                                         SimpleEntitySpec)
from curator_llm.services import schema_index

HUMAN = ExistingRef(db_id=48887, display_name='Homo sapiens', schema_class='Species')
_REG_CLASS = {'positive': 'PositiveRegulation', 'negative': 'NegativeRegulation',
              'requirement': 'Requirement'}
_REG_LABEL = {'positive': 'Positive regulation', 'negative': 'Negative regulation',
              'requirement': 'Requirement'}


def entity_display_name(p, name: Optional[str] = None, prefix: str = '') -> str:
    """Reactome-style entity display name: "<mods>NAME [compartment]". Also used by resolvers to
    look for an identical existing entity."""
    comp = p.compartment.display_name if p.compartment else p.compartment_name
    return f'{prefix}{name if name is not None else p.name}' + (f' [{comp}]' if comp else '')


@dataclass
class EmitResult:
    user_instances: dict
    evidence_links: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    key_to_db_id: Dict[str, int] = field(default_factory=dict)          # reaction key -> dbId
    participant_to_db_id: Dict[str, int] = field(default_factory=dict)  # participant key -> dbId


class Emitter:
    def __init__(self, draft: ReactomeDraft, evidence: Optional[Dict[str, Evidence]] = None,
                 first_db_id: int = -1):
        self.d = draft
        self.evidence = evidence or {}
        self.next_id = first_db_id
        self.new: List[dict] = []
        self.warnings: List[str] = []
        self._shells: Dict[str, dict] = {}       # participant/reaction key -> shell
        self._pubs: Dict[str, dict] = {}
        self._refs: Dict[str, dict] = {}         # memo for shared new instances (reference entities)
        self.links: List[dict] = []

    # ── building blocks ────────────────────────────────────────────────────
    @staticmethod
    def _shell_of(ref: ExistingRef) -> dict:
        return {'dbId': ref.db_id, 'displayName': ref.display_name, 'schemaClassName': ref.schema_class}

    def _add(self, cls: str, display: str, attrs: Dict[str, object]) -> dict:
        attrs = {k: v for k, v in attrs.items() if v not in (None, '', [])}
        inst = {'dbId': self.next_id, 'displayName': display, 'schemaClassName': cls, 'attributes': attrs}
        self.next_id -= 1
        problems = schema_index.check_instance(inst)
        if problems:
            raise ValueError('; '.join(problems))
        self.new.append(inst)
        return {'dbId': inst['dbId'], 'displayName': display, 'schemaClassName': cls}

    def _publication(self, pmid: str) -> dict:
        if pmid not in self._pubs:
            ref = self.d.publications.get(pmid)
            self._pubs[pmid] = self._shell_of(ref) if ref else self._add(
                'LiteratureReference', f'PMID:{pmid}', {'pubMedIdentifier': int(pmid)})
        return self._pubs[pmid]

    def _summation(self, text: str, pmids: List[str]) -> List[dict]:
        if not text:
            return []
        return [self._add('Summation', text[:60] + ('...' if len(text) > 60 else ''),
                          {'text': text, 'literatureReference': [self._publication(p) for p in pmids]})]

    # ── participants ───────────────────────────────────────────────────────
    def _participant(self, key: str) -> dict:
        if key in self._shells:
            return self._shells[key]
        p = self.d.participants[key]
        if p.existing is not None:
            shell = self._shell_of(p.existing)
        elif isinstance(p, EwasSpec):
            shell = self._ewas(p)
        elif isinstance(p, SimpleEntitySpec):
            shell = self._simple(p)
        elif isinstance(p, ComplexSpec):
            shell = self._complex(p)
        elif isinstance(p, DefinedSetSpec):
            members = [self._participant(m) for m in p.members]
            shell = self._add('DefinedSet', self._display(p.name, p), {
                'name': [p.name], 'hasMember': members, 'compartment': self._compartment(p),
                'species': [self._shell_of(HUMAN)]})
        else:
            raise ValueError(f'unsupported participant {key}')
        self._shells[key] = shell
        return shell

    def _compartment(self, p) -> List[dict]:
        if p.compartment is not None:
            return [self._shell_of(p.compartment)]
        if p.compartment_name:
            self.warnings.append(f'{p.name}: compartment "{p.compartment_name}" is not resolved to a '
                                 f'Compartment instance; set it manually')
        return []

    @staticmethod
    def _display(name: str, p, prefix: str = '') -> str:
        return entity_display_name(p, name, prefix)

    def _ewas(self, p: EwasSpec) -> dict:
        ref = None
        if p.reference_entity is not None:
            ref = self._shell_of(p.reference_entity)
        elif p.uniprot:
            memo = f'uniprot:{p.uniprot}'
            if memo not in self._refs:
                self._refs[memo] = self._add('ReferenceGeneProduct', f'UniProt:{p.uniprot} {p.name}', {
                    'identifier': p.uniprot, 'geneName': [p.name], 'species': self._shell_of(HUMAN)})
                self.warnings.append(f'{p.name}: new ReferenceGeneProduct for UniProt:{p.uniprot} - '
                                     f'its referenceDatabase must be set to UniProt by the curator')
            ref = self._refs[memo]
        else:
            self.warnings.append(f'{p.name}: no UniProt accession; referenceEntity left empty')
        mods = [self._modified_residue(m, ref) for m in p.modifications]
        prefix = ''.join(f'{m.short}-{m.residue or ""}{m.coordinate or ""}-' for m in p.modifications)
        return self._add('EntityWithAccessionedSequence', self._display(p.name, p, prefix), {
            'name': [p.name], 'referenceEntity': ref, 'species': self._shell_of(HUMAN) if p.species == 'Homo sapiens' else None,
            'compartment': self._compartment(p), 'hasModifiedResidue': mods,
            'startCoordinate': p.start_coordinate, 'endCoordinate': p.end_coordinate})

    def _modified_residue(self, m: ModifiedResidueSpec, ref: Optional[dict]) -> dict:
        label = m.mod_label or m.psi_mod or 'modification'
        if not m.psi_mod_ref:
            self.warnings.append(f'modified residue {label} at {m.coordinate}: PSI-MOD term {m.psi_mod} '
                                 f'is not resolved to an instance; set psiMod manually')
        return self._add('ModifiedResidue', f'{label} at {m.coordinate}' if m.coordinate else label, {
            'referenceSequence': ref, 'coordinate': m.coordinate, 'label': label,
            'psiMod': self._shell_of(m.psi_mod_ref) if m.psi_mod_ref else None})

    def _simple(self, p: SimpleEntitySpec) -> dict:
        ref = None
        if p.reference_entity is not None:
            ref = self._shell_of(p.reference_entity)
        elif p.chebi:
            memo = f'chebi:{p.chebi}'
            if memo not in self._refs:
                self._refs[memo] = self._add('ReferenceMolecule', f'ChEBI:{p.chebi.split(":")[-1]} {p.name}', {
                    'identifier': p.chebi.split(':')[-1], 'name': [p.name]})
                self.warnings.append(f'{p.name}: new ReferenceMolecule for {p.chebi} - '
                                     f'its referenceDatabase must be set to ChEBI by the curator')
            ref = self._refs[memo]
        else:
            self.warnings.append(f'{p.name}: no ChEBI id; referenceEntity left empty')
        return self._add('SimpleEntity', self._display(p.name, p), {
            'name': [p.name], 'referenceEntity': ref, 'compartment': self._compartment(p)})

    def _complex(self, p: ComplexSpec) -> dict:
        comps = [self._participant(k) for k in p.components]
        name = p.name or ':'.join(c['displayName'] for c in comps)
        return self._add('Complex', self._display(name, p), {
            'name': [name], 'hasComponent': comps, 'compartment': self._compartment(p),
            'species': [self._shell_of(HUMAN)]})

    # ── reactions ──────────────────────────────────────────────────────────
    def _reaction(self, r: ReactionSpec, key_to_shell: Dict[str, dict]) -> dict:
        if r.existing is not None:
            self.warnings.append(f'{r.name}: already in Reactome as {r.existing.display_name} '
                                 f'(dbId {r.existing.db_id}); not emitted as a new reaction')
            return self._shell_of(r.existing)
        # allocate the reaction's id first so its regulations can point back at it
        reaction_id = self.next_id
        self.next_id -= 1
        shell = {'dbId': reaction_id, 'displayName': r.name,
                 'schemaClassName': 'BlackBoxEvent' if r.reaction_type == 'blackBoxEvent' else 'Reaction'}
        claim_ids: Dict[str, int] = {}
        cats = []
        if r.catalyst is not None:
            act = None
            ga = r.catalyst.activity
            if ga is not None and ga.ref is not None:
                act = self._shell_of(ga.ref)
            elif ga is not None and ga.identifier:
                act = self._add('GO_MolecularFunction', ga.name or ga.identifier,
                                {'identifier': ga.identifier.split(':')[-1], 'name': ga.name})
                self.warnings.append(f'{ga.identifier}: new GO_MolecularFunction; link to the existing '
                                     f'GO term before committing')
            ent = self._participant(r.catalyst.entity)
            cat = self._add('CatalystActivity',
                            f'{(ga.name if ga and ga.name else "catalytic activity")} of {ent["displayName"]}',
                            {'physicalEntity': ent, 'activity': act,
                             'literatureReference': [self._publication(p) for p in r.pmids]})
            cats.append(cat); claim_ids['catalystActivity'] = cat['dbId']
        regs = []
        for i, g in enumerate(r.regulations):
            ent = self._participant(g.regulator)
            summ = self._summation(f'Curator note: {g.note}', []) if g.note else []
            reg = self._add(_REG_CLASS[g.kind], f'{_REG_LABEL[g.kind]} by {ent["displayName"]}', {
                'regulator': ent, 'regulatedEntity': [shell], 'summation': summ,
                'literatureReference': [self._publication(p) for p in r.pmids]})
            regs.append(reg); claim_ids[f'regulatedBy[{i}]'] = reg['dbId']
        preceding = [key_to_shell[k] for k in r.preceding if k in key_to_shell]
        attrs = {
            'name': [r.name],
            'input': [self._participant(k) for k in r.inputs],
            'output': [self._participant(k) for k in r.outputs],
            'catalystActivity': cats, 'regulatedBy': regs,
            'compartment': [self._shell_of(r.compartment)] if r.compartment else [],
            'summation': self._summation(r.summation, r.pmids),
            'literatureReference': [self._publication(p) for p in r.pmids],
            'precedingEvent': preceding,
            'inferredFrom': [self._shell_of(x) for x in r.inferred_from],
            'species': [self._shell_of(HUMAN)] if r.species == 'Homo sapiens' else [],
        }
        attrs = {k: v for k, v in attrs.items() if v not in (None, '', [])}
        inst = {'dbId': reaction_id, 'displayName': r.name, 'schemaClassName': shell['schemaClassName'],
                'attributes': attrs}
        problems = schema_index.check_instance(inst)
        if problems:
            raise ValueError('; '.join(problems))
        self.new.append(inst)
        self._link_evidence(r, reaction_id, claim_ids)
        return shell

    def _link_evidence(self, r: ReactionSpec, reaction_id: int, claim_ids: Dict[str, int]):
        for ev_id in r.evidence_ids:
            ev = self.evidence.get(ev_id)
            fields = (ev.supports if ev and ev.supports else ['reaction'])
            for f in fields:
                target = claim_ids.get(f, reaction_id)
                self.links.append({'evidenceId': ev_id, 'instanceDbId': target,
                                   'field': f if f in claim_ids else (f if f != 'reaction' else 'reaction')})

    def emit(self) -> EmitResult:
        key_to_shell: Dict[str, dict] = {}
        for r in self.d.reactions:
            key_to_shell[r.key] = self._reaction(r, key_to_shell)
        pw = self.d.pathway
        if pw is not None:
            if pw.existing is not None:
                self.warnings.append(
                    f'add the new reactions to the existing pathway "{pw.existing.display_name}" '
                    f'(dbId {pw.existing.db_id}) manually; its hasEvent was not rewritten')
            else:
                self._add('Pathway', pw.name, {
                    'hasEvent': [key_to_shell[k] for k in pw.reactions if k in key_to_shell],
                    'summation': self._summation(pw.summation, pw.pmids),
                    'literatureReference': [self._publication(p) for p in pw.pmids],
                    'species': [self._shell_of(HUMAN)]})
        ui = {'newInstances': self.new, 'updatedInstances': [], 'deletedInstances': [], 'bookmarks': []}
        return EmitResult(ui, self.links, self.warnings,
                          {k: s['dbId'] for k, s in key_to_shell.items()},
                          {k: s['dbId'] for k, s in self._shells.items() if k in self.d.participants})


def emit_user_instances(draft: ReactomeDraft, evidence: Optional[Dict[str, Evidence]] = None,
                        first_db_id: int = -1) -> EmitResult:
    return Emitter(draft, evidence, first_db_id).emit()
