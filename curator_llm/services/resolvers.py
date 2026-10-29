"""Deterministic identifier resolution for a ReactomeDraft. The LLM never supplies a dbId: names
become identifiers here, from gk_central first (so references reuse what already exists) and
external services second (UniProt, OLS). Anything that cannot be resolved with confidence is left
empty and listed in `needs_resolution` / the returned notes for the curator."""
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from curator_llm.models.reactome import (EwasSpec, ExistingRef, ModifiedResidueSpec, ReactomeDraft,
                                         SimpleEntitySpec)
from curator_llm.ports.external import OntologyClient, UniProtClient
from curator_llm.ports.lookup import InstanceLookup
from curator_llm.services.emitter import entity_display_name

_COMPARTMENT_SYNONYMS = {'cytoplasm': 'cytosol', 'outer mitochondrial membrane': 'mitochondrial outer membrane',
                         'omm': 'mitochondrial outer membrane', 'mitochondrial matrix space': 'mitochondrial matrix'}

# Common PSI-MOD terms: label -> (id, short prefix). Looked up by id in gk_central (class PsiMod).
PSI_MOD = {
    'MOD:00046': ('O-phospho-L-serine', 'p'), 'MOD:00047': ('O-phospho-L-threonine', 'p'),
    'MOD:00048': ('O4\'-phospho-L-tyrosine', 'p'), 'MOD:00064': ('N6-acetyl-L-lysine', 'ac'),
    'MOD:01148': ('N6-glycyl-L-lysine', 'ub'), 'MOD:00085': ('N6-methyl-L-lysine', 'me'),
}
_GENE_SYMBOL = re.compile(r'^[A-Za-z][A-Za-z0-9\-]{1,11}$')
_GENE_OR_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9\- ]{1,40}$')


def _ref(inst: Optional[dict]) -> Optional[ExistingRef]:
    if not inst:
        return None
    return ExistingRef(db_id=inst['dbId'], display_name=inst.get('displayName', ''),
                       schema_class=inst.get('schemaClassName', ''))


class Resolver:
    def __init__(self, lookup: InstanceLookup, uniprot: Optional[UniProtClient] = None,
                 ontology: Optional[OntologyClient] = None):
        self.lookup, self.uniprot, self.ontology = lookup, uniprot, ontology
        self.notes: List[str] = []
        self._down: set = set()     # services that failed once: not retried for the rest of this run
        self._tl = threading.local()    # per-task note buffer while lookups run in parallel
        self.workers = max(1, int(os.getenv('RESOLVE_WORKERS', '8')))

    def _note(self, msg: str):
        buf = getattr(self._tl, 'buf', None)
        (self.notes if buf is None else buf).append(msg)

    def _task(self, fn: Callable, *args):
        """Run one lookup task with its own note buffer; returns (result, notes). Notes are merged in task order by
        the caller, so the output does not depend on which thread finished first."""
        self._tl.buf = buf = []
        try:
            return fn(*args), buf
        finally:
            self._tl.buf = None

    def _call(self, service: str, fn, *args, default=None):
        """Call an external service. A failure becomes one note and `default`, never an exception:
        resolution is best-effort and a slow or down service must not lose the whole draft."""
        if service in self._down:
            return default
        try:
            return fn(*args)
        except Exception as e:
            self._down.add(service)
            self._note(f'{service} unavailable ({type(e).__name__}); identifiers it would supply '
                              f'were left unresolved')
            return default

    # ── single lookups ─────────────────────────────────────────────────────
    def compartment(self, name: str) -> Optional[ExistingRef]:
        n = _COMPARTMENT_SYNONYMS.get(name.strip().lower(), name.strip())
        return _ref(self.lookup.find_by_display_name(n, ['Compartment', 'GO_CellularComponent']))

    def _by_identifier(self, cls: str, identifier: str) -> Optional[ExistingRef]:
        hits = self.lookup.search(cls, 'identifier', identifier, 'equal', 10)
        # a class search also returns subclass instances (ReferenceIsoform shares its parent's
        # identifier), so prefer the exact class; several of the SAME class is ambiguous: no guess
        exact = [h for h in hits if h.get('schemaClassName') == cls] or hits
        return _ref(exact[0]) if len(exact) == 1 else None

    def reference_gene_product(self, accession: str) -> Optional[ExistingRef]:
        return self._by_identifier('ReferenceGeneProduct', accession)

    def reference_molecule(self, chebi: Optional[str], name: Optional[str]) -> Optional[ExistingRef]:
        if chebi:
            r = self._by_identifier('ReferenceMolecule', chebi.split(':')[-1])
            if r:
                return r
        if name:
            hits = self.lookup.search('ReferenceMolecule', 'name', name, 'equal', 2)
            if len(hits) == 1:
                return _ref(hits[0])
        return None

    def go_function(self, identifier: Optional[str], name: Optional[str]):
        """(identifier, existing ref). Resolves a missing identifier from the label through OLS."""
        if not identifier and name:
            ref = _ref(self.lookup.find_by_display_name(name, ['GO_MolecularFunction']))
            if ref:                                      # curated label match: no external call needed
                hit = self.lookup.find_by_db_id(ref.db_id) or {}
                ident = hit.get('identifier')
                return (ident if not ident or ident.startswith('GO:') else f'GO:{ident}'), ref
        if not identifier and name and self.ontology:
            hits = self._call('OLS', self.ontology.search, name, 'go', default=[])
            identifier = hits[0]['identifier'] if len(hits) == 1 else None
        ref = self._by_identifier('GO_MolecularFunction', identifier.split(':')[-1]) if identifier else None
        return identifier, ref

    def psi_mod(self, mod_id: str) -> Optional[ExistingRef]:
        return self._by_identifier('PsiMod', mod_id.split(':')[-1])

    def publication(self, pmid: str) -> Optional[ExistingRef]:
        hits = self.lookup.search('LiteratureReference', 'pubMedIdentifier', pmid, 'equal', 2)
        return _ref(hits[0]) if len(hits) == 1 else None

    # ── whole draft ────────────────────────────────────────────────────────
    def _resolve_participant(self, p):
        if p.compartment is None and p.compartment_name:
            p.compartment = self.compartment(p.compartment_name)
            if p.compartment is None:
                p.needs_resolution.append('compartment')
                self._note(f'{p.name}: compartment "{p.compartment_name}" not found in gk_central')
        if isinstance(p, EwasSpec):
            self._ewas(p)
        elif isinstance(p, SimpleEntitySpec):
            self._simple(p)

    def _resolve_catalyst(self, r):
        a = r.catalyst.activity
        a.identifier, a.ref = self.go_function(a.identifier, a.name)
        if a.ref is None:
            self._note(f'{r.name}: GO function "{a.name or a.identifier}" is not an existing GO_MolecularFunction')

    def resolve(self, draft: ReactomeDraft) -> List[str]:
        """Every lookup is independent (one participant, one GO function, one publication), and each is a network
        round trip, so they run `workers` at a time. Results and notes are applied in the original order."""
        pmids = list(dict.fromkeys([m for r in draft.reactions for m in r.pmids] +
                                   (draft.pathway.pmids if draft.pathway else [])))
        pmids = [m for m in pmids if m not in draft.publications]
        catalysts = [r for r in draft.reactions if r.catalyst and r.catalyst.activity]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            parts = [pool.submit(self._task, self._resolve_participant, p) for p in draft.participants.values()]
            cats = [pool.submit(self._task, self._resolve_catalyst, r) for r in catalysts]
            pubs = [(m, pool.submit(self._task, self.publication, m)) for m in pmids]
            for f in parts + cats:
                self.notes += f.result()[1]
            for m, f in pubs:
                ref, notes = f.result()
                self.notes += notes
                if ref:
                    draft.publications[m] = ref
        self.notes = list(dict.fromkeys(self.notes))     # the same finding for several entities reads once
        return self.notes

    def _graph_accession(self, name: str) -> Optional[str]:
        """The accession Reactome already uses for this protein name (no external call), when the lookup can say."""
        find = getattr(self.lookup, 'find_human_accession', None)
        return self._call('Reactome graph', find, name) if find and _GENE_OR_NAME.match(name) else None

    def _note_candidates(self, p: EwasSpec):
        """A name with no single accession may stand for a family (ubiquitin): tell the curator which genes Reactome
        groups under it, rather than guessing one."""
        find = getattr(self.lookup, 'candidate_genes', None)
        genes = self._call('Reactome graph', find, p.name) if find else None
        if genes:
            self._note(f'{p.name}: no single UniProt accession; Reactome has a set of that name whose members '
                              f'come from {", ".join(genes)}. Choose the gene(s) or use that set')

    def _ewas(self, p: EwasSpec):
        if p.molecule != 'protein':          # a gene or transcript has no UniProt entry; only look for an identical one
            if p.existing is None and p.compartment is not None:
                p.existing = _ref(self.lookup.find_by_display_name(
                    entity_display_name(p), ['EntityWithAccessionedSequence']))
            return
        if not p.uniprot and (acc := self._graph_accession(p.name)):
            p.uniprot = acc
            self._note(f'{p.name}: UniProt {acc} found in the Reactome graph by name; confirm')
        if self.uniprot:
            if p.uniprot:
                info = self._call('UniProt', self.uniprot.fetch, p.uniprot, default=False)
                if info is False:
                    p.needs_resolution.append('uniprot-unchecked')
                elif info is None:
                    self._note(f'{p.name}: UniProt accession {p.uniprot} does not exist or is obsolete; removed')
                    p.uniprot = None
                    p.needs_resolution.append('uniprot')
                elif p.name.upper() not in {g.upper() for g in info['genes']} and not any(
                        p.name.lower() in n.lower() for n in info['names']):
                    self._note(f'{p.name}: UniProt {p.uniprot} is gene(s) {info["genes"]}, '
                                      f'which does not match the name; check it')
                    p.needs_resolution.append('uniprot-mismatch')
            elif _GENE_SYMBOL.match(p.name):
                acc = self._call('UniProt', self.uniprot.search_gene, p.name)
                if acc:
                    p.uniprot = acc
                    self._note(f'{p.name}: UniProt {acc} found by gene-name search; confirm')
                else:
                    p.needs_resolution.append('uniprot')
                    self._note_candidates(p)
        if p.uniprot and p.reference_entity is None:
            p.reference_entity = self.reference_gene_product(p.uniprot)
        for m in p.modifications:
            self._modification(m)
        if not p.modifications and p.existing is None and p.compartment is not None:
            p.existing = _ref(self.lookup.find_by_display_name(
                entity_display_name(p), ['EntityWithAccessionedSequence']))

    def _modification(self, m: ModifiedResidueSpec):
        if m.psi_mod and m.psi_mod in PSI_MOD:
            label, short = PSI_MOD[m.psi_mod]
            m.mod_label = m.mod_label or label
            m.short = short
        if m.psi_mod and m.psi_mod_ref is None:
            m.psi_mod_ref = self.psi_mod(m.psi_mod)

    def _simple(self, p: SimpleEntitySpec):
        if p.reference_entity is None:
            p.reference_entity = self.reference_molecule(p.chebi, p.name)
        if p.reference_entity is None and not p.chebi and self.ontology:
            hits = self._call('OLS', self.ontology.search, p.name, 'chebi', default=[])
            if len(hits) == 1:
                p.chebi = hits[0]['identifier']
                self._note(f'{p.name}: ChEBI {p.chebi} suggested by label search; confirm the right charge form')
        if p.existing is None and p.compartment is not None:
            p.existing = _ref(self.lookup.find_by_display_name(
                entity_display_name(p), ['SimpleEntity']))
        if p.reference_entity is None and p.existing is None and not p.chebi:
            p.needs_resolution.append('chebi')
