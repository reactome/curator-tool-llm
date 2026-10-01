from curator_llm.models.evidence import Verification
from curator_llm.services.evidence_attach import attach_evidence, merge_details, split_evidence
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.paper_text import PaperText
from reaction_to_instances import _reaction_items, restore_evidence_ids
from reactome_llm.ReactomeModels import ReactomeDataModel, ReactomeReaction
from reactome_llm.ReactomeQA import ReactomeQA

PAGES = ["Results\nTcPINK1 WT, but not KD, incorporates 32P from radiolabeled ATP onto Ub (Fig. 3 B).\n"
         "CCCP treatment stabilised PINK1 on the outer mitochondrial membrane.\n"]
Q1 = "TcPINK1 WT, but not KD, incorporates 32P from radiolabeled ATP onto Ub"
Q2 = "CCCP treatment stabilised PINK1"


def reaction(evidence):
    return {'source': 'PMID:24751536',
            'annotation_result': {'name': 'PINK1 phosphorylates ubiquitin', 'evidence': evidence}}


def paper(_src):
    return PaperText.from_pages('24751536', PAGES)


def test_split_evidence_accepts_objects_and_strings():
    r = split_evidence({'evidence': [{'quote': Q1, 'supports': ['catalystActivity'],
                                      'experimental_species': 'Tribolium castaneum'}, Q2, Q1]})
    assert r['evidence'] == [Q1, Q2]
    assert r['evidence_details'][Q1]['experimental_species'] == 'Tribolium castaneum'


def test_merge_details_unions_supports():
    out = merge_details({'q': {'supports': ['a'], 'system': 'cellular'}},
                        {'q': {'supports': ['b'], 'system': 'cell_free'}, 'r': {}})
    assert out['q'] == {'supports': ['a', 'b'], 'system': 'cellular'} and 'r' in out


def test_attach_keeps_all_verified_drops_fabricated_and_carries_metadata():
    rx = reaction([{'quote': Q1, 'supports': ['catalystActivity'],
                    'experimental_species': 'Tribolium castaneum', 'system': 'in_vitro_recombinant'},
                   {'quote': Q2}, 'PINK1 phosphorylates Parkin in HeLa cells, as expected'])
    store = EvidenceStore()
    stats = attach_evidence([rx], store, paper)
    a = rx['annotation_result']
    assert stats == {'verified': 2, 'unverified': 0, 'rejected': 1}
    assert a['evidence_ids'] == ['ev-001', 'ev-002'] and a['evidence'] == [Q1, Q2]
    assert len(a['evidence_rejected']) == 1
    ev = store.get('ev-001')
    assert ev.verified == Verification.EXACT and ev.figure == '3B'
    assert ev.experimental_species == 'Tribolium castaneum' and ev.supports == ['catalystActivity']


def test_attach_without_paper_text_keeps_quotes_unverified():
    rx = reaction([Q1])
    stats = attach_evidence([rx], EvidenceStore(), lambda s: None)
    assert stats['unverified'] == 1 and rx['annotation_result']['evidence_ids'] == ['ev-001']


def test_llm_input_never_contains_quotes_when_ids_exist():
    rx = reaction([Q1]); attach_evidence([rx], EvidenceStore(), paper)
    shown = _reaction_items([rx])[0]['reaction']
    assert 'evidence' not in shown and shown['evidence_ids'] == ['ev-001']


def test_restore_overrides_dropped_or_invented_ids():
    rx = reaction([Q1, Q2]); attach_evidence([rx], EvidenceStore(), paper)
    dm = ReactomeDataModel(gene='PINK1', reactions=[
        ReactomeReaction(displayName='PINK1 phosphorylates ubiquitin.', evidence=['ev-001', 'ev-099'])])
    restore_evidence_ids(dm, [rx])
    assert dm.reactions[0].evidence == ['ev-001', 'ev-002']


def test_qa_hydrates_quotes_for_review_only():
    rx = reaction([Q1]); attach_evidence([rx], EvidenceStore(), paper)
    inst = {'reactions': [{'displayName': 'x', 'evidence': ['ev-001']}]}
    out = ReactomeQA._hydrate_evidence(inst, [rx])
    assert out['reactions'][0]['evidence'] == [Q1] and inst['reactions'][0]['evidence'] == ['ev-001']


def test_species_synonyms_are_normalised():
    rx = reaction([{'quote': Q1, 'experimental_species': 'human'}])
    store = EvidenceStore(); attach_evidence([rx], store, paper)
    assert store.get('ev-001').experimental_species == 'Homo sapiens'
