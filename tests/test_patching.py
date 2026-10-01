import pytest

from curator_llm.models.reactome import (CatalystSpec, EwasSpec, ReactionSpec, ReactomeDraft, RegulationSpec,
                                         SimpleEntitySpec)
from curator_llm.services.paper_text import PaperText
from curator_llm.services.patching import PatchError, apply_patch, describe_changes, touched_reactions


def draft():
    d = ReactomeDraft()
    d.participants = {'pink1': EwasSpec(key='pink1', name='PINK1'), 'ub': EwasSpec(key='ub', name='UB'),
                      'atp': SimpleEntitySpec(key='atp', name='ATP'), 'ccp': SimpleEntitySpec(key='ccp', name='CCCP')}
    d.reactions = [ReactionSpec(key='r0', name='PINK1 phosphorylates Ub', inputs=['ub', 'atp'], outputs=['ub'],
                                catalyst=CatalystSpec(entity='pink1'), evidence_ids=['ev-001']),
                   ReactionSpec(key='r1', name='second', inputs=['ub'])]
    return d


def test_edit_by_reaction_key_and_by_index_and_original_is_untouched():
    d = draft()
    new = apply_patch(d, [{'op': 'replace', 'path': '/reactions/r0/summation', 'value': 'new text'},
                          {'op': 'replace', 'path': '/reactions/1/name', 'value': 'renamed'}])
    assert new.reactions[0].summation == 'new text' and new.reactions[1].name == 'renamed'
    assert d.reactions[0].summation == ''


def test_add_regulation_with_condition_note():
    new = apply_patch(draft(), [{'op': 'add', 'path': '/reactions/r0/regulations/-',
                                 'value': {'kind': 'positive', 'regulator': 'ccp', 'note': 'seen only after CCCP'}}])
    assert new.reactions[0].regulations == [RegulationSpec(kind='positive', regulator='ccp', note='seen only after CCCP')]


def test_add_participant_then_use_it():
    new = apply_patch(draft(), [
        {'op': 'add', 'path': '/participants/adp', 'value': {'kind': 'simple', 'key': 'adp', 'name': 'ADP', 'chebi': 'CHEBI:456216'}},
        {'op': 'add', 'path': '/reactions/r0/outputs/-', 'value': 'adp'}])
    assert new.participants['adp'].name == 'ADP' and new.reactions[0].outputs == ['ub', 'adp']


@pytest.mark.parametrize('ops,why', [
    ([{'op': 'replace', 'path': '/reactions/r0/inputs', 'value': ['ghost']}], 'unknown participant'),
    ([{'op': 'remove', 'path': '/participants/pink1'}], 'unknown participant'),
    ([{'op': 'replace', 'path': '/reactions/r0/preceding', 'value': ['r0']}], 'precedes itself'),
    ([{'op': 'replace', 'path': '/reactions/r0/preceding', 'value': ['nope']}], 'unknown preceding'),
    ([{'op': 'replace', 'path': '/reactions/r0/evidence_ids', 'value': []}], 'evidence cannot be edited'),
    ([{'op': 'add', 'path': '/reactions/r0/evidence_ids/-', 'value': 'ev-9'}], 'evidence cannot be edited'),
    ([{'op': 'replace', 'path': '/reactions/r0/reaction_type', 'value': 5}], 'not a valid draft'),
    ([{'op': 'replace', 'path': '/reactions/r0/nonexistent', 'value': 1}], 'patch does not apply'),
    ([{'op': 'remove', 'path': '/reactions/9'}], 'patch does not apply'),
])
def test_bad_patches_are_refused_with_a_reason(ops, why):
    with pytest.raises(PatchError, match=why):
        apply_patch(draft(), ops)


def test_describe_and_touched():
    d = draft()
    new = apply_patch(d, [{'op': 'replace', 'path': '/participants/ub/name', 'value': 'UBB'},
                          {'op': 'replace', 'path': '/reactions/r1/name', 'value': 'renamed'}])
    lines = describe_changes(d, new)
    assert any('UBB' in l and '.name' in l for l in lines) and any('reaction r1.name' in l for l in lines)
    assert touched_reactions(d, new) == ['r0', 'r1']          # r0 uses the changed participant


def test_paper_text_round_trips_through_a_dict():
    p = PaperText.from_pages('1', ['Results\nfoo bar baz.\n', 'Discussion\nqux.\n'], first_page=3)
    q = PaperText.from_dict(p.to_dict())
    assert q.search('qux', section='Discussion')[0].page == 4 and q.locate((0, 5)) == p.locate((0, 5))
