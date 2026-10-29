import sys, os
from curator_llm.services.dedup_prefilter import candidate_pairs, may_be_duplicates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reactome_llm'))


def r(inp, out):
    return {'input': inp, 'output': out}


def test_synonyms_and_decorations_still_overlap():
    a = r(['Ubiquitin', 'ATP'], ['Phospho-ubiquitin (Ser65)'])
    b = r(['Ubiquitin'], ['Ubiquitin phosphorylated at Ser65'])
    assert may_be_duplicates(a, b)


def test_disjoint_inputs_or_outputs_are_ruled_out():
    assert not may_be_duplicates(r(['PINK1'], ['PINK1 dimer']), r(['Parkin'], ['PINK1 dimer']))
    assert not may_be_duplicates(r(['Parkin'], ['Mfn1 ubiquitinated']), r(['Parkin'], ['Mfn2 ubiquitinated']))


def test_fails_open_on_empty_sides():
    assert may_be_duplicates(r([], ['X']), r(['Y'], ['X']))
    assert may_be_duplicates(r(['ATP'], ['Z']), r(['Z2'], ['Z']))      # only stop-words on one side


def test_candidate_pairs_filters_given_pairs():
    ann = [r(['A1'], ['B1']), r(['A1'], ['B1']), r(['C9'], ['D9'])]
    assert candidate_pairs(ann, [(0, 1), (0, 2), (1, 2)]) == [(0, 1)]


def test_extraction_prompt_head_is_cacheable_and_raw_prev_note():
    import FullTextPDFPrompts as m
    h1, t1 = m.build_extraction_prompt('chunk one', ['notes'], 'next', split=True)
    h2, t2 = m.build_extraction_prompt('chunk two', ['p1', 'p2'], None, raw_prev=True, split=True)
    assert h1 == h2                                     # identical head: can be cached across chunks
    assert 'chunk two' in t2 and 'RAW text' in t2 and 'NOTE FOR THIS CALL' in t2
    assert m.build_extraction_prompt('chunk one', ['notes'], 'next') == h1 + t1


def test_enzyme_listed_as_input_in_one_record_and_as_catalyst_in_the_other():
    # NAMPT / NAD+ paper: the same biosynthesis, with the enzyme among the inputs in one record only
    a = {'input': ['NAD+ precursor (nicotinamide)'], 'output': ['NAD+'], 'catalystActivity': {'catalyst': 'NAMPT'}}
    b = {'input': ['NAMPT'], 'output': ['NAD+']}
    assert may_be_duplicates(a, b)
