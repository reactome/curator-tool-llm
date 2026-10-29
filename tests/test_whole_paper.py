import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reactome_llm'))
import pytest
from FullTextPDFSections import extract_results_section, whole_paper

BODY = ' '.join(['word'] * 200)
MANUSCRIPT = f"Title\nAuthors A, B\nAffiliation 1\nAbstract\n{BODY}\nSecond paragraph {BODY}\nReferences\n1. Ref one\n2. Ref two\n"


def test_whole_paper_drops_front_matter_and_references():
    t = whole_paper(MANUSCRIPT)
    assert t.startswith('Abstract') and 'Affiliation' not in t and 'Ref one' not in t and 'Second paragraph' in t


def test_a_stray_early_references_line_does_not_discard_the_paper():
    text = f"Abstract\nReferences\n{BODY}\n{BODY}"
    assert 'word' in whole_paper(text)


def test_no_results_heading_raises_unless_fallback_is_asked_for():
    with pytest.raises(ValueError):
        extract_results_section(MANUSCRIPT)
    text, how = extract_results_section(MANUSCRIPT, whole_paper_fallback=True)
    assert how == 'whole-paper' and 'Second paragraph' in text


def test_fallback_does_not_ask_the_llm_to_guess_boundaries():
    class NoCalls:
        class messages:
            @staticmethod
            def create(**kw):
                raise AssertionError('the LLM must not be called')
    text, how = extract_results_section(MANUSCRIPT, client=NoCalls(), model='m', whole_paper_fallback=True)
    assert how == 'whole-paper'
