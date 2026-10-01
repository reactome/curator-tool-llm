from curator_llm.models.evidence import ClaimOrigin, Evidence, Verification
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.paper_text import PaperText

PAGES = ["Abstract\nPINK1 is a kinase.\n",
         "Results\nTcPINK1 WT, but not KD, incorporates 32P from radiolabeled ATP onto Ub (Fig. 3 B).\n"
         "CCCP treatment stabilised PINK1 on the outer mitochondrial membrane.\n",
         "Discussion\nWe propose a feed-forward model of Parkin activation.\n"]


def paper():
    return PaperText.from_pages('24751536', PAGES, first_page=145)


def test_verify_fills_location():
    ev = paper().verify(Evidence(quote="incorporates 32P from radiolabeled ATP onto Ub"))
    assert ev.verified == Verification.EXACT
    assert ev.page == 146 and ev.section == 'Results' and ev.figure == '3B'
    assert ev.pmid == '24751536'


def test_section_and_page_for_discussion():
    ev = paper().verify(Evidence(quote="We propose a feed-forward model of Parkin activation"))
    assert (ev.page, ev.section) == (147, 'Discussion')


def test_store_assigns_ids_dedups_and_merges_supports():
    store = EvidenceStore(paper())
    a, _ = store.add(Evidence(quote="CCCP treatment stabilised PINK1", supports=['condition']))
    b, _ = store.add(Evidence(quote="cccp  treatment stabilised PINK1", supports=['regulatedBy[0]']))
    assert a.id == b.id == 'ev-001' and len(store) == 1
    assert set(store.get('ev-001').supports) == {'condition', 'regulatedBy[0]'}


def test_store_rejects_unverifiable_quote():
    store = EvidenceStore(paper())
    ev, kept = store.add(Evidence(quote="PINK1 phosphorylates Parkin in neurons from patients"))
    assert not kept and ev.verified == Verification.FAILED and len(store) == 0


def test_curator_assertion_is_kept_without_paper_text():
    store = EvidenceStore(paper())
    ev, kept = store.add(Evidence(quote="Curator knowledge", claim_origin=ClaimOrigin.CURATOR_ASSERTION))
    assert kept and ev.verified == Verification.UNVERIFIED


def test_no_truncation_many_quotes_kept():
    store = EvidenceStore(paper())
    for q in ["TcPINK1 WT, but not KD", "radiolabeled ATP onto Ub", "CCCP treatment stabilised PINK1",
              "outer mitochondrial membrane"]:
        store.add(Evidence(quote=q))
    assert len(store) == 4


def test_search_filters_by_section():
    hits = paper().search("feed-forward Parkin", section='Discussion')
    assert hits and hits[0].section == 'Discussion' and hits[0].page == 147
    assert paper().search("feed-forward Parkin", section='Results') == []
