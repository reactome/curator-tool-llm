from curator_llm.models.evidence import Verification
from curator_llm.services.quote_verifier import normalize_with_map, verify_quote

TEXT = ("Results and discussion\nWith the same in vitro system, we also found that TcPINK1 WT, "
        "but not KD, incorporates 32P from radiolabeled ATP onto Ub (Fig. 3 B).\n"
        "Phospho-\nrylation of Ub at Ser65 was detected by mass spec.")


def test_exact_match_returns_original_span():
    q = "TcPINK1 WT, but not KD, incorporates 32P from radiolabeled ATP onto Ub"
    m = verify_quote(q, TEXT)
    assert m.status == Verification.EXACT
    assert TEXT[m.span[0]:m.span[1]] == q


def test_whitespace_case_and_curly_quote_tolerant():
    m = verify_quote("tcpink1   WT,\nbut not KD", TEXT)
    assert m.status == Verification.EXACT


def test_dehyphenation_across_line_break():
    m = verify_quote("Phosphorylation of Ub at Ser65", TEXT)
    assert m.status == Verification.EXACT
    assert TEXT[m.span[0]:m.span[1]].startswith('Phospho-')


def test_fuzzy_match_on_small_difference():
    q = "we also found that TcPINK1 WT, but not KD, incorporated 32P from radiolabeled ATP onto Ub"
    m = verify_quote(q, TEXT)
    assert m.status == Verification.FUZZY and m.score >= 90


def test_fabricated_quote_fails():
    assert verify_quote("PINK1 phosphorylates Parkin at Ser65 in HeLa cells", TEXT).status == Verification.FAILED


def test_short_quote_never_fuzzy():
    assert verify_quote("Ser66", TEXT).status == Verification.FAILED


def test_ellipsis_pieces_must_appear_in_order():
    ok = verify_quote("we also found that TcPINK1 WT ... onto Ub (Fig. 3 B)", TEXT)
    assert ok.status == Verification.EXACT
    reordered = verify_quote("onto Ub (Fig. 3 B) ... we also found that TcPINK1 WT", TEXT)
    assert reordered.status == Verification.FAILED


def test_normalize_map_points_into_original():
    norm, idx = normalize_with_map("A  b\n C")
    assert norm == "a b c" and len(idx) == len(norm)


def test_pdf_column_break_hyphen_copied_by_quote_is_still_exact():
    text = "translocates to damaged mitochon-\ndria where it ubiquitinates proteins"
    m = verify_quote("translocates to damaged mitochon-dria where it ubiquitinates proteins", text)
    assert m.status == Verification.EXACT
