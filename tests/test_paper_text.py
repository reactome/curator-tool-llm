from curator_llm.services.paper_text import PaperText


def test_a_pdf_with_only_an_abstract_heading_does_not_label_the_whole_paper_abstract():
    page1 = 'Title\nAbstract\n' + 'Short abstract sentence. ' * 20
    page2 = 'Body text with results but no headings. ' * 200
    pt = PaperText.from_pages(None, [page1, page2])
    body = pt.text.index('Body text')
    assert pt.locate((body, body + 20))['section'] == 'Main Text'
    assert pt.locate((pt.text.index('Short'), pt.text.index('Short') + 10))['section'] == 'Abstract'


def test_a_real_section_structure_gets_no_main_text_label():
    pages = ['Abstract\n' + 'a. ' * 30 + '\nIntroduction\n' + 'x ' * 3000 + '\nResults\n' + 'y ' * 100]
    assert 'Main Text' not in [t for _, t in PaperText.from_pages(None, pages).section_starts]
