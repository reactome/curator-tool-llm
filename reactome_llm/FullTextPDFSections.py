"""Isolate the Results section of a full-text paper, plus section-title utilities.

extract_results_section: deterministic heading match FIRST, then an LLM pass that
decides the Results start/end (told to ignore everything before the start it finds).

is_section_title / map_section_titles: per-chunk check — "is this chunk a section
heading? if so, what's the title?" — used to verify where sections begin/end.

    from FullTextPDFSections import extract_results_section, is_section_title, map_section_titles
"""
import re
import json

# A heading line that is essentially just "Results" (optionally numbered, or
# "Results and Discussion"). Matches on its own line.
_RESULTS_HEADING = re.compile(
    r'^[ \t]*(?:\d+[.)]?\s*)?results(\s+and\s+discussion)?[ \t]*$', re.I | re.M)

# Headings that mark the END of the Results section.
_NEXT_HEADING = re.compile(
    r'^[ \t]*(?:\d+[.)]?\s*)?('
    r'discussion|materials\s+and\s+methods|methods|experimental\s+procedures|'
    r'acknowledg(?:e?ments)?|references|conclusions?|'
    r'author\s+contributions|data\s+availability|supplementary'
    r')[ \t]*$', re.I | re.M)



def _no_thinking(model):
    """Keyword args that turn extended thinking off. These calls have a tiny max_tokens (100-300) and thinking
    spends it all, leaving no text block at all (seen on papers with no 'Results' heading, whose section lookup
    falls back to these calls)."""
    try:
        from curator_llm.model_settings import thinking_off
    except ImportError:                       # run from a directory without the project root on sys.path
        return {'thinking': {'type': 'disabled'}}
    return {'thinking': thinking_off(model or '')}

def _heuristic(full_text):
    """Return the Results-section text via heading detection, or None if not confident."""
    m = _RESULTS_HEADING.search(full_text)
    if not m:
        return None
    start = m.end()
    combined = bool(m.group(1))  # "Results and Discussion" -> don't stop at 'Discussion'
    end = len(full_text)
    for nm in _NEXT_HEADING.finditer(full_text, start):
        if combined and nm.group(1).lower().startswith('discussion'):
            continue
        end = nm.start()
        break
    section = full_text[start:end].strip()
    return section if len(section.split()) >= 100 else None


def _parse_json(raw):
    txt = raw.replace('```json', '').replace('```', '').strip()
    if not txt.startswith('{'):
        s, e = txt.find('{'), txt.rfind('}')
        if s != -1 and e != -1:
            txt = txt[s:e + 1]
    return json.loads(txt)


def _llm_boundaries(full_text, client, model, hint=None):
    """LLM pass: decide the Results section start & end. Returns the sliced section or None.

    The model is told to find where Results BEGINS and, once found, to ignore all text
    before it (abstract/intro/methods). It returns short verbatim markers (cheap output);
    we slice locally so we never pay to regenerate the whole section.
    """
    hint_line = ""
    if hint:
        hint_line = f'\nA heuristic guessed the Results may start near: "{" ".join(hint.split()[:12])}"\n'
    head = full_text[:16000]
    tail = full_text[-16000:]
    prompt = f"""This is the text of a scientific paper. Find the RESULTS section
(it may be titled "Results" or "Results and Discussion").
{hint_line}
Rules:
- Once you locate where the Results section BEGINS, ignore everything before it — the
  abstract, introduction, and methods are NOT part of Results.
- The Results section ENDS where the next section begins (Discussion, Methods, References,
  Acknowledgments, etc.).

Return ONLY JSON, no markdown:
{{"start_marker": "<first ~8 words of the Results section, verbatim>",
  "end_marker": "<first ~8 words of the section that immediately FOLLOWS Results, verbatim>"}}
If there is no identifiable Results section, return {{"start_marker": null, "end_marker": null}}.

--- PAPER START ---
{head}
--- PAPER END ---
{tail}"""
    try:
        msg = client.messages.create(model=model, max_tokens=300, **_no_thinking(model),
                                     messages=[{'role': 'user', 'content': prompt}])
        raw = ''.join(b.text for b in msg.content if getattr(b, 'type', None) == 'text').strip()
        obj = _parse_json(raw)
    except Exception as ex:
        print(f"    [sections] LLM boundary pass failed ({type(ex).__name__}: {ex})")
        return None
    sm, em = obj.get('start_marker'), obj.get('end_marker')
    if not sm:
        return None
    # find the start; discard everything before it
    si = full_text.find(sm)
    if si == -1:
        si = full_text.lower().find(sm.lower())
    if si == -1:
        return None
    if em:
        # locate the end; if the end marker can't be found, BAIL — do NOT slice to end of
        # document (that would swallow Discussion/Methods and everything after).
        ei = full_text.find(em, si + len(sm))
        if ei == -1:
            ei = full_text.lower().find(em.lower(), si + len(sm))
        if ei == -1:
            print("    [sections] LLM end marker not found in text — bailing (won't slice to end)")
            return None
        section = full_text[si:ei].strip()
    else:
        # LLM explicitly reported no following section -> Results runs to the end
        section = full_text[si:].strip()
    return section if len(section.split()) >= 100 else None


_REFS_HEADING = re.compile(r'^\s*(references|bibliography|literature cited|references and notes)\s*$', re.I | re.M)
_ABSTRACT_HEADING = re.compile(r'^\s*abstract\s*$', re.I | re.M)


def whole_paper(full_text):
    """The paper's running text for a paper with no Results section (a short communication, a review): from the
    Abstract heading (dropping title, authors and affiliations) to the References heading (dropping the
    bibliography). Each cut is made only when it leaves most of the text, so a stray 'References' line cannot
    throw the paper away."""
    text = full_text
    m = _ABSTRACT_HEADING.search(text)
    if m and m.start() < len(text) * 0.25:
        text = text[m.start():]
    refs = [r for r in _REFS_HEADING.finditer(text)]
    if refs and refs[-1].start() > len(text) * 0.5:
        text = text[:refs[-1].start()]
    return text.strip()


def extract_results_section(full_text, client=None, model=None, whole_paper_fallback=False):
    """Return (results_text, method). method in {'llm','heuristic','whole-paper'}.

    Pass 1: deterministic heading match — AUTHORITATIVE when it succeeds (it scans the raw
            text, so it reliably catches a mid-line "Results" heading).
    Pass 2: ONLY if the heuristic finds no Results heading, let the LLM decide start/end
            (with a guard that a missed end marker never slices to end of document).
    With no Results heading: raises ValueError (so the pipeline stops instead of wasting tokens on the full text),
    or, with whole_paper_fallback=True, returns the whole paper (see whole_paper) as method 'whole-paper' without
    asking the LLM to guess.
    """
    heur = _heuristic(full_text)
    if heur:
        return heur, 'heuristic'
    if whole_paper_fallback:
        # No Results heading means a short communication, a review or a manuscript: analyse the whole paper. The LLM
        # boundary guess is skipped here because it is not repeatable on such papers (one run returned a span of
        # 1,266 of 1,669 words, the next none) and a wrong guess silently drops part of the paper.
        text = whole_paper(full_text)
        if len(text.split()) >= 100:
            print('    [sections] no Results section: using the whole paper', flush=True)
            return text, 'whole-paper'
    if client is not None and model is not None:
        llm = _llm_boundaries(full_text, client, model, hint=None)
        if llm:
            return llm, 'llm'
    raise ValueError('No Results section found — aborting to avoid extracting from full text.')


# ─────────────────────────────────────────────────────────────────────────────
# Section-title check: is a given chunk the start of a section, and what's its title?
# ─────────────────────────────────────────────────────────────────────────────
def is_section_title(chunk_text, client, model):
    """Ask the LLM whether this chunk BEGINS a new section of a paper.

    Returns a dict:
      {"is_section_title": True,  "title": "<section name>"}   if it starts a section
      {"is_section_title": False, "title": None}               otherwise
    On any failure, returns the False result (treated as 'no section title found').
    """
    prompt = f"""Below is a text chunk from a scientific paper. Decide whether this chunk
BEGINS a new top-level section (i.e. it starts with a section heading such as
Introduction, Results, Results and Discussion, Discussion, Materials and Methods, Methods,
Acknowledgments, References, Conclusion, etc.).

Return ONLY JSON, no markdown:
{{"is_section_title": <true or false>, "title": "<the exact section title if true, else null>"}}

Text chunk:
{chunk_text[:800]}"""
    try:
        msg = client.messages.create(model=model, max_tokens=100, **_no_thinking(model),
                                     messages=[{'role': 'user', 'content': prompt}])
        raw = ''.join(b.text for b in msg.content if getattr(b, 'type', None) == 'text').strip()
        obj = _parse_json(raw)
        if obj.get('is_section_title'):
            return {"is_section_title": True, "title": obj.get('title')}
    except Exception as ex:
        print(f"    [sections] is_section_title failed ({type(ex).__name__}: {ex})")
    return {"is_section_title": False, "title": None}


def map_section_titles(chunks, client, model):
    """Run is_section_title over a chunk array. Returns a list of results aligned to
    chunks: each item is {chunk_index, is_section_title, title}. Chunks with no section
    title are marked False (caller prints 'no section title found')."""
    out = []
    for i, ch in enumerate(chunks):
        r = is_section_title(ch, client, model)
        out.append({"chunk_index": i, **r})
    return out
