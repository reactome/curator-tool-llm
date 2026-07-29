"""Manual check for the sectioning logic on any full-text PDF.

Usage:
  python check_sections.py                 # defaults to PINK1.pdf
  python check_sections.py ZNFX1.pdf       # a file name in data/papers/
  python check_sections.py /abs/path/to/paper.pdf

PART 1 — ALL SECTIONS: chunk the whole paper and run the per-chunk section-title check.
  For each chunk: is_section_title -> {"is_section_title": bool, "title": ...}.
  Prints the title (from JSON) when true, else 'no section title found'.

PART 2 — RESULTS SECTION ONLY: run extract_results_section, return a JSON result, and
  print the whole Results chunk it found so you can manually verify start/end.

NOTE: PART 1 makes one LLM call per chunk (~1 min for the whole paper).
"""
import os, sys, json, argparse
PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)
import fitz, anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FullTextPDFSections import extract_results_section, map_section_titles

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
MODEL_NAME = 'claude-sonnet-5'


def resolve_pdf(arg):
    """Accept an absolute/relative path or a bare filename in data/papers/."""
    if os.path.isfile(arg):
        return arg
    candidate = os.path.join(PROJECT_ROOT, 'data', 'papers', arg)
    if os.path.isfile(candidate):
        return candidate
    if not arg.lower().endswith('.pdf') and os.path.isfile(candidate + '.pdf'):
        return candidate + '.pdf'
    raise FileNotFoundError(f"PDF not found: {arg!r} (looked in cwd and data/papers/)")


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('pdf', nargs='?', default='PINK1.pdf',
                    help='PDF path, or a file name inside data/papers/ (default: PINK1.pdf)')
args = parser.parse_args()
PDF_PATH = resolve_pdf(args.pdf)
print(f"[check_sections] using PDF: {PDF_PATH}")

doc = fitz.open(PDF_PATH)
full_text = ''.join(p.get_text() for p in doc); doc.close()

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=400, length_function=len,
                                          separators=['\n\n', '\n', '. ', ' ', ''])
chunks = splitter.split_text(full_text)

# ── PART 1: all sections (section-title check per chunk) ──────────────────────
print("=" * 70)
print(f"PART 1 — ALL SECTIONS  ({len(chunks)} chunks)")
print("=" * 70)
section_map = map_section_titles(chunks, client, MODEL_NAME)
for r in section_map:
    if r['is_section_title']:
        print(f"  chunk {r['chunk_index']:2d}:  is_section_title=True  -> title: {r['title']}")
    else:
        print(f"  chunk {r['chunk_index']:2d}:  no section title found")
found = [r['title'] for r in section_map if r['is_section_title']]
print(f"\n  ALL SECTIONS FOUND: {found}")

# ── PART 2: results section only (JSON result + full chunk) ───────────────────
results_text, how = extract_results_section(full_text, client=client, model=MODEL_NAME)
results_json = {
    "section": "Results",
    "method": how,
    "n_words": len(results_text.split()),
}
print("\n" + "=" * 70)
print("PART 2 — RESULTS SECTION (JSON result)")
print("=" * 70)
print(json.dumps(results_json, indent=2))
print("\n--- full Results chunk (manual check) ---")
print(results_text)

# ── PART 3: split Results into 300-token chunks ───────────────────────────────
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=0, separators=['\n\n', '\n', '. ', ' ', ''])
results_chunks = token_splitter.split_text(results_text)
print("\n" + "=" * 70)
print(f"PART 3 — RESULTS SPLIT INTO 300-TOKEN CHUNKS  ({len(results_chunks)} chunks)")
print("=" * 70)
for i, c in enumerate(results_chunks):
    print(f"\n--- chunk {i} ({len(c.split())} words) ---")
    print(c)
