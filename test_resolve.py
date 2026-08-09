"""Standalone full-text RESOLUTION tester — no retrieval, no LLM, no pipeline.

Force-feed a set of PMIDs and see exactly how the resolver maps each one to a full-text
source (local curator PDF / downloaded PMC XML / miss). Use this to iterate on the resolver
or to sanity-check a known PMID set without paying for a full `run_pipeline.py` run.

What it does, in order:
  1. Resolve the papers folder: --papers-dir if given, else the one saved in
     data/user_config.json (what the wizard persisted).
  2. build_index(folder) -> {pmid: filepath}  (uses/updates data/.indexdoc_cache.json;
     cached files are instant, only new/changed PDFs are parsed).
  3. resolve_fulltext(pmids, index) -> manifest {pmid: {source, path}}:
       - pmid in index                      -> source "pdf"  (curator's local file)
       - cached data/fulltext_cache/<pmid>.xml -> source "xml"
       - PMID->PMCID (idconv) -> efetch PMC XML (needs a real <body>) -> source "xml"
       - otherwise                          -> source "miss"
  4. Print the manifest as a table (in the order the PMIDs were given).

Run it in the env that has PyMuPDF (needed only to parse NEW PDFs):
    conda run -n paperqa python test_resolve.py 19481056 41043992 33568460 23100419 40892546
    conda run -n paperqa python test_resolve.py --papers-dir ~/Downloads/fullTextPapers 33568460
    conda run -n paperqa python test_resolve.py --no-index 33568460 19481056   # skip folder, force PMC path
"""
import argparse
import sys
from pathlib import Path

sys.path.append("reactome_llm")

from dotenv import load_dotenv
load_dotenv()

import FullTextResolver as R


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pmids", nargs="+", help="PMIDs to resolve (space separated).")
    ap.add_argument("--papers-dir", default=None,
                    help="Folder of full-text PDFs. Defaults to the path saved in "
                         "data/user_config.json.")
    ap.add_argument("--no-index", action="store_true",
                    help="Skip the local folder entirely and resolve via PMC only "
                         "(everything not already cached goes to idconv/efetch or miss).")
    args = ap.parse_args()

    if args.no_index:
        print("Skipping local index — resolving via PMC only.")
        index = {}
    else:
        papers_dir = args.papers_dir or R.load_config().get("papers_dir")
        if not papers_dir:
            print("No papers folder given and none saved in data/user_config.json. "
                  "Pass --papers-dir, or use --no-index to resolve via PMC only.")
            return
        print(f"Papers folder: {papers_dir}")
        index = R.build_index(papers_dir)

    print()
    manifest = R.resolve_fulltext(args.pmids, index, gene="test_resolve")
    print()
    print("  PMID        source  path")
    print("  " + "-" * 70)
    for pmid in args.pmids:
        e = manifest.get(str(pmid), {})
        print(f"  {pmid:<11} {e.get('source', '?'):<6}  {e.get('path', '')}")


if __name__ == "__main__":
    main()
