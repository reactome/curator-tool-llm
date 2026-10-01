"""Build a replayable snapshot from files an earlier run left behind, without calling any language model.

Verifies quotes, resolves identifiers and looks for existing reactions (Neo4j, UniProt, OLS: fast, no LLM) around
a draft and merged reactions you already have, then saves the result where SNAPSHOT_MODE=replay will find it.

  python scripts/make_snapshot.py --pmid 24751536 --focus PINK1 --pdf data/papers/PINK1.pdf \\
      --merged results/pink1_evtest_merged2.json --draft results/pink1_draft.json

--draft is the typed draft (ReactomeDraft JSON, as the draft builder produced it); --merged is the merged reaction
list the extractor produced. The snapshot is saved for the PMID with and without the focus, so either form of the
start request replays it.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'reactome_llm'))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, '.env'), override=True)

from curator_llm.adapters.neo4j_lookup import Neo4jInstanceLookup  # noqa: E402
from curator_llm.adapters.ols import OlsClient  # noqa: E402
from curator_llm.adapters.uniprot import RestUniProtClient  # noqa: E402
from curator_llm.models.reactome import ReactomeDraft  # noqa: E402
from curator_llm.ports.pipeline import PipelineInput  # noqa: E402
from curator_llm.services.paper_text import PaperText  # noqa: E402
from curator_llm.services.pipeline_default import DefaultPipeline  # noqa: E402
from curator_llm.services.snapshots import SnapshotStore, snapshot_key  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pmid', required=True)
    ap.add_argument('--focus')
    ap.add_argument('--pdf', required=True, help='the paper, for quote checking and search')
    ap.add_argument('--merged', required=True)
    ap.add_argument('--draft', required=True)
    ap.add_argument('--dir', default=os.getenv('SNAPSHOT_DIR', os.path.join(ROOT, 'data', 'snapshots')))
    a = ap.parse_args(argv)

    reactions = json.load(open(a.merged))
    draft = ReactomeDraft.model_validate_json(open(a.draft).read())
    for r in draft.reactions:                     # a PDF source tag carries no PMID; we know it
        r.pmids = r.pmids or [a.pmid]
    if draft.pathway:
        draft.pathway.pmids = draft.pathway.pmids or [a.pmid]

    lookup = Neo4jInstanceLookup.from_env()
    pipe = DefaultPipeline(lookup, RestUniProtClient(), OlsClient(), events=lookup)
    paper = PaperText.from_pdf(a.pmid, a.pdf)
    spec = PipelineInput(pmid=a.pmid, focus=a.focus)
    result = pipe.assemble(spec, reactions, lambda src: paper, {'main': paper}, lambda m: print(' ', m), draft=draft)

    store = SnapshotStore(a.dir)
    keys = {snapshot_key(spec), snapshot_key(PipelineInput(pmid=a.pmid))}
    for k in sorted(keys):
        print('saved', store.save(k, result))
    print(f'{len(result.draft.reactions)} reactions, {len(result.evidence)} quotes, {len(result.issues)} issues, '
          f'{len(result.existing)} existing-reaction matches')
    return 0


if __name__ == '__main__':
    sys.exit(main())
