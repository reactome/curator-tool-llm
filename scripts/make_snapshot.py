"""Build a replayable snapshot from files an earlier run left behind, without calling any language model.

Verifies quotes, resolves identifiers and looks for existing reactions (Neo4j, UniProt, OLS: fast, no LLM) around
a draft and merged reactions you already have, then saves the result where SNAPSHOT_MODE=replay will find it.

  python scripts/make_snapshot.py --pmid 24751536 --focus PINK1 --pdf data/papers/PINK1.pdf \\
      --merged results/pink1_evtest_merged2.json --draft results/pink1_draft.json

--draft is the typed draft (ReactomeDraft JSON, as the draft builder produced it); --merged is the merged reaction
list the extractor produced. To show token usage for the saved result, pass the logs of the runs that produced it,
one per step: --usage extraction=run_extraction.log --usage merge=run_merge.log (the "N call(s), tokens in X / out Y"
line is read from each). A step with no log has no usage row; nothing is estimated. The snapshot is saved for the PMID with and without the focus, so either form of the
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
from curator_llm.models.usage import entry_from_counts  # noqa: E402
from curator_llm.ports.pipeline import PipelineInput  # noqa: E402
from curator_llm.services.paper_text import PaperText  # noqa: E402
from curator_llm.services.extraction import parse_usage  # noqa: E402
from curator_llm.services.pipeline_default import DefaultPipeline  # noqa: E402
from curator_llm.services.snapshots import SnapshotStore, snapshot_key  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pmid', required=True)
    ap.add_argument('--focus')
    ap.add_argument('--pdf', required=True, help='the paper, for quote checking and search')
    ap.add_argument('--merged', required=True)
    ap.add_argument('--draft', required=True)
    ap.add_argument('--usage', action='append', default=[], metavar='STEP=LOGFILE',
                    help='token usage of a step, read from its log (repeatable): extraction, merge or review')
    ap.add_argument('--dir', default=os.getenv('SNAPSHOT_DIR', os.path.join(ROOT, 'data', 'snapshots')))
    a = ap.parse_args(argv)

    usage_rows = []                                # checked first: a typo should not cost a slow assembly
    for item in a.usage:
        step, _, path = item.partition('=')
        if step not in ('extraction', 'merge', 'review') or not os.path.isfile(path):
            ap.error(f'--usage expects STEP=LOGFILE with STEP one of extraction, merge, review and an existing file: {item!r}')
        counts = parse_usage(open(path, errors='ignore').read())
        if not any(counts.values()):
            ap.error(f'no "N call(s), tokens in X / out Y" line found in {path}')
        usage_rows.append(entry_from_counts(step, counts))

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

    result.usage = usage_rows + result.usage

    store = SnapshotStore(a.dir)
    keys = {snapshot_key(spec), snapshot_key(PipelineInput(pmid=a.pmid))}
    for k in sorted(keys):
        print('saved', store.save(k, result))
    print(f'{len(result.draft.reactions)} reactions, {len(result.evidence)} quotes, {len(result.issues)} issues, '
          f'{len(result.existing)} existing-reaction matches, usage rows: {[u.step for u in result.usage] or "none"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
