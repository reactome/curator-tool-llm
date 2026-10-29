"""Compare extraction/merge settings against what Reactome curators recorded for the same papers.

For each PMID: fetch the curated human reactions that cite it from the Reactome graph (Neo4j, .env), run extraction
and merge in several configurations, and score each result against the curated reactions.

  python scripts/ground_truth_eval.py run   PMID [PMID ...]   # extraction + merge for every configuration (resumable)
  python scripts/ground_truth_eval.py score PMID [PMID ...]   # table of scores

Scoring is deliberately lenient and name-based (curated names are symbols, papers use common names, so every alias of
a curated entity counts). A curated reaction is RECALLED when one extracted reaction mentions the catalyst (if any)
and at least 60% of the other entities. "Matches curated" is the same test the other way round: it is an upper
bound on precision only, because curators do not record every reaction a paper describes.
Entity recall ignores structure: it asks whether the extraction names the curated entities anywhere.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'reactome_llm'))
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'), override=True)

from curator_llm.services.dedup_prefilter import tokens

RESULTS = os.path.join(ROOT, 'results')
GENE = 'gt'      # label only: PMIDs carry no gene
# extraction configurations: tag -> environment
EXTRACTION = {
    'gtleg': {'EXTRACT_WORKERS': '1', 'LLM_THINKING_EXTRACTION': 'on'},     # the original: serial, thinking on
    'gtdef': {},                                                            # the new defaults
    'gt55': {'LLM_MODEL_EXTRACTION': 'claude-sonnet-5-5', 'LLM_THINKING_EXTRACTION': 'on'},
}
# merge configurations: variant -> (extra args, environment, which extraction tags it runs on)
MERGE = {
    'orig': (['--no-prefilter'], {'MERGE_THINKING': 'on'}, ['gtleg']),      # the original merge
    'new': ([], {}, ['gtleg', 'gtdef', 'gt55']),
    'nopf': (['--no-prefilter'], {}, ['gtdef']),
    'think': ([], {'MERGE_THINKING': 'on'}, ['gtdef']),
}


def stem(pmid):
    import PubMedFetcher as fetcher
    return fetcher.output_stem(str(pmid), gene=GENE)


def extraction_path(pmid, tag):
    return os.path.join(RESULTS, f'{stem(pmid)}_{tag}_extraction.json')


def merged_path(pmid, tag, variant):
    return os.path.join(RESULTS, f'{stem(pmid)}_{tag}_merged_{variant}.json')


# ── curated side ────────────────────────────────────────────────────────────
_CURATED = """
MATCH (e:ReactionLikeEvent)-[:literatureReference]->(l:LiteratureReference {pubMedIdentifier: $pmid})
WHERE (e)-[:species]->(:Species {displayName: 'Homo sapiens'})
OPTIONAL MATCH (e)-[:input]->(i) OPTIONAL MATCH (e)-[:output]->(o)
OPTIONAL MATCH (e)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(c)
RETURN e.stId AS st, e.displayName AS name, collect(DISTINCT i) AS ins, collect(DISTINCT o) AS outs,
       collect(DISTINCT c) AS cats
"""
_LEAVES = """
UNWIND $ids AS id MATCH (p:PhysicalEntity) WHERE p.dbId = id
OPTIONAL MATCH (p)-[:hasComponent|hasMember|hasCandidate*0..6]->(x)
WHERE x:EntityWithAccessionedSequence OR x:SimpleEntity OR x:OtherEntity
OPTIONAL MATCH (x)-[:referenceEntity]->(r)
RETURN id, collect(DISTINCT x.displayName) AS names, collect(DISTINCT r.geneName) AS genes,
       collect(DISTINCT r.name) AS rnames
"""


def _aliases(row):
    out = set()
    for n in row['names']:
        out |= tokens(n.split(' [')[0])
    for lst in (row['genes'], row['rnames']):
        for g in lst:
            for v in (g if isinstance(g, list) else [g]):
                out |= tokens(v or '')
    return frozenset(out)


def curated_reactions(pmid):
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(os.environ['REACTOME_NEO4J_URI'],
                             auth=(os.environ['REACTOME_NEO4J_USER'], os.environ['REACTOME_NEO4J_PWD']))
    out = []
    with d.session(database=os.environ.get('REACTOME_NEO4J_DATABASE')) as s:
        for ev in s.run(_CURATED, pmid=int(pmid)).data():
            ids = {k: [n['dbId'] for n in ev[k]] for k in ('ins', 'outs', 'cats')}
            alias, has_gene = {}, {}
            for r in s.run(_LEAVES, ids=sorted({i for v in ids.values() for i in v})).data():
                alias[r['id']] = _aliases(r)
                has_gene[r['id']] = any(g for g in r['genes'])
            both = [i for group in (ids['ins'], ids['outs']) for i in group]
            # Judge on proteins, genes and RNAs: ATP/GDP/phosphate would almost never be named by an extraction. Only a
            # reaction with no gene-bearing participant (a metabolic one) falls back to every entity.
            pick = [i for i in both if has_gene.get(i)] or both
            ents = [alias.get(i, frozenset()) for i in pick]
            ents = [e for e in ents if e]
            cats = [alias.get(i, frozenset()) for i in ids['cats']]
            out.append({'st': ev['st'], 'name': ev['name'], 'entities': ents, 'catalysts': [c for c in cats if c]})
    d.close()
    return out


# ── extracted side ──────────────────────────────────────────────────────────
def _strings(a):
    for k in ('input', 'output'):
        yield from a.get(k) or []
    cat = (a.get('catalystActivity') or {}).get('catalyst')
    if cat:
        yield cat
    for r in a.get('regulatedBy') or []:
        if isinstance(r, dict) and r.get('regulator'):
            yield r['regulator']
    yield a.get('name') or ''


def extracted_tokens(a):
    out = set()
    for s in _strings(a):
        out |= tokens(s)
    return out


def matches(curated, toks):
    """Does an extracted reaction (as a token set) cover this curated reaction?"""
    if any(c and not (c & toks) for c in curated['catalysts']):
        return False
    ents = curated['entities']
    if not ents:
        return bool(curated['catalysts'])
    return sum(1 for e in ents if e & toks) / len(ents) >= 0.6


def score(path, curated):
    rx = [x.get('annotation_result', x) for x in json.load(open(path))]
    toks = [extracted_tokens(a) for a in rx]
    recalled = sum(1 for c in curated if any(matches(c, t) for t in toks))
    supported = sum(1 for t in toks if any(matches(c, t) for c in curated))
    allt = set().union(*toks) if toks else set()
    ents = [e for c in curated for e in c['entities']]
    ent_found = sum(1 for e in ents if e & allt)
    return {'extracted': len(rx), 'curated': len(curated), 'recalled': recalled, 'matches_curated': supported,
            'entities': len(ents), 'entities_found': ent_found}


# ── running ─────────────────────────────────────────────────────────────────
def sh(cmd, env, log):
    e = dict(os.environ, **env)
    with open(log, 'w') as f:
        return subprocess.run(cmd, cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT).returncode


def run(pmids):
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for pmid in pmids:
        for tag, env in EXTRACTION.items():
            jobs.append((pmid, tag, env))

    def extract(job):
        pmid, tag, env = job
        out = extraction_path(pmid, tag)
        if os.path.exists(out) and os.path.getsize(out) > 2:
            return
        sh([sys.executable, 'run_extraction.py', str(pmid), '--gene', GENE, '--tag', tag, '--overwrite'], env,
           f'/tmp/gt_{pmid}_{tag}.log')
        print(f'extracted {pmid} {tag}: {"ok" if os.path.exists(out) else "NO OUTPUT"}', flush=True)

    # the serial ones are slow but light (one call at a time): run them all together; the parallel ones one at a time
    serial = [j for j in jobs if j[1] == 'gtleg']
    fast = [j for j in jobs if j[1] != 'gtleg']
    with ThreadPoolExecutor(max_workers=len(serial) + 1) as pool:
        futs = [pool.submit(extract, j) for j in serial]
        for j in fast:
            extract(j)
        for f in futs:
            f.result()

    for pmid in pmids:
        for variant, (args, env, tags) in MERGE.items():
            for tag in tags:
                src, dst = extraction_path(pmid, tag), merged_path(pmid, tag, variant)
                if not os.path.exists(src) or (os.path.exists(dst) and os.path.getsize(dst) > 2):
                    continue
                sh([sys.executable, 'run_merge.py', src, '--out', dst] + args, env, f'/tmp/gt_{pmid}_{tag}_{variant}.log')
                print(f'merged {pmid} {tag} {variant}: {"ok" if os.path.exists(dst) else "NO OUTPUT"}', flush=True)


def report(pmids):
    cur = {p: curated_reactions(p) for p in pmids}
    rows = [('original (serial, thinking on, no pre-filter)', 'gtleg', 'orig'),
            ('serial extraction + new merge', 'gtleg', 'new'),
            ('NEW DEFAULTS', 'gtdef', 'new'),
            ('defaults, merge without pre-filter', 'gtdef', 'nopf'),
            ('defaults, merge thinking on', 'gtdef', 'think'),
            ('sonnet-5.5 extraction (thinking on)', 'gt55', 'new')]
    print('curated reactions per paper:', {p: len(c) for p, c in cur.items()})
    tot = {}
    print(f'\n{"configuration":50} {"PMID":>9} extracted recalled(of curated) matches-curated entity-recall')
    for label, tag, variant in rows:
        for p in pmids:
            path = merged_path(p, tag, variant)
            if not os.path.exists(path):
                continue
            s = score(path, cur[p])
            t = tot.setdefault(label, dict.fromkeys(s, 0))
            for k, v in s.items():
                t[k] += v
            print(f'{label:50} {p:>9} {s["extracted"]:9} {s["recalled"]:>4}/{s["curated"]:<4} {s["matches_curated"]:>13} '
                  f'{s["entities_found"]}/{s["entities"]}')
    print('\nTOTALS')
    for label, t in tot.items():
        print(f'{label:50} extracted={t["extracted"]:4} recall={t["recalled"]}/{t["curated"]} '
              f'({100 * t["recalled"] / max(1, t["curated"]):.0f}%)  matches-curated={t["matches_curated"]}/{t["extracted"]} '
              f'({100 * t["matches_curated"] / max(1, t["extracted"]):.0f}%)  entity-recall={100 * t["entities_found"] / max(1, t["entities"]):.0f}%')
    # unmerged extraction too, to separate extraction from merge
    print('\nEXTRACTION ONLY (before merge)')
    for label, tag in (('original serial', 'gtleg'), ('new default', 'gtdef'), ('sonnet-5.5 thinking on', 'gt55')):
        r = c = x = 0
        for p in pmids:
            if os.path.exists(extraction_path(p, tag)):
                s = score(extraction_path(p, tag), cur[p])
                r += s['recalled']; c += s['curated']; x += s['extracted']
        print(f'{label:26} extracted={x:4} recall={r}/{c} ({100 * r / max(1, c):.0f}%)')


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('run', 'score'):
        print(__doc__)
        sys.exit(1)
    ids = sys.argv[2:]
    (run if sys.argv[1] == 'run' else report)(ids)
