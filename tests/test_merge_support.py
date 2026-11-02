"""Merging across papers: per-quote sources, exact duplicates kept from every paper, fast clustering, verdict cache."""
import itertools
import json
import random
import time

from curator_llm.models.evidence import Verification
from curator_llm.services.evidence_attach import attach_evidence, stamp_sources
from curator_llm.services.evidence_store import EvidenceStore
from curator_llm.services.merge_support import VerdictCache, collapse_exact_duplicates, conflict_free_clusters
from curator_llm.services.paper_text import PaperText
from curator_llm.services.sources import record_pmids, record_sources


# ── clustering ─────────────────────────────────────────────────────────────
def reference_clusters(n, SAME, CONF):
    """The algorithm as it was before: scan every pair of groups on every round."""
    groups = [[i] for i in range(n)]
    while True:
        best = None
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                cross = [(min(a, b), max(a, b)) for a in groups[gi] for b in groups[gj]]
                if not all(SAME.get(p) for p in cross):
                    continue
                weakest = min(CONF.get(p) or 0.0 for p in cross)
                if best is None or weakest > best[0]:
                    best = (weakest, gi, gj)
        if best is None:
            return sorted(groups, key=lambda g: (-len(g), g))
        _, gi, gj = best
        groups[gi] = sorted(groups[gi] + groups[gj])
        groups.pop(gj)


def random_verdicts(rng, n, density, tied):
    same, conf = {}, {}
    for a, b in itertools.combinations(range(n), 2):
        if rng.random() < density:
            same[(a, b)] = rng.random() < 0.7
            conf[(a, b)] = rng.choice([0.6, 0.8, 0.95]) if tied else rng.choice([None, round(rng.random(), 3)])
    return same, conf


def test_clustering_gives_exactly_the_groups_the_old_algorithm_gave():
    rng = random.Random(7)
    for trial in range(300):
        n = rng.randint(1, 14)
        same, conf = random_verdicts(rng, n, rng.choice([0.2, 0.5, 0.9]), tied=trial % 2 == 0)   # ties and conflicts both occur
        assert conflict_free_clusters(n, same, conf) == reference_clusters(n, same, conf), (trial, n, same)


def test_clustering_stays_fast_with_many_papers_in_one_file():
    # 20 papers x 30 reactions; reactions only ever compared within their own paper (cross-paper pairs are absent)
    rng = random.Random(3)
    same, conf, n = {}, {}, 600
    for p in range(20):
        ids, i = list(range(p * 30, p * 30 + 30)), 0
        while i < len(ids):
            k = rng.choice([1, 2, 2, 3])
            for a, b in itertools.combinations(ids[i:i + k], 2):
                same[(a, b)], conf[(a, b)] = True, 0.95
            i += k
    t = time.time()
    groups = conflict_free_clusters(n, same, conf)
    assert time.time() - t < 2.0                     # the old scan took about 13 s on this size
    assert sum(len(g) for g in groups) == n and len(groups) < n


# ── exact duplicates across papers ─────────────────────────────────────────
def rec(source, name, quotes, inputs=('A',), outputs=('B',)):
    a = {'name': name, 'input': list(inputs), 'output': list(outputs), 'evidence': list(quotes),
         'evidence_details': {q: {'supports': ['reaction']} for q in quotes}}
    return stamp_sources({'source': source, 'annotation_result': a})


def test_stamp_sources_labels_every_quote_and_is_idempotent():
    r = rec('PMID:11111111', 'x', ['q1', 'q2'])
    d = r['annotation_result']['evidence_details']
    assert d['q1']['source'] == d['q2']['source'] == 'PMID:11111111' and r['annotation_result']['sources'] == ['PMID:11111111']
    d['q1']['source'] = 'PMID:99999999'                    # an existing label is kept
    stamp_sources(r)
    assert d['q1']['source'] == 'PMID:99999999'


def test_an_exact_duplicate_from_another_paper_is_kept_not_dropped():
    a = rec('PMID:11111111', 'Parkin binds Ub', ['quote from paper one'])
    b = rec('PMID:22222222', 'Parkin Binds UB', ['quote from paper two'])      # same reaction (case differs only)
    c = rec('pink1.pdf', 'something else', ['unrelated'], inputs=('X',), outputs=('Y',))
    unique = collapse_exact_duplicates([a, b, c])
    assert len(unique) == 3                                                   # nothing is dropped, nothing crosses papers here
    assert unique[0]['annotation_result']['evidence'] == ['quote from paper one']
    assert unique[1]['annotation_result']['evidence'] == ['quote from paper two']
    assert unique[1]['annotation_result']['evidence_details']['quote from paper two']['source'] == 'PMID:22222222'
    assert record_pmids(unique[0]) == ['11111111'] and record_pmids(unique[1]) == ['22222222']
    assert record_pmids(unique[2]) == [] and record_sources(unique[2]) == ['pink1.pdf']       # a PDF has no PMID


def test_the_same_paper_twice_still_folds_its_quotes_without_repeating_the_source():
    unique = collapse_exact_duplicates([rec('PMID:11111111', 'r', ['q1']), rec('PMID:11111111', 'r', ['q2'])])
    ka = unique[0]['annotation_result']
    assert len(unique) == 1 and ka['evidence'] == ['q1', 'q2'] and ka['sources'] == ['PMID:11111111']


# ── each quote is verified against ITS paper ───────────────────────────────
def test_quotes_of_a_reaction_merged_across_papers_are_verified_against_their_own_paper():
    paper_one = PaperText.from_pages('11111111', ['Parkin binds phosphorylated ubiquitin in cells.'])
    paper_two = PaperText.from_pages('22222222', ['PINK1 phosphorylates ubiquitin at serine 65.'])
    papers = {'PMID:11111111': paper_one, 'PMID:22222222': paper_two}
    one = rec('PMID:11111111', 'r', ['Parkin binds phosphorylated ubiquitin in cells.'])
    two = rec('PMID:22222222', 'r', ['PINK1 phosphorylates ubiquitin at serine 65.'])
    one['annotation_result']['evidence'] += two['annotation_result']['evidence']      # as if the cross-paper stage had merged them
    one['annotation_result']['evidence_details'].update(two['annotation_result']['evidence_details'])
    merged = [one]
    store = EvidenceStore()
    stats = attach_evidence(merged, store, papers.get)
    assert stats == {'verified': 2, 'unverified': 0, 'rejected': 0}      # against ONE paper the second quote would be rejected
    by_quote = {e.quote: e for e in store.all()}
    assert by_quote['Parkin binds phosphorylated ubiquitin in cells.'].pmid == '11111111'
    assert by_quote['PINK1 phosphorylates ubiquitin at serine 65.'].pmid == '22222222'
    assert {e.source for e in store.all()} == {'PMID:11111111', 'PMID:22222222'}
    assert all(e.verified in (Verification.EXACT, Verification.FUZZY) for e in store.all())


def test_a_quote_without_a_source_label_falls_back_to_the_records_source():
    paper = PaperText.from_pages('11111111', ['Parkin binds phosphorylated ubiquitin in cells.'])
    r = {'source': 'PMID:11111111', 'annotation_result': {'name': 'r', 'evidence': ['Parkin binds phosphorylated ubiquitin in cells.'],
                                                     'evidence_details': {}}}          # an old extraction file: never stamped
    store = EvidenceStore()
    assert attach_evidence([r], store, {'PMID:11111111': paper}.get)['verified'] == 1
    assert store.all()[0].source == 'PMID:11111111'


def test_two_pdfs_with_the_same_quote_are_two_pieces_of_evidence():
    a = PaperText.from_pages(None, ['Parkin binds ubiquitin.'])
    b = PaperText.from_pages(None, ['Parkin binds ubiquitin.'])
    r1, r2 = rec('one.pdf', 'r1', ['Parkin binds ubiquitin.']), rec('two.pdf', 'r2', ['Parkin binds ubiquitin.'], inputs=('X',))
    store = EvidenceStore()
    attach_evidence([r1, r2], store, {'one.pdf': a, 'two.pdf': b}.get)
    assert len(store.all()) == 2 and {e.source for e in store.all()} == {'one.pdf', 'two.pdf'}     # no PMID to tell them apart


# ── verdict cache ──────────────────────────────────────────────────────────
def test_cache_key_ignores_pair_order_and_changes_with_model_thinking_or_rules():
    k = VerdictCache.key
    base = k('m1', True, 'rules1', 'brief A', 'brief B')
    assert base == k('m1', True, 'rules1', 'brief B', 'brief A')
    assert len({base, k('m2', True, 'rules1', 'brief A', 'brief B'), k('m1', False, 'rules1', 'brief A', 'brief B'),
                k('m1', True, 'rules2', 'brief A', 'brief B'), k('m1', True, 'rules1', 'brief A', 'brief C')}) == 5


def test_cache_survives_a_restart_and_a_final_verdict_replaces_the_first_vote(tmp_path):
    path = str(tmp_path / 'sub' / 'judge.jsonl')
    c = VerdictCache(path)
    c.put('k1', True, 0.7)                                    # first vote, shaky
    assert c.get('k1') == {'k': 'k1', 'same': True, 'conf': 0.7, 'final': False}
    c.put('k1', False, 0.7, final=True)                       # settled by the re-vote
    c.put('k2', False, 0.99)
    again = VerdictCache(path)                                # a new process
    assert again.get('k1')['same'] is False and again.get('k1')['final'] is True and len(again) == 2


def test_failed_calls_are_never_cached_and_a_torn_line_is_ignored(tmp_path):
    path = str(tmp_path / 'judge.jsonl')
    c = VerdictCache(path)
    c.put('bad', False, None)                                 # a failed verdict has no confidence
    assert len(c) == 0
    c.put('ok', True, 0.9)
    with open(path, 'a') as f:
        f.write('{"k": "half"')                               # a write cut short
    assert len(VerdictCache(path)) == 1


def test_a_cache_without_a_path_remembers_nothing():
    c = VerdictCache(None)
    c.put('k', True, 0.9)
    assert c.get('k') is None and len(c) == 0


# ── merging across papers ──────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from curator_llm.services.dedup_prefilter import may_be_duplicates  # noqa: E402
from curator_llm.services.merge_support import TfidfSimilarity, merge_across_papers, signature  # noqa: E402


def R(source, name, inputs, outputs, catalyst=None):
    a = {'name': name, 'input': inputs, 'output': outputs, 'merged_names': [name], 'sources': [source]}
    if catalyst:
        a['catalystActivity'] = {'catalyst': catalyst}
    return {'source': source, 'annotation_result': a}


def combine(rep, r):
    a, b = rep['annotation_result'], r['annotation_result']
    return {'source': rep['source'], 'annotation_result': {**a, 'merged_names': a['merged_names'] + b['merged_names'],
            'sources': list(dict.fromkeys(a['sources'] + b['sources']))}}


class FakeJudge:
    """Same when the first words of the two names match; records every pair it was asked about."""

    def __init__(self, conf=0.95, fail=False):
        self.calls, self.conf, self.fail = [], conf, fail

    def __call__(self, rep, r):
        self.calls.append((rep['annotation_result']['name'], r['annotation_result']['name']))
        if self.fail:
            raise RuntimeError('model down')
        same = rep['annotation_result']['name'].split()[0] == r['annotation_result']['name'].split()[0]
        return same, self.conf


def run_across(records, judge, **kw):
    sim = TfidfSimilarity([r['annotation_result'] for r in records])
    kw.setdefault('signature', lambda r: signature(r['annotation_result']))
    return merge_across_papers(records, prefilter=lambda r, g: may_be_duplicates(r['annotation_result'], g['annotation_result']),
                               similar=lambda r, g: sim(r['annotation_result'], g['annotation_result']),
                               judge=judge, combine=combine, **kw)


def test_a_reaction_of_a_later_paper_joins_the_matching_group_and_the_rest_stay_apart():
    recs = [R('PMID:11111111', 'alpha one', ['Parkin', 'Ub'], ['Parkin-Ub']), R('PMID:11111111', 'beta one', ['PINK1'], ['PINK1 p']),
            R('PMID:22222222', 'alpha two', ['Parkin', 'Ub'], ['Parkin-Ub']), R('PMID:22222222', 'gamma two', ['MFN2'], ['MFN2 Ub'])]
    j = FakeJudge()
    groups, stats = run_across(recs, j)
    assert sorted(len(g['members']) for g in groups) == [1, 1, 2] and stats['merged'] == 1
    merged = next(g for g in groups if len(g['members']) == 2)
    assert merged['rep']['annotation_result']['sources'] == ['PMID:11111111', 'PMID:22222222'] and merged['conf'] == 0.95
    assert merged['rep']['annotation_result']['merged_names'] == ['alpha one', 'alpha two']
    assert j.calls == [('alpha one', 'alpha two')]       # gamma shares no entity with any group, so it cost no call


def test_reactions_of_the_same_paper_are_never_compared_here():
    recs = [R('PMID:11111111', 'alpha one', ['Parkin'], ['Parkin p']), R('PMID:11111111', 'alpha uno', ['Parkin'], ['Parkin p'])]
    j = FakeJudge()
    groups, stats = run_across(recs, j)
    assert len(groups) == 2 and j.calls == [] and stats['merged'] == 0          # the within-paper merge already decided


def test_candidates_are_asked_best_first_stop_at_the_first_same_and_stop_at_the_cap():
    a_paper = [R('PMID:11111111', f'n{i} x', ['Parkin', f'Ent{i}'], ['Parkin p']) for i in range(7)]     # 7 groups, all share Parkin
    # the best-matching candidate shares the rare entity Ent3; only 'n3' will be called the same
    j = FakeJudge()
    hit = R('PMID:22222222', 'n3 y', ['Parkin', 'Ent3'], ['Parkin p'])
    groups, _ = run_across(a_paper + [hit], j, reconcile=False)
    assert j.calls[0] == ('n3 x', 'n3 y') and len(j.calls) == 1                  # best candidate first, then it stopped
    j2 = FakeJudge()
    miss = R('PMID:22222222', 'zzz y', ['Parkin', 'Ent9'], ['Parkin p'])
    run_across(a_paper + [miss], j2, reconcile=False, max_candidates=5)
    assert len(j2.calls) == 5                                                    # nothing matched: the cap, not all 7
    j3 = FakeJudge()
    run_across(a_paper + [miss], j3, max_candidates=2, reconcile=False)
    assert len(j3.calls) == 2


def test_a_group_takes_one_reaction_per_paper_the_more_sure_one_wins():
    class Conf(FakeJudge):
        def __call__(self, rep, r):
            same, _ = super().__call__(rep, r)
            return same, 0.9 if r['annotation_result']['name'].endswith('b') else 0.7
    recs = [R('PMID:11111111', 'alpha a', ['Parkin'], ['Parkin p']),
            R('PMID:22222222', 'alpha a2', ['Parkin'], ['Parkin p']), R('PMID:22222222', 'alpha b', ['Parkin'], ['Parkin p'])]
    groups, stats = run_across(recs, Conf())
    assert stats['merged'] == 1 and sorted(len(g['members']) for g in groups) == [1, 2]
    joined = next(g for g in groups if len(g['members']) == 2)
    assert joined['rep']['annotation_result']['merged_names'] == ['alpha a', 'alpha b']      # 0.9 beat 0.7


def test_a_third_paper_is_compared_with_the_merged_group_and_joins_it():
    recs = [R('PMID:11111111', 'alpha 1', ['Parkin'], ['Parkin p']), R('PMID:22222222', 'alpha 2', ['Parkin'], ['Parkin p']),
            R('PMID:33333333', 'alpha 3', ['Parkin'], ['Parkin p'])]
    j = FakeJudge()
    groups, stats = run_across(recs, j)
    assert len(groups) == 1 and stats['merged'] == 2
    assert groups[0]['rep']['annotation_result']['sources'] == ['PMID:11111111', 'PMID:22222222', 'PMID:33333333']
    assert j.calls == [('alpha 1', 'alpha 2'), ('alpha 1', 'alpha 3')]            # one call per later paper, not per pair of reactions


def test_a_failed_judge_call_is_not_a_merge_and_loses_nothing():
    recs = [R('PMID:11111111', 'alpha 1', ['Parkin'], ['Parkin p']), R('PMID:22222222', 'alpha 2', ['Parkin'], ['Parkin p'])]
    groups, stats = run_across(recs, FakeJudge(fail=True))
    assert len(groups) == 2 and stats['merged'] == 0 and sum(len(g['members']) for g in groups) == 2


def test_judging_a_paper_concurrently_gives_the_same_groups_as_one_at_a_time():
    papers = [[R(f'PMID:{p}0000000', f'n{i} p{p}', ['Parkin', f'Ent{i}'], ['Parkin p']) for i in range(6)] for p in (1, 2, 3)]
    recs = [r for paper in papers for r in paper]
    serial, _ = run_across(recs, FakeJudge())
    with ThreadPoolExecutor(8) as pool:
        parallel, _ = run_across(recs, FakeJudge(), map_fn=pool.map)
    assert [g['members'] for g in serial] == [g['members'] for g in parallel] and len(serial) == 6


def test_a_rare_shared_entity_ranks_above_a_common_one():
    pool = [R('p', f'x{i}', ['Parkin', 'Ub'], ['Parkin p']) for i in range(10)] + [R('p', 'rare', ['Parkin', 'MFN1'], ['Parkin p'])]
    sim = TfidfSimilarity([r['annotation_result'] for r in pool])
    query = R('q', 'q', ['Parkin', 'MFN1'], ['Parkin p'])['annotation_result']
    assert sim(query, pool[-1]['annotation_result']) > sim(query, pool[0]['annotation_result'])


def test_an_identical_reaction_in_another_paper_joins_its_group_without_a_judge_call():
    recs = [R('PMID:11111111', 'alpha one', ['Parkin', 'Ub'], ['Parkin-Ub']), R('PMID:22222222', 'alpha one', ['Parkin', 'Ub'], ['Parkin-Ub'])]
    j = FakeJudge()
    groups, stats = run_across(recs, j)
    assert len(groups) == 1 and j.calls == [] and stats['exact'] == 1 and groups[0]['conf'] == 1.0
    assert groups[0]['rep']['annotation_result']['sources'] == ['PMID:11111111', 'PMID:22222222']


def test_two_groups_for_the_same_reaction_are_reconciled_at_the_end():
    # paper 3's reaction can match only ONE of two groups made earlier; the groups themselves were never compared.
    # alpha-A (paper 1) and alpha-B (paper 2) share no entity words pairwise at first sight, so they start as two groups
    recs = [R('PMID:11111111', 'alpha a', ['Parkin', 'EntA'], ['Parkin p']), R('PMID:22222222', 'alpha b', ['Parkin', 'EntB'], ['Parkin p'])]

    # during the incremental pass the second reaction DOES match the first (both share Parkin): so make them not match then
    class Picky(FakeJudge):
        def __call__(self, rep, r):
            first = not self.calls
            self.calls.append((rep['annotation_result']['name'], r['annotation_result']['name']))
            return (not first), 0.9                    # the first question (incremental pass) says no, the next (reconcile) says yes
    j = Picky()
    groups, stats = run_across(recs, j)
    assert len(groups) == 1 and stats['reconciled'] == 1 and len(j.calls) == 2


def test_reconcile_never_joins_two_groups_that_share_a_paper():
    recs = [R('PMID:11111111', 'alpha a', ['Parkin', 'EntA'], ['Parkin p']), R('PMID:11111111', 'alpha b', ['Parkin', 'EntB'], ['Parkin p'])]
    j = FakeJudge()
    groups, stats = run_across(recs, j)                # two reactions of ONE paper: decided by the within-paper merge, not here
    assert len(groups) == 2 and j.calls == [] and stats['reconciled'] == 0


def test_reconcile_can_be_switched_off_and_reconciling_ends():
    recs = [R(f'PMID:{p}0000000', f'alpha {p}', ['Parkin', f'Ent{p}'], ['Parkin p']) for p in (1, 2, 3, 4)]
    j = FakeJudge()
    groups, stats = run_across(recs, j)                # everything matches everything: one group, and the loop terminates
    assert len(groups) == 1
    j2 = FakeJudge()
    _, stats2 = run_across(recs, j2, reconcile=False)
    assert stats2['reconciled'] == 0


def test_the_cap_is_counted_when_it_leaves_candidates_unasked():
    a_paper = [R('PMID:11111111', f'n{i} x', ['Parkin', f'Ent{i}'], ['Parkin p']) for i in range(7)]
    miss = R('PMID:22222222', 'zzz y', ['Parkin', 'Ent9'], ['Parkin p'])
    assert run_across(a_paper + [miss], FakeJudge(), reconcile=False, max_candidates=5)[1]['capped'] == 1   # 7 candidates, asked 5, no match
    assert run_across(a_paper + [miss], FakeJudge(), reconcile=False)[1]['capped'] == 0                 # the default (10) covers all 7
    hit = R('PMID:22222222', 'n3 y', ['Parkin', 'Ent3'], ['Parkin p'])
    assert run_across(a_paper + [hit], FakeJudge(), reconcile=False, max_candidates=5)[1]['capped'] == 0   # it matched: nothing was lost
