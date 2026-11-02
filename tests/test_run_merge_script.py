"""Run the real run_merge.py as a script with a stub in place of the Anthropic client (every call answers "same").

run_merge.py executes its whole driver when it is run, so the unit tests of its helper modules cannot see a mistake in
the script itself (an overwritten name once crashed it after clustering). This runs it end to end, offline."""
import json
import os
import subprocess
import sys
import textwrap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

STUB = textwrap.dedent('''
    import json, os
    class _Block:
        type = 'text'
        def __init__(self, t): self.text = t
    class _Usage: input_tokens = 1; output_tokens = 1; cache_read_input_tokens = 0; cache_creation_input_tokens = 0
    class _Msg:
        def __init__(self, t): self.content = [_Block(t)]; self.usage = _Usage(); self.stop_reason = 'end_turn'
    class _Messages:
        def create(self, **kw):
            with open(os.environ['FAKE_CALL_LOG'], 'a') as f: f.write('x\\n')
            return _Msg(json.dumps({'same': True, 'confidence': 0.95, 'reason': 'stub'}))
    class Anthropic:
        def __init__(self, *a, **k): self.messages = _Messages()
''')


def record(source, name, quote, inputs, outputs, catalyst='PINK1'):
    return {'source': source, 'chunk_index': 0, 'annotation_result': {
        'name': name, 'reactionType': 'transition', 'input': inputs, 'output': outputs,
        'catalystActivity': {'catalyst': catalyst}, 'summation': [{'text': name, 'literatureReference': []}],
        'evidence': [quote], 'evidence_details': {quote: {'supports': ['reaction']}}, 'confidence': 0.8}}


def run(tmp_path, extraction, out, calls, extra=()):
    stub = tmp_path / 'stub' / 'anthropic'
    stub.mkdir(parents=True, exist_ok=True)
    (stub / '__init__.py').write_text(STUB)
    env = dict(os.environ, PYTHONPATH=str(tmp_path / 'stub'), FAKE_CALL_LOG=str(calls), MERGE_CACHE_DIR=str(tmp_path / 'cache'),
               ANTHROPIC_API_KEY='stub')
    env.pop('MERGE_CACHE', None)
    files = [str(x) for x in (extraction if isinstance(extraction, (list, tuple)) else [extraction])]
    p = subprocess.run([sys.executable, 'run_merge.py', *files, '--out', str(out), '--no-validate', *extra], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    return p.stdout


def test_merge_across_papers_end_to_end_and_the_cache_makes_a_rerun_free(tmp_path):
    a1 = record('PMID:11111111', 'PINK1 phosphorylates ubiquitin', 'quote one', ['Ub', 'ATP'], ['pUb', 'ADP'])
    a2 = record('PMID:11111111', 'PINK1 phosphorylates Ub at Ser65', 'quote two', ['Ub', 'ATP'], ['pUb', 'ADP'])
    dup = record('PMID:22222222', 'PINK1 phosphorylates ubiquitin', 'quote from the other paper', ['Ub', 'ATP'], ['pUb', 'ADP'])
    other = record('PMID:22222222', 'Parkin binds pUb', 'quote four', ['Parkin', 'pUb'], ['Parkin:pUb'], catalyst=None)
    ext = tmp_path / 'x_extraction.json'
    ext.write_text(json.dumps([a1, a2, dup, other]))
    calls = tmp_path / 'calls.log'

    out1 = run(tmp_path, ext, tmp_path / 'm1.json', calls)
    merged = [r['annotation_result'] for r in json.load(open(tmp_path / 'm1.json'))]
    assert len(merged) == 2                                                      # PINK1 reactions in one, Parkin binding in the other
    pink = next(m for m in merged if 'PINK1' in m['name'])
    assert pink['sources'] == ['PMID:11111111', 'PMID:22222222']                 # the duplicate from paper two was folded in, not dropped
    assert pink['evidence_details']['quote from the other paper']['source'] == 'PMID:22222222'
    assert pink['evidence_details']['quote one']['source'] == 'PMID:11111111'
    first_calls = len(calls.read_text().split())
    assert first_calls > 0 and 'reused from the cache' not in out1

    out2 = run(tmp_path, ext, tmp_path / 'm2.json', calls)
    new_calls = len(calls.read_text().split()) - first_calls
    assert 'reused from the cache' in out2                                       # every first-pass verdict came from disk
    assert new_calls < first_calls                                               # only the consolidation call is left
    assert json.load(open(tmp_path / 'm2.json')) == json.load(open(tmp_path / 'm1.json'))


def test_several_extraction_files_are_merged_across_papers_unless_told_not_to(tmp_path):
    one = tmp_path / 'one_extraction.json'
    two = tmp_path / 'two_extraction.json'
    one.write_text(json.dumps([record('PMID:11111111', 'PINK1 phosphorylates ubiquitin', 'quote from one', ['Ub', 'ATP'], ['pUb', 'ADP']),
                               record('PMID:11111111', 'PINK1 phosphorylates Ub at Ser65', 'another from one', ['Ub', 'ATP'], ['pUb', 'ADP'])]))
    two.write_text(json.dumps([record('PMID:22222222', 'PINK1 phosphorylation of ubiquitin', 'quote from two', ['Ub', 'ATP'], ['pUb', 'ADP']),
                               record('PMID:22222222', 'PINK1 makes phospho-Ub', 'another from two', ['Ub', 'ATP'], ['pUb', 'ADP'])]))
    calls = tmp_path / 'calls.log'

    out = run(tmp_path, [one, two], tmp_path / 'across.json', calls)
    merged = [r['annotation_result'] for r in json.load(open(tmp_path / 'across.json'))]
    assert len(merged) == 1 and '[cross]' in out                                          # four reactions, two papers -> one
    a = merged[0]
    assert a['sources'] == ['PMID:11111111', 'PMID:22222222'] and len(a['merged_names']) == 4
    assert {d['source'] for d in a['evidence_details'].values()} == {'PMID:11111111', 'PMID:22222222'}
    assert a['evidence_details']['quote from two']['source'] == 'PMID:22222222'          # each quote keeps its own paper
    assert a['cross_paper_merge_confidence'] == 0.95

    run(tmp_path, [one, two], tmp_path / 'apart.json', calls, extra=['--no-cross-paper'])
    apart = [r['annotation_result'] for r in json.load(open(tmp_path / 'apart.json'))]
    assert len(apart) == 2 and all(len(x['sources']) == 1 for x in apart)                 # within each paper only, as before


def test_merging_several_files_needs_an_output_name(tmp_path):
    a, b = tmp_path / 'a_extraction.json', tmp_path / 'b_extraction.json'
    a.write_text('[]'); b.write_text('[]')
    p = subprocess.run([sys.executable, 'run_merge.py', str(a), str(b)], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and '--out' in (p.stdout + p.stderr)
