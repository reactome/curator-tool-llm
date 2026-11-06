"""Estimated cost of model calls from the usage records, and summary.tsv's token and cost columns."""
import csv
import json

from curator_llm.pricing import cost, cost_of_steps, prices
from curator_llm.spreadsheet import Paper, write_summary


def test_cost_counts_input_output_and_cache_at_their_rates():
    m = {'input_tokens': 1_000_000, 'output_tokens': 1_000_000, 'cache_write_tokens': 1_000_000, 'cache_read_tokens': 1_000_000}
    # claude-sonnet-5: $2 in, $10 out, cache write 1.25 x input, cache read 0.1 x input
    assert round(cost(m, 'claude-sonnet-5'), 4) == round(2 + 10 + 2.5 + 0.2, 4)
    assert cost(m, 'no-such-model') is None


def test_prices_can_be_changed_without_code(monkeypatch):
    monkeypatch.setenv('LLM_PRICES', json.dumps({'claude-sonnet-5': {'input': 3.0}, 'my-model': {'input': 1, 'output': 2}}))
    p = prices()
    assert p['claude-sonnet-5']['input'] == 3.0 and p['claude-sonnet-5']['output'] == 10.0
    assert round(p['claude-sonnet-5']['cache_read'], 6) == 0.3 and p['my-model']['cache_write'] == 1.25


def test_steps_are_priced_by_the_model_their_role_uses_and_unknown_ones_listed(monkeypatch):
    monkeypatch.setenv('LLM_MODEL_QA', 'claude-haiku-5-5')
    steps = [{'step': 'extraction', 'calls': 2, 'input_tokens': 1_000_000, 'output_tokens': 0},
             {'step': 'qa', 'calls': 1, 'input_tokens': 1_000_000},
             {'step': 'review', 'calls': 1, 'input_tokens': 5}]                 # an OpenAI model: no price here
    t = cost_of_steps(steps)
    assert t['calls'] == 4 and round(t['usd'], 4) == round(2.0 + 0.1, 4) and t['unpriced'] == ['review']


def test_summary_has_cost_per_annotation_once_and_a_total(tmp_path):
    out = tmp_path
    usage = {'steps': [{'step': 'extraction', 'calls': 10, 'input_tokens': 500_000, 'output_tokens': 100_000,
                        'cache_read_tokens': 0, 'cache_write_tokens': 0}]}
    for name in ('001_a', 'grp'):
        json.dump({'n_reactions': 3, 'usage': usage}, open(out / f'{name}.session.json', 'w'))
        json.dump({'status': 'complete', 'papers': []}, open(out / f'{name}.progress.json', 'w'))
    papers = [Paper(row='1', pmid='1'), Paper(row='2', pmid='2'), Paper(row='3', pmid='3')]
    name_of = {'1': '001_a', '2': 'grp', '3': 'grp'}                            # rows 2 and 3 are one annotation
    t = write_summary(str(out / 'summary.tsv'), papers, name_of, str(out), None,
                      extra={'combining papers (hierarchy)': {'calls': 4, 'input_tokens': 1000, 'usd': 0.5}})
    rows = list(csv.DictReader(open(out / 'summary.tsv'), delimiter='\t'))
    per = 0.5 * 2 + 0.1 * 10                                                   # 0.5M in at $2, 0.1M out at $10
    assert rows[0]['cost USD'] == f'{per:.4f}' and rows[1]['cost USD'] == f'{per:.4f}' and rows[2]['cost USD'] == ''
    assert rows[3]['annotation'] == 'combining papers (hierarchy)' and rows[3]['calls'] == '4'
    assert rows[-1]['row'] == 'TOTAL' and rows[-1]['calls'] == str(10 + 10 + 4)
    assert rows[-1]['cost USD'] == f'{2 * per + 0.5:.2f}' and round(t['usd'], 4) == round(2 * per + 0.5, 4)
