import json
import os

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from curator_llm.adapters.chat_anthropic import AnthropicChatModel
from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.models.reactome import EwasSpec, ReactionSpec, ReactomeDraft
from curator_llm.models.usage import UsageEntry, entry_from_counts, summarize
from curator_llm.ports.chat_model import ModelTurn, TokenUsage
from curator_llm.ports.extractor import ExtractionResult
from curator_llm.ports.pipeline import PipelineInput, PipelineResult
from curator_llm.services.draft_builder import DraftExtraction, LlmEntity, LlmReaction, build_draft
from curator_llm.services.pipeline_default import DefaultPipeline
from curator_llm.services.qa import qa_reaction
from curator_llm.services.snapshots import SnapshotPipeline, SnapshotStore, snapshot_key
from tests.fakes.ports import FakeAuthProvider, FakeInstanceLookup, SyncJobRunner
from tests.test_chat import ScriptedModel, chat, final, tool
from tests.test_edit_api import H, TOKENS, Pipe


def U(i=0, o=0, cr=0, cw=0):
    return TokenUsage(i, o, cr, cw)


class TestSummary:
    def test_per_step_totals_come_in_pipeline_order_and_skip_steps_that_did_not_run(self):
        entries = [UsageEntry(step='chat', calls=1, input_tokens=100, output_tokens=20),
                   UsageEntry(step='extraction', calls=20, input_tokens=142247, output_tokens=60586, cache_read_tokens=5),
                   UsageEntry(step='chat', calls=2, input_tokens=50, output_tokens=10, cache_write_tokens=3),
                   UsageEntry(step='merge', calls=231, input_tokens=390872, output_tokens=23737)]
        s = summarize(entries)
        assert [x['step'] for x in s['steps']] == ['extraction', 'merge', 'chat']
        chat_row = s['steps'][2]
        assert chat_row['calls'] == 3 and chat_row['input_tokens'] == 150 and chat_row['output_tokens'] == 30
        assert chat_row['cache_write_tokens'] == 3 and chat_row['total_tokens'] == 180
        t = s['totals']
        assert t['calls'] == 254 and t['input_tokens'] == 142247 + 390872 + 150 and t['cache_read_tokens'] == 5
        assert t['total_tokens'] == t['input_tokens'] + t['output_tokens']

    def test_nothing_recorded_summarises_to_nothing(self):
        zero = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cache_read_tokens': 0, 'cache_write_tokens': 0, 'total_tokens': 0}
        assert summarize([]) == {'steps': [], 'totals': zero, 'spent_now': zero}

    def test_script_output_counts_become_an_entry(self):
        e = entry_from_counts('merge', {'calls': 231, 'input': 390872, 'output': 23737, 'cache_read': 1, 'cache_write': 2})
        assert (e.step, e.calls, e.input_tokens, e.output_tokens, e.cache_read_tokens, e.cache_write_tokens) == ('merge', 231, 390872, 23737, 1, 2)


class TestSavedFlag:
    ENTRIES = [UsageEntry(step='extraction', calls=20, input_tokens=1000, output_tokens=200),
               UsageEntry(step='draft', calls=1, input_tokens=300, output_tokens=40),
               UsageEntry(step='qa', calls=1, input_tokens=50, output_tokens=5),
               UsageEntry(step='chat', calls=2, input_tokens=70, output_tokens=7)]

    def test_nothing_is_flagged_saved_for_a_live_run(self):
        s = summarize(self.ENTRIES)
        assert [r['saved'] for r in s['steps']] == [False] * 4 and s['spent_now'] == s['totals']

    def test_a_replay_flags_only_the_steps_that_produce_the_annotation_and_spent_now_leaves_them_out(self):
        s = summarize(self.ENTRIES, saved_pipeline=True)
        assert {r['step']: r['saved'] for r in s['steps']} == {'extraction': True, 'draft': True, 'qa': False, 'chat': False}
        assert s['totals']['input_tokens'] == 1000 + 300 + 50 + 70
        assert s['spent_now']['input_tokens'] == 50 + 70 and s['spent_now']['calls'] == 3


class TestExtractionSteps:
    def test_each_script_step_keeps_its_own_usage_as_well_as_the_sum(self, tmp_path):
        import subprocess
        from curator_llm.services.extraction import SubprocessExtractor
        lines = {'run_extraction.py': '20 call(s), tokens in 1,000 / out 200\n', 'run_merge.py': '231 call(s), tokens in 5,000 / out 300, cache r7/w9\n'}

        def run(cmd, cwd, timeout):
            name = os.path.basename(cmd[1])
            if name == 'run_extraction.py':
                json.dump([{'annotation_result': {'name': 'a'}}], open(os.path.join(str(tmp_path), 'g_pmid1_extraction.json'), 'w'))
            if name == 'run_merge.py':
                json.dump([{'annotation_result': {'name': 'a'}}], open(os.path.join(str(tmp_path), 'g_pmid1_merged.json'), 'w'))
            return subprocess.CompletedProcess(cmd, 0, lines[name], '')
        ex = SubprocessExtractor(root=str(tmp_path), results_dir=str(tmp_path), run=run, stem=lambda s, g: f'{g}_pmid{s}')
        res = ex.extract('1', None, 'g')
        assert res.step_usage['extraction']['input'] == 1000 and res.step_usage['merge']['cache_read'] == 7
        assert 'review' not in res.step_usage
        assert res.usage['input'] == 6000 and res.usage['calls'] == 251


class TestDraftStep:
    @staticmethod
    def reporting_model(model_name='claude-x', usage=None):
        usage = usage or {'input_tokens': 1200, 'output_tokens': 300, 'total_tokens': 1500,
                          'input_token_details': {'cache_read': 50, 'cache_creation': 10}}
        out = DraftExtraction(entities=[LlmEntity(key='a', kind='protein', name='PINK1')],
                              reactions=[LlmReaction(source_index=0, name='r', inputs=['a'])])

        class M:
            def with_structured_output(self, schema):
                return self

            def invoke(self, prompt, config=None):
                msg = AIMessage(content='', usage_metadata=usage, response_metadata={'model_name': model_name})
                for cb in (config or {}).get('callbacks', []):          # what LangChain does when the model call ends
                    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
                return out
        return M()

    REACTIONS = [{'source': 'PMID:1', 'annotation_result': {'name': 'r', 'evidence_ids': ['ev-001']}}]

    def test_the_draft_call_reports_its_tokens_by_model(self):
        used = []
        build_draft('PINK1', self.REACTIONS, None, self.reporting_model(), usage=used)
        assert len(used) == 1
        e = used[0]
        assert (e.step, e.model, e.calls, e.input_tokens, e.output_tokens, e.cache_read_tokens, e.cache_write_tokens) == \
               ('draft', 'claude-x', 1, 1200, 300, 50, 10)

    def test_it_works_without_asking_for_usage_and_with_a_model_that_reports_none(self):
        assert build_draft('PINK1', self.REACTIONS, None, self.reporting_model())[0].reactions
        used = []
        build_draft('PINK1', self.REACTIONS, None, TestDraftStep.silent_model(), usage=used)
        assert used == []

    @staticmethod
    def silent_model():
        out = DraftExtraction(entities=[LlmEntity(key='a', kind='protein', name='PINK1')], reactions=[LlmReaction(source_index=0, name='r', inputs=['a'])])

        class M:
            def with_structured_output(self, schema):
                return self

            def invoke(self, prompt, config=None):
                return out
        return M()


class TestPipeline:
    def test_run_lists_what_the_script_steps_spent_then_the_draft_and_skips_steps_that_did_not_run(self, monkeypatch):
        from curator_llm.services.paper_text import PaperText
        monkeypatch.setattr('curator_llm.services.pipeline_default.load_paper_text',
                            lambda src: PaperText.from_pages('1', ['Results\nTcPINK1 phosphorylates Ub.\n']))

        class Ext:
            def extract(self, pmid, pdf, gene):
                return ExtractionResult(ok=True, reactions=[{'source': 'PMID:1', 'annotation_result': {'name': 'r', 'input': ['Ub'], 'evidence': ['TcPINK1 phosphorylates Ub']}}],
                                        step_usage={'extraction': {'calls': 20, 'input': 1000, 'output': 200, 'cache_read': 0, 'cache_write': 0},
                                                    'merge': {'calls': 5, 'input': 400, 'output': 50, 'cache_read': 0, 'cache_write': 0},
                                                    'review': {'calls': 0, 'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0}})
        pipe = DefaultPipeline(FakeInstanceLookup([]), model=TestDraftStep.reporting_model(), extractor=Ext())
        out = pipe.run(PipelineInput(pmid='1'), lambda m: None)
        assert [u.step for u in out.usage] == ['extraction', 'merge', 'draft']
        assert out.usage[0].input_tokens == 1000 and out.usage[2].input_tokens == 1200
        assert out.replayed is False


def result_with_usage():
    d = ReactomeDraft()
    d.participants = {'p': EwasSpec(key='p', name='PINK1', uniprot='Q9BXM7')}
    d.reactions = [ReactionSpec(key='r0', name='x', inputs=['p'], pmids=['1'])]
    return PipelineResult(d, [], [], None, [], [entry_from_counts('extraction', {'calls': 20, 'input': 142247, 'output': 60586}),
                                               UsageEntry(step='draft', calls=1, input_tokens=9000, output_tokens=2000, model='claude-x')])


class TestSnapshots:
    spec = PipelineInput(pmid='24751536', focus='PINK1')

    def test_usage_survives_a_save_and_load_and_a_replay_says_it_is_the_original_runs(self, tmp_path):
        class Inner:
            def run(self, spec, report):
                return result_with_usage()
        p = SnapshotPipeline(Inner(), SnapshotStore(str(tmp_path)), 'replay')
        first = p.run(self.spec, lambda m: None)
        assert first.replayed is False and len(first.usage) == 2
        second = p.run(self.spec, lambda m: None)
        assert second.replayed is True
        assert [(u.step, u.input_tokens) for u in second.usage] == [('extraction', 142247), ('draft', 9000)]
        assert second.usage[1].model == 'claude-x'

    def test_a_snapshot_saved_before_usage_existed_still_loads(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        store.save('k', result_with_usage())
        doc = json.loads((tmp_path / 'k.json').read_text())
        del doc['usage']
        (tmp_path / 'k.json').write_text(json.dumps(doc))
        assert store.load('k').usage == []


class TestQaUsage:
    def draft(self):
        return result_with_usage().draft

    class Reviewer:
        def __init__(self, text, usage):
            self.text, self.usage = text, usage

        def turn(self, system, messages, tools, on_text):
            return ModelTurn(self.text, usage=self.usage)

    def test_the_review_reports_what_it_spent(self):
        r = qa_reaction(self.draft(), 'r0', [], self.Reviewer('{"verdict": "ok", "score": 0.9, "findings": []}', U(700, 80, 5, 6)))
        assert r.usage == {'input_tokens': 700, 'output_tokens': 80, 'cache_read_tokens': 5, 'cache_write_tokens': 6}

    def test_a_reply_that_cannot_be_parsed_still_cost_tokens(self):
        r = qa_reaction(self.draft(), 'r0', [], self.Reviewer('not json', U(700, 80)))
        assert not r.llm_used and r.usage['input_tokens'] == 700

    def test_rule_checks_alone_spend_nothing(self):
        assert qa_reaction(self.draft(), 'r0', [], None).usage is None


def turn_with_usage(model_turn, usage):
    model_turn.usage = usage
    return model_turn


def env(model):
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(), chat_model=model)
    c = TestClient(app)
    sid = c.post('/api/llm/sessions', json={'pmid': '24751536'}, headers=H()).json()['sessionId']
    return c, sid


class TestChatUsage:
    def test_a_turn_sums_every_model_call_and_is_recorded_and_reported(self):
        model = ScriptedModel([turn_with_usage(tool('list_reactions'), U(1000, 50)),
                               turn_with_usage(tool('get_reaction', key='r0'), U(1500, 70, cr=400)),
                               turn_with_usage(final('done'), U(1800, 90))])
        c, sid = env(model)
        events = chat(c, sid, 'what is here?')
        done = events[-1]
        assert done['event'] == 'done'
        assert done['data']['usage'] == {'calls': 3, 'input_tokens': 4300, 'output_tokens': 210, 'cache_read_tokens': 400, 'cache_write_tokens': 0}
        u = c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()
        chat_rows = [e for e in u['entries'] if e['step'] == 'chat']
        assert len(chat_rows) == 1 and chat_rows[0]['detail'] == 'turn 1' and chat_rows[0]['calls'] == 3
        assert u['totals']['input_tokens'] == 4300

    def test_each_turn_is_its_own_entry_and_the_step_totals_add_up(self):
        model = ScriptedModel([turn_with_usage(final('one'), U(100, 10)), turn_with_usage(final('two'), U(200, 20))])
        c, sid = env(model)
        chat(c, sid, 'first')
        chat(c, sid, 'second')
        u = c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()
        assert [e['detail'] for e in u['entries'] if e['step'] == 'chat'] == ['turn 1', 'turn 2']
        row = [s for s in u['steps'] if s['step'] == 'chat'][0]
        assert (row['calls'], row['input_tokens'], row['output_tokens']) == (2, 300, 30)

    def test_a_model_that_reports_no_usage_records_the_call_but_no_tokens(self):
        c, sid = env(ScriptedModel([final('hi')]))
        events = chat(c, sid, 'hello')
        assert events[-1]['data']['usage']['calls'] == 1 and events[-1]['data']['usage']['input_tokens'] == 0

    def test_a_turn_that_never_reached_the_model_records_nothing(self):
        c, sid = env(ScriptedModel([final('x')]))
        events = chat(c, sid, '')                                         # refused up front
        assert events[0]['event'] == 'error'
        assert c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()['entries'] == []


class TestReactionCheckUsage:
    def test_a_model_review_through_the_api_is_recorded_against_the_reaction_and_rule_checks_are_not(self):
        c, sid = env(ScriptedModel([ModelTurn('{"verdict": "ok", "score": 0.9, "findings": []}', usage=U(800, 60))]))
        c.post(f'/api/llm/sessions/{sid}/qa/r0?llm=false', headers=H())
        assert c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()['entries'] == []
        c.post(f'/api/llm/sessions/{sid}/qa/r0', headers=H())
        e = c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()['entries']
        assert len(e) == 1 and e[0]['step'] == 'qa' and e[0]['detail'] == 'r0' and e[0]['input_tokens'] == 800


class TestUsageEndpoint:
    def test_shape_privacy_and_a_session_with_nothing_recorded(self):
        c, sid = env(ScriptedModel([]))
        u = c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()
        assert set(u) == {'source', 'entries', 'steps', 'totals', 'spent_now'} and u['source'] == 'run' and u['steps'] == []
        assert c.get(f'/api/llm/sessions/{sid}/usage', headers=H('bob')).status_code == 404
        assert c.get(f'/api/llm/sessions/{sid}/usage').status_code == 401

    def test_a_replayed_session_reports_the_saved_runs_numbers_as_saved(self, tmp_path):
        spec = PipelineInput(pmid='24751536', focus='PINK1')
        SnapshotStore(str(tmp_path)).save(snapshot_key(spec), result_with_usage())

        class NoRun:
            def run(self, spec, report):
                raise AssertionError('must not run')
        app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(),
                         SnapshotPipeline(NoRun(), SnapshotStore(str(tmp_path)), 'replay'))
        c = TestClient(app)
        sid = c.post('/api/llm/sessions', json={'pmid': '24751536', 'focus': 'PINK1'}, headers=H()).json()['sessionId']
        u = c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()
        assert u['source'] == 'saved'
        assert [s['step'] for s in u['steps']] == ['extraction', 'draft']
        assert [s['saved'] for s in u['steps']] == [True, True]
        assert u['totals']['input_tokens'] == 142247 + 9000
        assert u['spent_now']['total_tokens'] == 0          # nothing was spent in this session yet


class TestAdapter:
    def test_the_anthropic_adapter_reports_the_providers_counts_including_cache(self):
        class B:
            def __init__(self, **k): self.__dict__.update(k)

        class Stream:
            text_stream = ['hi']
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get_final_message(self):
                return B(stop_reason='end_turn', content=[B(type='text', text='hi')],
                         usage=B(input_tokens=11, output_tokens=7, cache_read_input_tokens=3, cache_creation_input_tokens=None))

        class Client:
            class messages:
                @staticmethod
                def stream(**kw): return Stream()
        t = AnthropicChatModel(Client(), 'm', 10).turn('s', [], [], lambda d: None)
        assert t.usage == TokenUsage(11, 7, 3, 0)


class TestReplayThenLive:
    def test_chat_after_a_replay_is_live_while_the_replayed_steps_stay_saved(self, tmp_path):
        spec = PipelineInput(pmid='24751536', focus='PINK1')
        SnapshotStore(str(tmp_path)).save(snapshot_key(spec), result_with_usage())

        class NoRun:
            def run(self, spec, report):
                raise AssertionError('must not run')
        model = ScriptedModel([turn_with_usage(final('hi'), U(100, 10))])
        app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(),
                         SnapshotPipeline(NoRun(), SnapshotStore(str(tmp_path)), 'replay'), chat_model=model)
        c = TestClient(app)
        sid = c.post('/api/llm/sessions', json={'pmid': '24751536', 'focus': 'PINK1'}, headers=H()).json()['sessionId']
        chat(c, sid, 'hello')
        u = c.get(f'/api/llm/sessions/{sid}/usage', headers=H()).json()
        assert {s['step']: s['saved'] for s in u['steps']} == {'extraction': True, 'draft': True, 'chat': False}
        assert u['spent_now'] == {'calls': 1, 'input_tokens': 100, 'output_tokens': 10, 'cache_read_tokens': 0,
                                  'cache_write_tokens': 0, 'total_tokens': 110}
        assert u['totals']['input_tokens'] == 142247 + 9000 + 100
