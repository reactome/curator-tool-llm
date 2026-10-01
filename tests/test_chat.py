import json

from fastapi.testclient import TestClient

from curator_llm.adapters.sessions_memory import InMemorySessionStore
from curator_llm.api.app import create_app
from curator_llm.ports.chat_model import ModelTurn, ToolCall
from tests.fakes.ports import FakeAuthProvider, FakeInstanceLookup, FakeUniProt, SyncJobRunner
from tests.test_edit_api import Pipe, Q2, TOKENS, H
from curator_llm.services.resolvers import Resolver


class ScriptedModel:
    """Returns the scripted turns in order and records what it was sent."""

    def __init__(self, turns):
        self.turns, self.calls = list(turns), []

    def turn(self, system, messages, tools, on_text):
        self.calls.append({'system': system, 'messages': json.loads(json.dumps(messages)), 'tools': [t['name'] for t in tools]})
        t = self.turns.pop(0)
        for chunk in (t.text[:5], t.text[5:]) if t.text else ():
            on_text(chunk)
        return t


def tool(tool_name, **inp):
    return ModelTurn('', [ToolCall(f'tu_{tool_name}', tool_name, inp)], 'tool_use',
                     [{'type': 'tool_use', 'id': f'tu_{tool_name}', 'name': tool_name, 'input': inp}])


def final(text):
    return ModelTurn(text, [], 'end_turn', [{'type': 'text', 'text': text}])


CCCP = [{'op': 'add', 'path': '/reactions/r0/regulations/-', 'value': {'kind': 'positive', 'regulator': 'cccp', 'note': 'after CCCP'}}]


def setup(turns):
    model = ScriptedModel(turns)
    gk = [{'dbId': 17906, 'displayName': 'mitochondrial outer membrane', 'schemaClassName': 'Compartment'}]
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(),
                     resolver_factory=lambda: Resolver(FakeInstanceLookup(gk), FakeUniProt({}, {'PINK1': 'Q9BXM7'})),
                     chat_model=model)
    c = TestClient(app)
    sid = c.post('/api/llm/sessions', json={'pmid': '24751536'}, headers=H()).json()['sessionId']
    return c, sid, model


def chat(c, sid, message, token='alice', selected=()):
    r = c.post(f'/api/llm/sessions/{sid}/chat', json={'message': message, 'selectedDbIds': list(selected)}, headers=H(token))
    assert r.status_code == 200, r.text
    assert r.headers['content-type'].startswith('text/event-stream')
    events, cur = [], {}
    for line in r.text.splitlines():
        if line.startswith('event: '):
            cur = {'event': line[7:]}
        elif line.startswith('data: '):
            cur['data'] = json.loads(line[6:])
            events.append(cur)
    return events


def test_turn_reads_paper_proposes_and_streams_events_in_order():
    c, sid, model = setup([tool('search_paper', query='CCCP stabilised PINK1'),
                           tool('propose_patch', reason='add the damage condition', ops=CCCP,
                                evidence=[{'quote': Q2, 'supports': ['regulatedBy[0]']}]),
                           final('I proposed adding CCCP as a positive regulator; it awaits your review.')])
    ev = chat(c, sid, 'R1 only happens after mitochondrial damage - add that condition')
    kinds = [e['event'] for e in ev]
    assert kinds == ['tool', 'tool', 'proposal', 'text', 'text', 'done']
    assert ev[0]['data']['name'] == 'search_paper' and ev[2]['data']['id'] == 'p-001'
    assert ''.join(e['data']['delta'] for e in ev if e['event'] == 'text').startswith('I proposed')
    assert ev[-1]['data']['proposalIds'] == ['p-001']
    # the search result the model saw contains the quote it then used
    result = model.calls[1]['messages'][-1]['content'][0]['content']
    assert Q2[:30] in result
    # nothing changed yet; the proposal waits for a curator
    ui = c.get(f'/api/llm/sessions/{sid}/export', headers=H()).json()
    assert not any(i['schemaClassName'] == 'PositiveRegulation' for i in ui['newInstances'])
    assert c.get(f'/api/llm/sessions/{sid}/proposals?status=pending', headers=H()).json()[0]['actor'] == 'chat'


def test_tool_errors_go_back_to_the_model_which_can_retry():
    c, sid, model = setup([tool('propose_patch', reason='x', ops=CCCP),                      # no evidence: refused
                           tool('propose_patch', reason='x', ops=CCCP, evidence=[{'quote': Q2}]),
                           final('done')])
    ev = chat(c, sid, 'add CCCP')
    first_result = json.loads(model.calls[1]['messages'][-1]['content'][0]['content'])
    assert 'needs at least one verified quote' in first_result['error']
    assert [e['event'] for e in ev].count('proposal') == 1


def test_a_fabricated_quote_is_refused_even_if_the_model_insists():
    c, sid, model = setup([tool('propose_patch', reason='x', ops=CCCP, evidence=[{'quote': 'CCCP kills all PINK1 in neurons in a day'}]),
                           final('could not')])
    ev = chat(c, sid, 'add CCCP')
    assert 'not found in the paper' in json.loads(model.calls[1]['messages'][-1]['content'][0]['content'])['error']
    assert 'proposal' not in [e['event'] for e in ev]


def test_read_tools_and_selection_context():
    c, sid, model = setup([tool('list_reactions'), tool('get_reaction', key='r0'), tool('get_reaction', key='nope'),
                           tool('resolve_identifier', name='PINK1', type='uniprot'),
                           tool('resolve_identifier', name='mitochondrial outer membrane', type='compartment'),
                           tool('resolve_identifier', name='nothing', type='compartment'), final('ok')])
    r0 = c.get(f'/api/llm/sessions/{sid}', headers=H()).json()['reactions'][0]['dbId']
    chat(c, sid, 'what is selected?', selected=[r0, 424242])
    first_user = model.calls[0]['messages'][-1]['content']
    assert 'reaction r0' in first_user and 'PINK1 phosphorylates ubiquitin' in first_user and '424242' in first_user
    res = lambda i: json.loads(model.calls[i]['messages'][-1]['content'][0]['content'])
    assert res(1)[0]['key'] == 'r0' and res(1)[0]['catalyst'] == 'PINK1'
    assert res(2)['evidence'][0]['id'] == 'ev-001' and res(2)['evidence'][0]['page'] == 1
    assert 'no reaction with key' in res(3)['error']
    assert res(4)['found'] and res(4)['uniprot'] == 'Q9BXM7'
    assert res(5)['reactome']['db_id'] == 17906 and res(6)['found'] is False


def test_step_cap_stops_a_runaway_loop_and_history_is_kept_for_the_next_turn():
    c, sid, model = setup([tool('list_reactions')] * 8 + [final('second answer')])
    ev = chat(c, sid, 'loop forever')
    assert len(model.calls) == 8 and any('Stopped' in e['data'].get('delta', '') for e in ev if e['event'] == 'text')
    chat(c, sid, 'and now?')
    assert [m['content'] for m in model.calls[8]['messages'][:2]] == ['loop forever', model.calls[8]['messages'][1]['content']]
    assert model.calls[8]['messages'][0]['role'] == 'user' and model.calls[8]['messages'][1]['role'] == 'assistant'


def test_model_failure_is_reported_not_raised():
    class Boom:
        def turn(self, *a):
            raise RuntimeError('overloaded')
    app = create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe(), chat_model=Boom())
    c = TestClient(app)
    sid = c.post('/api/llm/sessions', json={'pmid': '24751536'}, headers=H()).json()['sessionId']
    ev = chat(c, sid, 'hello')
    assert ev[0]['event'] == 'error' and 'overloaded' in ev[0]['data']['message'] and ev[-1]['event'] == 'done'


def test_validation_privacy_and_auth():
    c, sid, _ = setup([final('x')])
    assert chat(c, sid, '')[0]['event'] == 'error'
    assert chat(c, sid, 'x' * 5000)[0]['event'] == 'error'
    assert c.post(f'/api/llm/sessions/{sid}/chat', json={'message': 'hi'}, headers=H('bob')).status_code == 404
    assert c.post(f'/api/llm/sessions/{sid}/chat', json={'message': 'hi'}).status_code == 401
    off = TestClient(create_app(FakeAuthProvider(TOKENS), InMemorySessionStore(), SyncJobRunner(), Pipe()))
    s2 = off.post('/api/llm/sessions', json={'pmid': '24751536'}, headers=H()).json()['sessionId']
    assert off.post(f'/api/llm/sessions/{s2}/chat', json={'message': 'hi'}, headers=H()).status_code == 503


def test_chat_history_is_saved_on_the_session():
    c, sid, _ = setup([tool('propose_patch', reason='r', ops=CCCP, evidence=[{'quote': Q2}]), final('proposed')])
    chat(c, sid, 'add CCCP')
    s = c.app.state.store.get(sid, 'alice')
    assert [m.role for m in s.chat] == ['user', 'assistant'] and s.chat[1].proposal_ids == ['p-001']


def test_anthropic_adapter_translates_stream_to_a_turn():
    from curator_llm.adapters.chat_anthropic import AnthropicChatModel

    class Block:
        def __init__(self, **k): self.__dict__.update(k)

    class Stream:
        text_stream = ['Hel', 'lo']
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get_final_message(self):
            return Block(stop_reason='tool_use', content=[Block(type='text', text='Hello'),
                         Block(type='tool_use', id='t1', name='get_reaction', input={'key': 'r0'})])

    class Client:
        class messages:
            kw = None
            @staticmethod
            def stream(**kw):
                Client.messages.kw = kw
                return Stream()
    got = []
    t = AnthropicChatModel(Client(), 'm', 100).turn('sys', [{'role': 'user', 'content': 'hi'}], [{'name': 'x'}], got.append)
    assert got == ['Hel', 'lo'] and t.text == 'Hello' and t.tool_calls[0].name == 'get_reaction'
    assert t.content[1] == {'type': 'tool_use', 'id': 't1', 'name': 'get_reaction', 'input': {'key': 'r0'}}
    assert Client.messages.kw['system'] == 'sys' and Client.messages.kw['model'] == 'm'
