"""Chat agent: a tool-use loop that helps a curator edit one session's draft.

The agent can READ (reactions, paper text, identifier resolvers) and PROPOSE (validated patches with
verified quotes). It cannot change the session: a proposal only takes effect when a curator accepts it
through the accept endpoint. All identifiers come from resolver tools, never from the model's memory.
"""
import json
import logging
import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional

from curator_llm.models.session import ChatMessage, Session
from curator_llm.ports.chat_model import ChatModel
from curator_llm.services.paper_text import PaperText
from curator_llm.services.session_service import ProposalError, SessionService

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 8000
HISTORY_MESSAGES = 10

SYSTEM = """You are a Reactome curation assistant working inside the curator tool, helping one curator refine the
draft annotation of a single paper. You can read the draft and the paper, and you can PROPOSE edits. You can never
apply an edit: the curator reviews each proposal and accepts or rejects it.

How to work
- Look before you answer: use list_reactions / get_reaction for the draft and search_paper for what the paper says.
  Do not answer from memory about what the paper reports.
- After an edit is accepted, or when asked whether a reaction is sound, call run_qa on it and report the findings.
- To change the draft call propose_patch. One focused proposal per request; explain it in `reason`.
- Any edit that changes what a reaction IS (inputs, outputs, catalyst, regulations, order) or adds/removes a
  reaction needs `evidence`: verbatim quotes from the paper, copied exactly from search_paper results (the server
  checks every quote and rejects any that is not in the paper). If the curator wants a claim the paper does not
  support, set curator_assertion=true and say so; never invent a quote.
- Identifiers (UniProt, ChEBI, GO, PSI-MOD, compartments) come only from resolve_identifier. If it finds nothing,
  say so and leave the field empty; do not guess an id.
- To see whether Reactome already has a reaction, call find_existing_reactome. Report what it finds and let the
  curator decide; only if they ask, record the decision with a patch that sets /reactions/<key>/existing to
  {db_id, display_name, schema_class} of that event (the reaction is then reused, not added as new).
- If a tool returns an error, read it, fix the problem and retry once; if it still fails, tell the curator why.

Curation rules
- Catalyst = the entity that directly performs the reaction. Anything that only promotes, enables or is required
  for it is a regulator (positive, negative, or requirement), not a catalyst.
- Hedged statements ("may", "is suggested to") are not evidence. A condition-dependent regulator gets a note
  naming the condition it was observed under.
- Keep entity names plain (gene symbols for proteins); a modified protein is the base protein plus modifications.

Draft JSON (what propose_patch edits; RFC 6902 ops, a reaction can be addressed by its key)
- /participants/<key>: {kind: ewas|simple|complex|set, key, name, uniprot?, chebi?, compartment_name?,
  modifications?: [{psi_mod, residue, coordinate}], components?: [keys], members?: [keys]}
- /reactions/<key>: {key, name, reaction_type, inputs: [participant keys], outputs: [participant keys],
  catalyst?: {entity: key, activity?: {name}}, regulations: [{kind: positive|negative|requirement, regulator: key,
  note?}], summation, preceding: [reaction keys]}
Evidence ids on reactions cannot be edited with a patch; attach quotes through propose_patch's `evidence`.
Keep replies short and concrete. When you have made a proposal, say what it does and that it awaits review."""

TOOLS: List[Dict[str, Any]] = [
    {'name': 'list_reactions', 'description': 'List every reaction in the draft with its key, name, inputs and outputs.',
     'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'get_reaction', 'description': 'Full detail of one reaction: fields, the participants it uses, and its evidence quotes with page and section.',
     'input_schema': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}},
    {'name': 'search_paper', 'description': 'Search the paper text. Returns passages with section, page and figure. Quote from these verbatim when you attach evidence.',
     'input_schema': {'type': 'object', 'properties': {'query': {'type': 'string'}, 'section': {'type': 'string', 'description': 'optional: Abstract, Results And Discussion, Materials And Methods, ...'}},
                      'required': ['query']}},
    {'name': 'find_existing_reactome', 'description': 'Check whether Reactome already has a draft reaction (matched on UniProt accessions). Omit key to check every reaction. Returns matches with level same | similar, the existing reaction name, stable id and why.',
     'input_schema': {'type': 'object', 'properties': {'key': {'type': 'string'}}}},
    {'name': 'run_qa', 'description': 'QA one reaction: rule checks (missing evidence, unsupported catalyst/regulation, no change between inputs and outputs, ...) plus a model review. Returns a verdict and findings; they are also recorded as issues.',
     'input_schema': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}},
    {'name': 'resolve_identifier', 'description': 'Look up an identifier in Reactome / UniProt. type: uniprot (gene or protein name), compartment, go_function, chebi (small molecule name), psi_mod (MOD:xxxxx).',
     'input_schema': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'type': {'type': 'string', 'enum': ['uniprot', 'compartment', 'go_function', 'chebi', 'psi_mod']}},
                      'required': ['name', 'type']}},
    {'name': 'propose_patch', 'description': 'Propose an edit to the draft. Nothing changes until the curator accepts it.',
     'input_schema': {'type': 'object', 'properties': {
         'reason': {'type': 'string'},
         'ops': {'type': 'array', 'items': {'type': 'object'}, 'description': 'RFC 6902 operations'},
         'evidence': {'type': 'array', 'items': {'type': 'object', 'properties': {
             'quote': {'type': 'string'}, 'supports': {'type': 'array', 'items': {'type': 'string'}},
             'system': {'type': 'string'}, 'experimental_species': {'type': 'string'}}, 'required': ['quote']}},
         'curator_assertion': {'type': 'boolean'}}, 'required': ['reason', 'ops']}},
]


def _clip(obj: Any) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= MAX_TOOL_RESULT_CHARS else s[:MAX_TOOL_RESULT_CHARS] + '... [truncated]'


class ChatAgent:
    def __init__(self, service: SessionService, model: ChatModel, max_steps: int = 8):
        self.service, self.model, self.max_steps = service, model, max_steps

    # ── context ────────────────────────────────────────────────────────────
    @staticmethod
    def _selection(s: Session, db_ids: List[int]) -> str:
        if not db_ids or not s.user_instances:
            return ''
        rev = {v: k for k, v in s.key_to_db_id.items()}
        by_id = {i['dbId']: i for i in s.user_instances['newInstances']}
        lines = []
        for d in db_ids[:20]:
            inst = by_id.get(d)
            if inst is None:
                lines.append(f'- dbId {d}: not part of this session (an existing Reactome instance, or unknown)')
            elif d in rev:
                lines.append(f'- reaction {rev[d]}: "{inst["displayName"]}"')
            else:
                lines.append(f'- {inst["schemaClassName"]} "{inst["displayName"]}" (dbId {d})')
        return 'The curator has these instances selected in the tool:\n' + '\n'.join(lines) + '\n\n'

    # ── tools ──────────────────────────────────────────────────────────────
    def _tool(self, owner: str, session_id: str, name: str, args: Dict[str, Any],
              proposals: List[str], emit: Callable[[str, dict], None]) -> Any:
        s = self.service.store.get(session_id, owner)
        d = s.draft
        if name == 'list_reactions':
            return [{'key': r.key, 'name': r.name, 'inputs': [d.participants[k].name for k in r.inputs],
                     'outputs': [d.participants[k].name for k in r.outputs],
                     'catalyst': d.participants[r.catalyst.entity].name if r.catalyst else None,
                     'n_regulations': len(r.regulations)} for r in d.reactions]
        if name == 'get_reaction':
            r = next((x for x in d.reactions if x.key == args.get('key')), None)
            if r is None:
                return {'error': f'no reaction with key {args.get("key")!r}; keys are {[x.key for x in d.reactions]}'}
            used = set(r.inputs + r.outputs + [g.regulator for g in r.regulations] + ([r.catalyst.entity] if r.catalyst else []))
            by_id = {e.id: e for e in s.evidence}
            return {'reaction': r.model_dump(mode='json'),
                    'participants': {k: d.participants[k].model_dump(mode='json') for k in used},
                    'evidence': [{'id': e.id, 'quote': e.quote, 'page': e.page, 'section': e.section,
                                  'figure': e.figure, 'supports': e.supports}
                                 for e in (by_id[i] for i in r.evidence_ids if i in by_id)]}
        if name == 'search_paper':
            if not s.paper:
                return {'error': 'this session has no paper text'}
            hits = PaperText.from_dict(s.paper).search(args.get('query', ''), args.get('section') or None, top_k=5)
            return [{'text': h.text[:700], 'section': h.section, 'page': h.page, 'figure': h.figure} for h in hits]
        if name == 'find_existing_reactome':
            if self.service.events is None:
                return {'error': 'the existing-event check is not configured'}
            key = args.get('key')
            if key and not any(r.key == key for r in d.reactions):
                return {'error': f'no reaction with key {key!r}; keys are {[x.key for x in d.reactions]}'}
            try:
                s = self.service.check_existing(owner, session_id, [key] if key else None)
            except ProposalError as e:
                return {'error': str(e)}
            return [m.model_dump(mode='json') for m in s.existing if not key or m.reaction_key == key] or \
                {'matches': [], 'note': 'no similar reaction found in Reactome'}
        if name == 'run_qa':
            try:
                return self.service.run_qa(owner, session_id, args.get('key', ''), self.model).model_dump(mode='json')
            except ProposalError as e:
                return {'error': str(e)}
        if name == 'resolve_identifier':
            return self._resolve(args.get('name', ''), args.get('type', ''))
        if name == 'propose_patch':
            try:
                p = self.service.propose(owner, session_id, args.get('ops') or [], args.get('reason', ''),
                                         args.get('evidence') or [], actor='chat',
                                         curator_assertion=bool(args.get('curator_assertion')))
            except ProposalError as e:
                return {'error': str(e)}
            proposals.append(p.id)
            emit('proposal', p.model_dump(mode='json'))
            return {'proposal_id': p.id, 'status': 'pending - awaiting the curator', 'summary': p.summary}
        return {'error': f'unknown tool {name}'}

    def _resolve(self, name: str, kind: str) -> Any:
        if self.service.resolver_factory is None:
            return {'error': 'identifier resolution is not configured'}
        r = self.service.resolver_factory()
        if kind == 'uniprot':
            if not r.uniprot:
                return {'error': 'UniProt is not configured'}
            acc = r._call('UniProt', r.uniprot.search_gene, name)
            if acc is None:
                return {'found': False, 'note': r.notes[-1] if r.notes else 'no unique reviewed human entry'}
            ref = r.reference_gene_product(acc)
            return {'found': True, 'uniprot': acc, 'reactome_reference': ref.model_dump() if ref else None}
        if kind == 'compartment':
            ref = r.compartment(name)
        elif kind == 'go_function':
            ident, ref = r.go_function(None, name)
            return {'found': bool(ident or ref), 'identifier': ident, 'reactome': ref.model_dump() if ref else None}
        elif kind == 'chebi':
            ref = r.reference_molecule(None, name)
        elif kind == 'psi_mod':
            ref = r.psi_mod(name)
        else:
            return {'error': f'unknown type {kind!r}'}
        return {'found': ref is not None, 'reactome': ref.model_dump() if ref else None}

    # ── the loop ───────────────────────────────────────────────────────────
    def run(self, owner: str, session_id: str, message: str,
            selected_db_ids: Optional[List[int]] = None) -> Iterator[Dict[str, Any]]:
        """Yield events: {'event': 'text'|'tool'|'proposal'|'error'|'done', 'data': {...}}."""
        message = (message or '').strip()
        s = self.service.store.get(session_id, owner)
        if s is None or s.draft is None:
            yield {'event': 'error', 'data': {'message': 'session is not ready for chat'}}
            return
        if not message or len(message) > MAX_MESSAGE_CHARS:
            yield {'event': 'error', 'data': {'message': f'message must be 1-{MAX_MESSAGE_CHARS} characters'}}
            return
        q: 'queue.Queue[Optional[dict]]' = queue.Queue()
        emit = lambda ev, data: q.put({'event': ev, 'data': data})
        worker = threading.Thread(target=self._loop, args=(owner, session_id, message, selected_db_ids or [], emit, q),
                                  daemon=True)
        worker.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    def _loop(self, owner, session_id, message, selected, emit, q):
        proposals: List[str] = []
        final_text = ''
        try:
            s = self.service.store.get(session_id, owner)
            history = [{'role': m.role, 'content': m.text} for m in s.chat[-HISTORY_MESSAGES:]]
            messages: List[Dict[str, Any]] = history + [{'role': 'user', 'content': self._selection(s, selected) + message}]
            for _ in range(self.max_steps):
                turn = self.model.turn(SYSTEM, messages, TOOLS, lambda t: emit('text', {'delta': t}))
                final_text = turn.text or final_text
                if not turn.tool_calls:
                    break
                messages.append({'role': 'assistant', 'content': turn.content})
                results = []
                for call in turn.tool_calls:
                    emit('tool', {'name': call.name, 'input': call.input})
                    try:
                        out = self._tool(owner, session_id, call.name, call.input, proposals, emit)
                    except Exception as e:                     # a tool bug must not kill the conversation
                        logger.exception('chat tool %s failed', call.name)
                        out = {'error': f'{type(e).__name__}: {e}'}
                    results.append({'type': 'tool_result', 'tool_use_id': call.id, 'content': _clip(out)})
                messages.append({'role': 'user', 'content': results})
            else:
                emit('text', {'delta': '\n\n(Stopped: too many steps for one request. Ask again to continue.)'})
                final_text += '\n\n(Stopped: too many steps for one request.)'
        except Exception as e:
            logger.exception('chat failed')
            emit('error', {'message': f'{type(e).__name__}: {e}'})
        finally:
            try:
                s = self.service.store.get(session_id, owner)
                s.chat.append(ChatMessage(role='user', text=message))
                s.chat.append(ChatMessage(role='assistant', text=final_text, proposal_ids=proposals))
                self.service.store.save(s)
            except Exception:
                logger.exception('could not save chat history')
            emit('done', {'proposalIds': proposals})
            q.put(None)
