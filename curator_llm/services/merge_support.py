"""The pure logic of run_merge.py, kept here so it can be tested (run_merge.py itself runs its driver at import).

  * collapse_exact_duplicates: identical reactions become one, quotes kept, each quote labelled with its paper.
  * conflict_free_clusters: group reactions so that every pair inside a group was judged "same".
  * merge_across_papers: merge the per-paper results into reactions that cite several papers, one paper at a time.
  * VerdictCache: remember the duplicate judge's verdicts on disk, so a re-run or an added paper never pays for a pair
    it has already judged.
"""
import hashlib
import json
import math
import os
import threading
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from curator_llm.services.dedup_prefilter import tokens
from curator_llm.services.evidence_attach import merge_details
from curator_llm.services.sources import record_sources


# ── exact duplicates ───────────────────────────────────────────────────────
def _norm(x) -> str:
    return (x or '').strip().lower()


def signature(a: dict) -> tuple:
    """What makes two extracted reactions identical: name, inputs and outputs (case and order ignored)."""
    return (_norm(a.get('name')),
            tuple(sorted(_norm(i) for i in (a.get('input') or []))),
            tuple(sorted(_norm(o) for o in (a.get('output') or []))))


def collapse_exact_duplicates(records: Sequence[dict]) -> List[dict]:
    """One record per (paper, signature), in first-seen order: an identical reaction found again in the SAME paper is
    folded into the kept record and its quotes are kept. An identical reaction in ANOTHER paper stays its own record
    (nothing is dropped, and each record keeps its own paper); the cross-paper stage merges the two later, for free,
    because their signatures match. Records are changed in place."""
    seen: Dict[tuple, dict] = {}
    unique: List[dict] = []
    for e in records:
        a = e.get('annotation_result') or {}
        key = (e.get('source'), signature(a))
        kept = seen.get(key)
        if kept is None:
            seen[key] = e
            unique.append(e)
            continue
        ka = kept['annotation_result']
        ka['evidence'] = list(dict.fromkeys((ka.get('evidence') or []) + (a.get('evidence') or [])))
        ka['evidence_details'] = merge_details(ka.get('evidence_details'), a.get('evidence_details'))
    return unique


# ── clustering ─────────────────────────────────────────────────────────────
def _components(n: int, same: Dict[Tuple[int, int], bool]) -> List[List[int]]:
    """Connected components of the "judged same" graph, each sorted, in order of their first member."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), v in same.items():
        if v:
            parent[find(a)] = find(b)
    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda c: c[0])


def conflict_free_clusters(n: int, same: Dict[Tuple[int, int], bool],
                           conf: Dict[Tuple[int, int], Optional[float]]) -> List[List[int]]:
    """Cluster reactions 0..n-1 so that every cluster is a clique in the "same" graph: no pair inside a cluster was
    judged different. Transitive closure would chain A~B and B~C into one group even when A vs C was judged different
    (on ZNFX1 it joined three distinct mechanistic steps), so this is agglomerative: repeatedly join the two groups that
    are FULLY connected to each other, taking the join whose weakest connecting verdict is strongest, so the result does
    not depend on the order pairs were listed in. A "same" with no confidence counts as 0.0: it can still join, but only
    after every join backed by an actual confidence.

    Two groups can only join if some pair across them is "same", so each connected component of the same-graph is
    clustered on its own, looking only at the group pairs that an edge connects. The result is identical to scanning
    every pair of groups, which was cubic in the number of reactions across ALL papers in the file (13 s at 600
    reactions); this is near-linear."""
    out: List[List[int]] = []
    for comp in _components(n, same):
        if len(comp) == 1:
            out.append(comp)
            continue
        members = set(comp)
        edges = [(a, b) for (a, b), v in same.items() if v and a in members and b in members]
        groups = [[i] for i in comp]
        while True:
            where = {i: gi for gi, g in enumerate(groups) for i in g}
            cand = sorted({(min(where[a], where[b]), max(where[a], where[b])) for a, b in edges if where[a] != where[b]})
            best = None
            for gi, gj in cand:
                cross = [(min(a, b), max(a, b)) for a in groups[gi] for b in groups[gj]]
                if not all(same.get(p) for p in cross):
                    continue
                weakest = min(conf.get(p) or 0.0 for p in cross)
                if best is None or weakest > best[0]:
                    best = (weakest, gi, gj)
            if best is None:
                break
            _, gi, gj = best
            groups[gi] = sorted(groups[gi] + groups[gj])
            groups.pop(gj)
        out.extend(groups)
    return sorted(out, key=lambda g: (-len(g), g))


# ── verdict cache ──────────────────────────────────────────────────────────
class VerdictCache:
    """Duplicate-judge verdicts on disk (one JSON object per line; a later line for a key replaces an earlier one).

    The key is a hash of everything that can change a verdict: the model and its thinking setting, a fingerprint of the
    judge's rules, and the two reactions' descriptions (order does not matter). An entry is `final` once any re-vote of
    a shaky verdict has been settled; before that it is only the first vote, so a run that stopped half way resumes
    from it but still re-votes. Passing path=None gives a cache that remembers nothing."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self.hits = 0
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            self._data[d['k']] = d
                        except (ValueError, KeyError):
                            continue                      # a torn or foreign line: ignore it

    @staticmethod
    def key(model: str, thinking: bool, rules_fingerprint: str, brief_a: str, brief_b: str) -> str:
        blob = json.dumps([model, bool(thinking), rules_fingerprint, sorted([brief_a, brief_b])])
        return hashlib.sha256(blob.encode()).hexdigest()

    def __len__(self):
        return len(self._data)

    def get(self, key: str) -> Optional[dict]:
        d = self._data.get(key)
        if d is not None:
            self.hits += 1
        return d

    def put(self, key: str, same: bool, conf: Optional[float], final: bool = False) -> None:
        if not self.path or conf is None:                 # nothing to remember, or a failed call: never cache a failure
            return
        entry = {'k': key, 'same': bool(same), 'conf': conf, 'final': bool(final)}
        with self._lock:
            self._data[key] = entry
            with open(self.path, 'a') as f:
                f.write(json.dumps(entry) + '\n')


# ── merging across papers ──────────────────────────────────────────────────
class TfidfSimilarity:
    """How alike two reactions are by the entities they name (inputs, outputs, catalyst), weighting rare entities up.
    Only used to decide WHICH candidates to ask the judge about first; the judge decides whether they are the same."""

    def __init__(self, annotations: Sequence[dict]):
        df: Dict[str, int] = {}
        for a in annotations:
            for t in self._tokens(a):
                df[t] = df.get(t, 0) + 1
        n = len(annotations)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._default = math.log(1 + n) + 1                    # an entity unseen at build time is the rarest kind

    @staticmethod
    def _tokens(a: dict):
        cat = (a.get('catalystActivity') or {}).get('catalyst') or ''
        return tokens(' '.join((a.get('input') or []) + (a.get('output') or []) + [cat]))

    def _vec(self, a: dict) -> Dict[str, float]:
        v = {t: self._idf.get(t, self._default) for t in self._tokens(a)}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def __call__(self, a: dict, b: dict) -> float:
        va, vb = self._vec(a), self._vec(b)
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(x * vb.get(t, 0.0) for t, x in va.items())


def merge_across_papers(records: Sequence[dict], *,
                        prefilter: Callable[[dict, dict], bool],
                        similar: Callable[[dict, dict], float],
                        judge: Callable[[dict, dict], Tuple[bool, Optional[float]]],
                        combine: Callable[[dict, dict], dict],
                        signature: Optional[Callable[[dict], object]] = None,
                        max_candidates: int = 10,
                        reconcile: bool = True,
                        map_fn: Callable = map) -> Tuple[List[dict], Dict[str, int]]:
    """Merge reactions that were already merged within each paper into reactions that may cite several papers.

    1. Papers are added one at a time, in the order they first appear in `records`. A reaction of the new paper is
       compared only with the groups found so far (not with every reaction of every paper): candidates are the groups
       that pass `prefilter` and share no paper with it (the within-paper merge already decided those), asked
       best-`similar`-first, at most `max_candidates` of them, stopping at the first the `judge` calls the same. A group
       that already holds a reaction with the same `signature` is joined without asking anyone.
       So the number of judge calls grows with the number of papers times a small constant, not with the square of the
       number of reactions. `combine(group_record, reaction)` folds a reaction into its group's record, which later
       reactions are compared to.
    2. A paper's reactions are all judged against the groups as they stood before that paper, concurrently (`map_fn`).
       A group takes at most one reaction per paper: if two of a paper's reactions pick the same group, the one the judge
       was more sure of joins and the other starts a group of its own.
    3. `reconcile`: groups never get re-compared in step 1, so the same reaction can end up in two groups (it arrived
       before either group could see the other). Finally the groups are compared with each other, the same way
       (candidates only from papers the two groups do not share, best first, at most `max_candidates`), until no two merge.
       Groups are far fewer than reactions, so this is cheap.

    Returns (groups, stats); a group is {'rep': merged record, 'members': record indices, 'papers': set of sources,
    'conf': weakest cross-paper verdict confidence or None}."""
    def first_source(r):
        src = record_sources(r)
        return src[0] if src else ''

    stats = {'judged': 0, 'merged': 0, 'candidates': 0, 'exact': 0, 'reconciled': 0, 'capped': 0}
    groups: List[dict] = []
    for paper in dict.fromkeys(first_source(r) for r in records):
        idxs = [i for i, r in enumerate(records) if first_source(r) == paper]
        snapshot = list(groups)

        def decide(i):
            r = records[i]
            mine = set(record_sources(r))
            cands = [g for g in snapshot if not (mine & g['papers']) and prefilter(r, g['rep'])]
            if signature is not None:
                sig = signature(r)
                for g in cands:
                    if sig in g['sigs']:
                        return g, 1.0, 0, len(cands), True, False          # the same reaction, word for word: no judge call
            cands.sort(key=lambda g: -similar(r, g['rep']))
            calls = 0
            for g in cands[:max_candidates]:
                try:
                    same, conf = judge(g['rep'], r)
                except Exception:                                   # one failed call must not lose the reaction
                    same, conf = False, None
                calls += 1
                if same:
                    return g, conf, calls, len(cands), False, False
            return None, None, calls, len(cands), False, len(cands) > max_candidates   # capped: groups left unasked

        decisions = list(map_fn(decide, idxs))
        stats['judged'] += sum(d[2] for d in decisions)
        stats['candidates'] += sum(d[3] for d in decisions)
        stats['capped'] += sum(1 for d in decisions if d[5])      # reactions that found no match with candidates left unasked
        winners: Dict[int, dict] = {}
        claimed = set()
        for k in sorted(range(len(idxs)), key=lambda k: -(decisions[k][1] or 0.0)):        # most sure first; ties keep order
            g = decisions[k][0]
            if g is not None and id(g) not in claimed:
                claimed.add(id(g))
                winners[k] = g
        for k, i in enumerate(idxs):
            r = records[i]
            sig = {signature(r)} if signature is not None else set()
            if k in winners:
                g, conf = winners[k], decisions[k][1]
                g['rep'] = combine(g['rep'], r)
                g['members'].append(i)
                g['papers'] |= set(record_sources(r))
                g['sigs'] |= sig
                g['conf'] = conf if g['conf'] is None else (g['conf'] if conf is None else min(g['conf'], conf))
                g['ver'] += 1
                stats['merged'] += 1
                stats['exact'] += 1 if decisions[k][4] else 0
            else:
                groups.append({'rep': r, 'members': [i], 'papers': set(record_sources(r)), 'conf': None, 'sigs': sig, 'ver': 0})

    if reconcile:
        tried = set()                                    # (group, its version) pairs already judged: a changed group is new again
        merged_any = True
        while merged_any:
            merged_any = False
            for a in list(groups):
                cands = [b for b in groups if b is not a and not (a['papers'] & b['papers']) and prefilter(a['rep'], b['rep'])
                         and frozenset(((id(a), a['ver']), (id(b), b['ver']))) not in tried]
                cands.sort(key=lambda b: -similar(a['rep'], b['rep']))
                capped = len(cands) > max_candidates
                for b in cands[:max_candidates]:
                    tried.add(frozenset(((id(a), a['ver']), (id(b), b['ver']))))
                    try:
                        same, conf = judge(a['rep'], b['rep'])
                    except Exception:
                        same, conf = False, None
                    stats['judged'] += 1
                    if same:
                        a['rep'] = combine(a['rep'], b['rep'])
                        a['members'] += b['members']
                        a['papers'] |= b['papers']
                        a['sigs'] |= b['sigs']
                        parts = [x for x in (a['conf'], b['conf'], conf) if x is not None]
                        a['conf'] = min(parts) if parts else None
                        a['ver'] += 1
                        groups.remove(b)
                        stats['reconciled'] += 1
                        merged_any = True
                        break
                else:
                    stats['capped'] += 1 if capped else 0     # asked its best candidates, none matched, more were left
                if merged_any:
                    break
    return groups, stats



# ── the duplicate judge's brief and rules (moved from run_merge.py, which runs on import; text unchanged, so the
# verdict cache keys stay the same) ──────────────────────────────────────────────────────────────────────────────
def _reaction_brief(a):
    ca = a.get('catalystActivity') or {}
    regs = a.get('regulatedBy') or []
    # the note carries any condition-dependence flag already recorded, so the judge can use it
    # instead of re-deriving why one regulator appears in two directions
    reg_str = ', '.join(
        f"{r.get('regulationType')} by {r.get('regulator')}"
        + (f" [note: {r.get('note')}]" if r.get('note') else '')
        for r in regs) if regs else 'none'
    return (f"name: {a.get('name','')}\n"
            f"type: {a.get('reactionType','')}\n"
            f"input: {', '.join(a.get('input',[]) or []) or 'none'}\n"
            f"output: {', '.join(a.get('output',[]) or []) or 'none'}\n"
            f"catalyst: {ca.get('catalyst') or 'none'} ({ca.get('molecularFunction') or 'none'})\n"
            f"regulation: {reg_str}\n"
            f"condition: {a.get('condition') or 'none'}\n"
            f"compartment: {a.get('compartment') or 'none'}\n"
            f"relationships: {' | '.join(a.get('relationships',[]) or []) or 'none'}")


def _conf(x):
    """Judge confidence clamped to 0-1, or None when the model returned none/garbage."""
    try:
        return round(min(1.0, max(0.0, float(x))), 3)
    except (TypeError, ValueError):
        return None


def _judge_prompt(a1, a2):
    return f"""You are a Reactome biocurator deduplicating extracted reactions.
Two records are DUPLICATES if they describe the SAME underlying biochemical event — the same
molecular transformation of the same participants — EVEN IF they differ in:
  - wording or synonyms
  - level of detail / granularity (e.g. one adds a qualifier like "leading to degradation")
  - reactionType label (e.g. one tagged "blackBoxEvent", the other "transition")
They are DIFFERENT reactions — do NOT merge them — if ANY of the following differs:
  - the participants (input) or the products (output)
  - the catalyst, A catalyst is the entity whose activity
    performs the input-to-output conversion; an entity that acts upstream, or that modulates
    whether or how fast the event happens, is REGULATION, not a catalyst — so a record naming
    X as catalyst and one naming X only as a regulator do not agree on the catalyst either.
  - a different regulationType on the SAME regulator (one says positiveRegulation by X, the
    other negativeRegulation by X) — that is a genuine contradiction, not extra detail.
    When the difference is due to conditional circumstances, such as a different experimental
    condition, add a "note" on that regulation flagging it for the curator.
  - the biological condition under which the reaction occurs — EXCEPT when the two records are
    the SAME core transformation (same participants, same catalyst, same molecular action) and
    the condition is the ONLY thing separating them. That is one reaction tested under two
    conditions, not two reactions: merge it, keep BOTH regulatory directions, and flag each
    with a "note" naming the condition it holds under.
  - the compartment, when the two are genuinely different locations (cytosol vs
    mitochondrion) — but NOT when one merely contains the other
  - the molecular action (e.g. phosphorylation vs ubiquitination)
They are also DIFFERENT if they merely share one participant.

Likewise these are the SAME, not different:
  - a domain or subunit vs its parent protein as catalyst (e.g. "ZNFX1 RZ domain" = "ZNFX1")
  - an experimental construct vs the protein it stands for, as substrate or catalyst: a
    truncation or fragment made for an assay ("CDH1 N186", "CDH1 (N-terminal fragment)" =
    "CDH1"), a tagged or recombinant form, or an ortholog from another species used as a stand-in
    ("Xenopus FZR (Cdh1)" = "CDH1"). One paper often shows the same event in vitro with such a
    construct and in cells with the native protein: that is one reaction. (A fragment that is a
    real product of the biology, such as a cleavage product, is a different entity.)
  - one record listing an accessory cofactor (E1/E2 enzyme, ATP, ubiquitin) the other omits
  - a compartment vs a sub-compartment of it ("mitochondrial outer membrane" = "mitochondrion")
  - DIFFERENT SETS OF REGULATORS. Regulation hangs OFF a reaction; it is not part of the
    reaction's identity. Two experiments on the same event naturally test different
    regulators — one knocks out X, another uses a phospho-null mutant of Y — so their
    regulator lists are additive and get unioned when the records merge. A regulator in one
    record and absent from the other, or two records naming entirely different regulators,
    is NOT a difference. (A conflicting regulationType on the same regulator still is; see
    above.)
  - GO molecular-function wording for the same catalytic act ("ubiquitin-protein ligase
    activity" = "ubiquitin-protein transferase activity")
  - the same modification of the same substrate by the same catalyst, named with FEWER sites
    or none: "phosphorylated at Tyr130" vs "phosphorylated at Tyr91 and Tyr130" vs
    "phosphorylated on tyrosine residues" vs "phosphorylated". One sentence of a paper often
    names only some of the sites another sentence lists: that is less detail, not a different
    product. Only two records naming DISJOINT site sets ("at Ser65" vs "at Ser228", with
    neither naming both) are different products.
Judge the CORE transformation: which entity acts on which, and what change it undergoes.

When in doubt, do NOT merge. Two records left separate can still be merged by a curator
later; a merge that should not have happened is harder to undo.

Reaction A:
{_reaction_brief(a1)}

Reaction B:
{_reaction_brief(a2)}

Answer with the "same" key FIRST so the verdict survives truncation.
Keep "reason" under 15 words. "confidence" is how sure you are OF THIS VERDICT, 0-1: high when
the two records plainly describe one event, low when you had to weigh a judgement call.
Return ONLY JSON: {{"same": <true|false>, "confidence": <0-1>, "reason": "<short phrase>"}}"""


def make_judge(client, model_kw: dict, cache: 'VerdictCache', revote_below: float = 0.9,
               on_usage: Optional[Callable] = None) -> Callable[[dict, dict], Tuple[bool, Optional[float]]]:
    """The duplicate judge as run_merge.py asks it, for use in-process: same rules and prompt caching, the verdict
    cache (shared with run_merge.py: same keys), and a shaky verdict (confidence below `revote_below`) asked twice more,
    majority wins. judge(a1, a2) takes two reaction annotations (run_merge's shape) and returns (same, confidence); a
    call that fails twice raises."""
    fp = hashlib.sha256(_judge_prompt({}, {}).split('Reaction A:')[0].encode()).hexdigest()[:16]

    def ask(a1, a2):
        prompt = _judge_prompt(a1, a2)
        cut = prompt.index('Reaction A:')
        last = None
        for _ in (1, 2):
            try:
                msg = client.messages.create(**model_kw, max_tokens=4000, messages=[{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt[:cut], 'cache_control': {'type': 'ephemeral'}},
                    {'type': 'text', 'text': prompt[cut:]}]}])
                if on_usage:
                    on_usage(msg)
                txt = next((b.text for b in msg.content if getattr(b, 'type', None) == 'text'), '')
                txt = txt.strip().replace('```json', '').replace('```', '').strip()
                try:
                    got = json.loads(txt)
                    return bool(got.get('same', False)), _conf(got.get('confidence'))
                except json.JSONDecodeError:
                    import re
                    m = re.search(r'"same"\s*:\s*(true|false)', txt, re.I)
                    if not m:
                        raise
                    c = re.search(r'"confidence"\s*:\s*([0-9.]+)', txt)
                    return m.group(1).lower() == 'true', _conf(c and c.group(1))
            except Exception as e:
                last = e
        raise RuntimeError(f'duplicate judge failed: {type(last).__name__}: {last}')

    def judge(a1, a2):
        key = VerdictCache.key(model_kw['model'], 'thinking' in model_kw, fp, _reaction_brief(a1), _reaction_brief(a2))
        hit = cache.get(key)
        if hit is not None and hit.get('final'):
            return hit['same'], hit['conf']
        same, conf = ask(a1, a2)
        if conf is not None and conf < revote_below:
            votes = [same] + [ask(a1, a2)[0] for _ in range(2)]
            same = sum(votes) * 2 > len(votes)
        cache.put(key, same, conf, final=True)
        return same, conf
    return judge
