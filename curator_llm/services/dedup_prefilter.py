"""Which pairs of extracted reactions are worth asking the duplicate judge about?

run_merge judges every same-source pair, which grows with the square of the reaction count and is
most of its LLM calls. The judge's own rules say two reactions are different when their inputs or
their outputs differ, so a pair whose inputs share no entity (or whose outputs share none) is
different without asking. The test works on name tokens, so synonyms and decorated names
("PINK1 (full-length)", "PTEN-induced kinase 1") still overlap; it is deliberately loose, and it
fails OPEN: a side with no usable tokens never rules a pair out.
"""
import re
from typing import Dict, FrozenSet, Iterable, List

# words that say HOW an entity is modified or where it is, or name a generic cofactor; they say
# nothing about WHICH entity it is, so sharing one is not evidence of the same reaction
_STOP = frozenset('''a an and of the to in on by with at from as or for into form forms full length wild type wt
mutant mut domain complex protein proteins phospho phosphorylated phosphorylation ubiquitinated
ubiquitination poly mono atp adp gtp gdp pi h2o water e1 e2 e3 enzyme cofactor active inactive free bound
mitochondrial mitochondria cytosolic cytosol nuclear nucleus membrane outer inner cleaved processed''' .split())
_TOKEN = re.compile(r'[a-z0-9]+')


def tokens(name: str) -> FrozenSet[str]:
    return frozenset(t for t in _TOKEN.findall((name or '').lower()) if len(t) > 1 and t not in _STOP)


def side_tokens(names: Iterable[str]) -> FrozenSet[str]:
    out = set()
    for n in names or []:
        out |= tokens(n)
    return frozenset(out)


def _sides(a: dict):
    # The extractor is inconsistent about the enzyme: one record lists it among the inputs, another names only the
    # substrate there and puts the enzyme in catalystActivity. Counting the catalyst on the input side keeps those
    # two records comparable (seen on the NAMPT / NAD+ paper, where it hid 3 true duplicates).
    cat = (a.get('catalystActivity') or {}).get('catalyst')
    return side_tokens(list(a.get('input') or []) + ([cat] if cat else [])), side_tokens(a.get('output'))


def may_be_duplicates(a1: dict, a2: dict) -> bool:
    """False only when the inputs, or the outputs, of the two reactions provably share no entity."""
    for t1, t2 in zip(_sides(a1), _sides(a2)):
        if t1 and t2 and not (t1 & t2):
            return False
    return True


def candidate_pairs(annotations: List[dict], pairs: Iterable) -> List:
    """The (i, j) index pairs, from `pairs`, that the judge still has to see."""
    sides: Dict[int, tuple] = {i: _sides(a) for i, a in enumerate(annotations)}

    def ok(i, j):
        return all(not (x and y) or (x & y) for x, y in zip(sides[i], sides[j]))
    return [(i, j) for i, j in pairs if ok(i, j)]
