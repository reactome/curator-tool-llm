"""Structure-first matching of extracted reactions to curated Reactome reactions.

Replaces "flatten both sides to prose, compare embeddings" with the pipeline:

    extracted reaction -> normalize entities -> structural filter
                       -> cosine (cheap rank) -> top-k -> cross-encoder rerank

A reaction is a structured object: catalyst, substrate, product, chemistry. Two
reactions are the same when those slots agree. Both sides of the comparison have
that structure available — the extraction has input/output/catalystActivity
fields, and Reactome stores participants as graph nodes — so the match is decided
on shared participants, and text models only rank what survives.

Entity-name facts this module works around, all confirmed against the graph:
  - Participants carry compartments:      'PRKN [cytosol]'
  - ...modifications and sites:           'p-S228,S402-PINK1'
  - ...and complex membership:            'PRKN:Ub-MOM proteins'
  - Reactome's ReferenceSequence.geneName holds symbols only (PRKN, PARK2) —
    the common name authors actually write ('Parkin') comes from UniProt.
  - Restricting participants to EntityWithAccessionedSequence drops ATP/ADP and
    other ubiquitous cofactors, which would otherwise make every kinase
    reaction look structurally identical.
"""
import os, re, json, time

import requests

PROJECT_ROOT = os.path.expanduser('~/curator-tool-llm')
UNIPROT_CACHE = os.path.join(PROJECT_ROOT, 'data', 'uniprot_names.json')
UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/{acc}.json'


# ─────────────────────────────────────────────────────────────────────────────
# 1. Entity normalization
# ─────────────────────────────────────────────────────────────────────────────
# Modification prefixes Reactome puts on participant names.
_MODS = {'p', 'ph', 'phospho', 'ac', 'acetyl', 'me', 'me1', 'me2', 'me3', 'ub',
         'sumo', 'sumo1', 'sumo2', 'sumo3', 'myr', 'palm', 'glcnac', 'farn',
         'gg', 'nt', 'ct'}

# The extraction names variants in ways Reactome never does, and these forms
# have to reduce to their parent entity or they match nothing at all:
#   'ParkinC431S'          mutant construct
#   'UbS65D'               phosphomimetic
#   'Parkin phospho-Ser65' modification as a SUFFIX (Reactome uses a prefix)
#   'Phospho-ubiquitin (UbS65D)'  parenthetical qualifier
_PAREN_SUFFIX = re.compile(r'\s*\([^)]*\)\s*$')
_MUTATION_SUFFIX = re.compile(r'[\s(]*([A-Z]\d{1,4}[A-Z])(\s*/\s*[A-Z]\d{1,4}[A-Z])*\)?\s*$')
_MOD_SUFFIX = re.compile(
    r'\s+(phospho|phosphorylated|acetyl|acetylated|ubiquitinated|ubiquitylated|'
    r'methylated|sumoylated)[-\s]?[A-Za-z]{0,3}\d*\s*$', re.I)
# Same vocabulary as a PREFIX, which is how papers usually write it:
# 'Phospho-ubiquitin', 'phospho-Parkin', 'p-S65-Ub'.
_MOD_PREFIX = re.compile(
    r'^(p|phospho|phosphorylated|ac|acetyl|acetylated|ub|ubiquitinated|'
    r'ubiquitylated|me\d?|methylated|sumo\d?|sumoylated)'
    r'(-[A-Za-z]?\d{1,4})?[-\s]', re.I)


def _strip_mutation(name):
    """Drop a trailing point-mutation/phosphomimetic tag, if it is safely one.

    A bare undelimited tag is ambiguous with ordinary gene symbols: 'UBE2N'
    matches the pattern ('E2N') and would wrongly reduce to 'UB'. HGNC symbols
    are all-uppercase and never append a bare mutation, whereas the mixed-case
    forms papers use ('UbS65D', 'ParkinC431S') always do. So a bare tag is only
    stripped off a base that is not itself all-uppercase; with a separator
    present ('PRKN C431S') the intent is unambiguous and casing is irrelevant.
    """
    m = _MUTATION_SUFFIX.search(name)
    if not m:
        return name
    base = name[:m.start()].strip(' -(')
    delimited = bool(re.match(r'[\s(]', m.group(0)))
    if not base or (not delimited and base.isupper()):
        return name
    return base


def strip_variant(name):
    """Progressively stripped forms of an entity name, least-stripped first.

        'ParkinC431S'                -> [..., 'Parkin']
        'Parkin phospho-Ser65'       -> [..., 'Parkin']
        'Phospho-ubiquitin (UbS65D)' -> [..., 'ubiquitin']
        'UBE2N'                      -> ['UBE2N']            (left alone)

    Callers take the first form the alias index RECOGNISES, not the shortest.
    """
    forms, cur = [name], name
    for pat in (_PAREN_SUFFIX, _MOD_SUFFIX, _MOD_PREFIX, _strip_mutation):
        nxt = (pat(cur) if callable(pat) else pat.sub('', cur)).strip(' -(')
        if nxt and nxt != cur:
            forms.append(nxt)
            cur = nxt
    return forms
# Residue/site tokens: S65, S228,S402, Y1234
_SITE = re.compile(r'^[A-Za-z]?\d+(,[A-Za-z]?\d+)*$')
_COMPARTMENT = re.compile(r'\s*\[[^\]]*\]\s*$')

# Generic words that are never a specific entity — matching on these is noise.
_STOPWORDS = {'protein', 'proteins', 'substrate', 'substrates', 'complex',
              'atp', 'adp', 'gtp', 'gdp', 'amp', 'pi', 'ppi', 'h2o', 'h+',
              'nad', 'nadh', 'nadp', 'nadph', 'coa', 'phosphate'}


def strip_compartment(name):
    """'PRKN [cytosol]' -> 'PRKN'"""
    return _COMPARTMENT.sub('', name or '').strip()


def strip_modifications(token, idx=None):
    """'p-S228,S402-PINK1' -> 'PINK1';  'polyUb-PARK2' -> 'PARK2'

    Leading modification markers and residue positions are peeled off until a
    real entity name remains. The modification itself is signal, but it belongs
    in a separate field — see split_entity().

    A prefix is treated as a modification when any of three things hold:
      1. it is in _MODS, seeded with common PTMs and extended at runtime from
         Reactome's own naming (see fetch_modifier_prefixes);
      2. it looks like a residue position, 'S65';
      3. `idx` is given and what remains is an entity the index recognises.

    (3) is the general case and needs no vocabulary at all: whatever prefix a
    paper or a pathway invents, 'polyUb-PARK2' is a modified PARK2 because PARK2
    is a real entity. (1) covers the reverse case, where the remainder is not
    itself indexed.
    """
    parts = token.split('-')
    while len(parts) > 1:
        head, rest = parts[0], '-'.join(parts[1:])
        if (head.lower() in _MODS or _SITE.match(head)
                or (idx is not None and idx.knows(rest.strip()))):
            parts.pop(0)
        else:
            break
    return '-'.join(parts).strip()


def split_entity(name, idx=None):
    """Break one participant name into its base entities and modifications.

    Returns (bases, mods):
      'p-S65-Ub:MFN1 [MOM]' -> ({'ub', 'mfn1'}, {'p-s65'})
    Complexes ('A:B') contribute every member, since a reaction touching the
    complex touches each of them. Passing `idx` lets modification stripping
    validate against real entity names instead of a fixed prefix list.
    """
    bases, mods = set(), set()
    for part in strip_compartment(name).split(':'):
        part = part.strip()
        if not part:
            continue
        base = strip_modifications(part, idx)
        # record what was peeled off, e.g. 'p-S65'
        if base != part:
            mods.add(part[:len(part) - len(base)].strip('-').lower())
        low = base.lower().strip()
        if low and low not in _STOPWORDS:
            bases.add(low)
    return bases, mods


# ─────────────────────────────────────────────────────────────────────────────
# 2. Alias index — Reactome gene symbols + UniProt common names
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache():
    if os.path.isfile(UNIPROT_CACHE):
        try:
            with open(UNIPROT_CACHE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(UNIPROT_CACHE), exist_ok=True)
    with open(UNIPROT_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)


def uniprot_names(accessions, verbose=True):
    """{uniprot_acc: [names]} — recommended and alternative protein names.

    Reactome only stores gene symbols, so 'Parkin' (what papers write) never
    maps to PRKN without this. Results are cached to disk; a lookup failure
    degrades to no extra aliases rather than aborting the run. Non-UniProt
    identifiers (Ensembl gene ids on 'X gene' entities) simply 404 and are
    cached as empty.
    """
    cache = _load_cache()
    todo = [a for a in accessions if a and a not in cache]
    if todo and verbose:
        print(f"[alias] fetching {len(todo)} UniProt record(s) for protein names...", flush=True)
    for acc in todo:
        names = []
        try:
            r = requests.get(UNIPROT_URL.format(acc=acc), timeout=20)
            r.raise_for_status()
            desc = r.json().get('proteinDescription') or {}
            rec = (desc.get('recommendedName') or {}).get('fullName', {}).get('value')
            if rec:
                names.append(rec)
            for alt in desc.get('alternativeNames') or []:
                v = (alt.get('fullName') or {}).get('value')
                if v:
                    names.append(v)
            for short in ((desc.get('recommendedName') or {}).get('shortNames') or []):
                if short.get('value'):
                    names.append(short['value'])
        except Exception:
            pass          # not a UniProt accession, or lookup failed — no aliases
        cache[acc] = names
        time.sleep(0.1)   # be polite to the UniProt API
    if todo:
        _save_cache(cache)
    return {a: cache.get(a, []) for a in accessions}


class AliasIndex:
    """Maps any surface form of an entity to a canonical token.

    Built from the participants of one gene's curated reactions: Reactome gene
    symbols give PRKN/PARK2, UniProt gives 'parkin' and 'E3 ubiquitin-protein
    ligase parkin'. Anything unknown maps to itself, so unmatched extraction
    terms still compare by string.
    """

    def __init__(self):
        self._to_canon = {}

    def add_group(self, canonical, surface_forms, overwrite=True):
        """Point every surface form at one canonical token.

        overwrite=False leaves a surface form already claimed by an earlier group
        alone. Needed when loading Reactome's whole vocabulary, where a synonym
        like 'ubiquitin' is claimed by several entities and last-write-wins would
        decide it arbitrarily.
        """
        canon = canonical.lower().strip()
        for s in list(surface_forms) + [canonical]:
            if not s:
                continue
            s = s.lower().strip()
            if s and s not in _STOPWORDS:
                if overwrite or s not in self._to_canon:
                    self._to_canon[s] = canon
                # 'e3 ubiquitin-protein ligase parkin' should also hit on 'parkin'
                last = s.split()[-1]
                if len(last) > 3 and last not in _STOPWORDS:
                    self._to_canon.setdefault(last, canon)

    def add_known(self, token):
        """Register a token as a real entity without overriding an existing alias.

        Participant names carry entities Reactome has no ReferenceSequence for —
        'Ub' is the common case, since ubiquitin is modelled as a modification
        rather than a gene product. Without these, variant stripping has nothing
        to validate against and 'UbS65D' never reduces to 'Ub'.
        """
        t = (token or '').lower().strip()
        if t and t not in _STOPWORDS:
            self._to_canon.setdefault(t, t)

    def knows(self, token):
        """True if this surface form resolves to a curated entity.

        Used to validate variant stripping — see strip_variant().
        """
        return token.lower().strip() in self._to_canon

    def canon(self, token):
        return self._to_canon.get(token.lower().strip(), token.lower().strip())

    def canon_set(self, tokens):
        return {self.canon(t) for t in tokens if t}

    def items(self):
        """(surface form, canonical) pairs, for merging one index into another."""
        return self._to_canon.items()

    def __len__(self):
        return len(self._to_canon)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ground truth WITH participants
# ─────────────────────────────────────────────────────────────────────────────
# Unlike query_reaction_roles_of_pathway (which returns only a displayName and
# the queried gene's role), this returns the participants that make a reaction
# identifiable. Complexes and sets are expanded so a reaction on 'PRKN:Ub-MOM
# proteins' is matchable by PRKN.
GT_QUERY = """
MATCH (ewas:EntityWithAccessionedSequence)-[:referenceEntity]->(g:ReferenceSequence)
WHERE g.geneName[0] = $gene
MATCH (p:Pathway)-[:hasEvent*]->(r:ReactionLikeEvent)
MATCH (r)-[:input|output|catalystActivity|physicalEntity|hasComponent|hasMember|hasCandidate*1..5]->(ewas)
// group by reaction here: a bare WITH DISTINCT r would drop p before collecting it
WITH r, collect(DISTINCT p.displayName) AS pathways
OPTIONAL MATCH (r)-[:input]->(i:PhysicalEntity)
OPTIONAL MATCH (r)-[:output]->(o:PhysicalEntity)
OPTIONAL MATCH (r)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(c:PhysicalEntity)
OPTIONAL MATCH (r)-[:input|output|catalystActivity|physicalEntity|hasComponent|hasMember|hasCandidate*1..5]
              ->(pe:EntityWithAccessionedSequence)-[:referenceEntity]->(ref:ReferenceSequence)
RETURN r.displayName            AS reaction,
       r.dbId                   AS db_id,
       r.stId                   AS st_id,
       pathways                 AS pathways,
       collect(DISTINCT i.displayName)  AS inputs,
       collect(DISTINCT o.displayName)  AS outputs,
       collect(DISTINCT c.displayName)  AS catalysts,
       collect(DISTINCT ref.identifier) AS uniprots,
       collect(DISTINCT ref.geneName)   AS gene_names,
       // Reactome's own synonym lists for the participants. This is the only
       // synonym source for entities with no ReferenceSequence -- Ub, small
       // molecules, named sets -- and it is per-entity data, so it generalizes
       // to whatever participants a gene happens to have.
       collect(DISTINCT i.name) + collect(DISTINCT o.name)
                                + collect(DISTINCT c.name) AS participant_names,
       // paired so a UniProt name is filed under Reactome's gene symbol
       // rather than becoming its own canonical form
       collect(DISTINCT [ref.identifier, ref.geneName[0]]) AS ref_pairs
"""

# Reactome keeps superseded records with these markers in the displayName.
# Scoring against a retired reaction is never meaningful.
_DEPRECATED = re.compile(r'^\s*clone of\b|\breplaced by\b|\bdeleted\b', re.I)


def is_deprecated(reaction_name):
    return bool(_DEPRECATED.search(reaction_name or ''))


# Reactome's whole entity-synonym vocabulary. PhysicalEntity.name is an array
# whose first element is the preferred label and the rest are synonyms, so the
# database already knows Ub = ubiquitin, ATP = adenosine 5'-triphosphate, and so
# on for every entity it curates.
# Species-filtered to match the corpus being matched against.
GLOBAL_ALIAS_QUERY = """
MATCH (e:PhysicalEntity)-[:species]->(:Species {displayName:$species})
WHERE e.name IS NOT NULL AND size(e.name) > 1
RETURN DISTINCT e.name AS names
"""


def fetch_global_aliases(neo4jutils, idx=None, species='Homo sapiens', verbose=True):
    """Alias index over every synonym Reactome records, for any entity.

    Harvesting only a gene's own participants is not enough: the reaction
    'PINK1 phosphorylates Ub on MOM proteins' has input 'Ub-MOM proteins', whose
    name array is just ['Ub-MOM proteins']. The array carrying ['Ub','ubiquitin']
    lives on a different node the gene never touches. One global pass picks it
    up, and does the same for every other entity, so nothing is hand-curated.

    Loaded BEFORE any gene-specific group, so a gene's own naming wins on
    conflict.
    """
    from neo4j import GraphDatabase
    idx = idx if idx is not None else AliasIndex()
    with GraphDatabase.driver(neo4jutils.URI, auth=neo4jutils.AUTH) as driver:
        with driver.session(database=neo4jutils.DB) as s:
            rows = [r['names'] for r in s.run(GLOBAL_ALIAS_QUERY, species=species)]

    groups = []
    for names in rows:
        names = [strip_compartment(x) for x in (names or []) if x]
        names = [x for x in names if x and x.lower() not in _STOPWORDS]
        if len(names) > 1:
            groups.append(names)

    # A synonym can be claimed by several entities, so the order groups are
    # loaded in decides the winner. Shortest preferred-name first, first write
    # wins: that is deterministic, and it favours the compact canonical label
    # ('Ub') over an incidental longer one that lists the same synonym.
    groups.sort(key=lambda ns: (len(ns[0]), ns[0].lower()))
    for names in groups:
        idx.add_group(names[0], names, overwrite=False)
    if verbose:
        print(f"[alias] {len(groups):,} synonym groups from Reactome's "
              f"{species} entity vocabulary", flush=True)
    fetch_modifier_prefixes(neo4jutils, idx, species=species, verbose=verbose)
    return idx


# Every participant name, to learn how Reactome writes modifications.
MODIFIER_SCAN_QUERY = """
MATCH (e:PhysicalEntity)-[:species]->(:Species {displayName:$species})
WHERE e.displayName CONTAINS '-'
RETURN DISTINCT e.displayName AS name
"""


def fetch_modifier_prefixes(neo4jutils, idx, species='Homo sapiens', verbose=True):
    """Learn Reactome's modification prefixes instead of listing them.

    Reactome writes a modified entity as '<modifier>-<entity>', and the entity
    part almost always exists in its own right. So scanning display names and
    keeping the prefix wherever the remainder is a known entity recovers the
    vocabulary from the data: polyUb, GlcNAc, and whatever else the database
    uses, without anyone enumerating PTMs by hand.

    Extends the module-level _MODS, which strip_modifications consults when the
    remainder is NOT indexed and rule (3) cannot fire.
    """
    from neo4j import GraphDatabase
    with GraphDatabase.driver(neo4jutils.URI, auth=neo4jutils.AUTH) as driver:
        with driver.session(database=neo4jutils.DB) as s:
            names = [r['name'] for r in s.run(MODIFIER_SCAN_QUERY, species=species)]

    found = {}
    for name in names:
        parts = strip_compartment(name).split('-')
        for k in range(1, len(parts)):
            head = '-'.join(parts[:k]).strip().lower()
            rest = '-'.join(parts[k:]).strip()
            # a plausible prefix: short, alphabetic-ish, and leaving a real entity
            if head and len(head) <= 14 and not _SITE.match(head) and idx.knows(rest):
                found[head] = found.get(head, 0) + 1

    # Seen once, it is as likely a hyphenated name as a modifier. Requiring a
    # few occurrences keeps the set to conventions rather than accidents.
    learned = {h for h, n in found.items() if n >= 3}
    _MODS.update(learned)
    if verbose:
        print(f"[mods] learned {len(learned)} modification prefix(es) from "
              f"Reactome naming", flush=True)
    return learned


def fetch_ground_truth(neo4jutils, gene, drop_deprecated=True, verbose=True,
                       base_idx=None):
    """Curated reactions for a gene, each with its participant entity sets.

    Returns (rows, alias_index). Each row carries the raw participant names
    ('inputs'/'outputs'/'catalysts'), normalized 'entities' and
    'catalyst_entities' sets for structural matching, a 'pathway' string and a
    'roles' label for reporting, and a 'text' field for the ranking stages.
    """
    import neo4j
    from neo4j import GraphDatabase

    with GraphDatabase.driver(neo4jutils.URI, auth=neo4jutils.AUTH) as driver:
        df = driver.execute_query(GT_QUERY, db=neo4jutils.DB, gene=gene,
                                  result_transformer_=neo4j.Result.to_df)
    if df is None or df.empty:
        return [], AliasIndex()

    rows, dropped = [], 0
    for _, r in df.iterrows():
        if drop_deprecated and is_deprecated(r['reaction']):
            dropped += 1
            continue
        rows.append(dict(r))
    if verbose and dropped:
        print(f"[neo4j] dropped {dropped} deprecated reaction(s) (Clone of / replaced by)",
              flush=True)

    # Alias index: Reactome symbols first, then UniProt common names filed under
    # the symbol Reactome uses for that accession. base_idx seeds it with a
    # wider vocabulary (see fetch_global_aliases) so participant entity sets are
    # built with those aliases already in force, not canonicalized afterwards.
    idx = base_idx if base_idx is not None else AliasIndex()
    for row in rows:
        for names in (row['gene_names'] or []):
            names = [n for n in (names or []) if n]
            if names:
                idx.add_group(names[0], names)
    acc2symbol = {}
    for row in rows:
        for pair in (row.get('ref_pairs') or []):
            if pair is not None and len(pair) == 2 and pair[0] and pair[1]:
                acc2symbol[pair[0]] = pair[1]
    accs = sorted({a for row in rows for a in (row['uniprots'] or []) if a})
    for acc, names in uniprot_names(accs, verbose=verbose).items():
        if names:
            idx.add_group(acc2symbol.get(acc, names[0]), names)

    # Reactome's per-entity synonym lists. Covers the participants that have no
    # gene symbol, which UniProt above cannot reach. Nothing here is curated by
    # hand, so it holds for any gene's participants.
    for row in rows:
        for names in (row.get('participant_names') or []):
            names = [n for n in (names or []) if n]
            if len(names) > 1:
                idx.add_group(strip_compartment(names[0]),
                              [strip_compartment(n) for n in names])

    gene_canon = idx.canon(gene)
    for row in rows:
        ents, mods = set(), set()
        for field in ('inputs', 'outputs', 'catalysts'):
            for name in (row[field] or []):
                b, m = split_entity(name, idx)
                ents |= b
                mods |= m
        cats = set()
        for name in (row['catalysts'] or []):
            cats |= split_entity(name, idx)[0]
        # canonicalize AFTER splitting so aliases apply to base names. A
        # modifier that is itself a curated entity counts as a participant:
        # 'Ub-MOM proteins' splits to base {MOM proteins} + modifier {Ub}, and a
        # reaction acting on the ubiquitin shares nothing without it.
        row['entities'] = idx.canon_set(ents) | {idx.canon(m) for m in mods if idx.knows(m)}
        row['catalyst_entities'] = idx.canon_set(cats)
        # Every curated participant is by definition a real entity, so register
        # it. This is what lets variant stripping validate against entities that
        # have no gene symbol (Ub, small complexes, named protein sets).
        for t in row['entities']:
            idx.add_known(t)
        row['modifications'] = mods
        paths = sorted(p for p in (row.get('pathways') or []) if p)
        row['pathway'] = '; '.join(paths[:3])
        row['roles'] = 'catalyst' if gene_canon in row['catalyst_entities'] else 'participant'
        row['text'] = _gt_text(row)
    return rows, idx


# Resolve free-text entity names to Reactome gene symbols
SYMBOL_QUERY = """
MATCH (g:ReferenceSequence)
WHERE any(n IN g.geneName WHERE toLower(n) IN $tokens)
   OR any(n IN g.name     WHERE toLower(n) IN $tokens)
RETURN DISTINCT g.geneName[0] AS symbol
"""


def resolve_gene_symbols(neo4jutils, tokens):
    """Which of these surface forms are real Reactome gene symbols.

    Canonicalize tokens through an AliasIndex first, so 'Parkin' arrives here as
    'prkn' — Reactome stores symbols, not protein common names.
    """
    import neo4j
    from neo4j import GraphDatabase
    toks = sorted({t.lower().strip() for t in tokens if t and t.lower() not in _STOPWORDS})
    if not toks:
        return set()
    with GraphDatabase.driver(neo4jutils.URI, auth=neo4jutils.AUTH) as driver:
        df = driver.execute_query(SYMBOL_QUERY, db=neo4jutils.DB, tokens=toks,
                                  result_transformer_=neo4j.Result.to_df)
    if df is None or df.empty:
        return set()
    return {s for s in df['symbol'].tolist() if s}


def fetch_ground_truth_multi(neo4jutils, genes, drop_deprecated=True, verbose=True,
                             base_idx=None):
    """Union of curated reactions across several genes, deduped by dbId.

    Scoping ground truth to one gene understates precision: a PINK1 paper also
    describes Parkin reactions, and if Reactome's Parkin reactions are never
    candidates, a correct extraction gets scored against the wrong target or
    counted as a miss.
    """
    merged, seen = [], set()
    idx = base_idx if base_idx is not None else AliasIndex()
    for gene in genes:
        rows, sub_idx = fetch_ground_truth(neo4jutils, gene,
                                           drop_deprecated=drop_deprecated,
                                           verbose=verbose, base_idx=base_idx)
        idx._to_canon.update(sub_idx._to_canon)
        added = 0
        for r in rows:
            key = r.get('db_id') or r['reaction']
            if key not in seen:
                seen.add(key)
                r['source_gene'] = gene
                merged.append(r)
                added += 1
        if verbose:
            print(f"[neo4j] {gene}: {len(rows)} reaction(s), {added} new", flush=True)
    return merged, idx


def _gt_text(row):
    """Compact prose for the cosine / cross-encoder stages.

    Deliberately parallel in shape to extracted_reaction_text() — comparing a
    20-word template against a 200-word blob inflates similarity through shared
    boilerplate rather than shared meaning.
    """
    def clean(names):
        return ', '.join(strip_compartment(n) for n in (names or [])[:6]) or 'none'
    return (f"{row['reaction']}. "
            f"Inputs: {clean(row['inputs'])}. "
            f"Outputs: {clean(row['outputs'])}. "
            f"Catalyst: {clean(row['catalysts'])}.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Extracted-side entities and text
# ─────────────────────────────────────────────────────────────────────────────
def resolve_entity(name, idx):
    """Canonical tokens for one extracted entity name.

    Tries progressively stripped variants and takes the first the alias index
    recognises, so 'ParkinC431S' and 'Parkin phospho-Ser65' both reach PRKN
    while 'UBE2N' is left alone.

    When nothing is recognised — a gene whose entities Reactome does not curate,
    or a synonym it does not record — fall back to the MOST-stripped form rather
    than the raw one. Stripping only removes decoration (compartments, mutation
    tags, modification affixes), so the reduced form is the more comparable
    token: 'Phospho-ubiquitin (UbS65D)' is worth carrying forward as
    'ubiquitin', not as 'ubiquitin (ubs65d)'.
    """
    fallback = set()
    for form in strip_variant(name):
        bases, _ = split_entity(form, idx)
        if any(idx.knows(b) for b in bases):
            return idx.canon_set(bases)
        if bases:
            fallback = idx.canon_set(bases)
    return fallback


def _participant(name, idx):
    """Canonical entities in one participant name, keeping a modifier that is
    itself a curated entity — see the matching note in fetch_ground_truth."""
    out = set(resolve_entity(name, idx))
    out |= {idx.canon(m) for m in split_entity(name, idx)[1] if idx.knows(m)}
    return out


def extracted_entities(annotation, idx):
    """Normalized entity set and catalyst set for one extracted reaction."""
    ents, cats = set(), set()
    for field in ('input', 'output'):
        for name in (annotation.get(field) or []):
            ents |= _participant(name, idx)
    ca = annotation.get('catalystActivity') or {}
    if ca.get('catalyst'):
        cats |= resolve_entity(ca['catalyst'], idx)
    for reg in (annotation.get('regulatedBy') or []):
        if reg.get('regulator'):
            ents |= resolve_entity(reg['regulator'], idx)
    ents |= cats
    return ents, cats


def extracted_reaction_text(a):
    """Compact prose mirroring _gt_text — name plus participants, nothing else.

    The long version (summation, relationships, evidence) mostly adds generic
    domain vocabulary, which raises similarity without adding discrimination.
    """
    ca = a.get('catalystActivity') or {}
    def clean(xs):
        return ', '.join(xs[:6]) if xs else 'none'
    return (f"{a.get('name', '')}. "
            f"Inputs: {clean(a.get('input') or [])}. "
            f"Outputs: {clean(a.get('output') or [])}. "
            f"Catalyst: {ca.get('catalyst') or 'none'}.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Structural filter
# ─────────────────────────────────────────────────────────────────────────────
def structural_candidates(ext_entities, ext_catalysts, gt_rows, min_shared=1,
                          require_catalyst=False, query_gene=None):
    """Indices of curated reactions that could plausibly be the same event.

    Returns (gt_index, n_shared, catalyst_match) per survivor.

    query_gene is excluded from the shared count: every curated reaction for
    gene G contains G by construction, so it carries no discriminating
    information between them. It still counts for catalyst_match, because
    whether G is the catalyst or the substrate genuinely distinguishes
    reactions. A shared catalyst is much stronger evidence than a shared
    participant, so it is reported separately and can be required outright.
    """
    ignore = {query_gene.lower().strip()} if query_gene else set()
    out = []
    for j, g in enumerate(gt_rows):
        shared = (ext_entities & g['entities']) - ignore
        cat_match = bool(ext_catalysts & g['catalyst_entities'])
        if require_catalyst and not cat_match:
            continue
        if len(shared) >= min_shared or cat_match:
            out.append((j, len(shared), cat_match))
    return out
