"""InstanceLookup backed directly by a Reactome Neo4j graph (read-only).

Much faster than going through ws, and needs no ws login. CAVEAT: the graph configured in .env
(REACTOME_NEO4J_*) is a Reactome release graph, not gk_central: work that curators have not yet
released is not in it, so a lookup miss here does not prove an instance is absent from gk_central.
dbIds of released objects are the same in both. Before committing, resolve against gk_central
(WsInstanceLookup) when unreleased duplicates matter.

Class and attribute names cannot be Cypher parameters, so they are checked against the schema and
a strict identifier pattern before being interpolated; values are always parameters."""
import re
from typing import Dict, List, Optional

from curator_llm.services import schema_index

_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_HUMAN = 'Homo sapiens'
_RETURN = ('RETURN n.dbId AS dbId, n.displayName AS displayName, '
           'coalesce(n.schemaClass, head([l IN labels(n) WHERE l <> "DatabaseObject"])) AS schemaClassName, '
           'n.speciesName AS species')


def _check(class_name: str, attribute: Optional[str] = None):
    if not _IDENT.match(class_name) or not schema_index.has_class(class_name):
        raise ValueError(f'unknown class {class_name!r}')
    if attribute is not None and (not _IDENT.match(attribute) or attribute not in schema_index.attributes_of(class_name)):
        raise ValueError(f'{attribute!r} is not an attribute of {class_name}')


def search_query(class_name: str, attribute: str, operand: str, limit: int):
    """(cypher, extra_params) for a single-attribute search. Works on Neo4j 4.4."""
    _check(class_name, attribute)
    is_list = attribute in schema_index.collection_attributes_of(class_name)
    ref = f'n.`{attribute}`'
    test = {'equal': 'toString({x}) = $v',
            'contains': 'toLower(toString({x})) CONTAINS toLower($v)',
            'regex': 'toString({x}) =~ $v'}.get(operand)
    if test is None:
        raise ValueError(f'unsupported operand {operand!r}')
    cond = f'any(x IN {ref} WHERE {test.format(x="x")})' if is_list else test.format(x=ref)
    return (f'MATCH (n:`{class_name}`) WHERE {ref} IS NOT NULL AND {cond} {_RETURN} LIMIT $limit',
            {'limit': int(limit)})


def _row(r) -> Dict:
    return {'dbId': r['dbId'], 'displayName': r['displayName'] or '',
            'schemaClassName': r['schemaClassName'] or '', 'species': r['species']}


class Neo4jInstanceLookup:
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None, driver=None):
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver, self.database = driver, database

    @classmethod
    def from_env(cls) -> 'Neo4jInstanceLookup':
        import os
        return cls(os.getenv('REACTOME_NEO4J_URI'), os.getenv('REACTOME_NEO4J_USER'),
                   os.getenv('REACTOME_NEO4J_PWD'), os.getenv('REACTOME_NEO4J_DATABASE'))

    def _run(self, cypher: str, **params) -> List[Dict]:
        with self.driver.session(database=self.database) as s:
            return [_row(r) for r in s.run(cypher, **params)]

    def close(self):
        self.driver.close()

    def find_by_display_name(self, display_name: str, class_names: List[str]) -> Optional[Dict]:
        for c in class_names:
            _check(c)
        rows = self._run(f'MATCH (n:DatabaseObject) WHERE n.displayName = $dn '
                         f'AND any(l IN labels(n) WHERE l IN $classes) {_RETURN} LIMIT 5',
                         dn=display_name, classes=list(class_names))
        human = [r for r in rows if r['species'] == _HUMAN]
        rows = human or rows
        return rows[0] if len(rows) == 1 else None          # several matches: ambiguous, no guess

    def search(self, class_name: str, attribute: str, value: str, operand: str = 'equal',
               limit: int = 5) -> List[Dict]:
        cypher, extra = search_query(class_name, attribute, operand, limit)
        return self._run(cypher, v=str(value), **extra)

    def find_human_accession(self, name: str) -> Optional[str]:
        """UniProt accession of the one human ReferenceGeneProduct in the graph known by `name`, or None.

        Tries gene names and synonyms first (PRKN, PARK2), then the UniProt short name kept in the description
        ("shortName: Parkin"). Several different accessions is ambiguous: no guess."""
        n = name.strip()
        if not n:
            return None
        base = ('MATCH (n:ReferenceGeneProduct)-[:species]->(:Taxon {displayName: $human}) WHERE n.identifier IS NOT NULL AND ')
        with self.driver.session(database=self.database) as s:
            ids = {r['id'] for r in s.run(
                base + 'any(g IN n.geneName WHERE toLower(g) = toLower($n)) RETURN DISTINCT n.identifier AS id',
                human=_HUMAN, n=n)}
            if not ids:
                short = re.compile(r'shortName: ' + re.escape(n) + r'(?= [A-Za-z]+(?: evidence=|:)|$)', re.I)
                ids = {r['id'] for r in s.run(
                    base + 'any(d IN n.description WHERE toLower(d) CONTAINS toLower($frag)) '
                    'RETURN n.identifier AS id, n.description AS d', human=_HUMAN, frag=f'shortName: {n}')
                    if any(short.search(d) for d in r['d'])}
        return next(iter(ids)) if len(ids) == 1 else None

    def candidate_genes(self, name: str) -> List[str]:
        """Gene symbols of the members of human Reactome sets named `name` (e.g. "Ub [cytosol]"), in alphabetical order.

        For a name that stands for a family of gene products and so has no single accession."""
        n = name.strip()
        if not n:
            return []
        with self.driver.session(database=self.database) as s:
            return sorted({r['g'] for r in s.run(
                'MATCH (s:DefinedSet)-[:species]->(:Taxon {displayName: $human}) '
                'WHERE toLower(s.displayName) STARTS WITH toLower($prefix) '
                'MATCH (s)-[:hasMember]->(:EntityWithAccessionedSequence)-[:referenceEntity]->(r:ReferenceEntity) '
                'WHERE r.geneName IS NOT NULL RETURN DISTINCT head(r.geneName) AS g', human=_HUMAN, prefix=f'{n} [')})

    def find_by_db_id(self, db_id: int) -> Optional[Dict]:
        rows = self._run(f'MATCH (n:DatabaseObject {{dbId: $id}}) {_RETURN} LIMIT 1', id=int(db_id))
        return rows[0] if rows else None


# ── existing-event lookup (EventLookup port) ────────────────────────────────────────────────────
_DEPRECATED = re.compile(r'^\s*clone of\b|\breplaced by\b|\bdeleted\b', re.I)
_SPAN = '-[:hasComponent|hasMember|hasCandidate*0..]->(:EntityWithAccessionedSequence)-[:referenceEntity]->(r:ReferenceEntity)'
_PARTICIPANTS = f"""
MATCH (e:ReactionLikeEvent) WHERE e.dbId IN $ids
CALL {{ WITH e OPTIONAL MATCH (e)-[:input]->(:PhysicalEntity){_SPAN} RETURN collect(DISTINCT r.identifier) AS ins }}
CALL {{ WITH e OPTIONAL MATCH (e)-[:output]->(:PhysicalEntity){_SPAN} RETURN collect(DISTINCT r.identifier) AS outs }}
CALL {{ WITH e OPTIONAL MATCH (e)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(:PhysicalEntity){_SPAN}
        RETURN collect(DISTINCT r.identifier) AS cats }}
RETURN e.dbId AS dbId, e.displayName AS displayName, e.stId AS stId, e.schemaClass AS schemaClass,
       ins, outs, cats
"""


def _event_rows(driver, database, ids):
    from curator_llm.ports.events import EventRow
    if not ids:
        return []
    with driver.session(database=database) as s:
        rows = [EventRow(r['dbId'], r['displayName'] or '', r['stId'] or '', r['schemaClass'] or '',
                         list(r['ins']), list(r['outs']), list(r['cats']))
                for r in s.run(_PARTICIPANTS, ids=list(ids))]
    return [r for r in rows if not _DEPRECATED.search(r.display_name)]


def events_citing(self, pmid: str):
    with self.driver.session(database=self.database) as s:
        ids = [r['id'] for r in s.run(
            'MATCH (e:ReactionLikeEvent)-[:literatureReference]->(:LiteratureReference {pubMedIdentifier: $p}) '
            'RETURN DISTINCT e.dbId AS id', p=int(pmid))]
    return _event_rows(self.driver, self.database, ids)


def candidate_events(self, catalyst_accessions, participant_accessions, limit: int = 100):
    with self.driver.session(database=self.database) as s:
        if catalyst_accessions:
            q = ('MATCH (e:ReactionLikeEvent)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(:PhysicalEntity)'
                 + _SPAN + ' WHERE r.identifier IN $accs RETURN DISTINCT e.dbId AS id LIMIT $limit')
            ids = [r['id'] for r in s.run(q, accs=list(catalyst_accessions), limit=int(limit))]
        elif participant_accessions:
            q = ('MATCH (e:ReactionLikeEvent)-[:input|output]->(:PhysicalEntity)' + _SPAN +
                 ' WHERE r.identifier IN $accs WITH e, count(DISTINCT r.identifier) AS k '
                 'ORDER BY k DESC LIMIT $limit RETURN e.dbId AS id')
            ids = [r['id'] for r in s.run(q, accs=list(participant_accessions), limit=int(limit))]
        else:
            ids = []
    return _event_rows(self.driver, self.database, ids)


Neo4jInstanceLookup.events_citing = events_citing
Neo4jInstanceLookup.candidate_events = candidate_events
