"""Identifier lookups run in parallel; the result must not depend on that."""
import threading
import time

from curator_llm.models.reactome import EwasSpec, ReactomeDraft, SimpleEntitySpec
from curator_llm.services.resolvers import Resolver
from tests.fakes.ports import FakeInstanceLookup, FakeUniProt


class SlowLookup(FakeInstanceLookup):
    """Every lookup takes `delay` seconds, like a network round trip; records how many ran at once."""

    def __init__(self, instances, delay):
        super().__init__(instances)
        self.delay, self.active, self.peak, self._lock = delay, 0, 0, threading.Lock()

    def _slow(self, fn, *a, **k):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        try:
            return fn(*a, **k)
        finally:
            with self._lock:
                self.active -= 1

    def find_by_display_name(self, *a, **k):
        return self._slow(super().find_by_display_name, *a, **k)

    def search(self, *a, **k):
        return self._slow(super().search, *a, **k)


def draft():
    d = ReactomeDraft()
    for i in range(8):
        d.participants[f'p{i}'] = EwasSpec(key=f'p{i}', name=f'GENE{i}', uniprot=f'Q0000{i}', compartment_name='cytosol')
    d.participants['atp'] = SimpleEntitySpec(key='atp', name='ATP', chebi='CHEBI:30616')
    return d


UNIPROT = {f'Q0000{i}': {'accession': f'Q0000{i}', 'genes': [f'WRONG{i}'], 'names': ['x'], 'reviewed': True,
                         'organism': 'Homo sapiens'} for i in range(8)}


def run(workers, delay=0.0):
    lookup = SlowLookup([{'dbId': 1, 'displayName': 'cytosol', 'schemaClassName': 'Compartment'}], delay)
    r = Resolver(lookup, FakeUniProt(UNIPROT))
    r.workers = workers
    d = draft()
    return d, r.resolve(d), lookup


def test_lookups_overlap_and_are_faster_than_one_at_a_time():
    t = time.time(); run(1, 0.05); serial = time.time() - t
    t = time.time(); _, _, lookup = run(8, 0.05); parallel = time.time() - t
    assert lookup.peak > 1
    assert parallel < serial / 2


def test_result_and_note_order_are_the_same_as_one_at_a_time():
    d1, n1, _ = run(1)
    d8, n8, _ = run(8, 0.01)
    assert n8 == n1 and len(n8) == 8                      # one "does not match the name" note per protein, in order
    assert [m.split(':')[0] for m in n8] == [f'GENE{i}' for i in range(8)]
    assert d8.model_dump() == d1.model_dump()


def test_a_service_failure_is_one_note_not_one_per_thread():
    class Down(FakeUniProt):
        def fetch(self, accession):
            raise ConnectionError('down')
    d = draft()
    r = Resolver(FakeInstanceLookup([]), Down({}))
    notes = r.resolve(d)
    assert notes.count('UniProt unavailable (ConnectionError); identifiers it would supply were left unresolved') == 1
