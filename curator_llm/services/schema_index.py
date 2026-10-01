"""Attribute validation against resources/reactome_domain_model.json (inherited fields included)."""
import json
import os
from functools import lru_cache
from typing import Dict, Set

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'reactome_domain_model.json')


@lru_cache(maxsize=1)
def _classes() -> Dict[str, dict]:
    with open(_PATH) as f:
        return {c['className']: c for c in json.load(f)['classes']}


def has_class(name: str) -> bool:
    return name in _classes()


@lru_cache(maxsize=None)
def attributes_of(class_name: str) -> Set[str]:
    out: Set[str] = set()
    k = class_name
    while k and k in _classes():
        out |= {f['name'] for f in _classes()[k].get('fields', [])}
        k = _classes()[k].get('extends')
    return out


def check_instance(inst: dict) -> list:
    """Problems with one emitted instance: unknown class, or attribute names not in that class."""
    cls = inst.get('schemaClassName')
    if not has_class(cls):
        return [f'unknown class {cls}']
    allowed = attributes_of(cls)
    return [f'{cls}.{a} is not a schema attribute' for a in (inst.get('attributes') or {}) if a not in allowed]


@lru_cache(maxsize=None)
def collection_attributes_of(class_name: str) -> Set[str]:
    """Attributes that are lists in the schema (e.g. name, hasComponent)."""
    out: Set[str] = set()
    k = class_name
    while k and k in _classes():
        out |= {f['name'] for f in _classes()[k].get('fields', []) if f.get('collection')}
        k = _classes()[k].get('extends')
    return out
