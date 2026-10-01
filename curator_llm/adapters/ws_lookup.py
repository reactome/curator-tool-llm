"""InstanceLookup backed by the existing curator-tool-ws read endpoints:
GET findByDisplayName?displayName=&classNames=  and  GET searchInstances/{class}/{skip}/{limit}."""
from typing import Dict, List, Optional

import requests


def _norm(inst: Optional[dict]) -> Optional[Dict]:
    if not inst or inst.get('dbId') is None:
        return None
    return {'dbId': inst['dbId'], 'displayName': inst.get('displayName') or '',
            'schemaClassName': inst.get('schemaClassName') or inst.get('className') or ''}


class WsInstanceLookup:
    def __init__(self, base_url: str, token_getter, timeout: float = 15.0, session=None):
        self.base = base_url.rstrip('/') + '/api/curation'
        self.token_getter, self.timeout = token_getter, timeout
        self.http = session or requests

    def _get(self, path: str, params=None):
        r = self.http.get(f'{self.base}/{path}', params=params, timeout=self.timeout,
                          headers={'Authorization': f'Bearer {self.token_getter()}'})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json() if r.content else None

    def find_by_display_name(self, display_name: str, class_names: List[str]) -> Optional[Dict]:
        return _norm(self._get('findByDisplayName',
                               {'displayName': display_name, 'classNames': ','.join(class_names)}))

    def search(self, class_name: str, attribute: str, value: str, operand: str = 'equal',
               limit: int = 5) -> List[Dict]:
        data = self._get(f'searchInstances/{class_name}/0/{int(limit)}',
                         {'attributes': attribute, 'operands': operand, 'searchKeys': value})
        return [i for i in (_norm(x) for x in (data or {}).get('instances', [])) if i]

    def find_by_db_id(self, db_id: int) -> Optional[Dict]:
        return _norm(self._get(f'findByDbId/{int(db_id)}'))
