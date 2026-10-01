"""MongoDB-backed SessionStore (the Mongo already used for the PubMed cache). One document per
session, keyed by the session id; owner is part of every read filter."""
import time
from typing import List, Optional

from curator_llm.models.session import Session


class MongoSessionStore:
    def __init__(self, collection):
        self.c = collection
        self.c.create_index([('owner', 1), ('created', -1)])

    @classmethod
    def from_env(cls, db_name: str = 'curator_llm', collection: str = 'sessions') -> 'MongoSessionStore':
        import os
        from pymongo import MongoClient
        return cls(MongoClient(os.getenv('PUBMED_MONGO_URI'), serverSelectionTimeoutMS=3000)[db_name][collection])

    def save(self, session: Session) -> None:
        session.updated = time.time()
        doc = session.model_dump(mode='json')
        doc['_id'] = session.id
        self.c.replace_one({'_id': session.id}, doc, upsert=True)

    def get(self, session_id: str, owner: str) -> Optional[Session]:
        doc = self.c.find_one({'_id': session_id, 'owner': owner})
        return Session.model_validate({k: v for k, v in doc.items() if k != '_id'}) if doc else None

    def list(self, owner: str) -> List[Session]:
        return [Session.model_validate({k: v for k, v in d.items() if k != '_id'})
                for d in self.c.find({'owner': owner}).sort('created', -1)]
