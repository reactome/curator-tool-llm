from typing import Dict, List, Optional, Protocol


class InstanceLookup(Protocol):
    """Read-only access to gk_central instances (through curator-tool-ws).

    Instances come back as dicts with at least dbId, displayName and schemaClassName."""

    def find_by_display_name(self, display_name: str, class_names: List[str]) -> Optional[Dict]:
        """The instance with exactly this displayName in one of the classes, or None."""
        ...

    def search(self, class_name: str, attribute: str, value: str, operand: str = 'equal',
               limit: int = 5) -> List[Dict]:
        """Instances of class_name whose attribute matches value (operand: equal | contains | regex)."""
        ...

    def find_by_db_id(self, db_id: int) -> Optional[Dict]: ...
