from workflows.events import StartEvent

from src.modules.datasets.feverous.models import EvidenceItem


class FactCheckStartEvent(StartEvent):
    context: str = ""
    claim: str = ""
    evidence: list[EvidenceItem] | str | None = None