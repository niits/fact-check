from pydantic import BaseModel


class EvidenceItem(BaseModel):
    """A single piece of evidence with its related context."""
    source: str = ""
    content: str
    context: str = ""


class FeverousSample(BaseModel):
    """A single FEVEROUS dataset sample."""
    claim: str
    evidence: list[EvidenceItem] | str | None = None
    label: str | None = None
