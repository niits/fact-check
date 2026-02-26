from enum import Enum

from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from workflows import Workflow, step
from workflows.events import StopEvent

from src.modules.prompts.simple import SIMPLE_USER, SIMPLE_REASONING_USER
from src.modules.prompts.evidence_simple import (
    EVIDENCE_SIMPLE_SYSTEM,
    EVIDENCE_SIMPLE_USER,
    EVIDENCE_SIMPLE_REASONING_USER,
)
from src.modules.prompts.evidence_structured import (
    EVIDENCE_STRUCTURED_SYSTEM,
    EVIDENCE_STRUCTURED_USER,
    EVIDENCE_STRUCTURED_REASONING_USER,
)
from ..events.base import FactCheckStartEvent


# ── Structured output models ─────────────────────────────────────────

class FactCheckLabel(str, Enum):
    """Constrained label enum for structured output."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT ENOUGH INFO"


class FactCheckResult(BaseModel):
    """Structured output for fact-check classification."""
    label: FactCheckLabel = Field(
        description="The fact-check verdict: SUPPORTS, REFUTES, or NOT ENOUGH INFO"
    )


class FactCheckReasoningResult(BaseModel):
    """Structured output for fact-check with reasoning."""
    reasoning: str = Field(
        description="Step-by-step reasoning explaining how the evidence relates to the claim"
    )
    label: FactCheckLabel = Field(
        description="The fact-check verdict: SUPPORTS, REFUTES, or NOT ENOUGH INFO"
    )


_LABEL_MAP = {
    FactCheckLabel.SUPPORTS: "SUPPORT",
    FactCheckLabel.REFUTES: "REFUTE",
    FactCheckLabel.NOT_ENOUGH_INFO: "NEI",
}


def _map_label(label: FactCheckLabel) -> str:
    return _LABEL_MAP[label]


class SimpleBaseFactCheck(Workflow):
    def __init__(
            self,
            llm: LLM,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        context = ev.context
        claim = ev.claim
        prompt = PromptTemplate(SIMPLE_USER)
        result = await self.llm.astructured_predict(
            FactCheckResult, prompt, context=context, claim=claim
        )
        return StopEvent(_map_label(result.label))


class SimpleReasoningFactCheck(Workflow):
    def __init__(
            self,
            llm: LLM,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        context = ev.context
        claim = ev.claim
        prompt = PromptTemplate(SIMPLE_REASONING_USER)
        result = await self.llm.astructured_predict(
            FactCheckReasoningResult, prompt, context=context, claim=claim
        )
        return StopEvent({
            "label": _map_label(result.label),
            "reasoning": result.reasoning,
        })


class EvidenceSimpleBaseFactCheck(Workflow):
    """Uses evidence_simple prompt. Pass evidence as list[EvidenceItem]."""

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @staticmethod
    def format_evidence(evidence: list) -> str:
        """Format list of EvidenceItem into a structured prompt string."""
        parts = []
        for i, ev in enumerate(evidence, 1):
            header = f"Evidence {i}"
            if ev.source:
                header += f" [Source: {ev.source}]"
            header += ":"
            section = [header, ev.content]
            if ev.context:
                section.append(f"Context: {ev.context}")
            parts.append("\n".join(section))
        return "\n\n".join(parts)

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        evidence_str = self.format_evidence(ev.evidence) if ev.evidence else ""
        claim = ev.claim
        prompt = PromptTemplate(
            EVIDENCE_SIMPLE_USER,
            metadata={"system_prompt": EVIDENCE_SIMPLE_SYSTEM},
        )
        result = await self.llm.astructured_predict(
            FactCheckResult, prompt, evidence=evidence_str, claim=claim
        )
        return StopEvent(_map_label(result.label))


class EvidenceSimpleReasoningFactCheck(Workflow):
    """Uses evidence_simple prompt with reasoning."""

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        evidence_str = EvidenceSimpleBaseFactCheck.format_evidence(ev.evidence) if ev.evidence else ""
        claim = ev.claim
        prompt = PromptTemplate(
            EVIDENCE_SIMPLE_REASONING_USER,
            metadata={"system_prompt": EVIDENCE_SIMPLE_SYSTEM},
        )
        result = await self.llm.astructured_predict(
            FactCheckReasoningResult, prompt, evidence=evidence_str, claim=claim
        )
        return StopEvent({
            "label": _map_label(result.label),
            "reasoning": result.reasoning,
        })


class StructuredEvidenceFactCheck(Workflow):
    """Uses structured XML evidence format with system prompt."""

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        evidence_str = ev.evidence if isinstance(ev.evidence, str) else ""
        claim = ev.claim
        prompt = PromptTemplate(
            EVIDENCE_STRUCTURED_USER,
            metadata={"system_prompt": EVIDENCE_STRUCTURED_SYSTEM},
        )
        result = await self.llm.astructured_predict(
            FactCheckResult, prompt, evidence=evidence_str, claim=claim
        )
        return StopEvent(_map_label(result.label))


class StructuredEvidenceReasoningFactCheck(Workflow):
    """Uses structured XML evidence format with step-by-step reasoning."""

    def __init__(self, llm: LLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def fact_check(self, ev: FactCheckStartEvent) -> StopEvent:
        evidence_str = ev.evidence if isinstance(ev.evidence, str) else ""
        claim = ev.claim
        prompt = PromptTemplate(
            EVIDENCE_STRUCTURED_REASONING_USER,
            metadata={"system_prompt": EVIDENCE_STRUCTURED_SYSTEM},
        )
        result = await self.llm.astructured_predict(
            FactCheckReasoningResult, prompt, evidence=evidence_str, claim=claim
        )
        return StopEvent({
            "label": _map_label(result.label),
            "reasoning": result.reasoning,
        })