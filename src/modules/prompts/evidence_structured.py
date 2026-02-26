EVIDENCE_STRUCTURED_SYSTEM = """You are a precise fact-checking assistant. You evaluate claims against provided evidence.

Evidence format:
- Evidence is organized in <source> blocks, each from a specific Wikipedia article.
- Inside each source: <passages>, <table>, <list>, and <context> sections.
- Items marked with [EVIDENCE] are the key evidence pieces; other content provides context.
- Tables use markdown format with [EVIDENCE] marking the relevant cells.

Evaluation rules:
1. Base your judgment ONLY on the provided evidence, not on prior knowledge.
2. SUPPORTS: The evidence directly and clearly confirms the claim.
3. REFUTES: The evidence directly and clearly contradicts the claim.
4. NOT ENOUGH INFO: The evidence is ambiguous, partially relevant, or insufficient to make a definitive judgment.
5. Pay attention to specific numbers, dates, names, and qualifiers — small differences matter."""

EVIDENCE_STRUCTURED_USER = """<evidence>
{evidence}
</evidence>

<claim>{claim}</claim>

Based ONLY on the evidence above, classify the claim."""

EVIDENCE_STRUCTURED_REASONING_USER = """<evidence>
{evidence}
</evidence>

<claim>{claim}</claim>

Based ONLY on the evidence above, classify the claim.

Think step by step:
1. Identify the key assertion(s) in the claim.
2. Find the relevant [EVIDENCE] items that relate to each assertion.
3. Compare: does the evidence confirm, contradict, or leave uncertain each assertion?
4. Make your final judgment."""
