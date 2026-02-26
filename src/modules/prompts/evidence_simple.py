EVIDENCE_SIMPLE_SYSTEM = """You are a precise fact-checking assistant. You evaluate claims against provided evidence.

Evaluation rules:
1. Base your judgment ONLY on the provided evidence, not on prior knowledge.
2. SUPPORTS: The evidence directly and clearly confirms the claim.
3. REFUTES: The evidence directly and clearly contradicts the claim.
4. NOT ENOUGH INFO: The evidence is ambiguous, partially relevant, or insufficient to make a definitive judgment.
5. Pay attention to specific numbers, dates, names, and qualifiers — small differences matter."""

EVIDENCE_SIMPLE_USER = """Given the following evidence and claim, determine whether the evidence supports, refutes, or is insufficient to verify the claim.

{evidence}

Claim: "{claim}"
"""

EVIDENCE_SIMPLE_REASONING_USER = """Given the following evidence and claim, determine whether the evidence supports, refutes, or is insufficient to verify the claim.

{evidence}

Claim: "{claim}"

Think step by step:
1. Identify the key assertion(s) in the claim.
2. Find the relevant evidence that relates to each assertion.
3. Compare: does the evidence confirm, contradict, or leave uncertain each assertion?
4. Make your final judgment."""