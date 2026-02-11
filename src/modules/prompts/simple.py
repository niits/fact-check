SIMPLE_USER = """
Determine whether the following claim is factually correct based on your internal knowledge.

Claim: "{claim}"

Respond with exactly one of the following:
- Yes
- No
- Not Enough Information

Do not provide any explanation.
"""

SIMPLE_REASONING_USER = """{context}
Choose your answer: based on the paragraph above can we conclude that "{claim}"?
OPTIONS:
- Yes: The context has information that SUPPORTS claim
- No: The context has information that CONTRADICTS the claim
- Not Enough Information: The context doesn't has enough information that support claim
Please answer using the following template
```
Reasoning: [Reason for the answer]
Answer: [one of Yes, No, Not Enough Information]
```
"""