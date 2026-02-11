SIMPLE_USER = """
Choose your answer: can we conclude that "{claim}"?
OPTIONS:
- Yes: I know the answer is yes base on my knowledge
- No: I know the answer is no base on my knowledge
- Not Enough Information: I don't have enough information to answer the question
I think the answer is [one of Yes, No, Not Enough Information without explanation]"""

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