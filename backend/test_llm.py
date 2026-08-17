from llm import LLM


llm = LLM()

answer = llm.generate(
    "Explain Retrieval-Augmented Generation in simple terms."
)

print(answer)