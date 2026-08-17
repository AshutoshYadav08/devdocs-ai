from document_loader import load_document
from chunker import chunk_text
from embeddings import EmbeddingModel
from vector_store import VectorStore
from retriever import Retriever
from prompt import build_rag_prompt
from llm import LLM


# ==========================================
# 1. Load document
# ==========================================

file_path = "data/documents/architecture.txt"

document = load_document(file_path)


# ==========================================
# 2. Chunk document
# ==========================================

chunks = chunk_text(
    document["text"],
    document["metadata"]["source"],
    chunk_size=300
)

print("Number of chunks:", len(chunks))


# ==========================================
# 3. Generate embeddings
# ==========================================

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.generate_embeddings(
    texts
)


# ==========================================
# 4. Create vector store
# ==========================================

vector_store = VectorStore(
    collection_name="devdocs",
    vector_size=len(embeddings[0])
)


# ==========================================
# 5. Store documents
# ==========================================

vector_store.add_documents(
    chunks,
    embeddings
)

print("Documents stored in Qdrant.")


# ==========================================
# 6. Create retriever
# ==========================================

retriever = Retriever(
    embedding_model,
    vector_store
)


# ==========================================
# 7. Ask question
# ==========================================

question = input("\nAsk a question: ")


# ==========================================
# 8. Retrieve relevant chunks
# ==========================================

results = retriever.retrieve(
    question,
    top_k=5,
    score_threshold=0.65
)


# ==========================================
# 9. Check whether relevant information exists
# ==========================================

if not results:

    print(
        "\nI don't have enough information "
        "in the provided documentation."
    )

else:

    # ======================================
    # 10. Display retrieved chunks
    # ======================================

    print("\n========== RETRIEVED CHUNKS ==========")

    for result in results:

        print("\n-----------------------------")

        print("Score:", round(result["score"], 4))
        print("Source:", result["source"])
        print("Chunk ID:", result["chunk_id"])

        print("\nText:")
        print(result["text"])


    # ======================================
    # 11. Build context
    # ======================================

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
Source: {result["source"]}
Chunk ID: {result["chunk_id"]}

{result["text"]}
"""
        )

    context = "\n\n".join(context_parts)


    # ======================================
    # 12. Build RAG prompt
    # ======================================

    prompt = build_rag_prompt(
        question,
        context
    )


    # ======================================
    # 13. Generate answer
    # ======================================

    llm = LLM()

    answer = llm.generate(prompt)


    # ======================================
    # 14. Display answer
    # ======================================

    print("\n========== ANSWER ==========")

    print(answer)


    # ======================================
    # 15. Display sources
    # ======================================

    print("\n========== SOURCES ==========")

    for result in results:

        print(
            f"- {result['source']} "
            f"(chunk {result['chunk_id']})"
        )