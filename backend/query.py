from embeddings import EmbeddingModel
from vector_store import VectorStore
from hybrid_retriever import HybridRetriever
from context_builder import build_context
from prompt import build_rag_prompt
from llm import LLM

from config import (
    EMBEDDING_DIMENSION,
    COLLECTION_NAME,
    TOP_K,
    SCORE_THRESHOLD
)


# ==========================================
# 1. Initialize embedding model
# ==========================================

embedding_model = EmbeddingModel()


# ==========================================
# 2. Connect to persistent Qdrant
# ==========================================

vector_store = VectorStore(
    collection_name=COLLECTION_NAME,
    vector_size=EMBEDDING_DIMENSION
)


try:

    # ======================================
    # 3. Create retriever
    # ======================================

    retriever = HybridRetriever(
        embedding_model,
        vector_store
    )


    # ======================================
    # 4. Create LLM
    # ======================================

    llm = LLM()


    # ======================================
    # 5. Ask question
    # ======================================

    question = input("\nAsk a question: ").strip()


    if not question:

        print("\nPlease enter a question.")

    else:

        # ==================================
        # 6. Retrieve relevant chunks
        # ==================================

        results = retriever.retrieve(
            question,
            top_k=TOP_K,
            score_threshold=SCORE_THRESHOLD
        )


        # ==================================
        # 7. Check retrieval results
        # ==================================

        if not results:

            print(
                "\nI don't have enough information "
                "in the provided documentation."
            )

        else:

            # ==================================
            # 8. Display retrieved chunks
            # ==================================

            print(
                "\n========== RETRIEVED CHUNKS =========="
            )

            for index, result in enumerate(
                results,
                start=1
            ):

                print("\n-----------------------------")

                print(
                    f"Result: [{index}]"
                )

                print(
                    "Score:",
                    round(result["score"], 4)
                )

                print(
                    "Source:",
                    result["source"]
                )

                print(
                    "Chunk ID:",
                    result["chunk_id"]
                )

                print("\nText:")

                print(result["text"])


            # ==================================
            # 9. Build context
            # ==================================

            context = build_context(
                results
            )


            # ==================================
            # 10. Build RAG prompt
            # ==================================

            prompt = build_rag_prompt(
                question,
                context
            )


            # ==================================
            # 11. Generate answer
            # ==================================

            answer = llm.generate(
                prompt
            )


            # ==================================
            # 12. Display answer
            # ==================================

            print(
                "\n========== ANSWER =========="
            )

            print(answer)


            # ==================================
            # 13. Display sources
            # ==================================

            print(
                "\n========== SOURCES =========="
            )

            for index, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"[{index}] "
                    f"{result['source']} "
                    f"(chunk {result['chunk_id']}, "
                    f"score={result['score']:.4f})"
                )


finally:

    # ======================================
    # 14. Always close Qdrant
    # ======================================

    vector_store.close()