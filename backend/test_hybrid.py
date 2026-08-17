from embeddings import EmbeddingModel
from vector_store import VectorStore
from hybrid_retriever import HybridRetriever

from config import (
    EMBEDDING_DIMENSION,
    COLLECTION_NAME
)


# ==========================================
# 1. Initialize embedding model
# ==========================================

embedding_model = EmbeddingModel()


# ==========================================
# 2. Connect to Qdrant
# ==========================================

vector_store = VectorStore(

    collection_name=COLLECTION_NAME,

    vector_size=EMBEDDING_DIMENSION
)


try:

    # ======================================
    # 3. Create hybrid retriever
    # ======================================

    retriever = HybridRetriever(

        embedding_model,

        vector_store
    )


    # ======================================
    # 4. Test question
    # ======================================

    question = "What is QdrantClient?"


    # ======================================
    # 5. Retrieve results
    # ======================================

    results = retriever.retrieve(

        question,

        top_k=5,

        score_threshold=0.65
    )


    # ======================================
    # 6. Display results
    # ======================================

    print(
        "\n========== HYBRID SEARCH RESULTS =========="
    )


    if not results:

        print(
            "\nNo results found."
        )

    else:

        for index, result in enumerate(
            results,
            start=1
        ):

            print(
                "\n-----------------------------"
            )

            print(
                f"Result: [{index}]"
            )

            print(
                "RRF Score:",
                round(
                    result["score"],
                    6
                )
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

            print(
                result["text"][:500]
            )


finally:

    # ======================================
    # 7. Close Qdrant
    # ======================================

    vector_store.close()