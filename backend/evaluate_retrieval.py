from embeddings import EmbeddingModel
from vector_store import VectorStore
from retriever import Retriever
from hybrid_retriever import HybridRetriever

from config import (
    EMBEDDING_DIMENSION,
    COLLECTION_NAME,
    SCORE_THRESHOLD
)


# ==========================================
# Evaluation questions
# ==========================================

TEST_QUERIES = [

    "What database is used by the Contest Compiler?",

    "How is authentication handled?",

    "What is Redis used for?",

    "How are code submissions executed?",

    "Which service executes submitted code?",

    "How is the leaderboard updated?"

]


# ==========================================
# Initialize models
# ==========================================

embedding_model = EmbeddingModel()

vector_store = VectorStore(
    collection_name=COLLECTION_NAME,
    vector_size=EMBEDDING_DIMENSION
)


try:

    # ======================================
    # Create both retrievers
    # ======================================

    semantic_retriever = Retriever(
        embedding_model,
        vector_store
    )

    hybrid_retriever = HybridRetriever(
        embedding_model,
        vector_store
    )


    # ======================================
    # Evaluate each query
    # ======================================

    for query in TEST_QUERIES:

        print("\n")
        print("=" * 70)
        print("QUERY:", query)
        print("=" * 70)


        # ==================================
        # Semantic results
        # ==================================

        semantic_results = (
            semantic_retriever.retrieve(

                query,

                top_k=3,

                score_threshold=SCORE_THRESHOLD
            )
        )


        print("\n========== SEMANTIC SEARCH ==========")


        for index, result in enumerate(
            semantic_results,
            start=1
        ):

            print(
                f"\n[{index}] "
                f"{result['source']} "
                f"(chunk {result['chunk_id']})"
            )

            print(
                result["text"][:250]
            )


        # ==================================
        # Hybrid results
        # ==================================

        hybrid_results = (
            hybrid_retriever.retrieve(

                query,

                top_k=3,

                score_threshold=SCORE_THRESHOLD
            )
        )


        print("\n========== HYBRID SEARCH ==========")


        for index, result in enumerate(
            hybrid_results,
            start=1
        ):

            print(
                f"\n[{index}] "
                f"{result['source']} "
                f"(chunk {result['chunk_id']})"
            )

            print(
                "RRF Score:",
                round(
                    result["score"],
                    6
                )
            )

            print(
                result["text"][:250]
            )


finally:

    vector_store.close()