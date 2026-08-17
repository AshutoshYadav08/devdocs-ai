import json

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
# Load evaluation dataset
# ==========================================

with open(
    "evaluation_dataset.json",
    "r",
    encoding="utf-8"
) as file:

    evaluation_data = json.load(file)


# ==========================================
# Initialize models
# ==========================================

embedding_model = EmbeddingModel()

vector_store = VectorStore(
    collection_name=COLLECTION_NAME,
    vector_size=EMBEDDING_DIMENSION
)


try:

    semantic_retriever = Retriever(
        embedding_model,
        vector_store
    )

    hybrid_retriever = HybridRetriever(
        embedding_model,
        vector_store
    )


    # ==========================================
    # Evaluation function
    # ==========================================

    def evaluate_retriever(
        retriever,
        name,
        top_k=5
    ):

        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0

        reciprocal_rank_sum = 0


        print(
            f"\n\n{'=' * 70}"
        )

        print(
            f"{name} EVALUATION"
        )

        print(
            f"{'=' * 70}"
        )


        for item in evaluation_data:

            question = item["question"]

            expected_sources = set(
                item["expected_sources"]
            )


            results = retriever.retrieve(

                question,

                top_k=top_k,

                score_threshold=SCORE_THRESHOLD
            )


            retrieved_sources = [
                result["source"]
                for result in results
            ]


            # ==================================
            # Find first relevant result
            # ==================================

            first_relevant_rank = None

            for rank, source in enumerate(
                retrieved_sources,
                start=1
            ):

                if source in expected_sources:

                    first_relevant_rank = rank

                    break


            # ==================================
            # Hit@1
            # ==================================

            if (
                first_relevant_rank is not None
                and first_relevant_rank <= 1
            ):

                hit_at_1 += 1


            # ==================================
            # Hit@3
            # ==================================

            if (
                first_relevant_rank is not None
                and first_relevant_rank <= 3
            ):

                hit_at_3 += 1


            # ==================================
            # Hit@5
            # ==================================

            if (
                first_relevant_rank is not None
                and first_relevant_rank <= 5
            ):

                hit_at_5 += 1


            # ==================================
            # MRR
            # ==================================

            if first_relevant_rank is not None:

                reciprocal_rank_sum += (
                    1 / first_relevant_rank
                )


            # ==================================
            # Display query result
            # ==================================

            print(
                f"\nQuery: {question}"
            )

            print(
                "Expected:",
                ", ".join(expected_sources)
            )

            print(
                "Retrieved:",
                ", ".join(retrieved_sources)
            )

            print(
                "First relevant rank:",
                first_relevant_rank
            )


        total = len(evaluation_data)


        # ==========================================
        # Final metrics
        # ==========================================

        hit_at_1_score = (
            hit_at_1 / total
        )

        hit_at_3_score = (
            hit_at_3 / total
        )

        hit_at_5_score = (
            hit_at_5 / total
        )

        mrr_score = (
            reciprocal_rank_sum / total
        )


        print(
            f"\n\n========== {name} METRICS =========="
        )

        print(
            f"Hit@1 : {hit_at_1_score:.3f}"
        )

        print(
            f"Hit@3 : {hit_at_3_score:.3f}"
        )

        print(
            f"Hit@5 : {hit_at_5_score:.3f}"
        )

        print(
            f"MRR   : {mrr_score:.3f}"
        )


        return {
            "hit_at_1": hit_at_1_score,
            "hit_at_3": hit_at_3_score,
            "hit_at_5": hit_at_5_score,
            "mrr": mrr_score
        }


    # ==========================================
    # Evaluate semantic retrieval
    # ==========================================

    semantic_metrics = evaluate_retriever(
        semantic_retriever,
        "SEMANTIC SEARCH"
    )


    # ==========================================
    # Evaluate hybrid retrieval
    # ==========================================

    hybrid_metrics = evaluate_retriever(
        hybrid_retriever,
        "HYBRID SEARCH"
    )


    # ==========================================
    # Compare
    # ==========================================

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 70
    )


    print(
        "\nMetric          Semantic       Hybrid"
    )

    print(
        "-" * 45
    )

    print(
        f"Hit@1          "
        f"{semantic_metrics['hit_at_1']:.3f}"
        f"          "
        f"{hybrid_metrics['hit_at_1']:.3f}"
    )

    print(
        f"Hit@3          "
        f"{semantic_metrics['hit_at_3']:.3f}"
        f"          "
        f"{hybrid_metrics['hit_at_3']:.3f}"
    )

    print(
        f"Hit@5          "
        f"{semantic_metrics['hit_at_5']:.3f}"
        f"          "
        f"{hybrid_metrics['hit_at_5']:.3f}"
    )

    print(
        f"MRR            "
        f"{semantic_metrics['mrr']:.3f}"
        f"          "
        f"{hybrid_metrics['mrr']:.3f}"
    )


finally:

    vector_store.close()