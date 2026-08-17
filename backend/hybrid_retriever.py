from bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store

        # Load all chunks from Qdrant
        self.documents = (
            self.vector_store.get_all_documents()
        )

        # Build BM25 index
        self.bm25 = BM25Retriever(
            self.documents
        )

    def retrieve(
        self,
        query,
        top_k=5,
        score_threshold=0.65
    ):

        # ==========================================
        # 1. Generate query embedding
        # ==========================================

        query_embedding = (
            self.embedding_model
            .generate_embeddings([query])[0]
        )


        # ==========================================
        # 2. Semantic search
        # ==========================================

        semantic_results = (
            self.vector_store.search(
                query_embedding,
                limit=top_k,
                score_threshold=score_threshold
            )
        )


        formatted_semantic = []

        for result in semantic_results:

            formatted_semantic.append({

                "text": result.payload["text"],

                "score": result.score,

                "source": (
                    result.payload["metadata"]["source"]
                ),

                "chunk_id": (
                    result.payload["metadata"]["chunk_id"]
                )

            })


        # ==========================================
        # 3. BM25 keyword search
        # ==========================================

        keyword_results = (
            self.bm25.search(
                query,
                top_k=top_k
            )
        )


        # ==========================================
        # 4. Combine using RRF
        # ==========================================

        scores = {}

        documents = {}

        rrf_constant = 60


        # ------------------------------------------
        # Semantic ranking
        # ------------------------------------------

        for rank, document in enumerate(
            formatted_semantic,
            start=1
        ):

            doc_id = document["chunk_id"]

            documents[doc_id] = document

            scores[doc_id] = (
                scores.get(doc_id, 0)
                +
                1 / (rrf_constant + rank)
            )


        # ------------------------------------------
        # BM25 ranking
        # ------------------------------------------

        for rank, document in enumerate(
            keyword_results,
            start=1
        ):

            doc_id = document["chunk_id"]

            documents[doc_id] = document

            scores[doc_id] = (
                scores.get(doc_id, 0)
                +
                1 / (rrf_constant + rank)
            )


        # ==========================================
        # 5. Sort by RRF score
        # ==========================================

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True
        )


        # ==========================================
        # 6. Create final results
        # ==========================================

        final_results = []

        for doc_id in ranked_ids[:top_k]:

            document = documents[doc_id].copy()

            document["score"] = scores[doc_id]

            final_results.append(
                document
            )


        # ==========================================
        # 7. Remove duplicate chunks
        # ==========================================

        return self.remove_duplicates(
            final_results
        )


    def remove_duplicates(
        self,
        results
    ):

        seen = set()

        unique_results = []

        for result in results:

            if result["text"] not in seen:

                seen.add(
                    result["text"]
                )

                unique_results.append(
                    result
                )

        return unique_results