class Retriever:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query,
        top_k=5,
        score_threshold=0.65
    ):

        query_embedding = (
            self.embedding_model
            .generate_embeddings([query])[0]
        )

        results = self.vector_store.search(
            query_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )

        formatted_results = []

        for result in results:

            formatted_results.append({
                "text": result.payload["text"],
                "score": result.score,
                "source": result.payload["metadata"]["source"],
                "chunk_id": result.payload["metadata"]["chunk_id"]
            })

        return self.remove_duplicates(
            formatted_results
        )

    def remove_duplicates(self, results):

        seen = set()
        unique_results = []

        for result in results:

            if result["text"] not in seen:

                seen.add(result["text"])

                unique_results.append(result)

        return unique_results