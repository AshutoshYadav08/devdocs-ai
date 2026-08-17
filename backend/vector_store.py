from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)


class VectorStore:

    def __init__(
        self,
        collection_name,
        vector_size
    ):

        self.client = QdrantClient(
            path="db/qdrant"
        )

        self.collection_name = collection_name

        if not self.client.collection_exists(
            collection_name
        ):

            self.client.create_collection(

                collection_name=collection_name,

                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def add_documents(
        self,
        chunks,
        embeddings,
        point_ids
    ):

        points = []

        for chunk, embedding, point_id in zip(
            chunks,
            embeddings,
            point_ids
        ):

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def delete_document(self, document_id):

        self.client.delete(
            collection_name=self.collection_name,

            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",

                        match=MatchValue(
                            value=document_id
                        )
                    )
                ]
            )
        )

    def search(
        self,
        query_vector,
        limit=5,
        score_threshold=None
    ):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )

        return results.points


    
    def get_all_documents(self):

        documents = []

        offset = None

        while True:

            points, offset = self.client.scroll(

                collection_name=self.collection_name,

                limit=100,

                offset=offset,

                with_payload=True,

                with_vectors=False
            )

            for point in points:

                payload = point.payload

                metadata = payload.get(
                    "metadata",
                    {}
                )

                documents.append({

                    "text": payload.get(
                        "text",
                        ""
                    ),

                    "source": metadata.get(
                        "source",
                        ""
                    ),

                    "chunk_id": metadata.get(
                        "chunk_id",
                        str(point.id)
                    )
                })

            if offset is None:
                break

        return documents


    def close(self):

        self.client.close()






