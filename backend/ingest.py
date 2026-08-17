from pathlib import Path

from document_loader import load_document
from chunker import chunk_text
from embeddings import EmbeddingModel
from vector_store import VectorStore
from id_generator import generate_chunk_id

from config import (
    EMBEDDING_DIMENSION,
    COLLECTION_NAME,
    CHUNK_SIZE
)


DOCUMENTS_DIR = Path(
    "data/documents"
)


# ==========================================
# 1. Find documents
# ==========================================

files = list(
    DOCUMENTS_DIR.glob("*.txt")
)

print(
    f"Found {len(files)} documents."
)


# ==========================================
# 2. Initialize models
# ==========================================

embedding_model = EmbeddingModel()

vector_store = VectorStore(
    collection_name=COLLECTION_NAME,
    vector_size=EMBEDDING_DIMENSION
)


try:

    for file_path in files:

        print(
            f"\nProcessing: {file_path.name}"
        )

        # ==================================
        # Document ID
        # ==================================

        document_id = file_path.stem


        # ==================================
        # Delete previous version
        # ==================================

        vector_store.delete_document(
            document_id
        )

        print(
            "Previous version removed."
        )


        # ==================================
        # Load document
        # ==================================

        document = load_document(
            str(file_path)
        )


        # ==================================
        # Chunk document
        # ==================================

        chunks = chunk_text(
            document["text"],
            document["metadata"]["source"],
            document_id,
            chunk_size=CHUNK_SIZE
        )

        print(
            f"Chunks created: {len(chunks)}"
        )


        # ==================================
        # Generate embeddings
        # ==================================

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            embedding_model
            .generate_embeddings(texts)
        )


        # ==================================
        # Generate stable IDs
        # ==================================

        point_ids = [

            generate_chunk_id(
                document_id,
                chunk["metadata"]["chunk_id"]
            )

            for chunk in chunks
        ]


        # ==================================
        # Store new version
        # ==================================

        vector_store.add_documents(
            chunks,
            embeddings,
            point_ids
        )


        print(
            "New version stored successfully."
        )


    print(
        "\nIngestion completed successfully."
    )


finally:

    vector_store.close()