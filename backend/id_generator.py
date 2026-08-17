import uuid


def generate_chunk_id(
    document_id,
    chunk_id
):

    value = f"{document_id}:{chunk_id}"

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            value
        )
    )