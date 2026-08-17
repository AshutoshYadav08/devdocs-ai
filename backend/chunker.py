import re


def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text,
    source,
    document_id,
    chunk_size=300
):

    sentences = split_sentences(text)

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        if (
            len(current_chunk) +
            len(sentence)
            <= chunk_size
        ):

            current_chunk += sentence + " "

        else:

            if current_chunk:

                chunks.append({
                    "text": current_chunk.strip(),

                    "metadata": {
                        "source": source,
                        "document_id": document_id,
                        "chunk_id": len(chunks)
                    }
                })

            current_chunk = sentence + " "

    # Store final chunk

    if current_chunk:

        chunks.append({
            "text": current_chunk.strip(),

            "metadata": {
                "source": source,
                "document_id": document_id,
                "chunk_id": len(chunks)
            }
        })

    return chunks