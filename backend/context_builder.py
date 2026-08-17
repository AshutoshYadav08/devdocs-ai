def build_context(results):

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE [{index}]
File: {result["source"]}
Chunk: {result["chunk_id"]}

{result["text"]}
"""
        )

    return "\n\n".join(
        context_parts
    )