from fastembed import TextEmbedding


class EmbeddingModel:

    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def generate_embeddings(self, texts):

        embeddings = self.model.embed(texts)

        return list(embeddings)