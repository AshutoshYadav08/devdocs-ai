import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        tokenized_documents = [
            self.tokenize(
                document["text"]
            )
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )


    def tokenize(self, text):

        text = text.lower()

        tokens = re.findall(
            r"[a-zA-Z0-9_]+",
            text
        )

        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
            "does",
            "do",
            "did",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "from",
            "by",
            "as",
            "it",
            "this",
            "that"
        }

        tokens = [
            token
            for token in tokens
            if token not in stopwords
        ]

        return tokens


    def search(self, query, top_k=5):

        tokenized_query = self.tokenize(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indexes[:top_k]:

            if scores[index] <= 0:
                continue

            result = (
                self.documents[index]
                .copy()
            )

            result["score"] = float(
                scores[index]
            )

            result["retrieval_method"] = "bm25"

            results.append(
                result
            )

        return results