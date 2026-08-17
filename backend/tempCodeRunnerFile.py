from vector_store import VectorStore

store = VectorStore()

documents = store.get_all_documents()

print("Total documents:", len(documents))

if documents:
    print(documents[0])