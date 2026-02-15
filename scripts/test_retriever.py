"""
test_retriever.py

Loads the Chroma vector database and tests semantic retrieval.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- CONFIG ----------------

PERSIST_DIR = "vector_db"
COLLECTION_NAME = "stablecoin_research"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------- LOAD VECTOR DB ----------------

def load_vectorstore():
    """Load existing Chroma database."""

    print("📦 Loading vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    return vectorstore


# ---------------- CREATE RETRIEVER ----------------

def create_retriever(vectorstore):
    """Create retriever abstraction."""

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    return retriever


# ---------------- QUERY LOOP ----------------

def interactive_query(retriever):

    print("\n🔎 Retriever Ready (type 'exit' to quit)\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)

        print("\n📚 Retrieved Context:\n")

        for i, doc in enumerate(docs):
            print(f"Result {i+1}")
            print(
                f"Source: {doc.metadata.get('source_file')} "
                f"| Page: {doc.metadata.get('page')} "
                f"| Chunk: {doc.metadata.get('chunk_id')}"
            )
            print(doc.page_content[:400])
            print("-" * 60)


# ---------------- MAIN ----------------

if __name__ == "__main__":

    vectorstore = load_vectorstore()
    retriever = create_retriever(vectorstore)

    interactive_query(retriever)