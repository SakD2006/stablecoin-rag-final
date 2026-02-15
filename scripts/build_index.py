"""
build_index.py

Creates a persistent Chroma vector database from research papers.

LangChain ingestion flow:
Loader → Splitter → Embeddings → Vector Store
"""

from pathlib import Path
import os

# ---- LangChain Imports (Modern structure) ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------- CONFIG ----------------

PAPERS_DIR = Path("data/papers")
PERSIST_DIR = Path("vector_db")
COLLECTION_NAME = "stablecoin_research"

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------- LOAD DOCUMENTS ----------------

def load_documents():
    """Load all PDFs using LangChain document loaders."""

    documents = []

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in {PAPERS_DIR.resolve()}")

    print(f"\n📚 Found {len(pdf_files)} papers\n")

    for pdf_path in pdf_files:
        try:
            print(f"📄 Loading {pdf_path.name}")

            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            # enrich metadata (preserve existing!)
            for d in docs:
                d.metadata.update({
                    "source_file": pdf_path.name,
                })

            documents.extend(docs)

        except Exception as e:
            print(f"⚠️ Failed to load {pdf_path.name}: {e}")

    print(f"\n✅ Total pages loaded: {len(documents)}")
    return documents


# ---------------- SPLIT DOCUMENTS ----------------

def split_documents(documents):
    """
    Split documents using RecursiveCharacterTextSplitter
    (recommended default splitter in LangChain docs)
    """

    print("\n✂️ Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # add chunk ids for traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


# ---------------- CREATE EMBEDDINGS ----------------

def create_embeddings():
    """Create HuggingFace embedding model."""

    print("\n🧠 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )

    return embeddings


# ---------------- BUILD VECTOR STORE ----------------

def build_vector_store(chunks, embeddings):
    """Create persistent Chroma vector store."""

    print("\n📦 Creating Chroma vector database...")

    # ensure directory exists
    os.makedirs(PERSIST_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"✅ Vector DB saved to: {PERSIST_DIR.resolve()}")

    return vectorstore


# ---------------- TEST RETRIEVAL ----------------

def test_retrieval(vectorstore):
    """Sanity check retrieval using retriever abstraction."""

    print("\n🔍 Testing retrieval...")

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        results = retriever.invoke("What causes stablecoin depegging?")

        print(f"✅ Retrieved {len(results)} results")

        if results:
            r = results[0]
            print(
                f"Top result → {r.metadata.get('source_file')} "
                f"(page {r.metadata.get('page', 'N/A')}, "
                f"chunk {r.metadata.get('chunk_id')})"
            )

    except Exception as e:
        print(f"⚠️ Retrieval test failed: {e}")


# ---------------- MAIN PIPELINE ----------------

def main():

    print("\n🚀 Building Stablecoin Research Index\n")

    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = create_embeddings()
    vectorstore = build_vector_store(chunks, embeddings)

    test_retrieval(vectorstore)

    print("\n🎉 Index successfully built!\n")


if __name__ == "__main__":
    main()