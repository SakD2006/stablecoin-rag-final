from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load paper
loader = PyPDFLoader("data/papers/ssrn-4700764.pdf")
docs = loader.load()

print("Pages loaded:", len(docs))

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))

# Inspect one chunk
print("\nExample chunk:\n")
print(chunks[5].page_content[:500])
print("\nMetadata:", chunks[5].metadata)