"""
research_rag.py

Reusable Research RAG Tool
LangChain v1.x LCEL implementation
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


# ---------------- CONFIG ----------------

PERSIST_DIR = "vector_db"
COLLECTION_NAME = "stablecoin_research"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------- LOAD VECTOR DB (ONCE) ----------------

print("🔎 Initializing Research RAG...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 12},
)


# ---------------- LLM ----------------

llm = ChatGroq(
    model="meta-llama/llama-guard-4-12b",
    temperature=0,
)


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_template("""
You are a professional financial risk analyst specializing in stablecoins.

Use ONLY the provided research context.

Explain concepts clearly using financial reasoning.

Context:
{context}

Question:
{question}

Provide:
- Clear explanation
- Key mechanisms
- Financial reasoning
""")


# ---------------- HELPERS ----------------

def format_docs(docs):
    """Combine retrieved documents into context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- RAG PIPELINE ----------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------- TOOL FUNCTION ----------------

def research_tool(question: str) -> str:
    """
    Provide academic and theoretical explanations about stablecoins.

    Use for:
    - definitions
    - mechanisms
    - financial theory
    - depegging causes
    - conceptual explanations

    This tool DOES NOT access live market data.
    """
    response = rag_chain.invoke(question)
    return response