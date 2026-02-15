"""
rag_chain.py
LangChain v1.x RAG using LCEL (Runnable pipelines)
"""

from dotenv import load_dotenv
load_dotenv()

# ---- LangChain imports (v1 style) ----
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------------- CONFIG ----------------

PERSIST_DIR = "vector_db"
COLLECTION_NAME = "stablecoin_research"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------- LOAD VECTOR DB ----------------

print("📦 Loading vector database...")

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

print("🧠 Loading Groq LLM...")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_template("""
You are a professional financial risk analyst specializing in stablecoins.

Use ONLY the provided research context to answer.

Context:
{context}

Question:
{question}

Provide:
- Clear explanation
- Key mechanisms
- Financial reasoning
""")


# ---------------- FORMAT DOCUMENTS ----------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- LCEL RAG PIPELINE ----------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------- INTERACTIVE LOOP ----------------

print("\n🚀 Stablecoin Research RAG Ready (type 'exit' to quit)\n")

while True:
    question = input("Ask: ")

    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)

    print("\n🧠 AI Analysis:\n")
    print(answer)
    print("\n" + "=" * 60)