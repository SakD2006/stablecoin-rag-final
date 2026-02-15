"""
FastAPI Backend for Stablecoin AI Agent
"""

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# load env vars
load_dotenv()

# import agent
from app.agent.financial_agent import ask_agent


# =====================================================
# APP INIT
# =====================================================

app = FastAPI(
    title="USDT Risk Intelligence API",
    description="AI Stablecoin Risk Analyst Backend",
    version="1.0.0"
)


# =====================================================
# REQUEST / RESPONSE MODELS
# =====================================================

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health():
    return {"status": "running"}


# =====================================================
# AGENT ENDPOINT
# =====================================================

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Main endpoint used by frontend.
    Sends user query to financial agent.
    """

    answer = ask_agent(request.question)

    return QueryResponse(answer=answer)