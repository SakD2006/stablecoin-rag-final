"""
financial_agent.py

Main USDT Financial Intelligence Agent

LangChain v1 Architecture:
User → Agent → Tool Selection → Tool Execution → Response

NO deprecated APIs used.
"""

# =====================================================
# ENVIRONMENT
# =====================================================

from dotenv import load_dotenv
load_dotenv()

# =====================================================
# LANGCHAIN IMPORTS (V1 STRUCTURE)
# =====================================================

from langchain.agents import create_agent
from langchain_groq import ChatGroq

# =====================================================
# TOOLS
# =====================================================

# Research explanations (RAG over papers)
from app.rag.research_rag import research_tool

# Live + historical market data (Supabase)
from app.tools.market_state_tool import market_state_tool


# =====================================================
# LLM CONFIG
# =====================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
)


# =====================================================
# SYSTEM PROMPT (AGENT BRAIN)
# =====================================================

SYSTEM_PROMPT = """
You are an AI Stablecoin Risk Analyst specializing ONLY in USDT.

You must reason like a quantitative crypto risk analyst.

You have access to two tools:

--------------------------------------------------
research_tool
--------------------------------------------------
Use when:
- explaining stablecoin concepts
- financial mechanisms
- academic reasoning
- theoretical causes
- market microstructure explanations

--------------------------------------------------
market_state_tool
--------------------------------------------------
Use when:
- user asks about NOW or current risk
- historical timestamps
- yesterday / last hour / specific time
- predictions or risk levels
- model outputs or features

--------------------------------------------------
Decision Rules:

1. Conceptual or theory question → research_tool
2. Time-based or live data question → market_state_tool
3. "Why" questions about real events → use BOTH tools
4. NEVER invent market data.
5. Always rely on tools for factual information.

--------------------------------------------------
Response Format:

1. Clear Explanation
2. Key Drivers
3. Financial Reasoning
4. Risk Interpretation (if applicable)

Be precise, analytical, and professional.
"""


# =====================================================
# CREATE AGENT (LANGCHAIN V1 OFFICIAL)
# =====================================================

agent = create_agent(
    model=llm,
    tools=[
        research_tool,
        market_state_tool,
    ],
    system_prompt=SYSTEM_PROMPT,
)


# =====================================================
# PUBLIC FUNCTION (USED BY API / CLI)
# =====================================================

def ask_agent(question: str) -> str:
    """
    Main interface for frontend/backend.

    Example:
        ask_agent("Why is USDT risky today?")
    """

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    # LangChain returns structured message list
    return response["messages"][-1].content

