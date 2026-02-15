from app.rag.research_rag import research_tool

answer = research_tool(
    "Why does liquidity stress cause stablecoin depegging?"
)

print("\n=== RESPONSE ===\n")
print(answer)