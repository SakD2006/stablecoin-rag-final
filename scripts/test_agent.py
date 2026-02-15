from app.agent.financial_agent import ask_agent

response = ask_agent(
    "how does a whale affect the prices of stablecoins?"
)

print("\n=== AGENT RESPONSE ===\n")
print(response)

#python -m scripts.test_agent