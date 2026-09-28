"""Ask the finance agent using local transactions and a persisted retrieval index."""

import argparse
from pathlib import Path

from agents import set_default_openai_key

from app.agents.context import FinanceAgentContext
from app.agents.finance_agent import run_finance_agent
from app.config import Settings
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.index import load_index
from app.tools.transactions import load_transactions_csv

TRANSACTIONS_PATH = Path("data/sample_transactions/sample_transactions.csv")
INDEX_PATH = Path("data/index/retrieval_index.json")


def main() -> None:
    """Load local dependencies, run one agent turn, and print grounded metadata."""
    parser = argparse.ArgumentParser(description="Ask the financialassist finance agent.")
    parser.add_argument("question", help="Finance or Northstar Financial question to answer.")
    arguments = parser.parse_args()

    settings = Settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required to run the finance agent.")
    if not INDEX_PATH.is_file():
        raise SystemExit(
            "Retrieval index not found. Run scripts/build_retrieval_index.py first."
        )

    set_default_openai_key(settings.openai_api_key)
    context = FinanceAgentContext(
        transactions=load_transactions_csv(TRANSACTIONS_PATH),
        retrieval_index=load_index(INDEX_PATH),
        embedding_provider=OpenAIEmbeddingProvider(
            model=settings.embedding_model, api_key=settings.openai_api_key
        ),
    )
    response = run_finance_agent(arguments.question, context, settings.agent_model)
    print(response.answer)
    print(f"\nTools used: {', '.join(response.tools_used) or 'none'}")
    print("Citations:")
    for citation in response.citations:
        print(f"- {citation}")


if __name__ == "__main__":
    main()
