"""Interactive CLI driver for the mutual fund analyzer ADK agent.

Usage:
    python main.py                       # interactive chat
    python main.py "How is AAPL doing?"  # one-shot query, then drop into chat

You can also run this agent with the ADK CLI/UI from the project root:
    adk run equity_portfolio_agent
    adk web
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from equity_portfolio_agent.agent import root_agent

APP_NAME = "equity_portfolio_analyzer"
USER_ID = "local_user"


async def _run_once(runner: Runner, session_id: str, query: str) -> None:
    """Send a single user message to the agent and stream its final reply."""
    content = types.Content(role="user", parts=[types.Part(text=query)])
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            text = event.content.parts[0].text or ""
            print(f"Agent: {text}\n")


async def chat() -> None:
    load_dotenv()

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print("=== Equity & Portfolio Analyzer Agent ===")
    print("Ask about any stock, ETF or mutual fund.")
    print("Type 'exit' or press Ctrl-D to quit.\n")

    # If the user passed a query on the command line, run it first.
    initial_query = " ".join(sys.argv[1:]).strip()
    if initial_query:
        print(f"You: {initial_query}")
        await _run_once(runner, session.id, initial_query)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        await _run_once(runner, session.id, query)


def main() -> None:
    asyncio.run(chat())


if __name__ == "__main__":
    main()
