from dotenv import load_dotenv
from agents import Agent, Runner, trace, add_trace_processor
from tracers import LogTracer, make_trace_id
import os
import asyncio

# Just for searching
from langchain_openai import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv(override=True)

from src.search_agent import search_agent

async def main():
    # result = await Runner.run(
    #     search_agent,
    #     "What is the oldest country in the world?"
    # )
    pass

    

if __name__ == "__main__":
    with trace("playground"):
        asyncio.run(main())
