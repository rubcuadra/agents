from langchain_community.utilities import GoogleSerperAPIWrapper
from agents import Agent, Runner, function_tool
import asyncio

SYSTEM_PROMPT = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

search = GoogleSerperAPIWrapper()

@function_tool
def google_search(query: str) -> str:
    """Useful for when you need more information from an online search"""
    return search.run(query)

tools = [google_search]

search_agent = Agent(
    name="Search Agent",
    instructions=SYSTEM_PROMPT,
    tools=tools,
    model="gpt-4o-mini"
)

async def run(query: str) -> str:
    result = await Runner.run(search_agent, query)
    return result.final_output

if __name__ == "__main__":
    res = asyncio.run(run("""
        What is the ancient capital of Mexico?
    """))
    print(res)