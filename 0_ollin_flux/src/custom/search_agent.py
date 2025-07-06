from langchain_community.utilities import GoogleSerperAPIWrapper
from agents import Agent, Runner, function_tool
import asyncio

SYSTEM_PROMPT = (
    "You are a researcher. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly, no need to have complete sentences or good "
    "grammar. Its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

search = GoogleSerperAPIWrapper()

@function_tool
def google_search(query: str) -> str:
    """Useful for when you need more information from an online search"""
    return search.run(query)

def get_search_agent(model: str = "gpt-4o-mini"):
    return Agent(
        name="Search Agent",
        instructions=SYSTEM_PROMPT,
        tools=[google_search],
        model=model
    )

if __name__ == "__main__":
    async def run_agent(query: str) -> str:
        result = await Runner.run(get_search_agent(), query)
        return result.final_output

    res = asyncio.run(run_agent("""
        What is the ancient capital of Mexico?
    """))
    print(res)