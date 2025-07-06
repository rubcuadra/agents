from dotenv import load_dotenv
from agents import Agent, Runner, trace, add_trace_processor
from tracers import LogTracer, make_trace_id
import os
import asyncio
import hashlib

# Just for searching
from langchain_openai import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv(override=True)

from src import (
    get_rivera_agent, 
    get_siqueiros_agent, 
    get_orozco_agent, 
    get_search_agent
)

model = "gpt-4o-mini"
MARKDOWN_FILE = 'summary.md'

# Create a function that generates a hash from a string
def generate_hash(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()

async def main(topic: str, overwrite: bool = False):
    # Create the output directory if it doesn't exist
    dir_name = f'{os.path.dirname(os.path.abspath(__file__))}/output/{model}'
    os.makedirs(dir_name, exist_ok=True)
    file_name = f'{dir_name}/{generate_hash(topic)}.md'
    if os.path.exists(file_name) and not overwrite:
        print(f'File {file_name} already exists, skipping...')
        return

    # 1. Get some information about the topic
    result = await Runner.run(
        get_search_agent(model),
        topic
    )

    # 2. Get the thoughts on the topic from the agents
    agents = [
        get_rivera_agent(model), 
        get_siqueiros_agent(model), 
        get_orozco_agent(model)
    ]
    message = f'''
        Give your thoughts on the following topic: {topic}
        <summary>{result.final_output}</summary>
    '''
    agent_thoughts = await asyncio.gather(*[Runner.run(agent, message) for agent in agents])

    # 3. Write a Markdown file with the whole discussion, the file is written in the output folder
    with open(file_name, 'w+') as f:
        f.write(f'# {topic}\n')
        f.write(f'## Summary\n')
        f.write(result.final_output)
        f.write(f'\n## Thoughts from the agents\n')
        for agent, agent_thought in zip(agents, agent_thoughts):
            f.write(f'### {agent.name}\n')
            f.write(agent_thought.final_output)
            f.write('\n')

if __name__ == "__main__":
    topic = 'Human Sacrifice in mesoamerican cultures'

    # https://platform.openai.com/traces
    with trace("Ollin Flux"):
        asyncio.run(main(topic))
