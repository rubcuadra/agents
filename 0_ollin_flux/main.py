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

from src.custom.rivera_agent import rivera_agent
from src.custom.siqueiros_agent import siqueiros_agent
from src.custom.orozco_agent import orozco_agent
from src.custom.search_agent import search_agent

agents = (
    rivera_agent, 
    siqueiros_agent, 
    orozco_agent,
    search_agent
)

MARKDOWN_FILE = 'summary.md'

# Create a function that generates a hash from a string
def generate_hash(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()

async def main(topic: str):
    # Include current directory for the whole path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = f'{current_dir}/output/{generate_hash(topic)}.md'
    if os.path.exists(file_name):
        print(f'File {file_name} already exists, skipping...')
        return

    # 1. Get some information about the topic
    result = await Runner.run(
        search_agent,
        topic
    )

    # 2. Get the thoughts on the topic from the agents
    agents = [rivera_agent, siqueiros_agent, orozco_agent]
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
        f.write(f'## Thoughts from the agents\n')
        for agent, agent_thought in zip(agents, agent_thoughts):
            f.write(f'### {agent.name}\n')
            f.write(agent_thought.final_output)
            f.write('\n')
    
    # Maybe TODO ask them to discuss the thoughts of the other agents

if __name__ == "__main__":
    topic = 'Monumental art in Aztec culture'
    # https://platform.openai.com/traces
    with trace("Ollin Flux"):
        asyncio.run(main(topic))
