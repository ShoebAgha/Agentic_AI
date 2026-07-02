from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

tavily = TavilySearch()

class Sources(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description='The url of the source')

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources: list[Sources] = Field(description="List of sources used to generate reponse to the query")

llm = ChatAnthropic(model = "claude-haiku-4-5-20251001")
tools=[tavily]

agent=create_agent(model=llm, tools=tools,response_format=AgentResponse)


def main():
    print('hello from langchain')
    result=agent.invoke({'messages':HumanMessage('Find 3 active AI Engineer role in Delhi NCR along with pay')})
    print(result)

if __name__ == '__main__':
    main()