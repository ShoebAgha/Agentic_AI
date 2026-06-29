from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

tavily = TavilySearch()


llm = ChatAnthropic(model = "claude-haiku-4-5-20251001")
tools=[tavily]

agent=create_agent(model=llm, tools=tools)


def main():
    print('hello from langchain')
    result=agent.invoke({'messages':HumanMessage('What is the weather in tokyo?')})
    print(result)

if __name__ == '__main__':
    main()