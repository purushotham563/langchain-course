from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel,Field

tavily=TavilyClient()

load_dotenv()


class Source(BaseModel):
    '''Schema for a source used by the agent'''
    url:str=Field(description="The URL for the source")

class AgentResponse(BaseModel):
    '''Schema for the agent response with answer and source'''
    answer:str=Field(description="The agent answer to the query")
    sources:List[Source]=Field(default_factory=list,description="List of source used to generate the answer")


@tool
def search(query:str)->str:
    '''Tool that searches over the internet
    Args:
        query:The query to search for
    Returns:
        the search result
    '''
    print(f"Searching for {query}")
    return tavily.search(query=query)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
tools=[search]
agent=create_agent(model=llm,tools=tools,response_format=AgentResponse)

def main():
    print("Hello from langchain course")
    result=agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")})
    print(result)
if __name__=="__main__":
    main()