from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
import asyncio


async def main():
    pass
if __name__=="__main__":
    asyncio.run(main())
