import asyncio

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

stdio_server_params=StdioServerParameters(command="python",args=["C:/Users/91879/Desktop/langchain-course/servers/math_server.py"])
async def main():
    async with stdio_client(stdio_server_params) as (read,write):
        async with ClientSession(read_stream=read,write_stream=write) as session:
            await session.initialize()
            print("Session Initialized")
            tools=await load_mcp_tools(session)
            agent=create_agent(llm,tools)
            result=await agent.ainvoke({"messages":[HumanMessage(content="what is 2 + 2 * 3?")]})
            print(result["messages"][-1].text)




if __name__=="__main__":
    asyncio.run(main())


    