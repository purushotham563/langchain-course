from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
load_dotenv()
@tool
def triple(num:float)->float:
    """
    param num : a number to triple
    retuns: the triple
    """
    return float(num)*3

tools=[TavilySearch(max_results=1),triple]
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite").bind_tools(tools)






