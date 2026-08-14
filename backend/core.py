import os 
from typing import Any,Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    api_key=os.environ.get("GOOGLE_API_KEY")  # type: ignore[arg-type]
)
vectorstrore=PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"),embedding=embeddings)
model=init_chat_model(model="gemini-3.5-flash-lite",model_provider="gemini")

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrive relevnt documnetation to help answer user queries about Langchain"""
    retrieved_docs=vectorstrore.as_retriever().invoke(query,k=4)
    serialized="\n\n".join((f"source:{doc.metadata.get("source","unknown")}\n\nContent:{doc.page_content}")for doc in retrieved_docs)
    return serialized,retrieved_docs

def run_llm(query:str)->Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    agent=create_agent(model,tools=[retrieve_context],system_prompt=system_prompt)
    messages=[{"role":"user","content":query}]
    response = agent.invoke({"messages": messages}) # type : ignore
    answer=response["messages"][-1].content
    context_docs=[]
    for message in response["messages"]:
        if isinstance(message,ToolMessage) and hasattr(message,"artifact"):
            if isinstance(message.artifact,list):
                context_docs.extend(message.artifact)
    return {"answer":answer,"context":context_docs}
if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)


    






        








