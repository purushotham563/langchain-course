import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()
print("Initilizing componets....")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    api_key=os.environ.get("GOOGLE_API_KEY") )   # type: ignore[arg-type]
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
vectorstore=PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)
retriver=vectorstore.as_retriever(search_kwargs={"k":3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)
def format_docs(docs):
    """Format retrived documnets into single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrievl_chain_without_lcel(query:str):
    """Simple retieval chain without LCEL Manually retrives documents,formats them and generates response
    """
    docs=retriver.invoke(query)
    contex=format_docs(docs)
    message=prompt_template.format_messages(contex=contex,question=query)
    response =llm.invoke(message)
    return response.content[0].get("text") # type: ignore

def create_retrieval_chain_with_lcel():
    retrieval_chain=(RunnablePassthrough.assign(context=itemgetter("question")|retriver|format_docs)|prompt_template|llm|StrOutputParser())
    return retrieval_chain




if __name__ == "__main__":
    print("Retrieving....")
    query="What is Pinecone in machine learning"
    print("\n"+"="*60)
    print("Implementation 0: Raw LLM Invocation (NO RAG)")
    print("="*70)
    result_raw=llm.invoke([HumanMessage(content=query)])
    print("\nAnswer")
    print(result_raw.content[0].get("text")) # type: ignore

    print("\n"+"="*60)
    print("Implementation 1: Without LECL")
    print("="*70)
    result_without_lcel=llm.invoke([HumanMessage(content=query)])
    print("\nAnswer")
    print(result_without_lcel.content[0].get("text")) # type: ignore
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)
    chain_with_lcel=create_retrieval_chain_with_lcel()
    result_with_lcel=chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_lcel)
