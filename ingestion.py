import os
from dotenv import load_dotenv
load_dotenv()
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai import types
if __name__=="__main__":
    print("Ingesting.....")
    loader=UnstructuredLoader(file_path="C:/Users/91879/Desktop/langchain-course/mediumblog.txt",chunking_strategy="basic",max_characters=100000,encoding="UTF-8")
    documet=loader.load()
    print("splitting........")
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts=text_splitter.split_documents(documet)
    contents = [doc.page_content for doc in texts]
    print(f"created {len(texts)} chunks")
    client = genai.Client()

    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=1536,
    api_key=os.environ.get("GOOGLE_API_KEY")  # type: ignore[arg-type]
)

    print("ingesting....")
    PineconeVectorStore.from_documents(texts,embeddings,index_name=os.environ["INDEX_NAME"])
    print("finish")


