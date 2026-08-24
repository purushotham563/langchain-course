from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a viral Twitter influencer grading a tweet. "
        "Generate critique and recommendations for the user's tweet. "
        "Always provide detailed recommendations, including requests "
        "for length, virality, style, etc.",
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "human",
        "Please critique the tweet above and provide specific recommendations for improvement."
    ),
])

generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Twitter tech influencer assistant tasked with writing "
        "excellent Twitter posts. Generate the best Twitter post possible "
        "for the user's request. If the user provides critique, respond "
        "with a revised version of your previous attempts."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm