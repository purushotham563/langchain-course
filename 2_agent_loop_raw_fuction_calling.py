import json

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

load_dotenv()
MAX_ITERATIONS = 10
MODEL = "gemini-3.5-flash-lite"

create_get_product_price = {
    "type": "function",
    "name": "get_product_price",
    "description": "Look up the price of a product in the catelog.",
    "parameters": {
        "type": "object",
        "properties": {
            "product": {
                "type": "string",
                "description": "The name of the product eg. Laptop , headphones etc",
            }
        },
        "required": ["product"],
    },
}
create_apply_discount_price = {
    "type": "function",
    "name": "apply_discount_price",
    "description": "Apply a discount tier to a price and return final price Avaliable tires: bronze, silver, gold.",
    "parameters": {
        "type": "object",
        "properties": {
            "price": {
                "type": "number",
                "description": "The original amount of the product eg. 1078.09, 78.2 etc",
            },
            "discount_tier": {
                "type": "string",
                "description": "The discount tier: 'bronze', 'silver', or 'gold'",
            },
        },
        "required": ["price", "discount_tier"],
    },
}


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catelog"""
    print(f"  >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.0, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount_price(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return final price
    Avaliable tires: bronze, silver, gold."""
    print(f"  >> apply_discount_price(price='{price}',discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [create_get_product_price, create_apply_discount_price]


@traceable(name="google_gen_ai", run_type="llm")
def google_chat_traced(messages):
    client = genai.Client()

    return client.interactions.create(model=MODEL, input=messages, tools=tools_for_llm)


@traceable(name="LangChain agent loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount_price": apply_discount_price,
    }
    print(f"Question {question}")
    print("=" * 60)
    messages = [
        {
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful shopping assistant.\n"
                        "You have access to a product catalog tool "
                        "and a discount tool.\n\n"
                        "STRICT RULES — you must follow these exactly:\n"
                        "1. NEVER guess or assume any product price. "
                        "You MUST call get_product_price first to get the real price.\n"
                        "2. Only call apply_discount_price AFTER you have received "
                        "a price from get_product_price. Pass the exact price "
                        "returned by get_product_price.\n"
                        "3. NEVER calculate discounts yourself using math. "
                        "Always use the apply_discount_price tool.\n"
                        "4. If the user does not specify a discount tier, "
                        "ask them which tier to use — do NOT assume one.\n\n"
                        f"User request: {question}"
                    ),
                }
            ],
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---- Iteration {iteration} ---")
        response = google_chat_traced(messages=messages)
        for step in response.steps:
            messages.append(step.model_dump())
        fc_step = next(
            (step for step in response.steps if step.type == "function_call"), None
        )
        if fc_step is None:
            print(f"\nFinal Answer: {response.output_text}")
            return response.output_text
        function_name = fc_step.name
        function_args = fc_step.arguments
        function_id = fc_step.id
        tool_to_use = tools_dict.get(function_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{function_name}' not found")
        observation = tool_to_use(**function_args)
        messages.append(
            {
                "type": "function_result",
                "name": function_name,
                "call_id": function_id,
                "result": [{"type": "text", "text": json.dumps(observation)}],
            }
        )

    print("ERROR : Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is price of a laptop after applying gold discount")
