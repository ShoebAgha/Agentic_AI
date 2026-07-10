from dotenv import load_dotenv

load_dotenv()

import anthropic
import json
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "anthropic:claude-haiku-4-5-20251001"

client = anthropic.Anthropic()


# --- Tools ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                }
            },
            "required": ["product"],
        },
    },
    {
        "name": "apply_discount",
        "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
        "input_schema": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The original price",
                },
                "discount_tier": {
                    "type": "string",
                    "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                },
            },
            "required": ["price", "discount_tier"],
        },
    },
]


# --- Helper: traced Claude call ---


@traceable(name="Claude Messages API", run_type="llm")
def claude_chat_traced(messages):
    return client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "You are a helpful shopping assistant. "
            "You have access to a product catalog tool "
            "and a discount tool.\n\n"
            "STRICT RULES — you must follow these exactly:\n"
            "1. NEVER guess or assume any product price. "
            "You MUST call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AFTER you have received "
            "a price from get_product_price. Pass the exact price "
            "returned by get_product_price — do NOT pass a made-up number.\n"
            "3. NEVER calculate discounts yourself using math. "
            "Always use the apply_discount tool.\n"
            "4. If the user does not specify a discount tier, "
            "ask them which tier to use — do NOT assume one."
        ),
        tools=tools_for_llm,
        messages=messages,
    )


# --- Agent Loop ---


@traceable(name="Claude Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = claude_chat_traced(messages=messages)

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        text_blocks = [block.text for block in response.content if block.type == "text"]

        if not tool_uses:
            final_text = "\n".join(text_blocks).strip()
            print(f"\nFinal Answer: {final_text}")
            return final_text

        tool_use = tool_uses[0]
        tool_name = tool_use.name
        tool_args = tool_use.input

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(observation),
                    }
                ],
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Claude Agent (tool use)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")