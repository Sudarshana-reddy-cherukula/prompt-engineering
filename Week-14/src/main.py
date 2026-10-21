import os
import ast
import operator
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# --------------------------------------------------
# 1. Environment Setup
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

MODEL = "qwen/qwen3.8-27b"


# --------------------------------------------------
# 2. Safe Calculator
# --------------------------------------------------

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_evaluate(expression: str):
    """Safely evaluate arithmetic expressions without using eval()."""

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError("Only numeric constants are allowed.")
            return node.value

        if isinstance(node, ast.BinOp):
            operation_type = type(node.op)

            if operation_type not in ALLOWED_OPERATORS:
                raise ValueError("Operator is not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operation_type is ast.Pow and abs(right) > 10:
                raise ValueError("Exponent is too large.")

            result = ALLOWED_OPERATORS[operation_type](left, right)

            if abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation_type = type(node.op)

            if operation_type not in ALLOWED_OPERATORS:
                raise ValueError("Unary operator is not allowed.")

            return ALLOWED_OPERATORS[operation_type](
                evaluate(node.operand)
            )

        raise ValueError("Unsupported expression.")

    return evaluate(tree)


@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression.

    Supports addition, subtraction, multiplication, division,
    powers, and parentheses. Example: '320 * 15 / 100'.
    """

    try:
        result = safe_evaluate(expression)
        return f"Calculation: {expression} = {result}"
    except Exception as error:
        return f"Calculation error: {error}"


# --------------------------------------------------
# 3. Weather Tool
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a city.

    Uses Open-Meteo geocoding and current weather services.
    Returns the city name, temperature in Celsius, and conditions.
    """

    try:
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

        geocoding_response = requests.get(
            geocoding_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=15,
        )
        geocoding_response.raise_for_status()

        locations = geocoding_response.json().get("results", [])

        if not locations:
            return f"Weather error: Could not find city '{city}'."

        location = locations[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code",
                "temperature_unit": "celsius",
            },
            timeout=15,
        )
        weather_response.raise_for_status()

        current = weather_response.json().get("current", {})

        temperature = current.get("temperature_2m")

        if temperature is None:
            return "Weather error: Current temperature is unavailable."

        return (
            f"City: {location['name']}, {location.get('country', '')}. "
            f"Current temperature: {temperature}°C. "
            f"Weather code: {current.get('weather_code', 'unknown')}."
        )

    except requests.RequestException as error:
        return f"Weather API error: {error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Could not process weather data: {error}"


# --------------------------------------------------
# 4. Initialize Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model=MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
    reasoning_effort="none",
)


# --------------------------------------------------
# 5. Create ReAct Agent
# --------------------------------------------------

tools = [
    get_weather,
    calculate,
]

agent = create_react_agent(
    model=llm,
    tools=tools,
)


# --------------------------------------------------
# 6. Run Agent
# --------------------------------------------------

def main():
    print("=" * 70)
    print("WEEK 14: LANGGRAPH REACT AGENT")
    print("=" * 70)

    query = (
        "Find the current temperature in London using the weather tool. "
        "The temperature tool returns Celsius. Convert the temperature "
        "to Fahrenheit using the calculator tool. Then calculate 15% "
        "of 320 using the calculator tool and add that result to the "
        "temperature in Fahrenheit. Show the temperature, the percentage "
        "calculation, and the final combined total. Use actual tool results."
    )

    print(f"\nModel: {MODEL}")
    print(f"\nUser Query:\n{query}")

    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": query}
                ]
            },
            config={
                "recursion_limit": 15,
            },
        )

        messages = result["messages"]

        print("\n" + "=" * 70)
        print("AGENT EXECUTION TRACE")
        print("=" * 70)

        for message in messages:
            message_type = type(message).__name__

            # Print tool calls and their arguments.
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    print(
                        f"\nAction: {tool_call['name']}"
                    )
                    print(
                        f"Action Input: {tool_call['args']}"
                    )

            # Print tool observations.
            elif message_type == "ToolMessage":
                print(f"\nObservation: {message.content}")

            # Print assistant's final answer.
            elif message_type == "AIMessage" and message.content:
                if not getattr(message, "tool_calls", None):
                    print(f"\nAssistant: {message.content}")

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(messages[-1].content)

        print("\nExperiment completed successfully.")

    except Exception as error:
        print("\nERROR: Agent execution failed.")
        print(f"Details: {error}")
        print(
            "\nCheck your Groq API key, model access, "
            "internet connection, and package installation."
        )


if __name__ == "__main__":
    main()