import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


def generate_response(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_completion_tokens=300
    )

    return response.choices[0].message.content.strip()


def check_json(response):
    try:
        json.loads(response)
        return "PASS"
    except json.JSONDecodeError:
        return "FAIL"


def run_test(title, original_prompt, improved_prompt, check_type):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    try:
        original_response = generate_response(original_prompt)
        improved_response = generate_response(improved_prompt)

        print("\nOriginal Prompt:")
        print(original_prompt)

        print("\nOriginal Response:")
        print(original_response)

        print("\nImproved Prompt:")
        print(improved_prompt)

        print("\nImproved Response:")
        print(improved_response)

        if check_type == "json":
            original_result = check_json(original_response)
            improved_result = check_json(improved_response)

        elif check_type == "bullets":
            original_result = (
                "PASS" if original_response.count("\n- ") >= 2
                else "CHECK"
            )
            improved_result = (
                "PASS" if improved_response.count("\n- ") >= 2
                else "CHECK"
            )

        else:
            original_result = "MANUAL REVIEW"
            improved_result = "MANUAL REVIEW"

        print("\nEvaluation:")
        print("Original:", original_result)
        print("Improved:", improved_result)

    except Exception as error:
        print("Error:", error)


# Test 1: Ambiguous instructions
run_test(
    "TEST 1: AMBIGUOUS INSTRUCTIONS",
    "Explain Python briefly.",
    (
        "Explain Python programming in exactly three bullet points. "
        "Each bullet must contain fewer than 15 words."
    ),
    "bullets"
)

# Test 2: JSON format violation
run_test(
    "TEST 2: JSON FORMAT",
    "Tell me about three programming languages.",
    (
        "Return only valid JSON with this structure: "
        '{"languages": [{"name": "Python", "purpose": "example"}]}. '
        "Include exactly three languages. Do not use Markdown."
    ),
    "json"
)

# Test 3: Contradictory instructions
run_test(
    "TEST 3: CONTRADICTORY INSTRUCTIONS",
    "Explain artificial intelligence in exactly 10 words and exactly 50 words.",
    (
        "Explain artificial intelligence in exactly 20 words. "
        "Use simple English and provide one complete sentence."
    ),
    "manual"
)

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)