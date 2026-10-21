import os
import json
from dotenv import load_dotenv
from groq import Groq

# --------------------------------------------------
# 1. Environment Setup
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=API_KEY)

MODEL = "qwen/qwen3.8-27b"
TEMPERATURE = 0
MAX_TOKENS = 800


# --------------------------------------------------
# 2. Define Prompts
# --------------------------------------------------

INITIAL_PROMPT = """
Generate a JSON array containing exactly three books.

Each book must be represented as an object with exactly these keys:
- "title": string
- "author": string
- "year": integer

Requirements:
1. Use real books and their correct publication years.
2. Use double quotes for all JSON keys and string values.
3. Do not include trailing commas.
4. Return only the JSON array.
5. Do not include Markdown code fences or explanations.

Expected structure:
[
  {
    "title": "Book Title",
    "author": "Author Name",
    "year": 2000
  }
]
"""

STRICT_PROMPT = """
Return ONLY one valid JSON array containing exactly three books.

Every array element must be a JSON object with exactly these keys:
"title", "author", "year"

Data types:
- title: string
- author: string
- year: integer

Rules:
- Use real books and accurate publication years.
- Use double quotes around all keys and string values.
- Do not use trailing commas.
- Do not include comments.
- Do not include Markdown code fences.
- Do not include explanations or text before or after the JSON.
- Do not add any other keys.
- The response must be directly parseable by Python's json.loads().

Return the JSON array only.
"""


# --------------------------------------------------
# 3. Generate JSON Response
# --------------------------------------------------

def generate_response(prompt: str) -> str:
    """Generate a response from the Groq model."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate valid JSON exactly as requested. "
                    "Never include Markdown or explanatory text."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_TOKENS,
        reasoning_effort="none",
    )

    if not response.choices:
        raise RuntimeError("The model returned no response choices.")

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("The model returned an empty response.")

    return content.strip()


# --------------------------------------------------
# 4. Validate JSON Structure
# --------------------------------------------------

def validate_books(data):
    """Validate the JSON data and required book fields."""

    if not isinstance(data, list):
        raise ValueError("The JSON root must be an array.")

    if len(data) != 3:
        raise ValueError("The JSON array must contain exactly 3 books.")

    required_keys = {"title", "author", "year"}

    for index, book in enumerate(data, start=1):

        if not isinstance(book, dict):
            raise ValueError(
                f"Book {index} must be a JSON object."
            )

        if set(book.keys()) != required_keys:
            raise ValueError(
                f"Book {index} must contain exactly these keys: "
                "'title', 'author', 'year'."
            )

        if not isinstance(book["title"], str) or not book["title"].strip():
            raise ValueError(
                f"Book {index}: title must be a non-empty string."
            )

        if not isinstance(book["author"], str) or not book["author"].strip():
            raise ValueError(
                f"Book {index}: author must be a non-empty string."
            )

        # bool is a subclass of int in Python, so reject it explicitly.
        if type(book["year"]) is not int:
            raise ValueError(
                f"Book {index}: year must be an integer."
            )

        if not 1 <= book["year"] <= 2100:
            raise ValueError(
                f"Book {index}: year is outside the accepted range."
            )

    return True


# --------------------------------------------------
# 5. Parse and Retry
# --------------------------------------------------

def generate_valid_books():
    """Generate JSON, parse it, and retry once if invalid."""

    prompts = [
        INITIAL_PROMPT,
        STRICT_PROMPT,
    ]

    for attempt, prompt in enumerate(prompts, start=1):

        print(f"\nAttempt {attempt} of {len(prompts)}")

        try:
            raw_response = generate_response(prompt)

            print("\nRaw model response:")
            print(raw_response)

            # Parse JSON.
            books = json.loads(raw_response)

            # Validate schema and data types.
            validate_books(books)

            print("\nJSON parsing and validation successful.")

            return books

        except json.JSONDecodeError as error:
            print("\nJSON parsing failed.")
            print(f"Error: {error}")

        except ValueError as error:
            print("\nJSON validation failed.")
            print(f"Error: {error}")

    raise RuntimeError(
        "Could not generate valid JSON after both attempts."
    )


# --------------------------------------------------
# 6. Main Experiment
# --------------------------------------------------

def main():
    print("=" * 60)
    print("WEEK 9: JSON GENERATION AND VALIDATION")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")

    try:
        books = generate_valid_books()

        print("\n" + "=" * 60)
        print("FINAL VALID JSON OUTPUT")
        print("=" * 60)

        # Print correctly formatted JSON.
        print(json.dumps(books, indent=2, ensure_ascii=False))

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print("Valid JSON syntax: PASS")
        print("Root is an array: PASS")
        print("Exactly three books: PASS")
        print("Required keys present: PASS")
        print("Data types correct: PASS")

        print("\nExperiment completed successfully.")

    except Exception as error:
        print("\nEXPERIMENT FAILED")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()