
import os
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
TEMPERATURE = 0.2
MAX_TOKENS = 500


# --------------------------------------------------
# 2. Define Prompts
# --------------------------------------------------

UNSTRUCTURED_PROMPT = """
Describe Python, Java, and C++ in a short paragraph.

For each language, briefly explain its typing system,
primary use cases, and general learning difficulty.

Keep the explanation concise and readable.
"""

STRUCTURED_PROMPT = """
Create a Markdown table comparing Python, Java, and C++.

The table must contain exactly these columns:
1. Language
2. Typing (static/dynamic)
3. Primary Use Case
4. Difficulty (Beginner/Intermediate/Advanced)

Include exactly three data rows, one for each language:
Python, Java, and C++.

Requirements:
- Use a valid Markdown table.
- Include all four specified column headings.
- Use Static or Dynamic in the Typing column.
- Use Beginner, Intermediate, or Advanced in the Difficulty column.
- Keep the primary use case concise.
- Do not include JSON.
- Do not add an introduction or conclusion outside the table.
"""


# --------------------------------------------------
# 3. Call Groq API
# --------------------------------------------------

def generate_response(prompt: str) -> str:
    """Send a prompt to the selected Groq model."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Follow the user's instructions carefully. "
                    "Return the requested output format."
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
# 4. Validate Structured Output
# --------------------------------------------------

def validate_markdown_table(response: str) -> dict:
    """Check the table's columns and language rows."""

    required_columns = [
        "Language",
        "Typing (static/dynamic)",
        "Primary Use Case",
        "Difficulty (Beginner/Intermediate/Advanced)",
    ]

    required_languages = {"Python", "Java", "C++"}

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    # Markdown table rows begin and end with vertical bars.
    table_rows = [
        line
        for line in lines
        if line.startswith("|") and line.endswith("|")
    ]

    header = table_rows[0] if table_rows else ""

    actual_columns = [
        cell.strip()
        for cell in header.strip("|").split("|")
    ]

    columns_correct = actual_columns == required_columns

    # Exclude the header and Markdown separator row.
    data_rows = []

    for row in table_rows[1:]:
        cells = [
            cell.strip()
            for cell in row.strip("|").split("|")
        ]

        if cells and all(
            set(cell) <= {"-", ":"}
            for cell in cells
        ):
            continue

        data_rows.append(cells)

    languages = {
        row[0]
        for row in data_rows
        if len(row) == len(required_columns)
    }

    rows_correct = (
        len(data_rows) == 3
        and languages == required_languages
        and all(len(row) == 4 for row in data_rows)
    )

    return {
        "valid_markdown_table": bool(table_rows),
        "correct_columns": columns_correct,
        "correct_language_rows": rows_correct,
        "all_checks_pass": (
            bool(table_rows)
            and columns_correct
            and rows_correct
        ),
    }


# --------------------------------------------------
# 5. Run Experiment
# --------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT: STRUCTURED VS. UNSTRUCTURED LLM OUTPUT")
    print("=" * 70)

    print(f"\nModel: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")

    try:
        # Generate the unstructured response.
        print("\nGenerating unstructured response...")
        unstructured_output = generate_response(
            UNSTRUCTURED_PROMPT
        )

        # Generate the structured response.
        print("Generating structured response...")
        structured_output = generate_response(
            STRUCTURED_PROMPT
        )

        # Display both outputs.
        print("\n" + "=" * 70)
        print("=== UNSTRUCTURED OUTPUT ===")
        print("=" * 70)
        print(unstructured_output)

        print("\n" + "=" * 70)
        print("=== STRUCTURED OUTPUT (MARKDOWN TABLE) ===")
        print("=" * 70)
        print(structured_output)

        # Validate the structured table.
        validation = validate_markdown_table(
            structured_output
        )

        print("\n" + "=" * 70)
        print("STRUCTURED OUTPUT VALIDATION")
        print("=" * 70)

        for check, passed in validation.items():
            status = "PASS" if passed else "FAIL"
            print(f"{check.replace('_', ' ').title()}: {status}")

        # Print the comparison analysis.
        print("\n" + "=" * 70)
        print("ANALYSIS: STRUCTURED VS. UNSTRUCTURED")
        print("=" * 70)

        print(
            "1. Readability: The paragraph provides a natural "
            "explanation, while the table makes information easier "
            "to scan."
        )
        print(
            "2. Comparison: The table places the three languages "
            "in separate rows, making differences easier to identify."
        )
        print(
            "3. Consistency: Explicit column names encourage the "
            "model to organize information consistently."
        )
        print(
            "4. Validation: The table can be checked automatically "
            "for required columns and rows."
        )

        if validation["all_checks_pass"]:
            print(
                "\nResult: The structured output passed all "
                "table validation checks."
            )
        else:
            print(
                "\nResult: The structured output did not pass every "
                "validation check. Review the generated table."
            )

        print("\nExperiment completed successfully.")

    except Exception as error:
        print("\nERROR: Experiment failed.")
        print(f"Details: {error}")
        print(
            "\nCheck your GROQ_API_KEY, model access, internet "
            "connection, and installed dependencies."
        )


if __name__ == "__main__":
    main()

