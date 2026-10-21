import os
import re
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
MAX_TOKENS = 500

EXPECTED_ANSWER = 11


# --------------------------------------------------
# 2. Define the Problem and Prompts
# --------------------------------------------------

PROBLEM = (
    "A farmer has 15 cows. He buys 7 more cows. "
    "Then he sells half of his cows. How many cows "
    "does he have left?"
)

DIRECT_PROMPT = PROBLEM

COT_PROMPT = (
    PROBLEM
    + "\n\nLet's think step by step. "
    "Show the calculation for each step, then clearly "
    "state the final numerical answer."
)


# --------------------------------------------------
# 3. Generate LLM Response
# --------------------------------------------------

def generate_response(prompt: str) -> str:
    """Send a prompt to the Groq model and return its response."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
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
# 4. Extract Final Numerical Answer
# --------------------------------------------------

def extract_final_answer(response: str):
    """
    Extract the final numerical answer.

    Prefer a number following phrases such as 'Final answer'
    or 'Answer'. Otherwise, use the last integer in the response.
    """

    patterns = [
        r"final\s+answer\s*(?:is|:|=)?\s*\**\s*(\d+)",
        r"\banswer\s*(?:is|:|=)\s*\**\s*(\d+)",
        r"\btherefore[,:\s]+(?:he has\s+)?(\d+)\s+cows\b",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            response,
            flags=re.IGNORECASE,
        )

        if matches:
            return int(matches[-1])

    # Fallback: use the last integer found in the response.
    numbers = re.findall(r"\b\d+\b", response)

    if numbers:
        return int(numbers[-1])

    return None


# --------------------------------------------------
# 5. Evaluate Answer
# --------------------------------------------------

def is_correct(answer):
    """Compare an extracted answer with the expected answer."""

    return answer == EXPECTED_ANSWER


# --------------------------------------------------
# 6. Optional: Decompose the Problem
# --------------------------------------------------

def solve_step_by_step():
    """Solve the problem through two smaller prompts."""

    step1_prompt = (
        "A farmer has 15 cows and buys 7 more cows. "
        "How many cows does he have now? "
        "Give the numerical answer and a short calculation."
    )

    step2_prompt = (
        "A farmer now has 22 cows and sells half of them. "
        "How many cows remain? "
        "Give the numerical answer and a short calculation."
    )

    print("\n" + "=" * 65)
    print("OPTIONAL: TASK DECOMPOSITION")
    print("=" * 65)

    print("\nStep 1: Calculate cows after buying.")
    result1 = generate_response(step1_prompt)
    print(result1)

    print("\nStep 2: Calculate cows after selling half.")
    result2 = generate_response(step2_prompt)
    print(result2)


# --------------------------------------------------
# 7. Main Experiment
# --------------------------------------------------

def main():
    print("=" * 65)
    print("WEEK 10: DIRECT PROMPTING VS. CHAIN-OF-THOUGHT")
    print("=" * 65)

    print(f"\nModel: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Expected answer: {EXPECTED_ANSWER} cows")

    try:
        # Direct prompting
        print("\n" + "=" * 65)
        print("=== DIRECT PROMPT ===")
        print("=" * 65)

        direct_response = generate_response(DIRECT_PROMPT)
        direct_answer = extract_final_answer(direct_response)

        print("\nModel response:")
        print(direct_response)

        print(f"\nExtracted answer: {direct_answer}")
        print(f"Correct: {'YES' if is_correct(direct_answer) else 'NO'}")

        # Chain-of-Thought prompting
        print("\n" + "=" * 65)
        print("=== CHAIN-OF-THOUGHT PROMPT ===")
        print("=" * 65)

        cot_response = generate_response(COT_PROMPT)
        cot_answer = extract_final_answer(cot_response)

        print("\nModel response:")
        print(cot_response)

        print(f"\nExtracted answer: {cot_answer}")
        print(f"Correct: {'YES' if is_correct(cot_answer) else 'NO'}")

        # Compare results
        print("\n" + "=" * 65)
        print("=== COMPARISON ===")
        print("=" * 65)

        print(
            f"Direct prompting: {direct_answer} "
            f"({'CORRECT' if is_correct(direct_answer) else 'INCORRECT'})"
        )

        print(
            f"CoT prompting: {cot_answer} "
            f"({'CORRECT' if is_correct(cot_answer) else 'INCORRECT'})"
        )

        print(f"Expected answer: {EXPECTED_ANSWER}")

        if (
            is_correct(direct_answer)
            and is_correct(cot_answer)
        ):
            print(
                "\nAnalysis: Both prompts produced the correct answer. "
                "CoT can make the calculation steps more transparent, "
                "but this single example does not prove greater accuracy."
            )
        elif (
            not is_correct(direct_answer)
            and is_correct(cot_answer)
        ):
            print(
                "\nAnalysis: CoT produced the correct answer, "
                "while direct prompting did not."
            )
        elif (
            is_correct(direct_answer)
            and not is_correct(cot_answer)
        ):
            print(
                "\nAnalysis: Direct prompting was correct, "
                "while CoT produced an incorrect answer."
            )
        else:
            print(
                "\nAnalysis: Neither prompt produced the expected answer. "
                "Inspect both responses and the answer-extraction results."
            )

        # Optional decomposition experiment
        solve_step_by_step()

        print("\nExperiment completed successfully.")

    except Exception as error:
        print("\nERROR: Experiment failed.")
        print(f"Details: {error}")
        print(
            "\nCheck your API key, model access, internet connection, "
            "and installed dependencies."
        )


if __name__ == "__main__":
    main()