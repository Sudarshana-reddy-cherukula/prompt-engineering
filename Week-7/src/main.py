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
        "GROQ_API_KEY not found. Check that your .env file "
        "exists in the project root and contains your API key."
    )

client = Groq(api_key=API_KEY)

MODEL = "qwen/qwen3.8-27b"


# --------------------------------------------------
# 2. Prompt Definitions
# --------------------------------------------------

INITIAL_PROMPT = """
Describe a smartwatch in exactly 3 bullet points.

Requirements:
- The entire response must contain no more than 60 words.
- Each bullet point must start with an action verb.
- Do not use the word "smart", regardless of capitalization.
- Output only the 3 bullet points.
"""

REFINED_PROMPT = """
Describe a wearable fitness watch.

Follow every constraint:
1. Output exactly 3 bullet points.
2. Start each bullet point with an action verb.
3. Keep the complete response at 60 words or fewer.
4. Do not use the word "smart" in any capitalization.
5. Do not include an introduction, conclusion, heading, or text
   outside the 3 bullet points.

Example of the required format:
- Track daily activity and movement.
- Monitor sleep patterns throughout the night.
- Display notifications directly on the wrist.

Now describe the wearable fitness watch.
Return only the 3 bullet points.
"""


# --------------------------------------------------
# 3. Generate Model Response
# --------------------------------------------------

def generate_response(prompt: str) -> str:
    """Generate a response using the configured Groq model."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Follow the user's instructions precisely. "
                    "Respect every output constraint."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=300,
        reasoning_effort="none",
    )

    if not response.choices:
        raise RuntimeError("The model returned no choices.")

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError(
            "The model returned an empty response. Try again."
        )

    return content.strip()


# --------------------------------------------------
# 4. Validate Constraints
# --------------------------------------------------

ACTION_VERBS = {
    "track",
    "monitor",
    "display",
    "measure",
    "record",
    "provide",
    "analyze",
    "count",
    "detect",
    "show",
    "sync",
    "support",
    "manage",
    "check",
    "connect",
    "deliver",
    "help",
    "enable",
    "measure",
    "notify",
    "calculate",
    "review",
    "guide",
    "offer",
    "capture",
    "estimate",
    "control",
    "log",
    "alert",
    "improve",
}


def validate_response(response: str) -> dict:
    """Check whether the response satisfies all constraints."""

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    # Remove Markdown bullet markers to count bullet content.
    bullet_pattern = re.compile(r"^[-*•]\s+(.+)$")
    bullets = []

    for line in lines:
        match = bullet_pattern.match(line)
        if match:
            bullets.append(match.group(1).strip())

    word_count = len(re.findall(r"\b[\w'-]+\b", response))

    exactly_three_bullets = (
        len(lines) == 3
        and len(bullets) == 3
        and all(bullet_pattern.match(line) for line in lines)
    )

    starts_with_action_verb = (
        len(bullets) == 3
        and all(
            bullet.split()[0].lower().strip(".,:;!?")
            in ACTION_VERBS
            for bullet in bullets
            if bullet.split()
        )
        and all(bullet.split() for bullet in bullets)
    )

    banned_word_absent = not re.search(
        r"\bsmart\b",
        response,
        flags=re.IGNORECASE,
    )

    within_word_limit = word_count <= 60

    return {
        "exactly_3_bullets": exactly_three_bullets,
        "starts_with_action_verb": starts_with_action_verb,
        "banned_word_absent": banned_word_absent,
        "within_60_words": within_word_limit,
        "word_count": word_count,
        "all_constraints_pass": (
            exactly_three_bullets
            and starts_with_action_verb
            and banned_word_absent
            and within_word_limit
        ),
    }


# --------------------------------------------------
# 5. Display Results
# --------------------------------------------------

def display_results(title: str, response: str) -> dict:
    """Print the response and its constraint-validation results."""

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print("\nMODEL RESPONSE:\n")
    print(response)

    results = validate_response(response)

    print("\nVALIDATION RESULTS:")
    print("-" * 45)

    checks = [
        ("Exactly 3 bullet points", "exactly_3_bullets"),
        ("Each bullet starts with an action verb", "starts_with_action_verb"),
        ("Banned word absent", "banned_word_absent"),
        ("60 words or fewer", "within_60_words"),
    ]

    for label, key in checks:
        status = "PASS" if results[key] else "FAIL"
        print(f"{label}: {status}")

    print(f"Word count: {results['word_count']}")

    overall = (
        "PASS"
        if results["all_constraints_pass"]
        else "FAIL"
    )
    print(f"Overall result: {overall}")

    return results


# --------------------------------------------------
# 6. Main Experiment
# --------------------------------------------------

def main():
    print("=" * 65)
    print("WEEK 7: CONSTRAINT SPECIFICATION AND ITERATIVE REFINEMENT")
    print("=" * 65)

    print(f"\nActive model: {MODEL}")

    try:
        print("\nRunning the initial prompt...")
        initial_response = generate_response(INITIAL_PROMPT)

        initial_results = display_results(
            "EXPERIMENT 1: INITIAL PROMPT",
            initial_response,
        )

        print("\nRunning the refined prompt...")
        refined_response = generate_response(REFINED_PROMPT)

        refined_results = display_results(
            "EXPERIMENT 2: REFINED PROMPT",
            refined_response,
        )

        print("\n" + "=" * 65)
        print("EXPERIMENT COMPARISON")
        print("=" * 65)

        print(
            "Initial prompt passed all constraints: "
            f"{initial_results['all_constraints_pass']}"
        )
        print(
            "Refined prompt passed all constraints: "
            f"{refined_results['all_constraints_pass']}"
        )

        improvement = (
            refined_results["all_constraints_pass"]
            and not initial_results["all_constraints_pass"]
        )

        if improvement:
            print(
                "\nConclusion: The refined prompt satisfied all "
                "constraints, while the initial prompt did not."
            )
        elif (
            initial_results["all_constraints_pass"]
            and refined_results["all_constraints_pass"]
        ):
            print(
                "\nConclusion: Both prompts satisfied all constraints. "
                "The refined prompt provides a more explicit structure."
            )
        elif refined_results["all_constraints_pass"]:
            print(
                "\nConclusion: The refined prompt satisfied all constraints."
            )
        else:
            print(
                "\nConclusion: One or more constraints were not satisfied. "
                "Inspect the validation results and refine the prompt."
            )

        print("\nWeek 7 experiment completed.")

    except Exception as error:
        print("\nERROR: The experiment could not be completed.")
        print(f"Details: {error}")
        print(
            "\nVerify your GROQ_API_KEY, model access, internet "
            "connection, and installed dependencies."
        )


if __name__ == "__main__":
    main()