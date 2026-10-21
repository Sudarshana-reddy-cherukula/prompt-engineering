import os
import re
from groq import Groq
from dotenv import load_dotenv

# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 400

client = Groq(
    api_key=API_KEY,
    timeout=30.0,
    max_retries=2
)


# =====================================================
# 2. DEFINE THE THREE PROMPTS
# =====================================================

baseline_prompt = (
    "Explain the importance of exercise."
)

role_based_prompt = (
    "You are a motivational fitness coach. "
    "Explain the importance of exercise."
)

role_negative_prompt = (
    "You are a motivational fitness coach. "
    "Do not use any medical jargon. "
    "Do not mention weight loss. "
    "Explain the importance of exercise."
)

prompts = [
    ("BASELINE", baseline_prompt),
    ("ROLE-BASED ONLY", role_based_prompt),
    ("ROLE + NEGATIVE PROMPTING", role_negative_prompt)
]


# =====================================================
# 3. GENERATE RESPONSE
# =====================================================

def generate_response(prompt):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise ValueError("The API returned an empty response.")

    return content.strip()


# =====================================================
# 4. NEGATIVE CONSTRAINT VALIDATION
# =====================================================

def check_negative_constraints(response):

    # Check whether the response mentions weight loss.
    mentions_weight_loss = bool(
        re.search(r"\bweight[\s-]+loss\b", response, re.IGNORECASE)
    )

    # Basic screening for common medical terms.
    # This is only a simple keyword check, not a complete
    # assessment of whether the response contains medical jargon.
    medical_terms = [
        "cardiovascular",
        "metabolism",
        "hypertension",
        "glucose regulation",
        "endocrine",
        "aerobic capacity",
        "physiological"
    ]

    found_medical_terms = [
        term
        for term in medical_terms
        if re.search(
            r"\b" + re.escape(term) + r"\b",
            response,
            re.IGNORECASE
        )
    ]

    return {
        "mentions_weight_loss": mentions_weight_loss,
        "found_medical_terms": found_medical_terms,
        "weight_loss_constraint_pass": not mentions_weight_loss,
        "medical_jargon_screen_pass": len(found_medical_terms) == 0
    }


# =====================================================
# 5. RUN EXPERIMENT
# =====================================================

def main():

    results = {}

    print("=" * 75)
    print("WEEK 6: ROLE-BASED AND NEGATIVE PROMPTING")
    print("=" * 75)

    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")

    for title, prompt in prompts:

        print("\n" + "=" * 75)
        print(title)
        print("=" * 75)

        print("\nPrompt:")
        print(prompt)

        try:
            output = generate_response(prompt)

            results[title] = {
                "prompt": prompt,
                "response": output
            }

            print("\nGenerated Response:")
            print(output)

            print("\nWord Count:", len(output.split()))

            if title == "ROLE + NEGATIVE PROMPTING":

                checks = check_negative_constraints(output)

                print("\nNegative Constraint Checks:")

                print(
                    "Weight-loss mention:",
                    "FOUND" if checks["mentions_weight_loss"]
                    else "NOT FOUND"
                )

                print(
                    "Medical-term screening:",
                    "TERMS FOUND"
                    if checks["found_medical_terms"]
                    else "NO LISTED TERMS FOUND"
                )

                if checks["found_medical_terms"]:
                    print(
                        "Flagged terms:",
                        ", ".join(checks["found_medical_terms"])
                    )

        except Exception as error:
            print("\nAPI Error:", repr(error))

            results[title] = {
                "prompt": prompt,
                "response": None,
                "error": str(error)
            }

    # =================================================
    # 6. COMPARISON SUMMARY
    # =================================================

    print("\n\n" + "=" * 75)
    print("COMPARISON SUMMARY")
    print("=" * 75)

    print(
        f"{'Condition':<30}"
        f"{'Word Count':<15}"
        f"{'Status':<15}"
    )

    print("-" * 60)

    for title, _ in prompts:

        result = results[title]
        output = result.get("response")

        if output:
            word_count = len(output.split())
            status = "Generated"
        else:
            word_count = "-"
            status = "Failed"

        print(
            f"{title:<30}"
            f"{str(word_count):<15}"
            f"{status:<15}"
        )

    # =================================================
    # 7. ANALYSIS
    # =================================================

    print("\n" + "=" * 75)
    print("ANALYSIS")
    print("=" * 75)

    print("\n1. Tone:")
    print(
        "Compare whether the role-based response sounds more "
        "motivational, encouraging, or conversational."
    )

    print("\n2. Vocabulary:")
    print(
        "Compare the language used in the baseline response "
        "with the language used by the fitness coach."
    )

    print("\n3. Constraint Adherence:")

    role_negative_result = results["ROLE + NEGATIVE PROMPTING"]
    output = role_negative_result.get("response")

    if output:
        checks = check_negative_constraints(output)

        print(
            "Weight-loss constraint:",
            "PASS" if checks["weight_loss_constraint_pass"]
            else "FAIL"
        )

        print(
            "Medical-term keyword screen:",
            "PASS" if checks["medical_jargon_screen_pass"]
            else "REVIEW REQUIRED"
        )

        print(
            "\nNote: Keyword screening cannot reliably determine "
            "whether every phrase is medical jargon. Review the "
            "response manually."
        )
    else:
        print("Constraint checks unavailable because generation failed.")

    print("\nExperiment completed.")


# =====================================================
# 8. ENTRY POINT
# =====================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")