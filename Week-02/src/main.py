import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

client = Groq(api_key=API_KEY)

# Use a model available in your Groq account
MODEL = "openai/gpt-oss-20b"


def generate_response(prompt):
    """Generate a response using Groq."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=500,
    )

    if not response.choices:
        raise RuntimeError("The model returned no choices.")

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("The model returned an empty response.")

    return answer.strip()


def evaluate_enhanced_response(response):
    """Check basic output requirements."""
    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    bullets = [
        line for line in lines
        if line.startswith(("-", "*", "•"))
    ]

    print("\nEnhanced Prompt Format Evaluation")
    print("-" * 45)

    print(
        "Exactly three bullet points:",
        "PASS" if len(bullets) == 3 else "FAIL"
    )

    if len(bullets) == 3:
        word_counts = [len(bullet.split()) for bullet in bullets]

        print("Words per bullet:", word_counts)

        print(
            "Each bullet under 20 words:",
            "PASS" if all(count < 20 for count in word_counts)
            else "FAIL"
        )

    print(
        "Relevance: Review whether the response answers the prompt."
    )
    print(
        "Accuracy: Verify the factual claims independently."
    )


def main():
    baseline_prompt = (
        "Tell me about climate change."
    )

    enhanced_prompt = (
        "You are a climate scientist. "
        "Explain the three main causes of climate change. "
        "Return exactly three bullet points. "
        "Each bullet must contain fewer than 20 words. "
        "Do not include an introduction or conclusion."
    )

    print("=" * 60)
    print("WEEK 02: BASELINE VS. ENHANCED PROMPTS")
    print("=" * 60)

    # Run baseline experiment
    print("\n1. BASELINE PROMPT")
    print("-" * 60)
    print("Prompt:", baseline_prompt)

    try:
        baseline = generate_response(baseline_prompt)

        print("\nBaseline Response:\n")
        print(baseline)

    except Exception as error:
        print("\nBaseline API Error:", error)
        baseline = None

    # Run enhanced experiment independently
    print("\n\n2. ENHANCED PROMPT")
    print("-" * 60)
    print("Prompt:", enhanced_prompt)

    try:
        enhanced = generate_response(enhanced_prompt)

        print("\nEnhanced Response:\n")
        print(enhanced)

        evaluate_enhanced_response(enhanced)

    except Exception as error:
        print("\nEnhanced API Error:", error)
        enhanced = None

    # Compare actual results
    print("\n\n3. COMPARISON")
    print("=" * 60)

    if baseline is not None and enhanced is not None:
        print("Baseline word count:", len(baseline.split()))
        print("Enhanced word count:", len(enhanced.split()))

        print("\nEvaluation Criteria")
        print("-" * 45)
        print("1. Relevance")
        print("2. Conciseness")
        print("3. Accuracy")
        print("4. Format adherence")

        print("\nRESULT")
        print("-" * 45)
        print(
            "Both prompts generated responses successfully. "
            "The enhanced response can be checked against "
            "the specified formatting constraints."
        )

    else:
        print(
            "Comparison incomplete because at least one "
            "API request failed. Check the error above."
        )


if __name__ == "__main__":
    main()