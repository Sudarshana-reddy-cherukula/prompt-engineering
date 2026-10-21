import os
import re
import json
from groq import Groq
from dotenv import load_dotenv

# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 100

client = Groq(
    api_key=API_KEY,
    timeout=30.0,
    max_retries=2
)

LABELS = ["Positive", "Neutral", "Negative"]


# ============================================================
# 2. TEST DATA WITH GROUND-TRUTH LABELS
# ============================================================

test_reviews = [
    {
        "review": "The battery life is amazing!",
        "ground_truth": "Positive"
    },
    {
        "review": "It works, nothing special.",
        "ground_truth": "Neutral"
    },
    {
        "review": "The phone stopped working after two days.",
        "ground_truth": "Negative"
    },
    {
        "review": "The camera takes beautiful pictures.",
        "ground_truth": "Positive"
    },
    {
        "review": "The screen is too small.",
        "ground_truth": "Negative"
    }
]


# ============================================================
# 3. FEW-SHOT EXAMPLES
# ============================================================

few_shot_examples = """
Example 1:
Review: "Best phone ever!"
Sentiment: Positive

Example 2:
Review: "The camera is slow and frustrating."
Sentiment: Negative

Example 3:
Review: "Average quality, nothing special."
Sentiment: Neutral
"""


# ============================================================
# 4. PROMPT GENERATION
# ============================================================

def build_prompt(review, few_shot=False):

    if few_shot:
        prompt = f"""
You are a customer review sentiment classifier.

Classify the review into exactly one category:
Positive, Neutral, or Negative.

Follow these examples:

{few_shot_examples}

Now classify this review:
Review: "{review}"

Return only one label:
Positive
Neutral
or Negative

Do not provide an explanation.
"""
    else:
        prompt = f"""
Classify the following customer review into exactly one
category: Positive, Neutral, or Negative.

Review: "{review}"

Return only one label:
Positive
Neutral
or Negative

Do not provide an explanation.
"""

    return prompt.strip()


# ============================================================
# 5. CALL GROQ API
# ============================================================

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


# ============================================================
# 6. VALIDATE MODEL OUTPUT
# ============================================================

def validate_prediction(raw_output):

    # Accept a response containing exactly one recognized label.
    matches = re.findall(
        r"\b(Positive|Neutral|Negative)\b",
        raw_output,
        flags=re.IGNORECASE
    )

    # A response containing explanations or multiple labels
    # is not considered format-compliant.
    if len(matches) == 1:
        prediction = matches[0].capitalize()

        if raw_output.strip().lower() == prediction.lower():
            format_ok = True
        else:
            format_ok = False

        return prediction, format_ok

    return "Invalid", False


# ============================================================
# 7. CLASSIFY A REVIEW
# ============================================================

def classify_review(review, few_shot=False):

    prompt = build_prompt(
        review=review,
        few_shot=few_shot
    )

    raw_output = generate_response(prompt)

    prediction, format_ok = validate_prediction(raw_output)

    return prediction, format_ok, raw_output


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    zero_shot_predictions = []
    few_shot_predictions = []

    zero_shot_format = []
    few_shot_format = []

    print("\n" + "=" * 85)
    print("WEEK 5: ZERO-SHOT VS FEW-SHOT PROMPTING")
    print("=" * 85)

    print(f"Model       : {MODEL}")
    print(f"Temperature : {TEMPERATURE}")
    print(f"Test reviews: {len(test_reviews)}")
    print("Few-shot examples: 3")

    # --------------------------------------------------------
    # Classify all test reviews
    # --------------------------------------------------------

    for index, item in enumerate(test_reviews, start=1):

        review = item["review"]
        ground_truth = item["ground_truth"]

        print("\n" + "-" * 85)
        print(f"Review {index}: {review}")
        print(f"Ground Truth: {ground_truth}")

        # Zero-shot classification
        print("\nRunning zero-shot classification...")

        try:
            zero_prediction, zero_format, zero_raw = (
                classify_review(
                    review,
                    few_shot=False
                )
            )

        except Exception as error:
            print("Zero-shot API error:", repr(error))

            zero_prediction = "Error"
            zero_format = False
            zero_raw = str(error)

        print("Zero-shot response:", zero_raw)
        print("Zero-shot prediction:", zero_prediction)

        zero_shot_predictions.append(zero_prediction)
        zero_shot_format.append(zero_format)

        # Few-shot classification
        print("\nRunning few-shot classification...")

        try:
            few_prediction, few_format, few_raw = (
                classify_review(
                    review,
                    few_shot=True
                )
            )

        except Exception as error:
            print("Few-shot API error:", repr(error))

            few_prediction = "Error"
            few_format = False
            few_raw = str(error)

        print("Few-shot response:", few_raw)
        print("Few-shot prediction:", few_prediction)

        few_shot_predictions.append(few_prediction)
        few_shot_format.append(few_format)

    # --------------------------------------------------------
    # Calculate accuracy
    # --------------------------------------------------------

    total = len(test_reviews)

    zero_correct = sum(
        prediction == item["ground_truth"]
        for prediction, item in zip(
            zero_shot_predictions,
            test_reviews
        )
    )

    few_correct = sum(
        prediction == item["ground_truth"]
        for prediction, item in zip(
            few_shot_predictions,
            test_reviews
        )
    )

    zero_accuracy = zero_correct / total * 100
    few_accuracy = few_correct / total * 100

    # --------------------------------------------------------
    # Calculate format adherence
    # --------------------------------------------------------

    zero_format_rate = (
        sum(zero_shot_format) / total * 100
    )

    few_format_rate = (
        sum(few_shot_format) / total * 100
    )

    # --------------------------------------------------------
    # Calculate agreement between methods
    # --------------------------------------------------------

    agreement_count = sum(
        zero == few
        for zero, few in zip(
            zero_shot_predictions,
            few_shot_predictions
        )
    )

    agreement_rate = agreement_count / total * 100

    # --------------------------------------------------------
    # Print comparison table
    # --------------------------------------------------------

    print("\n\n" + "=" * 85)
    print("COMPARISON TABLE")
    print("=" * 85)

    print(
        f"{'Review':<10}"
        f"{'Ground Truth':<18}"
        f"{'Zero-Shot':<18}"
        f"{'Few-Shot':<18}"
        f"{'Zero Correct':<15}"
        f"{'Few Correct':<15}"
    )

    print("-" * 94)

    for index, item in enumerate(test_reviews):

        ground_truth = item["ground_truth"]
        zero_prediction = zero_shot_predictions[index]
        few_prediction = few_shot_predictions[index]

        zero_result = (
            "Yes"
            if zero_prediction == ground_truth
            else "No"
        )

        few_result = (
            "Yes"
            if few_prediction == ground_truth
            else "No"
        )

        print(
            f"{index + 1:<10}"
            f"{ground_truth:<18}"
            f"{zero_prediction:<18}"
            f"{few_prediction:<18}"
            f"{zero_result:<15}"
            f"{few_result:<15}"
        )

    # --------------------------------------------------------
    # Final evaluation summary
    # --------------------------------------------------------

    print("\n" + "=" * 85)
    print("EVALUATION SUMMARY")
    print("=" * 85)

    print(f"Total reviews: {total}")

    print(
        f"Zero-shot correct predictions: "
        f"{zero_correct}/{total}"
    )

    print(
        f"Few-shot correct predictions: "
        f"{few_correct}/{total}"
    )

    print(f"Zero-shot accuracy: {zero_accuracy:.1f}%")
    print(f"Few-shot accuracy: {few_accuracy:.1f}%")

    print(
        f"Zero-shot format adherence: "
        f"{zero_format_rate:.1f}%"
    )

    print(
        f"Few-shot format adherence: "
        f"{few_format_rate:.1f}%"
    )

    print(
        f"Agreement between methods: "
        f"{agreement_rate:.1f}%"
    )

    print("\nCONCLUSION")
    print("-" * 85)

    if zero_correct == total and few_correct == total:
        print("Both methods correctly classified all reviews.")

    elif few_accuracy > zero_accuracy:
        print("Few-shot prompting achieved higher accuracy.")

    elif zero_accuracy > few_accuracy:
        print("Zero-shot prompting achieved higher accuracy.")

    else:
        print("Both methods achieved the same accuracy.")

    print(
        "\nNote: Results are based on five test reviews. "
        "They do not establish that one method is always superior."
    )


# ============================================================
# 9. EXECUTE PROGRAM
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nProgram interrupted. "
            "The experiment may not have completed."
        )

    except Exception as error:
        print("\nUnexpected error:", repr(error))