import os
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=500
    )
    return response.choices[0].message.content


# Iteration 1: Baseline prompt
prompt_1 = "Explain climate change."

# Iteration 2: Add context and structure
prompt_2 = (
    "You are an environmental science teacher. "
    "Explain climate change in three bullet points. "
    "Include its meaning, one major cause, and one impact."
)

# Iteration 3: Add word limits and improve clarity
prompt_3 = (
    "You are an environmental science teacher. "
    "Explain climate change using exactly three bullet points. "
    "The first bullet must define climate change, "
    "the second must explain one major cause, "
    "and the third must describe one impact. "
    "Use simple language and keep each bullet under 20 words."
)

prompts = [
    ("Iteration 1: Baseline", prompt_1),
    ("Iteration 2: Context and Structure", prompt_2),
    ("Iteration 3: Constraints and Clarity", prompt_3)
]

results = []

for title, prompt in prompts:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("PROMPT:", prompt)

    try:
        output = generate_response(prompt)
        word_count = len(output.split())

        print("\nRESPONSE:")
        print(output)
        print("\nWord count:", word_count)

        results.append({
            "iteration": title,
            "word_count": word_count,
            "response": output
        })

    except Exception as error:
        print("Error:", error)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for result in results:
    print(f"{result['iteration']}: {result['word_count']} words")