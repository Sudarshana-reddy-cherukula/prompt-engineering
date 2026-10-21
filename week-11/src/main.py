import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Environment Setup
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

MODEL = "qwen/qwen3.8-27b"


# --------------------------------------------------
# 2. Define Prompt Template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    "Write a very short story (2-3 sentences) about {topic}."
)


# --------------------------------------------------
# 3. Initialize Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model=MODEL,
    temperature=0.2,
    api_key=API_KEY,
    reasoning_effort="none",
)


# --------------------------------------------------
# 4. Initialize Output Parser
# --------------------------------------------------

parser = StrOutputParser()


# --------------------------------------------------
# 5. Build LCEL Chain
# --------------------------------------------------

chain = prompt | llm | parser


# --------------------------------------------------
# 6. Define Topics
# --------------------------------------------------

topics = [
    "a dragon and a knight",
    "a lost spaceship",
    "a robot chef",
]


# --------------------------------------------------
# 7. Execute Experiment
# --------------------------------------------------

def main():
    print("=" * 60)
    print("WEEK 11: LANGCHAIN LCEL WITH GROQ")
    print("=" * 60)

    print(f"\nModel: {MODEL}")
    print("Chain: prompt | llm | parser")

    for topic in topics:
        print("\n" + "-" * 60)
        print(f"Topic: {topic}")

        try:
            story = chain.invoke({"topic": topic})

            print("\nGenerated Story:")
            print(story)

        except Exception as error:
            print(f"\nError generating story: {error}")

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()