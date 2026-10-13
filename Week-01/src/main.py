
import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors


def main():
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    # Initialize Gemini
    client = genai.Client(api_key=api_key)

    # Updated model
    model = "gemini-3.8-flash"

    # System instructions for prompt engineering
    instructions = """
    You are a helpful AI assistant and prompt engineering tutor.
    Give clear, accurate answers with examples.
    Explain technical concepts step by step.
    """

    print("=" * 50)
    print(" GEMINI PROMPT ENGINEERING ASSISTANT")
    print("=" * 50)
    print("Enter a prompt or type 'exit' to quit.")

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if prompt.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not prompt:
                print("Please enter a prompt.")
                continue

            # Use the Interactions API
            response = client.interactions.create(
                model=model,
                input=prompt,
                system_instruction=instructions,
            )

            print("\nGemini:")
            print(response.output_text or "No text response returned.")

        except errors.APIError as error:
            print(f"\nGemini API error: {error}")

        except KeyboardInterrupt:
            print("\nProgram stopped.")
            break

        except Exception as error:
            print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
