
"""
Week-15: Multimodal Receipt Information Extraction
---------------------------------------------------
Model: qwen/qwen3.8-27b
API: Groq
Input: Week-15/data/receipt.jpg
Output: Week-15/output/receipt_extracted.json

Features:
- Image-based receipt understanding
- Structured JSON extraction
- Required-field validation
- Total amount consistency check
- Error handling
"""

import os
import sys
import json
import base64
import mimetypes
from pathlib import Path
from decimal import Decimal, InvalidOperation

from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

WEEK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEEK_DIR.parent

IMAGE_PATH = WEEK_DIR / "data" / "receipt.jpg"
OUTPUT_DIR = WEEK_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "receipt_extracted.json"

MODEL = "qwen/qwen3.8-27b"

# Load the .env file from the project root.
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# 2. Utility functions
# --------------------------------------------------

def encode_image(image_path: Path) -> tuple[str, str]:
    """Encode an image into Base64 and determine its MIME type."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Receipt image not found:\n{image_path}\n\n"
            "Place your receipt image in Week-15/data/receipt.jpg."
        )

    mime_type, _ = mimetypes.guess_type(str(image_path))

    supported_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if mime_type not in supported_types:
        raise ValueError(
            f"Unsupported image format: {mime_type}\n"
            "Use a JPG, JPEG, PNG, or WebP image."
        )

    image_bytes = image_path.read_bytes()

    # Groq documents a 20 MB limit for requests containing an image URL.
    # Keep the local image comfortably below that limit.
    if len(image_bytes) > 15 * 1024 * 1024:
        raise ValueError(
            "Image is larger than 15 MB. "
            "Please resize or compress the image and try again."
        )

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    return encoded_image, mime_type


def extract_json(response_text: str) -> dict:
    """Parse JSON returned by the model."""

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The model did not return valid JSON.\n"
            f"Model response:\n{response_text}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from the model.")

    return result


def validate_receipt(data: dict) -> dict:
    """Validate extracted fields and check arithmetic where possible."""

    required_fields = [
        "merchant_name",
        "receipt_date",
        "currency",
        "items",
        "subtotal",
        "tax",
        "discount",
        "total",
        "payment_method",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        raise ValueError(
            "Required fields missing from model response: "
            + ", ".join(missing_fields)
        )

    if not isinstance(data["items"], list):
        raise ValueError("'items' must be a JSON list.")

    for index, item in enumerate(data["items"], start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Item {index} must be a JSON object."
            )

        for field in ("description", "quantity", "unit_price", "amount"):
            if field not in item:
                raise ValueError(
                    f"Item {index} is missing field '{field}'."
                )

    # Do not force arithmetic checks when an amount is unknown.
    arithmetic_check = {
        "performed": False,
        "passed": None,
        "message": "One or more amounts are unavailable.",
    }

    try:
        subtotal = Decimal(str(data["subtotal"]))
        tax = Decimal(str(data["tax"]))
        discount = Decimal(str(data["discount"]))
        total = Decimal(str(data["total"]))

        if all(value >= 0 for value in (subtotal, tax, discount, total)):
            expected_total = subtotal + tax - discount

            arithmetic_check = {
                "performed": True,
                "passed": abs(expected_total - total) <= Decimal("0.02"),
                "expected_total": str(expected_total),
                "reported_total": str(total),
                "message": (
                    "Amounts are consistent."
                    if abs(expected_total - total) <= Decimal("0.02")
                    else "Reported total differs from subtotal + tax - discount."
                ),
            }

    except (InvalidOperation, ValueError, TypeError):
        arithmetic_check["message"] = (
            "Arithmetic check skipped because an amount "
            "was missing or could not be parsed."
        )

    data["validation"] = {
        "required_fields_present": True,
        "items_structure_valid": True,
        "arithmetic_check": arithmetic_check,
        "note": (
            "AI-extracted values may be inaccurate. "
            "Compare them with the original receipt."
        ),
    }

    return data


# --------------------------------------------------
# 3. Receipt extraction
# --------------------------------------------------

def extract_receipt() -> dict:
    """Send the receipt image to Groq and extract structured data."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY was not found.\n"
            f"Check your .env file at: {PROJECT_ROOT / '.env'}"
        )

    print("\n" + "=" * 60)
    print("WEEK-15: MULTIMODAL RECEIPT EXTRACTION")
    print("=" * 60)
    print(f"Model       : {MODEL}")
    print(f"Receipt     : {IMAGE_PATH}")
    print("Status      : Encoding image...")

    encoded_image, mime_type = encode_image(IMAGE_PATH)

    client = Groq(api_key=api_key)

    prompt = """
You are an accurate receipt information extraction system.

Read the attached receipt image and extract the information into
a single valid JSON object.

Use exactly this structure:

{
  "merchant_name": "string or null",
  "receipt_date": "date as printed or null",
  "currency": "currency code or symbol, or null",
  "items": [
    {
      "description": "item name",
      "quantity": 1,
      "unit_price": 0.00,
      "amount": 0.00
    }
  ],
  "subtotal": 0.00,
  "tax": 0.00,
  "discount": 0.00,
  "total": 0.00,
  "payment_method": "string or null"
}

Rules:
1. Extract only information visible in the image.
2. Do not invent missing information. Use null when unknown.
3. Use an empty list if no line items can be identified.
4. Use numbers for quantities and monetary amounts when readable.
5. Do not confuse subtotal, tax, discount, and final total.
6. Preserve the currency shown on the receipt.
7. If the receipt shows no tax or discount, use 0.00 only when
   that interpretation is supported by the receipt.
8. If a value cannot be read, use null.
9. Return JSON only. Do not use Markdown fences or extra commentary.
"""

    print("Status      : Sending image to Groq...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract receipt data faithfully. "
                    "Never invent unreadable values."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,{encoded_image}"
                            )
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=2048,
        reasoning_effort="none",
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response.")

    print("Status      : Parsing JSON...")

    receipt_data = extract_json(content)
    receipt_data = validate_receipt(receipt_data)

    # Include basic run metadata.
    receipt_data["metadata"] = {
        "model": MODEL,
        "input_image": str(IMAGE_PATH.relative_to(WEEK_DIR)),
        "extraction_method": "Groq multimodal vision",
    }

    return receipt_data


# --------------------------------------------------
# 4. Save and display results
# --------------------------------------------------

def main() -> int:
    try:
        receipt_data = extract_receipt()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                receipt_data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print("\n" + "=" * 60)
        print("EXTRACTED RECEIPT DATA")
        print("=" * 60)

        print(
            json.dumps(
                receipt_data,
                indent=4,
                ensure_ascii=False,
            )
        )

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"JSON saved to: {OUTPUT_PATH}")

        check = receipt_data["validation"]["arithmetic_check"]

        if check["performed"]:
            print(f"Arithmetic check: {check['message']}")
        else:
            print(f"Arithmetic check: {check['message']}")

        print("\nReceipt extraction completed successfully.")
        return 0

    except FileNotFoundError as exc:
        print(f"\nFILE ERROR:\n{exc}", file=sys.stderr)
        return 1

    except EnvironmentError as exc:
        print(f"\nCONFIGURATION ERROR:\n{exc}", file=sys.stderr)
        return 1

    except RateLimitError:
        print(
            "\nGROQ RATE LIMIT: Your API request limit was reached. "
            "Wait and try again, or check your Groq Console limits.",
            file=sys.stderr,
        )
        return 1

    except APIConnectionError:
        print(
            "\nCONNECTION ERROR: Could not connect to Groq. "
            "Check your internet connection and retry.",
            file=sys.stderr,
        )
        return 1

    except APIStatusError as exc:
        print(
            f"\nGROQ API ERROR: HTTP {exc.status_code}\n{exc}",
            file=sys.stderr,
        )
        return 1

    except (ValueError, OSError) as exc:
        print(f"\nPROCESSING ERROR:\n{exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        print(
            f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
