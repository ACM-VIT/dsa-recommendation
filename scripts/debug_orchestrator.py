"""Debug the complete AI orchestration pipeline."""

import asyncio
from pprint import pprint

from app.models.request_schemas import AnalyzeRequest
from app.orchestrator.orchestrator import analyze_submission
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


async def main() -> None:
    print("=" * 80)
    print("BACKEND PAYLOAD")
    print("=" * 80)

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)

    pprint(request.model_dump())

    print()

    print("=" * 80)
    print("ORCHESTRATOR")
    print("=" * 80)

    response = await analyze_submission(request)

    print()

    print("=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)

    pprint(response.model_dump())

    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Submission ID     : {response.submission_id}")
    print(f"Processing Status : {response.processing_status}")
    print(f"Model Used        : {response.model_used}")
    print(f"Processing Time   : {response.processing_ms} ms")
    print(f"Error Category    : {response.error_category}")
    print(f"Reasoning Quality : {response.reasoning_quality}")

    print()

    print("Feedback")
    print("-" * 80)
    print(response.feedback_text)

    print()

    print("Hint")
    print("-" * 80)
    print(response.hint_text)

    print()

    print("Concept Gaps")
    print("-" * 80)
    pprint(response.concept_gaps)


if __name__ == "__main__":
    asyncio.run(main())