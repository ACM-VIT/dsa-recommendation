"""Security sanitizer tests."""

from app.security.sanitizer import SAFE_LLM_OUTPUT_FALLBACK, scrub_llm_output

SOURCE_CODE = """def solve(nums):
    total = 0
    for value in nums:
        total += value
    return total
"""


def test_scrubs_feedback_with_many_source_lines() -> None:
    """Output containing 4+ source lines is withheld as a leak risk."""

    feedback = """
    You can write:
    def solve(nums):
    total = 0
    for value in nums:
    total += value
    return total
    """

    assert scrub_llm_output(feedback, SOURCE_CODE) == SAFE_LLM_OUTPUT_FALLBACK


def test_allows_incidental_source_line_matches() -> None:
    """One or two matching lines are allowed to avoid false positives."""

    feedback = "Your accumulator starts at total = 0 and then should return total."

    assert scrub_llm_output(feedback, SOURCE_CODE) == feedback
