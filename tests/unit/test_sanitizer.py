"""Security sanitizer tests."""

from app.security.sanitizer import (
    SAFE_LLM_OUTPUT_FALLBACK,
    scrub_llm_output,
    scrub_llm_output_pair,
)

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


def test_combined_leak_across_fields_triggers_fallback() -> None:
    """Combined leak detection: 3 lines in feedback + 3 different lines in hint → both fallback.

    Under the old per-field logic each field individually stayed under the threshold
    and would have passed. The combined check correctly catches this split leak.
    """
    feedback = (
        "Your accumulator starts at total = 0. "
        "Then for value in nums, you add it up. "
        "Finally you return total."
    )

    hint = (
        "Look at def solve(nums): — that's the entry point. "
        "Inside, total += value accumulates the sum."
    )

    scrubbed_feedback, scrubbed_hint = scrub_llm_output_pair(feedback, hint, SOURCE_CODE)

    assert scrubbed_feedback == SAFE_LLM_OUTPUT_FALLBACK
    assert scrubbed_hint == SAFE_LLM_OUTPUT_FALLBACK


def test_combined_pair_allows_safe_content() -> None:
    """scrub_llm_output_pair passes through clean content unchanged."""

    feedback = "Think about how you're iterating."
    hint = "Consider the loop bounds carefully."

    scrubbed_feedback, scrubbed_hint = scrub_llm_output_pair(feedback, hint, SOURCE_CODE)

    assert scrubbed_feedback == feedback
    assert scrubbed_hint == hint
