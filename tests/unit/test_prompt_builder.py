"""Prompt builder tests."""

import logging
from types import SimpleNamespace

from app.models.request_schemas import AnalyzeRequest
from app.parser.normalizer import normalize
from app.prompt_builder import builder
from app.prompt_builder.builder import build_prompt
from app.rule_engine.engine import run_rules
from tests.fixtures.sample_payloads import VALID_WRONG_ANSWER_PAYLOAD


def test_prompt_contains_source_code_verbatim() -> None:
    """The prompt includes the submitted source code unchanged."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    assert VALID_WRONG_ANSWER_PAYLOAD["source_code"] in prompt.user


def test_prompt_includes_problem_statement_before_source_code() -> None:
    """The prompt gives the model problem context before submitted code."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    problem_index = prompt.user.index("## Problem Statement")
    source_index = prompt.user.index("## Submitted Code")
    assert VALID_WRONG_ANSWER_PAYLOAD["problem_statement"] in prompt.user
    assert problem_index < source_index


def test_prompt_contains_json_schema_instruction() -> None:
    """The system prompt includes the required JSON response schema."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    assert '"feedback_text": "string"' in prompt.system
    assert '"hint_text": "string"' in prompt.system
    assert '"error_category": "one of:' in prompt.system


def test_system_prompt_includes_reasoning_scratchpad_instruction() -> None:
    """The system prompt asks the model to reason before final fields."""

    assert '"_reasoning_scratchpad"' in builder.SYSTEM_PROMPT
    assert 'Fill in "_reasoning_scratchpad" first' in builder.SYSTEM_PROMPT
    assert "1. Understand the problem" in builder.SYSTEM_PROMPT
    assert "2. Trace the code" in builder.SYSTEM_PROMPT
    assert "5. Commit to ONE primary root cause" in builder.SYSTEM_PROMPT


def test_system_prompt_includes_reasoning_examples() -> None:
    """Few-shot examples are present for simple and complex reasoning patterns."""

    assert "Example 1 - simple loop bound" in builder.SYSTEM_PROMPT
    assert "for i in range(len(nums) - 1)" in builder.SYSTEM_PROMPT
    assert "Example 2 - complex recursive state" in builder.SYSTEM_PROMPT
    assert "The `path` list is passed by reference and mutated without backtracking" in builder.SYSTEM_PROMPT


def test_system_prompt_includes_algorithmic_concepts() -> None:
    """The system prompt contains the algorithmic concept/invariant reference list."""
    
    assert "Binary Search -> search interval invariant" in builder.SYSTEM_PROMPT
    assert "Sliding Window -> left boundary must never move backwards" in builder.SYSTEM_PROMPT
    assert "Two Pointers -> pointer ordering invariant" in builder.SYSTEM_PROMPT


def test_system_prompt_includes_banned_phrases() -> None:
    """The system prompt contains the banned-phrase list."""
    
    assert "BANNED PHRASES: Do not use the following phrases" in builder.SYSTEM_PROMPT
    assert "Check your logic." in builder.SYSTEM_PROMPT
    assert "Your algorithm is incorrect." in builder.SYSTEM_PROMPT


def test_system_prompt_forbids_hint_code_changes() -> None:
    """The system prompt's instructions for hint_text explicitly forbid suggesting direct code changes."""
    
    assert "Never suggest a code change, point at a specific line to edit, or contain direct instructions like" in builder.SYSTEM_PROMPT


def test_system_prompt_concept_gaps_instructions() -> None:
    """The system prompt's instructions for concept_gaps contain the concept-level reference list."""
    
    assert "sliding window boundary management" in builder.SYSTEM_PROMPT
    assert "Do not use implementation-level nouns like" in builder.SYSTEM_PROMPT


def test_prompt_forbids_full_solution() -> None:
    """The system prompt explicitly prevents solution leakage."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)

    prompt = build_prompt(submission, outcome)

    assert "Never provide a full corrected solution" in prompt.system
    assert "working code that solves the problem" in prompt.system


def test_prompt_includes_test_summary_and_failed_cases() -> None:
    """The user prompt includes aggregate and representative failure context."""

    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "wrong_answer",
        "test_summary": {
            "total_test_cases": 30,
            "passed_test_cases": 24,
            "failed_test_cases": 6,
        },
        "sample_failed_cases": [
            {
                "stdin": "nums=[1]\ntarget=1",
                "expected_output": "0",
                "actual_output": "-1",
            },
        ],
    }
    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)
    summary = payload["test_summary"]

    prompt = build_prompt(submission, outcome)

    assert f"total_test_cases={summary['total_test_cases']}" in prompt.user
    assert f"passed_test_cases={summary['passed_test_cases']}" in prompt.user
    assert f"failed_test_cases={summary['failed_test_cases']}" in prompt.user
    assert "expected_output:" in prompt.user
    assert "actual_output:" in prompt.user


def test_truncation_preserves_source_over_sample_stdin(monkeypatch) -> None:
    """Large sample stdin is removed before source code is truncated."""

    source_code = "def solve():\n    print(42)"
    large_stdin = "x" * 5000
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "verdict": "wrong_answer",
        "source_code": source_code,
        "sample_failed_cases": [
            {
                "stdin": large_stdin,
                "expected_output": "42\n",
                "actual_output": "41\n",
            },
        ],
    }
    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)
    max_chars = len(builder.SYSTEM_PROMPT) + 1800
    monkeypatch.setattr(
        builder,
        "get_settings",
        lambda: SimpleNamespace(prompt_max_chars=max_chars),
    )

    prompt = build_prompt(submission, outcome)

    assert source_code in prompt.user
    assert large_stdin not in prompt.user
    assert len(prompt.system) + len(prompt.user) <= max_chars


def test_truncates_source_code_as_last_resort(monkeypatch) -> None:
    """Oversized source code is truncated when non-essential output is already absent."""

    source_code = "print('x')\n" * 1000
    max_chars = len(builder.SYSTEM_PROMPT) + 700
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "problem_statement": "Return the requested value.",
        "source_code": source_code,
        "stdout": "",
        "stderr": "",
        "compile_output": "",
        "sample_failed_cases": [],
    }
    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)
    monkeypatch.setattr(
        builder,
        "get_settings",
        lambda: SimpleNamespace(prompt_max_chars=max_chars),
    )

    prompt = build_prompt(submission, outcome)

    assert len(prompt.system) + len(prompt.user) <= max_chars
    assert builder.SOURCE_TRUNCATION_MARKER in prompt.user


def test_truncates_problem_statement_before_source_code(monkeypatch) -> None:
    """Large problem text is shortened before source code is touched."""

    source_code = "def solve():\n    return 42"
    problem_statement = "Explain the task. " * 200
    max_chars = len(builder.SYSTEM_PROMPT) + 1400
    payload = VALID_WRONG_ANSWER_PAYLOAD | {
        "problem_statement": problem_statement,
        "source_code": source_code,
        "stdout": "",
        "stderr": "",
        "compile_output": "",
        "sample_failed_cases": [],
    }
    request = AnalyzeRequest.model_validate(payload)
    submission = normalize(request)
    outcome = run_rules(submission)
    monkeypatch.setattr(
        builder,
        "get_settings",
        lambda: SimpleNamespace(prompt_max_chars=max_chars),
    )

    prompt = build_prompt(submission, outcome)

    assert len(prompt.system) + len(prompt.user) <= max_chars
    assert builder.PROBLEM_STATEMENT_TRUNCATION_MARKER in prompt.user
    assert source_code in prompt.user
    assert builder.SOURCE_TRUNCATION_MARKER not in prompt.user


def test_normal_sized_prompt_is_unmodified_without_warning(monkeypatch, caplog) -> None:
    """Prompt builder does not truncate or warn when the prompt starts under budget."""

    request = AnalyzeRequest.model_validate(VALID_WRONG_ANSWER_PAYLOAD)
    submission = normalize(request)
    outcome = run_rules(submission)
    original_prompt = builder._compose_user_prompt(
        submission,
        outcome,
        include_case_stdin=True,
        include_stderr=True,
    )
    monkeypatch.setattr(
        builder,
        "get_settings",
        lambda: SimpleNamespace(prompt_max_chars=15_000),
    )

    with caplog.at_level(logging.WARNING):
        prompt = build_prompt(submission, outcome)

    assert prompt.user == original_prompt
    assert builder.SOURCE_TRUNCATION_MARKER not in prompt.user
    assert not caplog.records


def test_system_prompt_includes_self_consistency_check() -> None:
    """The system prompt includes the 6th step for self-consistency check."""
    
    assert "6. Self-consistency check:" in builder.SYSTEM_PROMPT
    assert "Explicitly re-read your committed diagnosis against the actual submitted code." in builder.SYSTEM_PROMPT


def test_system_prompt_includes_uncertainty_handling() -> None:
    """The system prompt contains the uncertainty handling instructions."""
    
    assert "Uncertainty Handling:" in builder.SYSTEM_PROMPT
    assert "state the single most likely explanation in feedback_text with honest confidence" in builder.SYSTEM_PROMPT
    assert "A vague-but-honest answer is always better than a confident-but-wrong answer." in builder.SYSTEM_PROMPT


def test_system_prompt_includes_test_case_grounding() -> None:
    """The system prompt instructs the model to walk through a specific failed test case when helpful."""
    
    assert "Whenever a specific failed test case helps explain why the diagnosed root cause produces the observed wrong behavior, walk through that specific test case." in builder.SYSTEM_PROMPT


def test_system_prompt_includes_ambiguous_evidence_example() -> None:
    """The third few-shot example for ambiguous evidence is present."""
    
    assert "Example 3 - ambiguous evidence:" in builder.SYSTEM_PROMPT
    assert "The most likely cause of the time limit exceeded error is that your search interval fails to shrink" in builder.SYSTEM_PROMPT
