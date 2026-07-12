"""Build mentor-style prompts for LLM analysis."""

from app.config.settings import get_settings
from app.logging.logger import get_logger
from app.models.domain import LLMPrompt, NormalizedSubmission, RuleEngineOutcome

logger = get_logger(__name__)

PROBLEM_STATEMENT_TRUNCATION_MARKER = (
    "\n... [problem statement truncated for length] ..."
)
SOURCE_TRUNCATION_MARKER = "\n... [source truncated for length] ..."

SYSTEM_PROMPT = """You are a patient coding mentor for a learner.
Never provide a full corrected solution or working code that solves the problem.
Explain the mistake conceptually and give a nudge or hint, not the answer.
Always respond in strict JSON only, matching this exact schema:
{
  "_reasoning_scratchpad": "private step-by-step reasoning about the bug; never shown",
  "feedback_text": "string",
  "hint_text": "string",
  "error_category": "one of: wrong_answer_logic | off_by_one | edge_case_missing | \
wrong_algorithm | time_limit_exceeded | memory_limit_exceeded | runtime_error | \
compilation_error | unknown",
  "reasoning_quality": "one of: strong | partial | weak | unknown",
  "concept_gaps": ["array", "of", "short", "strings"]
}

Fill in "_reasoning_scratchpad" first, structuring your reasoning as follows:
1. Understand the problem: Restate what the problem statement is actually asking for.
2. Trace the code: Walk through what the submitted code actually does, referencing specific function names, variables, loop conditions, or pointer updates.
3. Compare against the evidence: Cross-reference the trace against expected vs actual output, stderr, and deterministic rules.
4. Identify candidate causes: Note plausible alternatives if any.
5. Commit to ONE primary root cause: Explicitly state the single diagnosis that best explains the evidence. If multiple explanations are plausible, state the most likely cause, do not hedge.
6. Self-consistency check: Explicitly re-read your committed diagnosis against the actual submitted code. Does the code actually contain the specific construct you referenced? Does your explanation describe behavior the code could actually produce? Is this diagnosis directly supported by concrete evidence? If the check fails, revise your diagnosis here. Never describe behavior in the code that the code does not actually exhibit.

Uncertainty Handling:
- If multiple explanations are equally plausible and evidence doesn't clearly favor one, state the single most likely explanation in feedback_text with honest confidence (e.g., "the most likely cause is..."). Do not invent false certainty.
- If evidence is insufficient to responsibly make a specific diagnosis, say so honestly rather than fabricating an explanation.
- A vague-but-honest answer is always better than a confident-but-wrong answer.

For "feedback_text" (3-6 sentences):
1. What is the primary issue: Name it specifically using evidence from the code.
2. Why it causes incorrect behavior: Explain the causal chain from the construct to the wrong output.
3. What consequence it produces on the observed test case: Tie back to the actual expected vs actual output of the submission. Whenever a specific failed test case helps explain why the diagnosed root cause produces the observed wrong behavior, walk through that specific test case.
4. What algorithmic concept or invariant is violated: Name the concept explicitly (e.g., Binary Search -> search interval invariant; Sliding Window -> left boundary must never move backwards; DFS/BFS -> traversal/visited-state invariants; Dynamic Programming -> recurrence relation and optimal substructure; Two Pointers -> pointer ordering invariant; Prefix Sum -> accumulated prefix consistency; Merge-based problems -> maintaining correct position in the conceptual merged sequence). Explain the root cause, not just describe the symptom.

BANNED PHRASES: Do not use the following phrases UNLESS immediately followed by a concrete, submission-specific explanation in the same sentence:
- "Check your logic."
- "Review your algorithm."
- "Handle edge cases."
- "Maintain state."
- "Consider duplicates."
- "Review your loop conditions."
- "Your algorithm is incorrect."

For "hint_text": Guide the student's reasoning process and underlying concept. Never suggest a code change, point at a specific line to edit, or contain direct instructions like "change X to Y" (even for small fixes).

For "concept_gaps": Use concept-level terms (e.g., "sliding window boundary management", "binary search", "two pointers", "graph traversal", "recursion", "dynamic programming", "hash maps", "prefix sums", "boundary conditions", "off-by-one reasoning", "greedy strategy", "backtracking", "invariant maintenance"). Do not use implementation-level nouns like "array", "string", "loop", or "variable".

Examples are illustrative only; do not copy their facts into the current task.

Example 1 - simple loop bound:
Buggy code:
for i in range(len(nums) - 1):
    total += nums[i]
Good JSON:
{
  "_reasoning_scratchpad": "Understand: Sum all elements in the array. Trace: The loop uses `range(len(nums) - 1)`, which goes up to `len(nums) - 2`. Evidence: The actual output is smaller than expected by exactly the last element's value. Candidates: Loop stops early, or subtraction inside loop. Root cause: The upper bound of the loop is off by one, causing the last index to be excluded.",
  "feedback_text": "The loop terminates one iteration early because the upper bound in `range(len(nums) - 1)` excludes the last index. Since `range` is already exclusive at the upper bound, subtracting 1 prevents the loop from processing the final element. For the given test case, this causes the final sum to be smaller than the expected output because the last number is never added. This violates the boundary condition for fully traversing the array.",
  "hint_text": "Think about what the maximum value of `i` will be when `range(len(nums) - 1)` is used, and compare that to the last index of the array.",
  "error_category": "off_by_one",
  "reasoning_quality": "strong",
  "concept_gaps": ["off-by-one boundary handling", "array traversal invariants"]
}

Example 2 - complex recursive state:
Buggy code:
def dfs(node):
    if node is None:
        path.append(0)
        return
    add_children(node, path)
    dfs(node.left)
    dfs(node.right)
Good JSON:
{
  "_reasoning_scratchpad": "Understand: Traverse tree and track paths. Trace: The `path` list is appended to in the base case, but never popped or reverted when the function returns. Evidence: The actual output shows paths from the left branch bleeding into the right branch. Candidates: Global state mutation, or incorrect base case. Root cause: The `path` list is passed by reference and mutated without backtracking, violating traversal state invariants.",
  "feedback_text": "The `path` list accumulates values across different branches because it is mutated in the base case without ever reverting the state. When `dfs(node.left)` returns, the appended values remain in the list when `dfs(node.right)` is called, polluting its path. In the failed test case, this causes the right branch to output a sequence that incorrectly includes nodes from the left branch. This violates the invariant of graph traversal where each recursive branch should maintain its own independent state or explicitly backtrack.",
  "hint_text": "Consider what happens to the shared `path` list after returning from the left recursive call, and how that affects the starting state for the right recursive call.",
  "error_category": "wrong_answer_logic",
  "reasoning_quality": "strong",
  "concept_gaps": ["backtracking", "invariant maintenance", "mutable state"]
}

Example 3 - ambiguous evidence:
Buggy code:
def search(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid
        else:
            r = mid
    return -1
Good JSON:
{
  "_reasoning_scratchpad": "Understand: Binary search for target in sorted array. Trace: `l = mid` instead of `l = mid + 1`. Evidence: Time Limit Exceeded on basic test case. Candidates: Infinite loop because `l` doesn't advance when `mid == l`, or issue with `r`. Commit: `l = mid` causes infinite loop. Self-check: Code uses `l = mid`. If `l=0, r=1`, `mid=0`. If `nums[0] < target`, `l` remains `0`, infinite loop. Matches diagnosis.",
  "feedback_text": "The most likely cause of the time limit exceeded error is that your search interval fails to shrink when only two elements remain. Specifically, the assignment `l = mid` will cause an infinite loop because integer division `(l + r) // 2` rounds down. For example, if `l = 0` and `r = 1`, `mid` becomes `0`. If `nums[0] < target`, `l` remains `0` on the next iteration, and the loop never terminates. This violates the binary search invariant that the search interval must strictly decrease in size each step.",
  "hint_text": "Consider what happens to the value of `mid` when `l` and `r` are adjacent (e.g., `l=0` and `r=1`), and how the assignment `l = mid` affects the search interval.",
  "error_category": "time_limit_exceeded",
  "reasoning_quality": "partial",
  "concept_gaps": ["binary search", "invariant maintenance", "loop bounds"]
}

No markdown fences, no prose before or after the JSON."""


def _format_failed_cases(sub: NormalizedSubmission, include_stdin: bool) -> str:
    """Format representative failed cases for the user prompt."""

    if not sub.sample_failed_cases:
        return "No representative failed cases were provided."

    sections: list[str] = []
    for index, failed_case in enumerate(sub.sample_failed_cases[:3], start=1):
        lines = [f"Case {index}:"]
        if include_stdin and failed_case.stdin:
            lines.append(f"stdin:\n{failed_case.stdin}")
        lines.append(f"expected_output:\n{failed_case.expected_output}")
        lines.append(f"actual_output:\n{failed_case.actual_output}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _compose_user_prompt(
    sub: NormalizedSubmission,
    rule_outcome: RuleEngineOutcome,
    include_case_stdin: bool,
    include_stderr: bool,
    problem_statement: str | None = None,
    source_code: str | None = None,
) -> str:
    """Compose the user prompt with optional non-essential sections."""

    parts = [
        f"Language: {sub.language}",
        f"Verdict: {sub.verdict}",
        "## Problem Statement",
        problem_statement if problem_statement is not None else sub.problem_statement,
        "## Submitted Code",
        source_code if source_code is not None else sub.source_code,
        "Test summary:",
        (
            f"total_test_cases={sub.test_summary.total_test_cases}, "
            f"passed_test_cases={sub.test_summary.passed_test_cases}, "
            f"failed_test_cases={sub.test_summary.failed_test_cases}"
        ),
        "Representative failed cases:",
        _format_failed_cases(sub, include_stdin=include_case_stdin),
    ]

    if sub.output_diff_summary:
        parts.extend(["Output diff summary:", sub.output_diff_summary])

    diagnostic_output = sub.normalized_stderr or sub.stderr or sub.compile_output
    if include_stderr and diagnostic_output:
        parts.extend(["stderr / compile_output:", diagnostic_output])

    if rule_outcome.rule_notes:
        parts.extend(
            [
                "Deterministic analysis notes:",
                "\n".join(f"- {note}" for note in rule_outcome.rule_notes),
            ],
        )

    parts.append("Respond with the JSON object only.")
    return "\n\n".join(parts)


def _fits_budget(prompt: str, max_chars: int) -> bool:
    """Return whether the complete prompt fits within the configured budget."""

    return len(SYSTEM_PROMPT) + len(prompt) <= max_chars


def _compose_with_truncated_source(
    sub: NormalizedSubmission,
    rule_outcome: RuleEngineOutcome,
    max_chars: int,
    problem_statement: str | None = None,
) -> str:
    """Compose a prompt that fits by truncating source code as the final fallback."""

    low = 0
    high = len(sub.source_code)
    best_prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=False,
        include_stderr=False,
        problem_statement=problem_statement,
        source_code=SOURCE_TRUNCATION_MARKER,
    )

    while low <= high:
        mid = (low + high) // 2
        truncated_source = sub.source_code[:mid] + SOURCE_TRUNCATION_MARKER
        prompt = _compose_user_prompt(
            sub,
            rule_outcome,
            include_case_stdin=False,
            include_stderr=False,
            problem_statement=problem_statement,
            source_code=truncated_source,
        )
        if _fits_budget(prompt, max_chars):
            best_prompt = prompt
            low = mid + 1
        else:
            high = mid - 1

    if not _fits_budget(best_prompt, max_chars):
        user_budget = max(0, max_chars - len(SYSTEM_PROMPT))
        return best_prompt[:user_budget]

    return best_prompt


def _compose_with_truncated_problem_statement(
    sub: NormalizedSubmission,
    rule_outcome: RuleEngineOutcome,
    max_chars: int,
) -> str:
    """Compose a prompt that fits by shortening problem text before source code."""

    low = 0
    high = len(sub.problem_statement)
    best_prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=False,
        include_stderr=False,
        problem_statement=PROBLEM_STATEMENT_TRUNCATION_MARKER,
    )

    while low <= high:
        mid = (low + high) // 2
        truncated_problem_statement = (
            sub.problem_statement[:mid] + PROBLEM_STATEMENT_TRUNCATION_MARKER
        )
        prompt = _compose_user_prompt(
            sub,
            rule_outcome,
            include_case_stdin=False,
            include_stderr=False,
            problem_statement=truncated_problem_statement,
        )
        if _fits_budget(prompt, max_chars):
            best_prompt = prompt
            low = mid + 1
        else:
            high = mid - 1

    if _fits_budget(best_prompt, max_chars):
        return best_prompt

    return _compose_with_truncated_source(
        sub,
        rule_outcome,
        max_chars,
        problem_statement=PROBLEM_STATEMENT_TRUNCATION_MARKER,
    )


def build_prompt(
    sub: NormalizedSubmission,
    rule_outcome: RuleEngineOutcome,
) -> LLMPrompt:
    """Build a strict JSON prompt from a normalized submission and rule outcome."""

    settings = get_settings()
    prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=True,
        include_stderr=True,
    )

    if _fits_budget(prompt, settings.prompt_max_chars):
        return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

    prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=False,
        include_stderr=True,
    )
    if _fits_budget(prompt, settings.prompt_max_chars):
        logger.warning("prompt truncated sample failed case stdin")
        return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

    prompt = _compose_user_prompt(
        sub,
        rule_outcome,
        include_case_stdin=False,
        include_stderr=False,
    )
    if _fits_budget(prompt, settings.prompt_max_chars):
        logger.warning("prompt truncated stderr and compile output")
        return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

    prompt = _compose_with_truncated_problem_statement(
        sub,
        rule_outcome,
        settings.prompt_max_chars,
    )
    if _fits_budget(prompt, settings.prompt_max_chars) and (
        SOURCE_TRUNCATION_MARKER not in prompt
    ):
        logger.warning("prompt truncated problem_statement")
        return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)

    prompt = _compose_with_truncated_source(
        sub,
        rule_outcome,
        settings.prompt_max_chars,
        problem_statement=PROBLEM_STATEMENT_TRUNCATION_MARKER,
    )
    logger.warning("prompt truncated problem_statement")
    logger.warning("prompt truncated source_code as last resort")
    return LLMPrompt(system=SYSTEM_PROMPT, user=prompt)
