"""
DSA Engine -- Qdrant Readiness Validator
=========================================
Validates every vector pool row before it can enter Qdrant.

HARD GATES (row is REJECTED if any fail):
  G1  problem_id          non-empty string
  G2  title               non-empty string
  G3  description         >= 30 chars
  G4  canonical_solution  non-None AND >= 20 chars
  G5  difficulty_score    float in [0.0, 1.0]
  G6  topic_tags          at least 1 tag
  G7  algorithm_tags OR data_structure_tags -- at least 1 combined
  G8  companies           must be a list (empty is allowed)

SOFT WARNINGS (logged, row still passes):
  W1  solution_signature  empty
  W2  patterns            empty list
  W3  techniques          empty list
  W4  skill_tags          fewer than 2 tags
  W5  similar_problem_ids empty list
  W6  description         < 100 chars (short but above minimum)
  W7  companies           empty list
  W8  frequency           None (missing from dataset)
  W9  rating              None (missing from dataset)
  W10 explanation_text    empty
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import pandas as pd

MIN_DESC_CHARS   = 30
MIN_SOL_CHARS    = 20
IDEAL_DESC_CHARS = 100


@dataclass
class ValidationResult:
    problem_id:    str
    passed:        bool
    hard_failures: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "[OK] PASS" if self.passed else "[X] REJECT"
        lines  = [f"[{status}] {self.problem_id}"]
        for f in self.hard_failures:
            lines.append(f"  [X] {f}")
        for w in self.soft_warnings:
            lines.append(f"  [!] {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Row-level validator
# ---------------------------------------------------------------------------

def validate_row(row: Dict[str, Any]) -> ValidationResult:
    pid      = str(row.get("problem_id", "")).strip()
    failures: List[str] = []
    warnings: List[str] = []

    # G1 -- problem_id
    if not pid:
        failures.append("G1: problem_id is empty or missing")

    # G2 -- title
    if not str(row.get("title", "")).strip():
        failures.append("G2: title is empty or missing")

    # G3 -- description
    desc = str(row.get("description", "") or "").strip()
    if len(desc) < MIN_DESC_CHARS:
        failures.append(f"G3: description too short ({len(desc)} chars, min={MIN_DESC_CHARS})")
    elif len(desc) < IDEAL_DESC_CHARS:
        warnings.append(f"W6: description short ({len(desc)} chars, ideal>={IDEAL_DESC_CHARS})")

    # G4 -- canonical_solution
    sol = row.get("canonical_solution")
    if sol is None:
        failures.append("G4: canonical_solution is None -- no Python solution exists (SQL/Shell problem?)")
    elif len(str(sol).strip()) < MIN_SOL_CHARS:
        failures.append(f"G4: canonical_solution too short ({len(str(sol).strip())} chars, min={MIN_SOL_CHARS})")

    # G5 -- difficulty_score
    score = row.get("difficulty_score")
    try:
        sf = float(score)
        if not (0.0 <= sf <= 1.0):
            failures.append(f"G5: difficulty_score={sf} out of range [0.0, 1.0]")
    except (TypeError, ValueError):
        failures.append(f"G5: difficulty_score is not numeric ({score!r})")

    # G6 -- topic_tags
    topic_tags = row.get("topic_tags") or []
    if not isinstance(topic_tags, list) or len(topic_tags) == 0:
        failures.append("G6: topic_tags is empty -- cannot categorise for rec engine")

    # G7 -- algorithm_tags OR data_structure_tags
    if len(row.get("algorithm_tags") or []) == 0 and len(row.get("data_structure_tags") or []) == 0:
        failures.append("G7: both algorithm_tags and data_structure_tags are empty")

    # G8 -- companies
    companies = row.get("companies")
    if companies is None:
        failures.append("G8: companies is None (must be a list)")
    elif not isinstance(companies, list):
        failures.append(f"G8: companies is not a list ({type(companies).__name__})")
    elif len(companies) == 0:
        warnings.append("W7: companies list is empty")

    # W1 -- solution_signature
    if not str(row.get("solution_signature", "") or "").strip():
        warnings.append("W1: solution_signature is empty")

    # W2 -- patterns
    if len(row.get("patterns") or []) == 0:
        warnings.append("W2: patterns list is empty")

    # W3 -- techniques
    if len(row.get("techniques") or []) == 0:
        warnings.append("W3: techniques list is empty")

    # W4 -- skill_tags
    skill_tags = row.get("skill_tags") or []
    if len(skill_tags) < 2:
        warnings.append(f"W4: skill_tags has only {len(skill_tags)} tag(s)")

    # W5 -- similar_problem_ids
    if len(row.get("similar_problem_ids") or []) == 0:
        warnings.append("W5: similar_problem_ids is empty")

    # W8 -- frequency
    if row.get("frequency") is None:
        warnings.append("W8: frequency is missing (50% coverage in dataset)")

    # W9 -- rating
    if row.get("rating") is None:
        warnings.append("W9: rating is missing (50% coverage in dataset)")

    # W10 -- explanation_text
    if not str(row.get("explanation_text", "") or "").strip():
        warnings.append("W10: explanation_text is empty")

    return ValidationResult(
        problem_id    = pid or "UNKNOWN",
        passed        = len(failures) == 0,
        hard_failures = failures,
        soft_warnings = warnings,
    )


# ---------------------------------------------------------------------------
# DataFrame-level validation
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[ValidationResult]]:
    """
    Validate all rows. Returns (clean_df, rejected_df, results).
    clean_df    -- passed all hard gates -> safe to upload to Qdrant
    rejected_df -- failed one or more hard gates -> needs fixing
    results     -- full ValidationResult per row for reporting
    """
    results:   List[ValidationResult] = []
    pass_mask: List[bool]             = []

    for _, row in df.iterrows():
        result = validate_row(row.to_dict())
        results.append(result)
        pass_mask.append(result.passed)

    clean_df    = df[pass_mask].copy().reset_index(drop=True)
    rejected_df = df[[not p for p in pass_mask]].copy().reset_index(drop=True)

    if verbose:
        _print_report(results, clean_df, rejected_df)

    return clean_df, rejected_df, results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(
    results: List[ValidationResult],
    clean_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> None:
    from collections import Counter

    total    = len(results)
    passed   = len(clean_df)
    rejected = len(rejected_df)
    warned   = sum(1 for r in results if r.passed and r.soft_warnings)

    print("\n" + "="*62)
    print("  DSA ENGINE -- QDRANT READINESS REPORT")
    print("="*62)
    print(f"  Total problems   : {total}")
    print(f"  [OK] Qdrant-ready   : {passed}  ({100*passed/max(total,1):.1f}%)")
    print(f"  [X] Rejected       : {rejected}  ({100*rejected/max(total,1):.1f}%)")
    print(f"  [!] Passed w/ warn : {warned}")
    print("="*62)

    if rejected > 0:
        print("\n-- REJECTED PROBLEMS --------------------------------------")
        for r in results:
            if not r.passed:
                print(r.summary())

    print("\n-- REJECTION BREAKDOWN ------------------------------------")
    gate_counts: Counter = Counter()
    for r in results:
        for f in r.hard_failures:
            gate_counts[f.split(":")[0].strip()] += 1
    if gate_counts:
        for gate, count in sorted(gate_counts.items()):
            print(f"  {gate}: {count} problem(s)")
    else:
        print("  None -- all problems passed hard gates.")

    print("\n-- WARNING BREAKDOWN --------------------------------------")
    warn_counts: Counter = Counter()
    for r in results:
        for w in r.soft_warnings:
            warn_counts[w.split(":")[0].strip()] += 1
    if warn_counts:
        for code, count in sorted(warn_counts.items()):
            print(f"  {code}: {count} problem(s)")
    else:
        print("  None.")
    print()


def rejection_manifest(
    rejected_df: pd.DataFrame,
    results: List[ValidationResult],
    output_path: str,
) -> None:
    """Write rejected.json -- fix list with per-problem failure reasons."""
    import json
    rejected_ids = set(rejected_df["problem_id"].tolist())
    manifest = [
        {
            "problem_id":    r.problem_id,
            "hard_failures": r.hard_failures,
            "soft_warnings": r.soft_warnings,
        }
        for r in results if r.problem_id in rejected_ids
    ]
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[[OK]] Rejection manifest -> {output_path}  ({len(manifest)} rejected)")
