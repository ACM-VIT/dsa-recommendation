"""
tests/test_bkt.py

Tests pipeline/recommender/bkt.py's update_bkt: the smoothing cap
(MAX_MASTERY_DELTA -- "each submission causes only small incremental
changes") and difficulty-aware dampening of trivial solves.

Run:
    python -m pytest tests/test_bkt.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.bkt import (
    update_bkt, MAX_MASTERY_DELTA, LEARNING_TRANSITION_THRESHOLD,
)


class TestSmoothingCap(unittest.TestCase):

    def test_strong_observation_from_cold_start_is_capped(self):
        # Uncapped Bayesian math would jump this well past MAX_MASTERY_DELTA
        # off a single strong observation from a cold-start P_L.
        new_p_l = update_bkt(current_p_l=0.15, observed=0.95)
        self.assertLessEqual(new_p_l - 0.15, MAX_MASTERY_DELTA + 1e-9)

    def test_delta_never_exceeds_cap_across_a_range_of_inputs(self):
        for current_p_l in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9):
            for observed in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
                new_p_l = update_bkt(current_p_l, observed)
                delta = abs(new_p_l - current_p_l)
                self.assertLessEqual(
                    delta, MAX_MASTERY_DELTA + 1e-9,
                    msg=f"delta {delta} exceeded cap for current_p_l={current_p_l}, observed={observed}",
                )

    def test_still_moves_upward_on_success(self):
        new_p_l = update_bkt(current_p_l=0.3, observed=0.9)
        self.assertGreater(new_p_l, 0.3)

    def test_still_bounded_0_to_1(self):
        self.assertLessEqual(update_bkt(0.95, 1.0), 1.0)
        self.assertGreaterEqual(update_bkt(0.05, 0.0), 0.0)

    def test_cold_start_delta_varies_with_difficulty_despite_saturating_cap(self):
        """
        Regression test: a cold-start user's raw Bayesian delta is large
        enough to saturate ANY flat cap regardless of difficulty, which
        previously made every confident cold-start solve read as an
        identical +MAX_MASTERY_DELTA no matter how hard the problem was.
        The cap now scales with difficulty, so this differentiation
        survives even when the raw delta would otherwise swamp it.
        """
        easy = update_bkt(current_p_l=0.15, observed=0.95, difficulty=0.05)
        hard = update_bkt(current_p_l=0.15, observed=0.95, difficulty=0.95)
        self.assertLess(easy - 0.15, MAX_MASTERY_DELTA)
        self.assertGreater(hard - 0.15, MAX_MASTERY_DELTA)
        self.assertLess(easy, hard)

    def test_repeated_strong_submissions_converge_gradually_not_instantly(self):
        """Multiple capped-delta submissions should take several rounds to
        reach mastery, not one -- the whole point of the cap."""
        p_l = 0.15
        history = [p_l]
        for _ in range(8):
            p_l = update_bkt(p_l, observed=0.95)
            history.append(p_l)
        # each step bounded
        for a, b in zip(history, history[1:]):
            self.assertLessEqual(b - a, MAX_MASTERY_DELTA + 1e-9)
        # but overall trend is upward and meaningful progress was made
        self.assertGreater(history[-1], history[0])


class TestDifficultyDampening(unittest.TestCase):
    """
    Two distinct, stacking mechanisms:
    1. Cap-scaling by ABSOLUTE difficulty (_CAP_SCALE_MIN.._CAP_SCALE_MAX):
       symmetric -- a low-difficulty problem carries weaker evidence in
       EITHER direction (smaller gain on success, smaller loss on
       failure), a high-difficulty one carries stronger evidence either
       way. difficulty=0.5 is neutral (matches difficulty=None).
    2. Trivial-gap dampening by RELATIVE difficulty-vs-mastery
       (_TRIVIAL_GAP_THRESHOLD): asymmetric, only ever shrinks a POSITIVE
       delta further, on top of #1 -- solving something well below your
       own current mastery teaches less, regardless of the problem's
       absolute difficulty.
    """

    def test_trivial_problem_dampens_positive_delta(self):
        # low absolute difficulty (cap-scaling) AND a large relative gap
        # below current mastery (trivial-gap dampening) both apply here --
        # delta should end up smaller than the difficulty-agnostic case.
        without_difficulty = update_bkt(current_p_l=0.6, observed=0.9, difficulty=None)
        with_trivial_difficulty = update_bkt(current_p_l=0.6, observed=0.9, difficulty=0.1)
        self.assertLess(with_trivial_difficulty, without_difficulty)

    def test_neutral_difficulty_matches_no_difficulty(self):
        # difficulty=0.5 is the cap-scaling formula's neutral point --
        # should reproduce difficulty=None's cap exactly.
        without_difficulty = update_bkt(current_p_l=0.3, observed=0.9, difficulty=None)
        with_neutral_difficulty = update_bkt(current_p_l=0.3, observed=0.9, difficulty=0.5)
        self.assertAlmostEqual(with_neutral_difficulty, without_difficulty, places=4)

    def test_above_neutral_difficulty_raises_the_cap(self):
        # difficulty above 0.5, gap is positive (not trivial) -- cap-scaling
        # alone should raise the ceiling above the neutral cap.
        without_difficulty = update_bkt(current_p_l=0.3, observed=0.9, difficulty=None)
        with_hard_difficulty = update_bkt(current_p_l=0.3, observed=0.9, difficulty=0.9)
        self.assertGreater(with_hard_difficulty, without_difficulty)

    def test_below_neutral_difficulty_lowers_the_cap_even_without_trivial_gap(self):
        # difficulty=0.4 is below the 0.5 neutral point but gap=0.4-0.3=0.1
        # doesn't cross _TRIVIAL_GAP_THRESHOLD -- cap-scaling alone (not
        # trivial-gap dampening) should still shrink the delta slightly.
        without_difficulty = update_bkt(current_p_l=0.3, observed=0.9, difficulty=None)
        with_matched_difficulty = update_bkt(current_p_l=0.3, observed=0.9, difficulty=0.4)
        self.assertLess(with_matched_difficulty, without_difficulty)

    def test_low_difficulty_dampens_a_decrease_too(self):
        # observed below the learning-transition threshold -> a big negative
        # raw delta, clipped by the cap. Cap-scaling is symmetric, so a low
        # difficulty problem's SMALLER cap means a SMALLER mastery loss too
        # (weak evidence either way) -- this is intentional, not a bug: it's
        # a distinct mechanism from trivial-gap dampening, which only ever
        # touches positive deltas.
        without_difficulty = update_bkt(current_p_l=0.6, observed=0.1, difficulty=None)
        with_low_difficulty = update_bkt(current_p_l=0.6, observed=0.1, difficulty=0.05)
        self.assertGreater(with_low_difficulty, without_difficulty)   # smaller magnitude drop -> higher new_p_l

    def test_never_dampens_below_floor(self):
        # extreme trivial gap should still leave some positive credit, not zero it
        new_p_l = update_bkt(current_p_l=0.9, observed=0.95, difficulty=0.0)
        self.assertGreaterEqual(new_p_l, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
