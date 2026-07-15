"""
tests/test_no_stdout_pollution.py

Regression test for a real bug found while testing run_full_pipeline.py
end-to-end: bkt.py/hlr.py had module-level print() calls (the
"Loaded topic mappings for N problems" diagnostic, and a missing-file
warning) that land on STDOUT. run_full_pipeline.py's contract is "stdout
is ALWAYS pure JSON, nothing else" -- these prints silently corrupted that,
breaking `json.loads()` on the script's own output.

This test imports bkt.py and hlr.py in a FRESH subprocess (so their
module-level code actually executes -- re-importing an already-cached
module in-process wouldn't re-run it) and asserts stdout contains nothing
but the diagnostic-free expected output, i.e. nothing at all.

Run:
    python -m pytest tests/test_no_stdout_pollution.py -v
"""

from __future__ import annotations

import subprocess
import sys
import unittest


class TestNoStdoutPollution(unittest.TestCase):

    def _stdout_for_import(self, module: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=f"import failed: {result.stderr}")
        return result.stdout

    def test_bkt_import_prints_nothing_to_stdout(self):
        stdout = self._stdout_for_import("pipeline.recommender.bkt")
        self.assertEqual(stdout, "",
                         f"bkt.py wrote to stdout at import time: {stdout!r} -- "
                         f"this corrupts run_full_pipeline.py's pure-JSON stdout contract")

    def test_hlr_import_prints_nothing_to_stdout(self):
        stdout = self._stdout_for_import("pipeline.recommender.hlr")
        self.assertEqual(stdout, "",
                         f"hlr.py wrote to stdout at import time: {stdout!r} -- "
                         f"this corrupts run_full_pipeline.py's pure-JSON stdout contract")

    def test_state_update_service_import_prints_nothing_to_stdout(self):
        """Broader net: the whole chain run_full_pipeline.py actually imports."""
        stdout = self._stdout_for_import("pipeline.recommender.services.state_update_service")
        self.assertEqual(stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
