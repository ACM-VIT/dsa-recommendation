"""
tests/test_catalog_metadata.py

Tests training/catalog_metadata.py::load_catalog_metadata_by_problem_id()
without any real file I/O or live Postgres connection -- mocks
pipeline.ingestion.ingest.load_manifest and
database.postgres.db.resolve_problem_ids_by_title_slugs (the two real
functions it reuses), proving:
  - it resolves manifest records to real problem_ids via title_slug (not
    the manifest's own internal problem_id)
  - it caches after the first call (module-level cache, mirroring the
    LightGBMRanker singleton pattern)
  - it degrades to {} (not a fabricated value) if the manifest is missing

Run:
    python -m pytest tests/test_catalog_metadata.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from training import catalog_metadata


class TestLoadCatalogMetadataByProblemId(unittest.TestCase):
    def setUp(self):
        catalog_metadata.reset_catalog_metadata_cache()

    def tearDown(self):
        catalog_metadata.reset_catalog_metadata_cache()

    def test_resolves_via_title_slug_not_manifest_problem_id(self):
        fake_records = [
            {"problem_id": "manifest_internal_hash_1", "title_slug": "two-sum",
             "companies": ["Google", "Amazon"], "frequency": 0.9, "rating": 0.95,
             "asked_by_faang": True},
        ]
        with patch.object(catalog_metadata, "load_manifest", return_value=fake_records), \
             patch("database.postgres.db.resolve_problem_ids_by_title_slugs",
                   return_value={"two-sum": "real_backend_cuid_123"}) as mock_resolve:
            result = catalog_metadata.load_catalog_metadata_by_problem_id()

        mock_resolve.assert_called_once_with(["two-sum"])
        self.assertNotIn("manifest_internal_hash_1", result)
        self.assertIn("real_backend_cuid_123", result)
        self.assertEqual(result["real_backend_cuid_123"]["companies"], ["Google", "Amazon"])
        self.assertEqual(result["real_backend_cuid_123"]["frequency"], 0.9)
        self.assertEqual(result["real_backend_cuid_123"]["rating"], 0.95)
        self.assertTrue(result["real_backend_cuid_123"]["asked_by_faang"])

    def test_unresolved_slug_is_simply_absent_not_fabricated(self):
        fake_records = [
            {"problem_id": "x", "title_slug": "not-in-backend-yet",
             "companies": ["Meta"], "frequency": 0.5, "rating": 0.5, "asked_by_faang": False},
        ]
        with patch.object(catalog_metadata, "load_manifest", return_value=fake_records), \
             patch("database.postgres.db.resolve_problem_ids_by_title_slugs", return_value={}):
            result = catalog_metadata.load_catalog_metadata_by_problem_id()

        self.assertEqual(result, {})

    def test_missing_manifest_file_degrades_to_empty_dict(self):
        with patch.object(catalog_metadata, "load_manifest", side_effect=FileNotFoundError()):
            result = catalog_metadata.load_catalog_metadata_by_problem_id()
        self.assertEqual(result, {})

    def test_result_is_cached_across_calls(self):
        fake_records = [
            {"problem_id": "x", "title_slug": "two-sum", "companies": ["Google"],
             "frequency": 0.9, "rating": 0.95, "asked_by_faang": True},
        ]
        with patch.object(catalog_metadata, "load_manifest", return_value=fake_records) as mock_load, \
             patch("database.postgres.db.resolve_problem_ids_by_title_slugs",
                   return_value={"two-sum": "real_id"}) as mock_resolve:
            first = catalog_metadata.load_catalog_metadata_by_problem_id()
            second = catalog_metadata.load_catalog_metadata_by_problem_id()

        self.assertEqual(first, second)
        mock_load.assert_called_once()
        mock_resolve.assert_called_once()

    def test_reset_cache_forces_recompute(self):
        fake_records = [
            {"problem_id": "x", "title_slug": "two-sum", "companies": ["Google"],
             "frequency": 0.9, "rating": 0.95, "asked_by_faang": True},
        ]
        with patch.object(catalog_metadata, "load_manifest", return_value=fake_records) as mock_load, \
             patch("database.postgres.db.resolve_problem_ids_by_title_slugs",
                   return_value={"two-sum": "real_id"}):
            catalog_metadata.load_catalog_metadata_by_problem_id()
            catalog_metadata.reset_catalog_metadata_cache()
            catalog_metadata.load_catalog_metadata_by_problem_id()

        self.assertEqual(mock_load.call_count, 2)


if __name__ == "__main__":
    unittest.main()
