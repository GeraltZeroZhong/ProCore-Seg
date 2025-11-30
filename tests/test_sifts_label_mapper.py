"""Smoke tests for the SIFTS label mapper."""
from __future__ import annotations

from importlib import util
from pathlib import Path
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "01_data_curation" / "02_sifts_label_mapper.py"


class FetchInstanceFeaturesSmokeTest(unittest.TestCase):
    def setUp(self) -> None:  # pragma: no cover - trivial
        spec = util.spec_from_file_location("sifts_label_mapper", _MODULE_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load SIFTS module from {_MODULE_PATH}")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module

    def test_fetch_instance_features_smoke(self) -> None:
        fetch_instance_features = getattr(self.module, "fetch_instance_features")

        try:
            instances = fetch_instance_features("2KGW", timeout=10)
        except RuntimeError as exc:
            # Network or GraphQL schema issues should be surfaced as RuntimeError
            self.assertTrue(str(exc))
            return

        self.assertIsInstance(instances, list)
        self.assertTrue(instances)
        self.assertTrue(all("auth_asym_id" in inst for inst in instances))


if __name__ == "__main__":  # pragma: no cover - test runner helper
    unittest.main()
