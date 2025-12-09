import sys
import os
import unittest

# Ensure the 'packages' directory is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


class TestPackageImports(unittest.TestCase):

    def test_import_pfc(self):
        import packages.pfc

        self.assertTrue(hasattr(packages, "pfc"))

    def test_import_pyjuice(self):
        import packages.pyjuice

        self.assertTrue(hasattr(packages, "pyjuice"))

    def test_import_model_submodule(self):
        from packages.pfc.models import (
            EinsumNet,
            LinearSplineEinsumFlow,
        )  # replace with actual module

        self.assertTrue(EinsumNet is not None)
        self.assertTrue(LinearSplineEinsumFlow is not None)


if __name__ == "__main__":
    unittest.main()
