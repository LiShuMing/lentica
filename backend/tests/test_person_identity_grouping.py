from __future__ import annotations

import unittest

from backend.app.ingestion.faces import _identity_key


class PersonIdentityGroupingTests(unittest.TestCase):
    def test_shared_display_name_forms_one_logical_identity(self) -> None:
        self.assertEqual(_identity_key(1, "person_001", ["乔乔"]), _identity_key(2, "person_002", ["乔乔"]))

    def test_explicit_cluster_stays_distinct(self) -> None:
        self.assertEqual(_identity_key(1, "person_001", ["person_001"]), "cluster:1")


if __name__ == "__main__":
    unittest.main()
