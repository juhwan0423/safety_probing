from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from safety_probing.protocol_lock import verify_lock


class ProtocolLockTest(unittest.TestCase):
    def test_verify_detects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            path.write_text("{}\n", encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            lock = {
                "protocol_id": "test",
                "files": [{"path": "config.json", "sha256": expected}],
            }
            self.assertTrue(verify_lock(root, lock)["valid"])
            path.write_text('{"changed": true}\n', encoding="utf-8")
            report = verify_lock(root, lock)
            self.assertFalse(report["valid"])
            self.assertEqual(report["mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()

