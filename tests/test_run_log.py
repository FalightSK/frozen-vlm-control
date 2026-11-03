import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_log


class AtomicJsonWriteTests(unittest.TestCase):
    def test_retries_transient_windows_replace_denial(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "progress.json"
            real_replace = run_log.os.replace
            calls = 0

            def fail_twice_then_replace(source, target):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise PermissionError("simulated transient Windows sharing denial")
                real_replace(source, target)

            with patch.object(run_log.os, "replace", side_effect=fail_twice_then_replace):
                run_log.write_json_atomic(destination, {"optimizer_step": 7})

            self.assertEqual(calls, 3)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")),
                             {"optimizer_step": 7})
            self.assertFalse(any(path.name.endswith(".tmp") for path in Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
