import tempfile
import unittest
from pathlib import Path

from cli.bootstrap import ensure_env_file


class EnsureEnvFileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_copies_example_when_env_missing(self):
        (self.root / ".env.example").write_text("KEY=value\n", encoding="utf-8")

        ensure_env_file(str(self.root))

        self.assertTrue((self.root / ".env").exists())
        self.assertEqual((self.root / ".env").read_text(encoding="utf-8"), "KEY=value\n")

    def test_does_not_overwrite_existing_env(self):
        (self.root / ".env").write_text("EXISTING=1\n", encoding="utf-8")
        (self.root / ".env.example").write_text("KEY=value\n", encoding="utf-8")

        ensure_env_file(str(self.root))

        self.assertEqual((self.root / ".env").read_text(encoding="utf-8"), "EXISTING=1\n")

    def test_no_op_when_no_example_present(self):
        ensure_env_file(str(self.root))
        self.assertFalse((self.root / ".env").exists())


if __name__ == "__main__":
    unittest.main()
