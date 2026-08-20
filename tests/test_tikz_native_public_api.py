"""Public package-surface regressions for fail-closed compiler errors."""

from __future__ import annotations

import unittest

import tikz_native
from tikz_native.compiler import TikzNativeError


class TikzNativePublicApiTests(unittest.TestCase):
    def test_compiler_error_is_exported_from_package_root(self) -> None:
        self.assertIs(tikz_native.TikzNativeError, TikzNativeError)
        self.assertIn("TikzNativeError", tikz_native.__all__)


if __name__ == "__main__":
    unittest.main()
