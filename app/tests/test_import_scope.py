"""Project boundaries and positive import resolution, independent of repository names."""
import tempfile
import unittest
from pathlib import Path

from kb_pipeline.graph.import_resolution import ImportResolver


class ImportScopeTests(unittest.TestCase):
    def resolver(self, files, markers=()):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel in [*files, *markers]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return ImportResolver(set(files), lambda rel: root / rel)

    def test_package_name_never_falls_back_to_unrelated_basename(self):
        for group in ("bundle", "renamed/deep/group"):
            a, b = f"{group}/alpha", f"{group}/beta"
            r = self.resolver([f"{a}/scripts/run.py", f"{b}/validators/deck.py", f"{b}/http/models.py"],
                              [f"{a}/pyproject.toml", f"{b}/pyproject.toml"])
            for module in ("deck", "client.http.models", "http.models"):
                result = r.resolve(f"{a}/scripts/run.py", module)
                self.assertEqual(result.status, "external")
                self.assertIsNone(result.target)

    def test_same_project_absolute_relative_and_namespace_modules(self):
        r = self.resolver(["project/src/pkg/run.py", "project/src/pkg/util.py", "project/src/pkg/nested/item.py"],
                          ["project/pyproject.toml"])
        for module, level, name, expected in [("pkg.util", 0, None, "util.py"), ("util", 1, None, "util.py"),
                                              ("pkg.nested", 0, "item", "nested/item.py")]:
            self.assertEqual(r.resolve("project/src/pkg/run.py", module, level, name=name).target, "project/src/pkg/" + expected)

    def test_ambiguity_is_not_resolved_by_shortest_path_or_order(self):
        files = ["project/pkg/main.py", "project/pkg/util.py", "project/util.py"]
        for order in (files, list(reversed(files))):
            r = self.resolver(order, ["project/pyproject.toml"])
            result = r.resolve(files[0], "util")
            self.assertEqual(result.status, "ambiguous")
            self.assertIsNone(result.target)
            self.assertEqual(len(result.candidates), 2)

    def test_relative_missing_or_escaping_project_is_unresolved(self):
        r = self.resolver(["one/run.py", "two/helper.py"], ["one/package.json", "two/package.json"])
        for module in ("./missing", "../two/helper"):
            self.assertEqual(r.resolve("one/run.py", module).status, "unresolved")

    def test_javascript_relative_import_is_preserved(self):
        r = self.resolver(["one/index.ts", "one/lib/worker.ts", "two/lib/worker.ts"], ["one/package.json", "two/package.json"])
        self.assertEqual(r.resolve("one/index.ts", "./lib/worker").target, "one/lib/worker.ts")
