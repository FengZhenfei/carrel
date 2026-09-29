"""Deployment assets and the machine-adaptive bits around them (2026-09-28,
open-source preparation): the console only manages the service rows listed in
KB_CONSOLE_SERVICES, the pipeline asks for whichever MinerU backend the parser
container chose, compose paths are relative, systemd units are templated, and
no personal or machine-specific trace is left in tracked files."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]


# Patterns that are personal wherever they appear; the machine-specific list
# lives outside the repository (see test_no_personal_traces_in_tracked_files).
GENERIC_PRIVATE_PATTERNS = ("privaterelay", "iCloud", "Mobile Documents", "/Users/", "/home/", ".ts.net", "192.168.", "aliyuncs.com")


def private_patterns() -> list[str]:
    raw = os.environ.get("CARREL_PRIVATE_PATTERNS", "").strip()
    path = Path(raw).expanduser() if raw else Path.home() / ".config" / "carrel" / "private-patterns.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")



def _scratch_dir(case: unittest.TestCase) -> Path:
    """A temporary directory removed when the test case ends."""
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    return Path(tmp.name)


class ConsoleServicesConfigTests(unittest.TestCase):
    def test_parse_console_services(self) -> None:
        from kb_pipeline.config import CONSOLE_SERVICES_DEFAULT, parse_console_services

        self.assertEqual(parse_console_services(None), CONSOLE_SERVICES_DEFAULT)
        self.assertEqual(CONSOLE_SERVICES_DEFAULT, ("database", "mineru"))
        self.assertEqual(parse_console_services(""), ())
        self.assertEqual(parse_console_services("vlm, database ,bogus,Reranker"), ("database", "vlm", "reranker"))
        example = _read("config/knowledge-base.env.example")
        self.assertIn("KB_CONSOLE_SERVICES=database,mineru", example)

    def test_unmanaged_rows_are_not_probed_and_cannot_be_restarted(self) -> None:
        from kb_server import service

        def cfg(**over):
            base = dict(qdrant_url="http://q", opensearch_url="http://o", neo4j_uri="", mineru_url="http://m",
                        embedding_base_url="http://e/v1", vlm_base_url="http://v/v1", visual_embedding_enabled=True,
                        visual_embedding_base_url="http://ve/v1", parse_enabled=True,
                        state_db=_scratch_dir(self) / "s.db", reranker_base_url="http://r/v1",
                        visual_reranker_base_url="http://vr/v1", console_services=("database", "mineru"),
                        runtime_dir=_scratch_dir(self), log_dir=_scratch_dir(self))
            base.update(over)
            return SimpleNamespace(**base)

        seen: list[str] = []

        def fake_get(url, timeout=0):
            seen.append(url)
            return SimpleNamespace(status_code=200)

        with mock.patch.object(service.requests, "get", side_effect=fake_get), \
                mock.patch.object(service, "_probe_neo4j", return_value=False), \
                mock.patch.object(service, "_systemctl_show", return_value={}), \
                mock.patch.object(service, "_maintenance_defers", return_value={}), \
                mock.patch.object(service, "timer_health", return_value=[]), \
                mock.patch.object(service, "settings", return_value=cfg()):
            service._health_cache["value"] = None
            h = service.health()
            self.assertEqual(h["managed"], ["database", "mineru"])
            self.assertTrue(h["qdrant"] and h["opensearch"] and h["mineru"])
            for key in ("embedding", "vlm", "visual_embedding", "reranker", "visual_reranker"):
                self.assertIsNone(h[key], key)
            self.assertFalse([u for u in seen if "://e/" in u or "://v/" in u or "://ve/" in u or "://r/" in u])
            self.assertEqual(h["mineru_backend"], "pipeline")          # nothing published, MINERU_BACKEND unset
            self.assertEqual(h["mineru_backend_source"], "default")
            self.assertEqual(service._all_service_containers(),
                             ["carrel-qdrant", "carrel-opensearch", "carrel-neo4j", "carrel-mineru"])
            with self.assertRaises(KeyError):
                service.restart_service("vlm", force=True)
            service._health_cache["value"] = None
        # with a model row managed as well: the health probe and restart both cover it
        with mock.patch.object(service.requests, "get", side_effect=fake_get), \
                mock.patch.object(service, "_probe_neo4j", return_value=False), \
                mock.patch.object(service, "_systemctl_show", return_value={}), \
                mock.patch.object(service, "_maintenance_defers", return_value={}), \
                mock.patch.object(service, "timer_health", return_value=[]), \
                mock.patch.object(service, "settings", return_value=cfg(console_services=("database", "mineru", "vlm"))):
            service._health_cache["value"] = None
            h = service.health()
            self.assertTrue(h["vlm"])
            self.assertIsNone(h["embedding"])
            self.assertIn("http://v/health", seen)
            self.assertEqual(service._all_service_containers()[-1], "carrel-vlm")
        service._health_cache["value"] = None

    def test_console_draws_buttons_only_for_managed_rows(self) -> None:
        js = _read("app/kb_server/static/app.js")
        health = js.split("async function _refreshHealth", 1)[1].split("\nasync function ", 1)[0]
        self.assertIn("h.managed", health)
        self.assertIn("managed.has(key)", health)                  # per-row restart button
        self.assertIn("managed.size ?", health)                    # restart-all / shutdown only appear when there are managed rows
        self.assertIn("h.mineru_backend", health)                  # the parser service row shows the backend mode
        bulk = js.split("async function onBulkService", 1)[1].split("\nasync function ", 1)[0]
        self.assertIn('t("关闭全部系统服务?\\n\\n{0}都会停止;解析队列会等服务回来再继续", managedNames)', bulk)
        i18n = _read("app/kb_server/static/i18n.js")
        self.assertIn('"关闭全部系统服务?\\n\\n{0}都会停止;解析队列会等服务回来再继续"', i18n)
        self.assertNotIn("iCloud", js)
        self.assertNotIn("iCloud", i18n)


class MineruBackendResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        from kb_pipeline.parsers import mineru_backend

        mineru_backend._cache.update(path=None, mtime=None, value=None)
        self.tmp = _scratch_dir(self)
        self.info = self.tmp / "mineru-output" / "carrel-mineru.json"
        self.info.parent.mkdir(parents=True)

    def _write(self, payload: dict) -> None:
        self.info.write_text(json.dumps(payload), encoding="utf-8")
        # mtime resolution on some filesystems is coarse; force a change
        stamp = time.time() + self._bump()
        os.utime(self.info, (stamp, stamp))

    _n = 0

    def _bump(self) -> int:
        MineruBackendResolverTests._n += 1
        return MineruBackendResolverTests._n

    def test_explicit_env_wins(self) -> None:
        from kb_pipeline.parsers.mineru_backend import resolve_backend

        self._write({"backend": "pipeline"})
        with mock.patch.dict(os.environ, {"MINERU_BACKEND": "vlm-engine"}):
            self.assertEqual(resolve_backend(self.tmp), ("vlm-engine", "env"))

    def test_auto_follows_the_container_and_notices_changes(self) -> None:
        from kb_pipeline.parsers.mineru_backend import resolve_backend

        with mock.patch.dict(os.environ, {"MINERU_BACKEND": "auto"}):
            self.assertEqual(resolve_backend(self.tmp), ("pipeline", "default"))     # no file yet
            self._write({"backend": "vlm-engine", "device": "cuda"})
            self.assertEqual(resolve_backend(self.tmp), ("vlm-engine", "container"))
            self._write({"backend": "pipeline"})
            self.assertEqual(resolve_backend(self.tmp), ("pipeline", "container"))
            self._write({"backend": "something-else"})
            self.assertEqual(resolve_backend(self.tmp), ("pipeline", "default"))      # unknown names are not taken at face value
            self.info.write_text("{not json", encoding="utf-8")
            os.utime(self.info, (time.time() + 99, time.time() + 99))
            self.assertEqual(resolve_backend(self.tmp), ("pipeline", "default"))
        with mock.patch.dict(os.environ, {"MINERU_BACKEND": "", "MINERU_INFO_FILE": str(self.info)}):
            self._write({"backend": "hybrid-engine"})
            self.assertEqual(resolve_backend(None), ("hybrid-engine", "container"))   # file location given explicitly

    def test_client_asks_for_the_resolved_backend(self) -> None:
        src = _read("app/kb_pipeline/parsers/service_clients.py")
        self.assertIn("backend, _backend_source = resolve_backend()", src)
        self.assertNotIn('os.getenv("MINERU_BACKEND"', src)


class DeploymentAssetTests(unittest.TestCase):
    def test_compose_uses_relative_paths_and_carrel_names(self) -> None:
        from kb_server.service import SERVICE_CONTAINERS

        compose = _read("deployment/compose/docker-compose.yml")
        self.assertIn("name: carrel", compose)
        self.assertNotIn("/home/", compose)
        self.assertNotIn("/Users/", compose)
        for names in SERVICE_CONTAINERS.values():
            for name in names:
                self.assertIn(f"container_name: {name}", compose, name)
        mineru = compose.split("\n  mineru:", 1)[1].split("\n  qdrant:", 1)[0]
        self.assertNotIn("gpus:", mineru)                          # the GPU reservation lives in the overlay
        self.assertIn("gpus: all", _read("deployment/compose/compose.gpu.yml"))
        self.assertIn("MINERU_BACKEND_POLICY", mineru)
        self.assertIn("../../models/mineru", mineru)
        self.assertIn("../../runtime/mineru-output", mineru)
        example = _read("deployment/compose/.env.example")
        self.assertIn("COMPOSE_FILE=docker-compose.yml", example)
        # 2026-09-28 security review F06 / F08 / F09 / .gitignore
        self.assertIn('KB_HOST_HOUSEKEEPING:-0', _read("scripts/kb-cleanup.sh"))
        self.assertIn('pdf-images = ["pymupdf', _read("app/pyproject.toml"))
        deploy = _read("deploy.sh")
        self.assertIn("--with-pdf-images) WITH_PDF_IMAGES=1", deploy)       # AGPL extra is opt-in
        self.assertNotIn('app[pdf-images]" pytest', deploy)
        self.assertIn("checks[\"pdf_images\"]", _read("app/kb_search/service.py"))
        self.assertTrue((REPO / "THIRD_PARTY_LICENSES/Apache-2.0.txt").exists())
        self.assertTrue((REPO / "THIRD_PARTY_LICENSES/GraphRAG-MIT.txt").exists())
        self.assertIn("PyMuPDF", _read("NOTICE.md"))
        ignore = _read(".gitignore")
        for rule in (".env", ".env.*", "*.env", "!*.env.example"):
            self.assertIn(rule + "\n", ignore)
        self.assertIn("MINERU_FLAVOR=gpu", example)
        self.assertIn("NEO4J_PASSWORD=\n", example)

    def test_mineru_detector_decides_from_policy_and_hardware(self) -> None:
        script = REPO / "deployment/compose/mineru/detect.py"
        base_env = {k: v for k, v in os.environ.items() if not k.startswith(("MINERU_", "CARREL_"))}

        def run(**env):
            out = subprocess.run([sys.executable, str(script), "--json"], env={**base_env, **env},
                                 capture_output=True, text=True, timeout=120, check=True)
            return json.loads(out.stdout)

        # a machine without a GPU (the test box) or the cpu flavor: auto → pipeline
        d = run(CARREL_MINERU_FLAVOR="cpu", MINERU_BACKEND_POLICY="auto")
        self.assertEqual(d["backend"], "pipeline")
        self.assertFalse(d["capable"])
        self.assertEqual(d["device"], "cpu" if not d["gpu"].get("cuda") else "cuda")
        d = run(CARREL_MINERU_FLAVOR="cpu", MINERU_BACKEND_POLICY="vlm-engine")
        self.assertEqual(d["backend"], "vlm-engine")                # a forced value is passed through as is; whether it can run is up to the entrypoint
        self.assertFalse(d["capable"])
        d = run(CARREL_MINERU_FLAVOR="gpu", MINERU_BACKEND_POLICY="pipeline", MINERU_GPU_MEMORY_UTILIZATION="0.33")
        self.assertEqual(d["backend"], "pipeline")
        self.assertEqual(d["gpu_memory_utilization"], 0.33)
        shell = subprocess.run([sys.executable, str(script), "--shell"], env={**base_env, "CARREL_MINERU_FLAVOR": "cpu"},
                               capture_output=True, text=True, timeout=120, check=True).stdout
        self.assertIn('export CARREL_BACKEND="pipeline"', shell)
        self.assertIn("export CARREL_DEVICE=", shell)

    def test_shell_scripts_parse(self) -> None:
        for rel in ("deploy.sh", "deployment/compose/mineru/entrypoint.sh", "scripts/install-systemd.sh"):
            proc = subprocess.run(["bash", "-n", str(REPO / rel)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"{rel}: {proc.stderr}")
        entry = _read("deployment/compose/mineru/entrypoint.sh")
        self.assertIn("mineru-models-download -s", entry)
        self.assertIn("carrel-mineru.json", entry)
        self.assertIn("MINERU_TOOLS_CONFIG_JSON", entry)

    def test_systemd_units_are_templated_on_the_repo_path(self) -> None:
        units = sorted((REPO / "deployment/systemd").glob("*.service"))
        self.assertTrue(units)
        for unit in units:
            text = unit.read_text(encoding="utf-8")
            self.assertIn("__CARREL_HOME__", text, unit.name)
            self.assertNotIn("%h/", text, unit.name)
        installer = _read("scripts/install-systemd.sh")
        self.assertIn("__CARREL_HOME__", installer)
        self.assertIn("--uninstall", installer)
        # 2026-09-28 security review F05: units are named carrel-*; the installer neither overwrites nor removes
        # units that belong to another checkout
        self.assertTrue(all(u.name.startswith("carrel-") for u in units))
        self.assertIn("owned_by_us", installer)
        # timers carry no ExecStart path, so every template starts with an ownership marker that renders the path
        for unit in sorted((Path(__file__).resolve().parents[2] / "deployment" / "systemd").glob("carrel-*")):
            self.assertTrue(unit.read_text(encoding="utf-8").startswith("# Installed from __CARREL_HOME__/"), unit.name)
        self.assertIn("refusing to overwrite", installer)
        self.assertNotIn("knowledge-base-", _read("app/kb_server/service.py"))
        self.assertNotIn("knowledge-base-", _read("scripts/kb-pipeline-scan.sh"))

    def test_installer_refuses_units_of_another_checkout(self) -> None:
        """Simulation: a same-named unit from an old directory is installed; the new checkout's installer does not
        overwrite it (only --force does), and uninstall only removes its own units."""
        with tempfile.TemporaryDirectory() as tmp:
            xdg = Path(tmp) / "xdg"; units = xdg / "systemd" / "user"; units.mkdir(parents=True)
            fake_bin = Path(tmp) / "bin"; fake_bin.mkdir()
            (fake_bin / "systemctl").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8"); (fake_bin / "systemctl").chmod(0o755)
            (units / "carrel-web.service").write_text("[Service]\nWorkingDirectory=/srv/other-checkout/app\n", encoding="utf-8")
            env = {**os.environ, "XDG_CONFIG_HOME": str(xdg), "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}", "CARREL_HOME": str(REPO)}
            proc = subprocess.run(["bash", str(REPO / "scripts/install-systemd.sh")], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("another checkout", proc.stderr)
            self.assertIn("/srv/other-checkout", (units / "carrel-web.service").read_text(encoding="utf-8"))
            proc = subprocess.run(["bash", str(REPO / "scripts/install-systemd.sh"), "--uninstall"], env=env, capture_output=True, text=True, timeout=60)
            self.assertTrue((units / "carrel-web.service").exists())            # someone else's unit is not removed
            proc = subprocess.run(["bash", str(REPO / "scripts/install-systemd.sh"), "--force"], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn(str(REPO) + "/", (units / "carrel-web.service").read_text(encoding="utf-8"))
            proc = subprocess.run(["bash", str(REPO / "scripts/install-systemd.sh"), "--uninstall"], env=env, capture_output=True, text=True, timeout=60)
            self.assertFalse((units / "carrel-web.service").exists())           # our own unit can be removed

    def test_no_personal_traces_in_tracked_files(self) -> None:
        """The public tree must not carry cloud-sync paths, home directories, tailnet
        names, LAN addresses or relay e-mail domains. Machine- and person-specific
        strings (user names, host names, device codes) are deliberately NOT listed
        here: a list of them would itself be a leak. Put them one per line in the
        file named by CARREL_PRIVATE_PATTERNS (default
        ~/.config/carrel/private-patterns.txt, never committed) and this test scans
        for them too on the machines that have it."""
        try:
            listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                     cwd=REPO, capture_output=True, text=True, timeout=60, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git unavailable")
        patterns = list(GENERIC_PRIVATE_PATTERNS) + private_patterns()
        skip = {"app/tests/test_deployment.py"}
        hits: list[str] = []
        for rel in listing.splitlines():
            if not rel or rel in skip or rel.startswith("app/.venv"):
                continue
            path = REPO / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for pat in patterns:
                if pat in text:
                    line = next(i for i, l in enumerate(text.splitlines(), 1) if pat in l)
                    hits.append(f"{rel}:{line}: {pat if pat in GENERIC_PRIVATE_PATTERNS else '<private pattern>'}")
        self.assertEqual(hits, [])

    def test_private_pattern_file_is_read_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p.txt"
            f.write_text("# comment\n\nalpha-host\n  beta_user \n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CARREL_PRIVATE_PATTERNS": str(f)}):
                self.assertEqual(private_patterns(), ["alpha-host", "beta_user"])
            with mock.patch.dict(os.environ, {"CARREL_PRIVATE_PATTERNS": str(Path(tmp) / "missing.txt")}):
                self.assertEqual(private_patterns(), [])


class SearchTemplateDefaultsTests(unittest.TestCase):
    """The search block of config/knowledge-base.env.example lists every search key of
    app/kb_search/config.py with its code default. Only the address and the token are live lines; the
    tuning keys stay commented out, so the calibrated defaults apply until someone uncomments one.
    deploy.sh copies the template into the live env file, so a listed value that drifted from the code
    would override a tuned default as soon as it is uncommented (the fourth review of 2026-09-28 found
    five such keys while they were still live lines)."""

    def test_template_search_values_are_the_code_defaults(self) -> None:
        from dataclasses import asdict

        from kb_search.config import load_search_settings

        text = _read("config/knowledge-base.env.example")
        template = {key: value.strip() for key, value in re.findall(r"(?m)^#?\s*(KB_SEARCH_[A-Z0-9_]+)=(.*)$", text)}
        self.assertGreaterEqual(len(template), 10)
        self.assertEqual(re.findall(r"(?m)^(KB_SEARCH_[A-Z0-9_]+)=", text),
                         ["KB_SEARCH_HOST", "KB_SEARCH_PORT", "KB_SEARCH_TOKEN"])
        named = set(re.findall(r"\"(KB_SEARCH_[A-Z0-9_]+)\"", _read("app/kb_search/config.py")))
        self.assertEqual(sorted(set(template) - named), [], "the template names search keys the code never reads")
        self.assertEqual(sorted(named - set(template)), [], "the template leaves out search keys the code reads")
        clean = {k: v for k, v in os.environ.items() if not k.startswith("KB_SEARCH_")}
        with mock.patch.dict(os.environ, clean, clear=True):
            defaults = asdict(load_search_settings())
        with mock.patch.dict(os.environ, {**clean, **template}, clear=True):
            listed = asdict(load_search_settings())
        drift = [f"{name}: template={listed[name]!r} code={value!r}" for name, value in defaults.items()
                 if listed[name] != value]
        self.assertEqual(drift, [])


if __name__ == "__main__":
    unittest.main()
