#!/usr/bin/env python3
"""Write the installed dependency closure of pyproject.toml, plus pytest.
Use --output to atomically replace a lock only after the complete closure is resolved.
"""
import argparse
import os
import tempfile
from pathlib import Path
import tomllib
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def closure(roots: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    seen_extras: dict[str, set[str]] = {}
    todo = [Requirement(r) for r in roots]
    while todo:
        req = todo.pop()
        name = canonicalize_name(req.name)
        requested = {"", *req.extras}
        fresh = requested - seen_extras.get(name, set())
        if not fresh:
            continue
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
            raise RuntimeError(f"not installed: {req.name}")
        seen_extras.setdefault(name, set()).update(fresh)
        out[name] = f"{dist.metadata['Name']}=={dist.version}"
        for line in dist.requires or []:
            sub = Requirement(line)
            if sub.marker and not any(sub.marker.evaluate({"extra": extra}) for extra in fresh):
                continue          # inactive extras and other platforms
            todo.append(sub)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("extras", nargs="*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = tomllib.loads(args.project.read_text(encoding="utf-8"))["project"]
    roots = list(project.get("dependencies") or [])
    for extra in args.extras:
        roots += project.get("optional-dependencies", {}).get(extra, [])
    roots.append("pytest")
    pins = closure(roots)
    body = f"# {len(pins)} packages, resolved from the tested environment.\n"
    body += "# Regenerate: app/.venv/bin/python scripts/kb-lock-deps.py app/pyproject.toml --output app/requirements.lock\n"
    body += "\n".join(pins[key] for key in sorted(pins)) + "\n"
    if args.output is None:
        print(body, end="")
    else:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as tmp:
            tmp.write(body)
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            os.replace(tmp.name, args.output)
        finally:
            Path(tmp.name).unlink(missing_ok=True)



if __name__ == "__main__":
    main()
