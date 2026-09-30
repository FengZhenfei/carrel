#!/usr/bin/env python3
"""Write the exact versions of the installed dependency closure of pyproject.toml (plus pytest) as a lock file.
Run it from the tested virtual environment:  app/.venv/bin/python scripts/kb-lock-deps.py app/pyproject.toml > app/requirements.lock
Only the packages the declared dependencies actually pull in are listed; tools that happen to be installed are not."""
import re
import sys
import tomllib
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def closure(roots: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    todo = [Requirement(r) for r in roots]
    while todo:
        req = todo.pop()
        name = canonicalize_name(req.name)
        if name in out:
            continue
        try:
            dist = metadata.distribution(req.name)
        except metadata.PackageNotFoundError:
            print(f"not installed: {req.name}", file=sys.stderr)
            continue
        out[name] = f"{dist.metadata['Name']}=={dist.version}"
        for line in dist.requires or []:
            sub = Requirement(line)
            if sub.marker and not sub.marker.evaluate({"extra": ""}):
                continue          # optional extras and other platforms
            todo.append(sub)
    return out


def main() -> None:
    project = tomllib.loads(open(sys.argv[1], "rb").read().decode("utf-8"))["project"]
    roots = list(project.get("dependencies") or [])
    for extra in sys.argv[2:]:
        roots += project.get("optional-dependencies", {}).get(extra, [])
    roots.append("pytest")
    pins = closure(roots)
    print(f"# {len(pins)} packages, the dependency closure of pyproject.toml as installed in the tested environment.")
    print("# Regenerate with: app/.venv/bin/python scripts/kb-lock-deps.py app/pyproject.toml > app/requirements.lock")
    for key in sorted(pins):
        print(pins[key])


if __name__ == "__main__":
    main()
