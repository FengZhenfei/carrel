#!/usr/bin/env python3
"""Small read-only client for the Carrel search service. Python 3.9+, standard library only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward bearer credentials to redirected destinations.


def read_object(path):
    raw = sys.stdin.read() if path == "-" else Path(path).expanduser().read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def settings(args):
    explicit = args.config or os.environ.get("CARREL_SEARCH_CONFIG")
    path = Path(explicit).expanduser() if explicit else Path.home() / ".config/carrel-search/config.json"
    cfg = read_object(str(path)) if explicit or path.exists() else {}
    url = str(args.base_url or os.environ.get("CARREL_SEARCH_BASE_URL") or cfg.get("base_url")
              or "http://127.0.0.1:9810").rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an HTTP(S) service URL without credentials, query or fragment")
    timeout = float(args.timeout if args.timeout is not None else cfg.get("timeout_seconds", 60))
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("timeout must be greater than 0 and at most 300 seconds")
    token = os.environ.get("CARREL_SEARCH_TOKEN") or os.environ.get(str(cfg.get("token_env") or "KB_SEARCH_TOKEN"), "")
    if not token and cfg.get("token_file"):
        token = Path(cfg["token_file"]).expanduser().read_text(encoding="utf-8").strip()
    if "\n" in token or "\r" in token:
        raise ValueError("token must be a single line")
    return url, token.strip(), timeout


def payload(args):
    data = read_object(args.request) if getattr(args, "request", None) else {}
    if args.command == "search":
        if args.question is not None:
            data["question"] = args.question
        if args.kb:
            data["kbs"] = args.kb
        if args.top_k is not None:
            data["top_k"] = args.top_k
        if args.no_context:
            data["context"] = False
        if args.explain:
            data["explain"] = True
        if args.image:
            image = Path(args.image).expanduser()
            if image.stat().st_size > 9_000_000:
                raise ValueError("image is too large for the API's base64 limit")
            data["image_b64"] = base64.b64encode(image.read_bytes()).decode("ascii")
        if not isinstance(data.get("question"), str) or not data["question"].strip():
            raise ValueError("search requires --question or a question in --request")
    if args.command == "neighbors":
        if args.kb:
            data["kb_id"] = args.kb
        if args.entity:
            data["entity"] = args.entity
        if args.entity_id:
            data["entity_id"] = args.entity_id
        if args.limit is not None:
            data["limit"] = args.limit
        if args.type:
            data["types"] = args.type
        if args.direction:
            data["direction"] = args.direction
        if not data.get("kb_id") or not (data.get("entity") or data.get("entity_id")):
            raise ValueError("neighbors requires --kb and one of --entity / --entity-id")
    return data


def request(url, token, timeout, method, route, data=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url + route, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        return response.read(), response.headers


def save_new(path, content):
    destination = Path(path).expanduser().absolute()
    # Exclusive creation: a mistyped output path must not replace user data.
    with destination.open("xb") as f:
        f.write(content)
    return str(destination)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="JSON config path; otherwise CARREL_SEARCH_CONFIG or ~/.config/carrel-search/config.json")
    p.add_argument("--base-url", help="Override service URL; default http://127.0.0.1:9810")
    p.add_argument("--timeout", type=float, help="HTTP timeout seconds (default 60); no automatic retries")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("health", "catalog", "search", "context", "image", "crop", "neighbors"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", required=name in ("image", "crop"), help="Save to a new file; binary image commands require this")
        if name in ("search", "context", "crop", "neighbors"):
            cmd.add_argument("--request", required=name in ("context", "crop"), help="UTF-8 JSON request file, or - for stdin")
        if name == "neighbors":
            cmd.add_argument("--kb", help="KB ID that has a graph")
            cmd.add_argument("--entity", help="Entity title or alias (case-insensitive)")
            cmd.add_argument("--entity-id", help="Entity id from a previous response")
            cmd.add_argument("--limit", type=int, choices=range(1, 101), metavar="1..100")
            cmd.add_argument("--type", action="append", help="Only these predicates; repeat to allow several")
            cmd.add_argument("--direction", choices=("both", "out", "in"))
        if name == "search":
            cmd.add_argument("--question")
            cmd.add_argument("--kb", action="append", help="Explicit KB ID; repeat to select multiple")
            cmd.add_argument("--top-k", type=int, choices=range(1, 51), metavar="1..50")
            cmd.add_argument("--image", help="Local query image file; sent to /search only")
            cmd.add_argument("--no-context", action="store_true")
            cmd.add_argument("--explain", action="store_true")
        if name == "image":
            cmd.add_argument("--kb", required=True)
            cmd.add_argument("--point-id", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    token = ""
    call_id = uuid.uuid4().hex[:12]
    try:
        if args.output and Path(args.output).expanduser().exists():
            raise FileExistsError("output already exists; choose a new filename")
        url, token, timeout = settings(args)
        binary = args.command in ("image", "crop")
        if args.command == "image":
            quote = lambda s: urllib.parse.quote(s, safe="")
            route = "/image/" + quote(args.kb) + "/" + quote(args.point_id)
            method, data = "GET", None
        elif args.command == "neighbors":
            route, method, data = "/graph/neighbors", "POST", payload(args)
        elif args.command in ("search", "context", "crop"):
            route, method, data = "/" + args.command, "POST", payload(args)
        else:
            route, method, data = "/" + args.command, "GET", None
        raw, headers = request(url, token, timeout, method, route, data)
        envelope = {"call_id": call_id, "operation": args.command}
        if binary:
            mime = headers.get_content_type()
            if not mime.startswith("image/"):
                raise ValueError("server did not return an image; no file saved")
            envelope["result"] = {
                "path": save_new(args.output, raw), "mime_type": mime,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": headers.get("X-Image-Source"),
                "width": headers.get("X-Image-Width"), "height": headers.get("X-Image-Height"),
                "crop_box": headers.get("X-Crop-Box"),
            }
        else:
            envelope["result"] = json.loads(raw)
            if args.output:
                encoded = (json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
                path = save_new(args.output, encoded)
                envelope = {"call_id": call_id, "operation": args.command, "response_file": path,
                            "note": "Read response_file for the complete API response."}
        print(json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            message = f"HTTP {exc.code}"
            if exc.code in (401, 403):
                message += ": check the configured bearer token; do not print it"
            else:
                detail = exc.read(2000).decode("utf-8", errors="replace")
                message += ": " + detail[:600]
        else:
            message = f"{type(exc).__name__}: {exc}"
        if token:
            message = message.replace(token, "[redacted]")
        print(json.dumps({"call_id": call_id, "operation": args.command, "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
