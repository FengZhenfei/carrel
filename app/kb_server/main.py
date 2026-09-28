from __future__ import annotations

import hmac
import os
from pathlib import Path

import sqlite3
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import service
from .api import router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        service.init_state()
    except Exception as exc:  # a cold state dir must not keep the UI down
        print(f"[web] state init skipped: {exc!r}", flush=True)
    yield


def _origin_tuple(scheme: str, host: str, port: int | None) -> tuple[str, str, int]:
    scheme = (scheme or "http").lower()
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, (host or "").lower(), int(port)


def _request_origin(request: Request) -> tuple[str, str, int]:
    """The origin the browser sees this console as: scheme + host + port, honouring
    a reverse proxy's X-Forwarded-Proto / X-Forwarded-Host."""
    scheme = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    host_header = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip()
    try:
        parts = urlsplit(f"//{host_header}")
        host, port = parts.hostname or "", parts.port
    except ValueError:
        host, port = "", None
    return _origin_tuple(scheme, host, port)


def _same_origin(request: Request) -> bool:
    """Write operations must come from the console itself: scheme + host + port of the Origin (or
    Referer) must all match this service, with no exceptions like "same host, different port" or
    "localhost always allowed" -- otherwise another web page on the same machine could use a simple
    POST without preflight to close a knowledge base, trigger a full re-parse or restart containers
    (kb_id is a sequential number and can be enumerated). Requests without Origin/Referer (curl /
    scripts / same-origin navigation) are allowed: this layer blocks cross-site browser requests and
    is not authentication; authentication is KB_WEB_TOKEN."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if not parts.scheme or not parts.hostname:
        return False
    try:
        port = parts.port
    except ValueError:
        return False
    return _origin_tuple(parts.scheme, parts.hostname, port) == _request_origin(request)


def create_app() -> FastAPI:
    app = FastAPI(title="Carrel console", docs_url="/api/docs", openapi_url="/api/openapi.json",
                  lifespan=_lifespan)
    app.include_router(router)
    # KB_WEB_TOKEN: when set, all of /api requires a Bearer token; the static pages open as usual, and
    # the front end asks for the token once and remembers it in the browser. It can stay unset while
    # listening on 127.0.0.1 only (the default); it should be set when opening up to the LAN
    # (KB_WEB_HOST=0.0.0.0).
    token = os.getenv("KB_WEB_TOKEN", "").strip()

    @app.middleware("http")
    async def _require_token(request: Request, call_next):
        if token and request.url.path.startswith("/api"):
            auth = request.headers.get("authorization") or ""
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if not given or not hmac.compare_digest(given.encode(), token.encode()):
                return JSONResponse(status_code=401, content={"detail": "console token required"})
        return await call_next(request)

    @app.middleware("http")
    async def _guard_writes(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _same_origin(request):
            return JSONResponse(
                status_code=403,
                content={"detail": "Cross-site write request rejected (the console only accepts same-origin operations)"},
            )
        return await call_next(request)

    @app.exception_handler(Exception)
    async def _readable_errors(request: Request, exc: Exception):
        # Every exception other than KeyError/ValueError used to become a generic 500 "Internal Server
        # Error"; the front-end toast could only show HTTP 500 and the user had no idea of the cause.
        if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
            return JSONResponse(status_code=503, content={"detail": "The state database is busy; try again later"})
        if isinstance(exc, (requests.RequestException, ConnectionError)):
            return JSONResponse(status_code=502, content={"detail": f"External service unreachable: {exc}"})
        if isinstance(exc, (OSError, RuntimeError, sqlite3.Error)):
            return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
        raise exc

    @app.middleware("http")
    async def _fresh_static(request, call_next):
        # No build chain and no fingerprinted filenames: HTML/JS are always no-cache (the browser
        # revalidates with a 304 every time), otherwise Safari pairs an old app.js with a new
        # index.html (it has actually broken this way).
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def run() -> None:
    import uvicorn

    # Pull KB_WEB_HOST/PORT (and everything else) from the env file before
    # reading them: load_settings loads KB_ENV_FILE into os.environ.
    from kb_pipeline.config import load_settings

    try:
        load_settings()
    except Exception as exc:
        # A bad config used to make the process exit outright, and systemd Restart=always looped it
        # every 3 seconds while the console never came up and the user never saw the cause. Start the
        # service first and let the API report the error faithfully.
        print(f"[web] load_settings failed at startup: {exc!r}; the console still starts and reports the error through its API",
              flush=True)
    uvicorn.run(
        create_app(),
        host=os.getenv("KB_WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("KB_WEB_PORT", "9800")),
        log_level="info",
    )
