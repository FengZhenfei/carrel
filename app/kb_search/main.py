from __future__ import annotations

import os
import sqlite3

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from .api import router, require_token


def create_app() -> FastAPI:
    app = FastAPI(title="Carrel search service", docs_url="/docs", openapi_url="/openapi.json")
    app.include_router(router)

    @app.middleware("http")
    async def _authenticate_before_body(request: Request, call_next):
        from starlette._utils import get_route_path
        path = get_route_path(request.scope)
        if path not in ("/health", "/openapi.json", "/docs", "/docs/oauth2-redirect") or request.method != "GET":
            try:
                require_token(request)
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return await call_next(request)

    @app.exception_handler(Exception)
    async def _readable_errors(request: Request, exc: Exception):
        if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
            return JSONResponse(status_code=503, content={"detail": "state database busy, retry shortly"})
        if isinstance(exc, (requests.RequestException, ConnectionError)):
            return JSONResponse(status_code=502, content={"detail": f"upstream service unreachable: {exc}"})
        if isinstance(exc, (OSError, RuntimeError, sqlite3.Error)):
            return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
        raise exc

    return app


def run() -> None:
    import uvicorn

    from kb_pipeline.config import load_settings

    from . import service

    try:
        load_settings()
        service.runtime()       # read the settings once while the env file has just been loaded into the process environment: only now can we tell which KB_SEARCH_* keys were set explicitly
    except Exception as exc:
        print(f"[search] load_settings failed at startup: {exc!r}; the service still starts and reports the error through its API", flush=True)
    uvicorn.run(create_app(), host=os.getenv("KB_SEARCH_HOST", "0.0.0.0"), port=int(os.getenv("KB_SEARCH_PORT", "9810")),
                log_level="info")
