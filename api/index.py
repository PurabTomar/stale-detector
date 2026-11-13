"""
api/index.py - Vercel Serverless Function entry point for Stale-Answer Detector.
Routes all incoming HTTP requests to the FastAPI ASGI application in api.py.
"""

import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api import app as base_app


class VercelPathNormalizerMiddleware:
    """
    Normalize route paths and query strings for Vercel Serverless Functions.

    When Vercel rewrites requests via `vercel.json` to `/api/index.py?__path__=/$1`,
    this middleware extracts `__path__`, sets `scope["path"]` to the original
    requested route (e.g. `/health`, `/docs`, `/verify`), and strips `__path__`
    from `scope["query_string"]` so FastAPI routing and query parameters work
    identically locally and on Vercel.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            target_path = None

            # 1. Check for `__path__` injected by vercel.json rewrite rule
            raw_qs = scope.get("query_string", b"")
            if b"__path__" in raw_qs:
                try:
                    qs_str = raw_qs.decode("latin-1")
                    parsed = parse_qsl(qs_str, keep_blank_values=True)
                    remaining = []
                    for k, v in parsed:
                        if k == "__path__":
                            target_path = v
                        else:
                            remaining.append((k, v))
                    if target_path is not None:
                        scope["query_string"] = urlencode(remaining).encode("latin-1")
                except Exception:
                    pass

            # 2. If no `__path__`, check x-matched-path header from Vercel
            if not target_path:
                headers = dict(scope.get("headers", []))
                raw_matched = headers.get(b"x-matched-path")
                if raw_matched:
                    matched = raw_matched.decode("utf-8", errors="ignore")
                    if "?" in matched:
                        matched = matched.split("?", 1)[0]
                    # Only use matched path if it is NOT the function entrypoint itself
                    if matched not in ("/api/index.py", "/api/index", "/api"):
                        target_path = matched

            # 3. Fallback to scope["path"]
            if not target_path:
                target_path = scope.get("path", "/")

            # Strip any /api/index.py or /api/index prefixes
            for pfx in ("/api/index.py", "/api/index"):
                if target_path == pfx:
                    target_path = "/"
                    break
                elif target_path.startswith(pfx + "/"):
                    target_path = target_path[len(pfx):]
                    break

            if not target_path.startswith("/"):
                target_path = "/" + target_path

            # Clean duplicate slashes like `//` -> `/`
            while "//" in target_path:
                target_path = target_path.replace("//", "/")

            scope["path"] = target_path
            scope["raw_path"] = target_path.encode("latin-1")

        await self.app(scope, receive, send)


app = VercelPathNormalizerMiddleware(base_app)
