"""
api/index.py - Vercel Serverless Function entry point for Stale-Answer Detector.
Routes all incoming HTTP requests to the FastAPI ASGI application in api.py.
"""

import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api import app as base_app


class StripPrefixMiddleware:
    """Normalize Vercel function route path prefixes before FastAPI router evaluation."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            # Vercel provides the client-requested URL path in x-matched-path
            raw_matched = headers.get(b"x-matched-path")
            if raw_matched:
                matched_path = raw_matched.decode("utf-8", errors="ignore")
                if "?" in matched_path:
                    matched_path = matched_path.split("?", 1)[0]
                # Strip function entrypoint prefix if present
                for pfx in ("/api/index.py", "/api/index"):
                    if matched_path == pfx:
                        matched_path = "/"
                        break
                    elif matched_path.startswith(pfx + "/"):
                        matched_path = matched_path[len(pfx):]
                        break
                scope["path"] = matched_path or "/"
            else:
                path = scope.get("path", "")
                for prefix in ("/api/index.py", "/api/index", "/api"):
                    if path == prefix:
                        scope["path"] = "/"
                        break
                    elif path.startswith(prefix + "/"):
                        scope["path"] = path[len(prefix):]
                        break
        await self.app(scope, receive, send)


app = StripPrefixMiddleware(base_app)
