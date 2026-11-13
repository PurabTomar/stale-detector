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

from api import app
