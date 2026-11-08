"""Vercel-recognized ASGI entry point and local uvicorn entry point."""
from geovalue.web import app

__all__ = ["app"]
