"""Uvicorn: ``uvicorn cem.api.app:create_app --factory --port 8000``"""
from cem.api.app import create_app

__all__ = ["create_app"]
