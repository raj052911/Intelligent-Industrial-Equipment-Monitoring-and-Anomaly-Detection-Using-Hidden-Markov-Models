"""Application configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    SECRET_KEY = "local-development-key-change-before-deployment"
    DEBUG = True
    JSON_SORT_KEYS = False
