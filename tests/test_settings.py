import importlib
import os
import sys

import pytest


def load_settings_with_env(env_updates):
    original_values = {
        key: os.environ.get(key)
        for key in {
            "OPENAI_API_KEY",
            "VECTOR_STORE_BACKEND",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_HOST",
        }
    }
    for key, value in env_updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    sys.modules.pop("app.core.settings", None)
    settings_module = importlib.import_module("app.core.settings")
    settings = settings_module.Settings()

    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    return settings


def test_settings_require_pinecone_host_for_pinecone_backend():
    settings = load_settings_with_env(
        {
            "OPENAI_API_KEY": "test-key",
            "VECTOR_STORE_BACKEND": "pinecone",
            "PINECONE_API_KEY": "pinecone-key",
            "PINECONE_INDEX_HOST": None,
        }
    )

    with pytest.raises(ValueError, match="PINECONE_INDEX_HOST is required"):
        settings.validate()


def test_settings_accept_pinecone_backend_when_required_envs_exist():
    settings = load_settings_with_env(
        {
            "OPENAI_API_KEY": "test-key",
            "VECTOR_STORE_BACKEND": "pinecone",
            "PINECONE_API_KEY": "pinecone-key",
            "PINECONE_INDEX_HOST": "index-host",
        }
    )

    settings.validate()
