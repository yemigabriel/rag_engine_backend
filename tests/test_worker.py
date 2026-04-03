import importlib
import sys


def test_get_worker_class_defaults_to_simple_on_current_settings():
    sys.modules.pop("worker", None)
    sys.modules.pop("app.infrastructure.queue.rq", None)
    worker_module = importlib.import_module("worker")
    settings_module = importlib.import_module("app.core.settings")

    expected_names = {
        "worker": "Worker",
        "simple": "SimpleWorker",
        "spawn": "SpawnWorker",
    }

    assert (
        worker_module.get_worker_class().__name__
        == expected_names[settings_module.settings.rq_worker_class]
    )
