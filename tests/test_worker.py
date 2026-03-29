import importlib
import sys


def test_get_worker_class_defaults_to_simple_on_current_settings():
    sys.modules.pop("worker", None)
    sys.modules.pop("app.infrastructure.queue.rq", None)
    worker_module = importlib.import_module("worker")

    assert worker_module.get_worker_class().__name__ == "SimpleWorker"
