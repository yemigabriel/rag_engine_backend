from rq import SimpleWorker, SpawnWorker, Worker

from app.core.settings import settings
from app.infrastructure.queue.rq import get_redis_connection


def get_worker_class():
    worker_classes = {
        "worker": Worker,
        "simple": SimpleWorker,
        "spawn": SpawnWorker,
    }
    return worker_classes[settings.rq_worker_class]


def main():
    settings.validate()
    settings.validate_queue()
    worker_class = get_worker_class()
    worker = worker_class(
        [settings.rq_queue_name],
        connection=get_redis_connection(),
    )
    worker.work()


if __name__ == "__main__":
    main()
