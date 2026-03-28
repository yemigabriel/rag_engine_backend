from rq import Worker

from app.core.settings import settings
from app.infrastructure.queue.rq import get_redis_connection


def main():
    settings.validate_queue()
    worker = Worker(
        [settings.rq_queue_name],
        connection=get_redis_connection(),
    )
    worker.work()


if __name__ == "__main__":
    main()
