from functools import lru_cache

from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.core.settings import settings


@lru_cache
def get_redis_connection() -> Redis:
    settings.validate_queue()
    return Redis.from_url(settings.redis_url)


@lru_cache
def get_ingestion_queue() -> Queue:
    return Queue(
        name=settings.rq_queue_name,
        connection=get_redis_connection(),
        default_timeout=settings.rq_job_timeout,
    )


def fetch_job(job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=get_redis_connection())
    except NoSuchJobError:
        return None
