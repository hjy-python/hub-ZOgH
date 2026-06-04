import random
import threading
import time


class RateLimiter:
    def __init__(self, calls_per_minute):
        self.interval = 60.0 / calls_per_minute if calls_per_minute else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if wait:
            time.sleep(wait)


def retry_call(fn, retries, base_sleep, max_sleep, retry_statuses, jitter):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status in retry_statuses or status is None
            if attempt >= retries or not retryable:
                raise
            retry_after = getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
            if retry_after:
                sleep = float(retry_after)
            else:
                sleep = min(max_sleep, base_sleep * (2 ** attempt)) + random.uniform(0, jitter)
            time.sleep(sleep)
