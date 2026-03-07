"""Retry decorator with exponential backoff for resilient API calls."""

import functools
import random
import time
from typing import Callable, Tuple, Type

from src.logger import get_logger

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to delay to prevent thundering herd (default: True)
        exceptions: Tuple of exception types to catch and retry (default: all)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(requests.Timeout, requests.ConnectionError))
        def fetch_data():
            return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            str(e),
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay,
                    )

                    # Add jitter (0-25% of delay)
                    if jitter:
                        delay = delay * (1 + random.uniform(0, 0.25))

                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.2fs...",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        str(e),
                        delay,
                    )

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper

    return decorator


def retry_on_rate_limit(
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Callable:
    """
    Specialized retry decorator for API rate limiting.

    Uses longer delays and more retries appropriate for rate-limited APIs.

    Args:
        max_retries: Maximum retry attempts (default: 5)
        base_delay: Initial delay in seconds (default: 2.0)

    Returns:
        Decorated function with rate-limit aware retry logic
    """
    # Import here to avoid circular imports
    import openai

    rate_limit_exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
    )

    return retry_with_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
        exceptions=rate_limit_exceptions,
    )
