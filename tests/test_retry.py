"""Unit tests for retry decorator."""

import time
from unittest.mock import patch

import pytest

from src.utils.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff decorator."""

    def test_succeeds_on_first_try(self):
        """Function that succeeds immediately should not retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Function should be retried up to max_retries times."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        assert fn() == "ok"
        assert call_count == 3  # 1 initial + 2 retries

    def test_raises_after_max_retries(self):
        """Function should raise after exhausting retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fail")

        with pytest.raises(ValueError, match="always fail"):
            fn()

        assert call_count == 3  # 1 initial + 2 retries

    def test_only_catches_specified_exceptions(self):
        """Unspecified exceptions should not be caught."""
        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def fn():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            fn()

    def test_exponential_backoff_timing(self):
        """Verify delays increase exponentially (roughly)."""
        call_times = []

        @retry_with_backoff(max_retries=2, base_delay=0.05, jitter=False)
        def fn():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("fail")
            return "ok"

        fn()
        assert len(call_times) == 3
        # Second delay should be larger than first
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        assert delay2 > delay1 * 1.2  # Exponential growth (relaxed for CI timing)

    def test_preserves_function_name(self):
        """Decorator should preserve the function's __name__."""
        @retry_with_backoff()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestRetryWithBackoffEdgeCases:
    """Edge case tests for retry decorator."""

    def test_zero_retries(self):
        """With max_retries=0, function should only be called once."""
        call_count = 0

        @retry_with_backoff(max_retries=0, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            fn()

        assert call_count == 1

    def test_passes_args_and_kwargs(self):
        """Arguments should be passed through to the decorated function."""
        @retry_with_backoff(max_retries=0)
        def fn(a, b, c=None):
            return (a, b, c)

        assert fn(1, 2, c=3) == (1, 2, 3)
