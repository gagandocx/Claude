"""
=============================================================
  NeuroX - Pipeline Architecture (Threading)

  Overlaps data fetch with model compute using ThreadPoolExecutor.
  This eliminates the serial bottleneck where I/O-bound data fetching
  blocks CPU-bound model inference.

  Architecture:
    Main loop submits data fetch tasks to the pipeline.
    While data is being fetched, the previous cycle's features
    can still be processed by models.

  Usage:
    pipeline = PipelineManager(PipelineConfig())
    future = pipeline.submit_data_fetch(fetcher.fetch_ohlcv, ...)
    # ... do other compute while fetch runs ...
    df = pipeline.get_result(future)
=============================================================
"""

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PipelineConfig

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Threading-based pipeline to overlap data fetch with model compute.

    Uses a ThreadPoolExecutor with configurable worker count to submit
    I/O-bound tasks (yfinance fetch, sentiment fetch, alt data fetch)
    in parallel with CPU-bound tasks (feature computation, model inference).

    Thread safety:
        - Each submitted task is independent (no shared mutable state)
        - Results are retrieved via Future objects
        - The pipeline does not modify any external state itself

    Example:
        pipeline = PipelineManager()

        # Submit multiple data fetches in parallel
        ohlcv_future = pipeline.submit_data_fetch(
            fetcher.fetch_ohlcv, interval="1h", period="3mo"
        )
        m1_future = pipeline.submit_data_fetch(
            fetcher.fetch_ohlcv, interval="1m", period="5d"
        )
        sentiment_future = pipeline.submit_task(
            sentiment.get_sentiment_score
        )

        # Retrieve results (blocks until available or timeout)
        df = pipeline.get_result(ohlcv_future)
        df_m1 = pipeline.get_result(m1_future)
        sent = pipeline.get_result(sentiment_future)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="NeuroX-Pipeline"
        )
        self._active_futures: List[Future] = []
        self._stats: Dict[str, float] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_fetch_time": 0.0,
        }
        logger.info(
            f"[Pipeline] Initialized with {self.config.max_workers} workers, "
            f"timeout={self.config.fetch_timeout_seconds}s"
        )

    def submit_data_fetch(self, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a data fetch task to the thread pool.

        Args:
            fn: Callable to execute (e.g., fetcher.fetch_ohlcv)
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Future object to retrieve the result later
        """
        future = self._executor.submit(self._timed_execute, fn, *args, **kwargs)
        self._active_futures.append(future)
        self._stats["total_tasks"] += 1
        return future

    def submit_task(self, fn: Callable, *args, **kwargs) -> Future:
        """
        Submit a generic task to the thread pool.

        Alias for submit_data_fetch, for semantic clarity when submitting
        non-fetch tasks (e.g., sentiment analysis, feature computation).
        """
        return self.submit_data_fetch(fn, *args, **kwargs)

    def get_result(self, future: Future, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a submitted task.

        Args:
            future: Future object from submit_data_fetch/submit_task
            timeout: Timeout in seconds. Defaults to config.fetch_timeout_seconds.

        Returns:
            Result of the callable, or None if it failed/timed out

        Raises:
            No exceptions raised - returns None on failure with a warning log
        """
        timeout = timeout or self.config.fetch_timeout_seconds
        try:
            result = future.result(timeout=timeout)
            self._stats["completed_tasks"] += 1
            return result
        except FuturesTimeout:
            logger.warning(
                f"[Pipeline] Task timed out after {timeout}s"
            )
            self._stats["failed_tasks"] += 1
            return None
        except Exception as e:
            logger.warning(f"[Pipeline] Task failed: {e}")
            self._stats["failed_tasks"] += 1
            return None

    def get_results(self, futures: List[Future],
                    timeout: Optional[float] = None) -> List[Any]:
        """
        Get results from multiple futures.

        Args:
            futures: List of Future objects
            timeout: Timeout per future in seconds

        Returns:
            List of results (None for failed/timed out tasks)
        """
        return [self.get_result(f, timeout) for f in futures]

    def submit_parallel_fetches(
        self,
        tasks: List[Tuple[Callable, tuple, dict]]
    ) -> List[Future]:
        """
        Submit multiple tasks in parallel.

        Args:
            tasks: List of (callable, args_tuple, kwargs_dict) tuples

        Returns:
            List of Future objects in the same order as tasks
        """
        futures = []
        for fn, args, kwargs in tasks:
            future = self.submit_data_fetch(fn, *args, **kwargs)
            futures.append(future)
        return futures

    def cleanup_futures(self) -> None:
        """Remove completed futures from the active list."""
        self._active_futures = [
            f for f in self._active_futures if not f.done()
        ]

    def get_stats(self) -> Dict[str, float]:
        """Return pipeline performance statistics."""
        return self._stats.copy()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        self._executor.shutdown(wait=wait)
        logger.info("[Pipeline] Shut down")

    def _timed_execute(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute a function and track timing."""
        start = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - start

            # Update rolling average fetch time
            prev_avg = self._stats["avg_fetch_time"]
            completed = self._stats["completed_tasks"] + 1
            self._stats["avg_fetch_time"] = (
                prev_avg + (elapsed - prev_avg) / completed
            )

            if elapsed > 5.0:
                logger.debug(
                    f"[Pipeline] Slow task ({elapsed:.1f}s): {fn.__name__}"
                )
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.debug(
                f"[Pipeline] Task {fn.__name__} failed after {elapsed:.1f}s: {e}"
            )
            raise
