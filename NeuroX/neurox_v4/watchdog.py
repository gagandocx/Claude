"""
=============================================================
  NeuroX - Process Watchdog / Supervisor
  Standalone supervisor that launches main.py as a subprocess,
  monitors its health (process alive check, stdout heartbeat),
  and auto-restarts on crash with exponential backoff.

  Usage:
    python watchdog.py --live
    python watchdog.py --paper
    python watchdog.py --live --interval 5
=============================================================
"""

import os
import sys
import time
import signal
import logging
import subprocess
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class WatchdogConfig:
    """Watchdog supervisor configuration."""

    def __init__(self):
        # Backoff schedule (seconds between restart attempts)
        self.backoff_schedule = [5, 10, 30, 60]
        # Max consecutive restarts before giving up
        self.max_consecutive_restarts = 20
        # Reset backoff counter after this many seconds of healthy running
        self.healthy_reset_seconds = 300
        # Heartbeat timeout: if subprocess produces no output for this long, restart
        self.heartbeat_timeout_seconds = 120
        # Log directory
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
def setup_watchdog_logging(config: WatchdogConfig) -> logging.Logger:
    """Configure logging for the watchdog."""
    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(
        config.log_dir, f"watchdog_{datetime.now().strftime('%Y%m%d')}.log"
    )

    logger = logging.getLogger("NeuroXWatchdog")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [WATCHDOG] %(levelname)s %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# ─────────────────────────────────────────────
#  WATCHDOG CLASS
# ─────────────────────────────────────────────
class ProcessWatchdog:
    """
    Process supervisor for NeuroX main.py.

    Responsibilities:
      - Launch main.py as a subprocess
      - Monitor process health (alive check)
      - Detect crashes (process exit with non-zero code)
      - Auto-restart with exponential backoff
      - Log all restarts and crash details
      - Graceful shutdown on SIGINT/SIGTERM
    """

    def __init__(self, config: Optional[WatchdogConfig] = None,
                 extra_args: Optional[list] = None):
        self.config = config or WatchdogConfig()
        self.logger = setup_watchdog_logging(self.config)
        self.extra_args = extra_args or []
        self._running = False
        self._process: Optional[subprocess.Popen] = None
        self._restart_count = 0
        self._total_restarts = 0
        self._last_healthy_time = 0.0
        self._start_time = 0.0

    def _get_backoff_seconds(self) -> int:
        """Get current backoff delay based on restart count."""
        schedule = self.config.backoff_schedule
        idx = min(self._restart_count, len(schedule) - 1)
        return schedule[idx]

    def _launch_process(self) -> subprocess.Popen:
        """Launch main.py as a subprocess."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = os.path.join(script_dir, "main.py")

        cmd = [sys.executable, "-u", main_script] + self.extra_args

        self.logger.info(f"Launching: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=script_dir,
        )
        return process

    def _monitor_process(self) -> int:
        """
        Monitor the running subprocess.

        Reads stdout line-by-line and forwards it. Detects process
        termination and returns the exit code.

        Returns:
            Process exit code (0 = clean, non-zero = crash)
        """
        last_output_time = time.time()

        while self._running:
            # Check if process is still alive
            retcode = self._process.poll()
            if retcode is not None:
                # Process terminated - read remaining output
                remaining = self._process.stdout.read()
                if remaining:
                    for line in remaining.strip().split("\n"):
                        if line.strip():
                            print(line)
                return retcode

            # Read output (non-blocking via readline with timeout behavior)
            try:
                line = self._process.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    last_output_time = time.time()
                else:
                    # No output available, brief sleep
                    time.sleep(0.1)
            except (IOError, OSError):
                time.sleep(0.1)

            # Check heartbeat timeout
            elapsed_silent = time.time() - last_output_time
            if elapsed_silent > self.config.heartbeat_timeout_seconds:
                self.logger.warning(
                    f"No output for {elapsed_silent:.0f}s "
                    f"(timeout={self.config.heartbeat_timeout_seconds}s). "
                    f"Process may be hung."
                )
                # Kill hung process
                self._kill_process()
                return -1

        # Watchdog stopping - return clean code
        return 0

    def _kill_process(self):
        """Forcefully terminate the subprocess."""
        if self._process and self._process.poll() is None:
            self.logger.info("Terminating subprocess...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Process did not terminate, killing...")
                self._process.kill()
                self._process.wait(timeout=5)

    def run(self):
        """
        Main watchdog loop. Launches, monitors, and restarts the process.
        Blocks until stopped via signal or max restarts exceeded.
        """
        self._running = True
        self._start_time = time.time()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self.logger.info("=" * 55)
        self.logger.info("  NeuroX Watchdog Starting")
        self.logger.info(f"  Max restarts: {self.config.max_consecutive_restarts}")
        self.logger.info(f"  Backoff schedule: {self.config.backoff_schedule}s")
        self.logger.info(f"  Args: {self.extra_args}")
        self.logger.info("=" * 55)

        while self._running:
            # Check max restart limit
            if self._restart_count >= self.config.max_consecutive_restarts:
                self.logger.error(
                    f"Max consecutive restarts ({self.config.max_consecutive_restarts}) "
                    f"reached. Stopping watchdog."
                )
                break

            # Launch process
            try:
                self._process = self._launch_process()
                self._last_healthy_time = time.time()
            except Exception as e:
                self.logger.error(f"Failed to launch process: {e}")
                backoff = self._get_backoff_seconds()
                self.logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)
                self._restart_count += 1
                continue

            # Monitor process
            exit_code = self._monitor_process()

            if not self._running:
                # Clean shutdown requested
                self._kill_process()
                break

            # Process crashed/exited
            run_duration = time.time() - self._last_healthy_time
            self._total_restarts += 1

            # Reset backoff if process ran long enough (was healthy)
            if run_duration >= self.config.healthy_reset_seconds:
                self._restart_count = 0
                self.logger.info(
                    f"Process ran for {run_duration:.0f}s before crash. "
                    f"Backoff reset."
                )
            else:
                self._restart_count += 1

            self.logger.warning(
                f"Process exited with code {exit_code} "
                f"after {run_duration:.1f}s. "
                f"Restart #{self._total_restarts} "
                f"(consecutive: {self._restart_count})"
            )

            # Apply backoff delay
            backoff = self._get_backoff_seconds()
            self.logger.info(f"Restarting in {backoff}s...")
            # Sleep in small increments to allow interrupt
            for _ in range(backoff * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        self.logger.info("Watchdog stopped.")
        total_runtime = time.time() - self._start_time
        self.logger.info(
            f"Total runtime: {total_runtime:.0f}s, "
            f"Total restarts: {self._total_restarts}"
        )

    def stop(self):
        """Stop the watchdog and terminate the subprocess."""
        self._running = False
        self._kill_process()

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    """Watchdog entry point."""
    print("=" * 60)
    print("  NeuroX Watchdog - Process Supervisor")
    print("  Auto-restart on crash with exponential backoff")
    print("=" * 60)
    print()

    # Pass through all command-line args to main.py
    extra_args = sys.argv[1:]

    config = WatchdogConfig()
    watchdog = ProcessWatchdog(config=config, extra_args=extra_args)
    watchdog.run()


if __name__ == "__main__":
    main()
