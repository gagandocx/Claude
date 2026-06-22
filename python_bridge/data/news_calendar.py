"""
=============================================================
  Python ML Bridge - News Calendar Filter
  Prevents trading during high-impact economic events like
  NFP, FOMC, CPI, ECB rate decisions. Professional traders
  never hold positions through these releases due to extreme
  volatility and unpredictable slippage.

  Data Source:
    - ForexFactory JSON calendar (free, no auth required)
    - Cached locally with configurable refresh interval

  Usage:
    filter = NewsCalendarFilter()
    if filter.is_high_impact_window():
        logger.info("Skipping trade - high impact event window")
        return
=============================================================
"""

import json
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import NewsFilterConfig


class NewsCalendarFilter:
    """
    Filters trade signals around high-impact economic events.

    Professional risk management requires avoiding trades during
    major news releases. This class:
    1. Fetches the economic calendar from ForexFactory
    2. Identifies high-impact events (NFP, FOMC, CPI, etc.)
    3. Creates buffer zones (default 30min before/after)
    4. Gates all trade entries during these windows
    5. Caches calendar data to minimize API calls
    """

    def __init__(self, config: Optional[NewsFilterConfig] = None):
        self.config = config or NewsFilterConfig()
        self._calendar: List[Dict[str, Any]] = []
        self._last_refresh: Optional[datetime] = None
        self._cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.config.cache_file
        )

    def fetch_calendar(self) -> List[Dict[str, Any]]:
        """
        Fetch economic calendar from ForexFactory JSON API.

        Returns a list of event dictionaries with fields:
            - title: Event name (e.g., "Non-Farm Employment Change")
            - country: Currency code (e.g., "USD")
            - date: ISO datetime string
            - impact: "High", "Medium", "Low"
            - forecast: Expected value
            - previous: Previous value

        Returns:
            List of event dicts, empty list on failure
        """
        try:
            response = requests.get(
                self.config.calendar_url,
                timeout=10,
                headers={"User-Agent": "PythonMLBridge/1.0"}
            )
            response.raise_for_status()
            events = response.json()

            # Normalize and filter events
            normalized = []
            for event in events:
                normalized.append({
                    "title": event.get("title", ""),
                    "country": event.get("country", ""),
                    "date": event.get("date", ""),
                    "impact": event.get("impact", "Low"),
                    "forecast": event.get("forecast", ""),
                    "previous": event.get("previous", ""),
                })

            self._calendar = normalized
            self._last_refresh = datetime.now(timezone.utc)
            self._save_cache(normalized)
            return normalized

        except Exception as e:
            print(f"[NewsCalendar] Error fetching calendar: {e}")
            # Try loading from cache
            cached = self._load_cache()
            if cached:
                self._calendar = cached
                return cached
            return []

    def _save_cache(self, events: List[Dict[str, Any]]) -> None:
        """Save calendar data to local JSON cache."""
        try:
            cache_data = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "events": events
            }
            with open(self._cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"[NewsCalendar] Error saving cache: {e}")

    def _load_cache(self) -> List[Dict[str, Any]]:
        """Load calendar data from local cache if available and fresh."""
        try:
            if not os.path.exists(self._cache_path):
                return []

            with open(self._cache_path, "r") as f:
                cache_data = json.load(f)

            fetched_at = datetime.fromisoformat(cache_data["fetched_at"])
            age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600

            # Only use cache if within refresh interval
            if age_hours < self.config.refresh_interval_hours * 2:
                return cache_data.get("events", [])

            return []
        except Exception:
            return []

    def _needs_refresh(self) -> bool:
        """Check if calendar data needs refreshing."""
        if self._last_refresh is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_refresh).total_seconds()
        return elapsed > self.config.refresh_interval_hours * 3600

    def _ensure_calendar_loaded(self) -> None:
        """Ensure calendar is loaded, refresh if needed."""
        if self._needs_refresh():
            self.fetch_calendar()
        elif not self._calendar:
            # Try cache first
            cached = self._load_cache()
            if cached:
                self._calendar = cached
            else:
                self.fetch_calendar()

    def _is_high_impact_event(self, event: Dict[str, Any]) -> bool:
        """
        Determine if an event is high-impact and relevant.

        Checks:
        1. Impact level is "High"
        2. Event title matches known high-impact event names
        3. Event currency is in monitored currencies list

        In strict mode, medium-impact USD events are also flagged
        since USD events directly affect gold prices.

        Args:
            event: Event dictionary from calendar

        Returns:
            True if event is high-impact and should gate trading
        """
        # Check impact level
        impact = event.get("impact", "").lower()
        country = event.get("country", "")

        # Check if currency is monitored
        if country not in self.config.monitored_currencies:
            return False

        if impact == "high":
            # All high-impact events on monitored currencies are flagged
            return True

        if impact == "medium" and self.config.strict_mode:
            # In strict mode, medium-impact USD events are also flagged
            # (USD moves directly impact gold pricing)
            if country == "USD":
                return True

        # Not high or medium-strict, check title match as fallback
        if impact not in ("high", "medium"):
            return False

        title = event.get("title", "").upper()
        for event_name in self.config.high_impact_events:
            if event_name.upper() in title:
                return True

        return False

    def _parse_event_time(self, event: Dict[str, Any]) -> Optional[datetime]:
        """Parse event datetime from calendar data."""
        date_str = event.get("date", "")
        if not date_str:
            return None

        try:
            # ForexFactory format: "2024-01-05T13:30:00-05:00" or similar
            dt = datetime.fromisoformat(date_str)
            # Ensure timezone aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            try:
                # Try alternate formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                            "%b %d, %Y %H:%M"]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            except Exception:
                pass
            return None

    def is_high_impact_window(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if the given time falls within a high-impact event window.

        A window extends from (event_time - minutes_before) to
        (event_time + minutes_after). During this window, no new
        trades should be opened.

        NOTE: This method may trigger a blocking network fetch if calendar
        data is stale. For the hot signal loop, prefer is_high_impact_window_cached()
        which only reads from already-loaded data.

        Args:
            check_time: Time to check (defaults to current UTC time)

        Returns:
            True if currently in a high-impact event window
        """
        self._ensure_calendar_loaded()
        return self._check_window(check_time)

    def is_high_impact_window_cached(self, check_time: Optional[datetime] = None) -> bool:
        """
        Non-blocking check: only uses already-loaded calendar data.

        This method never makes a network call. It reads from whatever
        calendar data is currently in memory (populated by background
        refresh or a previous fetch_calendar() call). If no data is loaded,
        it attempts to read from the local file cache only.

        Use this in the main signal loop to avoid blocking on HTTP requests.
        Schedule background refreshes separately.

        Args:
            check_time: Time to check (defaults to current UTC time)

        Returns:
            True if currently in a high-impact event window
        """
        # If no calendar data in memory, try local cache file (no network)
        if not self._calendar:
            cached = self._load_cache()
            if cached:
                self._calendar = cached
        return self._check_window(check_time)

    def _check_window(self, check_time: Optional[datetime] = None) -> bool:
        """Internal: check if time is in a high-impact event window."""

        if check_time is None:
            check_time = datetime.now(timezone.utc)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)

        buffer_before = timedelta(minutes=self.config.minutes_before)
        buffer_after = timedelta(minutes=self.config.minutes_after)

        for event in self._calendar:
            if not self._is_high_impact_event(event):
                continue

            event_time = self._parse_event_time(event)
            if event_time is None:
                continue

            window_start = event_time - buffer_before
            window_end = event_time + buffer_after

            if window_start <= check_time <= window_end:
                return True

        return False

    def get_upcoming_events(self, hours_ahead: int = 24,
                            check_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get high-impact events in the next N hours.

        Used for dashboard display and pre-session planning.

        NOTE: This method may trigger a blocking fetch via _ensure_calendar_loaded().
        For the hot signal loop, prefer get_upcoming_events_cached().

        Args:
            hours_ahead: How many hours ahead to look
            check_time: Reference time (defaults to now UTC)

        Returns:
            List of upcoming high-impact events with parsed times
        """
        self._ensure_calendar_loaded()
        return self._get_upcoming_events_internal(hours_ahead, check_time)

    def get_upcoming_events_cached(self, hours_ahead: int = 24,
                                   check_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Non-blocking variant of get_upcoming_events().

        Only reads from already-loaded calendar data in memory. If no data is
        loaded, attempts to read from the local file cache (no network call).
        Use this in the main signal loop to avoid blocking on HTTP requests.

        Args:
            hours_ahead: How many hours ahead to look
            check_time: Reference time (defaults to now UTC)

        Returns:
            List of upcoming high-impact events with parsed times
        """
        # If no calendar data in memory, try local cache file only (no network)
        if not self._calendar:
            cached = self._load_cache()
            if cached:
                self._calendar = cached
        return self._get_upcoming_events_internal(hours_ahead, check_time)

    def _get_upcoming_events_internal(self, hours_ahead: int = 24,
                                      check_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Internal implementation for upcoming events lookup."""

        if check_time is None:
            check_time = datetime.now(timezone.utc)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)

        lookahead = timedelta(hours=hours_ahead)
        upcoming = []

        for event in self._calendar:
            if not self._is_high_impact_event(event):
                continue

            event_time = self._parse_event_time(event)
            if event_time is None:
                continue

            if check_time <= event_time <= check_time + lookahead:
                upcoming.append({
                    "title": event.get("title", ""),
                    "country": event.get("country", ""),
                    "time": event_time.isoformat(),
                    "impact": event.get("impact", ""),
                    "minutes_until": int((event_time - check_time).total_seconds() / 60),
                })

        # Sort by time
        upcoming.sort(key=lambda x: x["minutes_until"])
        return upcoming

    def get_next_event_info(self, check_time: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about the next high-impact event.

        Returns:
            Dict with event info and time until event, or None
        """
        upcoming = self.get_upcoming_events(hours_ahead=48, check_time=check_time)
        return upcoming[0] if upcoming else None

    def get_safe_trading_windows(self, date: Optional[datetime] = None,
                                 hours: int = 24) -> List[Dict[str, str]]:
        """
        Get safe trading windows for the day (periods without high-impact events).

        Useful for planning session times and understanding available
        trading hours.

        Args:
            date: Start date (defaults to today UTC)
            hours: Hours to analyze

        Returns:
            List of safe window dicts with 'start' and 'end' ISO strings
        """
        self._ensure_calendar_loaded()

        if date is None:
            date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        elif date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)

        end_time = date + timedelta(hours=hours)

        # Collect all blocked windows
        blocked = []
        buffer_before = timedelta(minutes=self.config.minutes_before)
        buffer_after = timedelta(minutes=self.config.minutes_after)

        for event in self._calendar:
            if not self._is_high_impact_event(event):
                continue

            event_time = self._parse_event_time(event)
            if event_time is None:
                continue

            if date <= event_time <= end_time:
                blocked.append((event_time - buffer_before, event_time + buffer_after))

        # Merge overlapping blocked windows
        blocked.sort(key=lambda x: x[0])
        merged = []
        for start, end in blocked:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Compute safe windows between blocked periods
        safe = []
        current = date
        for block_start, block_end in merged:
            if current < block_start:
                safe.append({
                    "start": current.isoformat(),
                    "end": block_start.isoformat()
                })
            current = block_end

        if current < end_time:
            safe.append({
                "start": current.isoformat(),
                "end": end_time.isoformat()
            })

        return safe
