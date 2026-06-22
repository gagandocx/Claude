"""
=============================================================
  Python ML Bridge - News Calendar Filter Tests
  Tests for NewsCalendarFilter:
    - High-impact event detection
    - Event window gating logic
    - Calendar fetching and caching
    - Upcoming events query
    - Safe trading window computation
=============================================================
"""

import pytest
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import NewsFilterConfig
from data.news_calendar import NewsCalendarFilter


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────
@pytest.fixture
def news_config():
    """News filter config for testing."""
    return NewsFilterConfig(
        minutes_before=30,
        minutes_after=30,
        strict_mode=False,
    )


@pytest.fixture
def strict_config():
    """Strict mode news filter config."""
    return NewsFilterConfig(
        minutes_before=30,
        minutes_after=30,
        strict_mode=True,
    )


@pytest.fixture
def sample_calendar():
    """Sample economic calendar events."""
    base_time = datetime(2024, 6, 7, 13, 30, tzinfo=timezone.utc)
    return [
        {
            "title": "Non-Farm Employment Change",
            "country": "USD",
            "date": base_time.isoformat(),
            "impact": "High",
            "forecast": "180K",
            "previous": "175K",
        },
        {
            "title": "FOMC Statement",
            "country": "USD",
            "date": (base_time + timedelta(days=5, hours=5)).isoformat(),
            "impact": "High",
            "forecast": "",
            "previous": "",
        },
        {
            "title": "CPI m/m",
            "country": "USD",
            "date": (base_time + timedelta(days=3)).isoformat(),
            "impact": "High",
            "forecast": "0.3%",
            "previous": "0.4%",
        },
        {
            "title": "ECB Interest Rate Decision",
            "country": "EUR",
            "date": (base_time + timedelta(days=2)).isoformat(),
            "impact": "High",
            "forecast": "4.25%",
            "previous": "4.50%",
        },
        {
            "title": "Unemployment Claims",
            "country": "USD",
            "date": (base_time + timedelta(hours=1)).isoformat(),
            "impact": "Medium",
            "forecast": "220K",
            "previous": "215K",
        },
        {
            "title": "Trade Balance",
            "country": "AUD",
            "date": (base_time + timedelta(days=1)).isoformat(),
            "impact": "Low",
            "forecast": "",
            "previous": "",
        },
        {
            "title": "BOE Interest Rate Decision",
            "country": "GBP",
            "date": (base_time + timedelta(days=4)).isoformat(),
            "impact": "High",
            "forecast": "5.00%",
            "previous": "5.25%",
        },
    ]


@pytest.fixture
def filter_with_events(news_config, sample_calendar):
    """Create a NewsCalendarFilter pre-loaded with sample events."""
    nf = NewsCalendarFilter(config=news_config)
    nf._calendar = sample_calendar
    nf._last_refresh = datetime.now(timezone.utc)
    return nf


@pytest.fixture
def strict_filter_with_events(strict_config, sample_calendar):
    """Create a strict-mode NewsCalendarFilter pre-loaded with sample events."""
    nf = NewsCalendarFilter(config=strict_config)
    nf._calendar = sample_calendar
    nf._last_refresh = datetime.now(timezone.utc)
    return nf


# ─────────────────────────────────────────────
#  HIGH IMPACT EVENT DETECTION TESTS
# ─────────────────────────────────────────────
class TestHighImpactDetection:
    """Tests for _is_high_impact_event method."""

    def test_nfp_is_high_impact(self, filter_with_events, sample_calendar):
        """Test that NFP is identified as high-impact."""
        nfp_event = sample_calendar[0]
        assert filter_with_events._is_high_impact_event(nfp_event)

    def test_fomc_is_high_impact(self, filter_with_events, sample_calendar):
        """Test that FOMC is identified as high-impact."""
        fomc_event = sample_calendar[1]
        assert filter_with_events._is_high_impact_event(fomc_event)

    def test_cpi_is_high_impact(self, filter_with_events, sample_calendar):
        """Test that CPI is identified as high-impact."""
        cpi_event = sample_calendar[2]
        assert filter_with_events._is_high_impact_event(cpi_event)

    def test_ecb_is_high_impact(self, filter_with_events, sample_calendar):
        """Test that ECB rate decision is high-impact."""
        ecb_event = sample_calendar[3]
        assert filter_with_events._is_high_impact_event(ecb_event)

    def test_boe_is_high_impact(self, filter_with_events, sample_calendar):
        """Test that BOE rate decision is high-impact."""
        boe_event = sample_calendar[6]
        assert filter_with_events._is_high_impact_event(boe_event)

    def test_low_impact_not_flagged(self, filter_with_events, sample_calendar):
        """Test that low-impact events are not flagged."""
        low_event = sample_calendar[5]  # AUD Trade Balance
        assert not filter_with_events._is_high_impact_event(low_event)

    def test_medium_impact_not_flagged_non_strict(self, filter_with_events, sample_calendar):
        """Test that medium-impact events are not flagged in non-strict mode."""
        medium_event = sample_calendar[4]  # Unemployment Claims
        assert not filter_with_events._is_high_impact_event(medium_event)

    def test_medium_impact_usd_flagged_strict(self, strict_filter_with_events, sample_calendar):
        """Test that medium-impact USD events ARE flagged in strict mode."""
        medium_event = sample_calendar[4]  # Unemployment Claims (USD, Medium)
        assert strict_filter_with_events._is_high_impact_event(medium_event)

    def test_unmonitored_currency_not_flagged(self, filter_with_events):
        """Test that events for unmonitored currencies are not flagged."""
        event = {
            "title": "Interest Rate Decision",
            "country": "NZD",
            "date": "2024-06-07T03:00:00+00:00",
            "impact": "High",
        }
        assert not filter_with_events._is_high_impact_event(event)


# ─────────────────────────────────────────────
#  EVENT WINDOW GATING TESTS
# ─────────────────────────────────────────────
class TestEventWindow:
    """Tests for is_high_impact_window method."""

    def test_during_nfp_window(self, filter_with_events, sample_calendar):
        """Test that time during NFP window is correctly identified."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 10 minutes before NFP
        check_time = nfp_time - timedelta(minutes=10)
        assert filter_with_events.is_high_impact_window(check_time)

    def test_exactly_at_event_time(self, filter_with_events, sample_calendar):
        """Test that exactly at event time is in window."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        assert filter_with_events.is_high_impact_window(nfp_time)

    def test_after_event_in_buffer(self, filter_with_events, sample_calendar):
        """Test that time after event but within buffer is in window."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 15 minutes after NFP (within 30-min buffer)
        check_time = nfp_time + timedelta(minutes=15)
        assert filter_with_events.is_high_impact_window(check_time)

    def test_before_buffer_window(self, filter_with_events, sample_calendar):
        """Test that time well before event is NOT in window."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 2 hours before NFP
        check_time = nfp_time - timedelta(hours=2)
        assert not filter_with_events.is_high_impact_window(check_time)

    def test_after_buffer_window(self, filter_with_events, sample_calendar):
        """Test that time well after event is NOT in window."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 2 hours after NFP
        check_time = nfp_time + timedelta(hours=2)
        assert not filter_with_events.is_high_impact_window(check_time)

    def test_at_buffer_boundary_before(self, filter_with_events, sample_calendar):
        """Test at exact boundary of before-buffer."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # Exactly 30 minutes before (should be in window)
        check_time = nfp_time - timedelta(minutes=30)
        assert filter_with_events.is_high_impact_window(check_time)

    def test_at_buffer_boundary_after(self, filter_with_events, sample_calendar):
        """Test at exact boundary of after-buffer."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # Exactly 30 minutes after (should be in window)
        check_time = nfp_time + timedelta(minutes=30)
        assert filter_with_events.is_high_impact_window(check_time)

    def test_just_outside_before_buffer(self, filter_with_events, sample_calendar):
        """Test just outside the before-buffer (should be safe to trade)."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 31 minutes before
        check_time = nfp_time - timedelta(minutes=31)
        assert not filter_with_events.is_high_impact_window(check_time)

    @patch("data.news_calendar.requests.get")
    def test_no_events_always_safe(self, mock_get, news_config):
        """Test that empty calendar means always safe to trade."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        nf = NewsCalendarFilter(config=news_config)
        nf._calendar = []
        nf._last_refresh = datetime.now(timezone.utc)
        # Use far-future time to avoid any event match
        far_future = datetime(2099, 6, 15, 12, 0, tzinfo=timezone.utc)
        assert not nf.is_high_impact_window(far_future)

    def test_naive_datetime_handled(self, filter_with_events, sample_calendar):
        """Test that naive datetime (no timezone) is handled correctly."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # Pass naive datetime (timezone stripped)
        check_time = nfp_time.replace(tzinfo=None)
        # Should still work (adds UTC timezone internally)
        result = filter_with_events.is_high_impact_window(check_time)
        assert result  # Should be in window


# ─────────────────────────────────────────────
#  UPCOMING EVENTS TESTS
# ─────────────────────────────────────────────
class TestUpcomingEvents:
    """Tests for get_upcoming_events method."""

    def test_get_upcoming_events_returns_list(self, filter_with_events, sample_calendar):
        """Test that upcoming events returns a list."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # Check from 2 hours before NFP
        check_time = nfp_time - timedelta(hours=2)
        upcoming = filter_with_events.get_upcoming_events(
            hours_ahead=24, check_time=check_time
        )
        assert isinstance(upcoming, list)
        assert len(upcoming) > 0

    def test_upcoming_events_sorted_by_time(self, filter_with_events, sample_calendar):
        """Test that upcoming events are sorted by time (nearest first)."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        check_time = nfp_time - timedelta(hours=5)
        upcoming = filter_with_events.get_upcoming_events(
            hours_ahead=48, check_time=check_time
        )
        if len(upcoming) > 1:
            for i in range(len(upcoming) - 1):
                assert upcoming[i]["minutes_until"] <= upcoming[i + 1]["minutes_until"]

    def test_upcoming_events_contains_nfp(self, filter_with_events, sample_calendar):
        """Test that NFP appears in upcoming events."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        check_time = nfp_time - timedelta(hours=3)
        upcoming = filter_with_events.get_upcoming_events(
            hours_ahead=12, check_time=check_time
        )
        titles = [e["title"] for e in upcoming]
        assert any("Non-Farm" in t for t in titles)

    def test_upcoming_events_empty_for_past(self, filter_with_events, sample_calendar):
        """Test that past events don't appear in upcoming."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # Check from well after all events
        check_time = nfp_time + timedelta(days=30)
        upcoming = filter_with_events.get_upcoming_events(
            hours_ahead=24, check_time=check_time
        )
        assert len(upcoming) == 0

    def test_get_next_event_info(self, filter_with_events, sample_calendar):
        """Test get_next_event_info returns nearest event."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        check_time = nfp_time - timedelta(hours=2)
        next_event = filter_with_events.get_next_event_info(check_time)
        assert next_event is not None
        assert "title" in next_event
        assert "minutes_until" in next_event

    @patch("data.news_calendar.requests.get")
    def test_get_next_event_info_none_when_empty(self, mock_get, news_config):
        """Test get_next_event_info returns None with empty calendar."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        nf = NewsCalendarFilter(config=news_config)
        nf._calendar = []
        nf._last_refresh = datetime.now(timezone.utc)
        # Use far-future time so no events match
        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        result = nf.get_next_event_info(check_time=far_future)
        assert result is None


# ─────────────────────────────────────────────
#  CALENDAR FETCHING TESTS
# ─────────────────────────────────────────────
class TestCalendarFetching:
    """Tests for calendar fetching and caching."""

    @patch("data.news_calendar.requests.get")
    def test_fetch_calendar_success(self, mock_get, news_config, sample_calendar):
        """Test successful calendar fetch from API."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_calendar
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        nf = NewsCalendarFilter(config=news_config)
        # Use temp file for cache
        nf._cache_path = os.path.join(tempfile.gettempdir(), "test_news_cache.json")
        events = nf.fetch_calendar()
        assert len(events) == 7
        assert nf._last_refresh is not None

    @patch("data.news_calendar.requests.get")
    def test_fetch_calendar_network_error(self, mock_get, news_config):
        """Test calendar fetch falls back to cache on network error."""
        mock_get.side_effect = Exception("Connection timeout")

        nf = NewsCalendarFilter(config=news_config)
        nf._cache_path = os.path.join(tempfile.gettempdir(), "test_news_no_cache.json")
        # Remove any existing cache
        if os.path.exists(nf._cache_path):
            os.remove(nf._cache_path)

        events = nf.fetch_calendar()
        assert events == []  # No cache available either

    @patch("data.news_calendar.requests.get")
    def test_cache_save_and_load(self, mock_get, news_config, sample_calendar):
        """Test that calendar data is cached and loadable."""
        mock_response = MagicMock()
        mock_response.json.return_value = sample_calendar
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        cache_path = os.path.join(tempfile.gettempdir(), "test_news_cache_rw.json")

        nf = NewsCalendarFilter(config=news_config)
        nf._cache_path = cache_path
        nf.fetch_calendar()

        # Verify cache file exists
        assert os.path.exists(cache_path)

        # Load cache
        loaded = nf._load_cache()
        assert len(loaded) == 7

        # Clean up
        os.remove(cache_path)

    def test_needs_refresh_on_init(self, news_config):
        """Test that refresh is needed on fresh initialization."""
        nf = NewsCalendarFilter(config=news_config)
        assert nf._needs_refresh()

    def test_no_refresh_needed_after_fetch(self, news_config, sample_calendar):
        """Test that refresh is not needed right after fetch."""
        nf = NewsCalendarFilter(config=news_config)
        nf._calendar = sample_calendar
        nf._last_refresh = datetime.now(timezone.utc)
        assert not nf._needs_refresh()


# ─────────────────────────────────────────────
#  SAFE TRADING WINDOWS TESTS
# ─────────────────────────────────────────────
class TestSafeTradingWindows:
    """Tests for safe trading window computation."""

    def test_safe_windows_with_events(self, filter_with_events, sample_calendar):
        """Test that safe windows are computed around events."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        start_of_day = nfp_time.replace(hour=0, minute=0, second=0)
        safe = filter_with_events.get_safe_trading_windows(date=start_of_day)
        assert isinstance(safe, list)
        # Should have at least one safe window (before NFP)
        assert len(safe) > 0

    @patch("data.news_calendar.requests.get")
    def test_safe_windows_no_events(self, mock_get, news_config):
        """Test that no events means entire day is safe."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        nf = NewsCalendarFilter(config=news_config)
        nf._calendar = []
        nf._last_refresh = datetime.now(timezone.utc)
        start = datetime(2099, 6, 10, 0, 0, tzinfo=timezone.utc)
        safe = nf.get_safe_trading_windows(date=start, hours=24)
        assert len(safe) == 1
        # Single window spanning entire period


# ─────────────────────────────────────────────
#  CONFIGURATION TESTS
# ─────────────────────────────────────────────
class TestNewsFilterConfig:
    """Tests for NewsFilterConfig defaults and customization."""

    def test_default_high_impact_events(self):
        """Test default list includes key events."""
        config = NewsFilterConfig()
        assert "NFP" in config.high_impact_events
        assert "FOMC" in config.high_impact_events
        assert "CPI" in config.high_impact_events
        assert "ECB" in config.high_impact_events
        assert "BOE" in config.high_impact_events
        assert "BOJ" in config.high_impact_events

    def test_default_buffer_times(self):
        """Test default buffer times."""
        config = NewsFilterConfig()
        assert config.minutes_before == 30
        assert config.minutes_after == 30

    def test_default_monitored_currencies(self):
        """Test default monitored currencies."""
        config = NewsFilterConfig()
        assert "USD" in config.monitored_currencies
        assert "EUR" in config.monitored_currencies
        assert "GBP" in config.monitored_currencies
        assert "JPY" in config.monitored_currencies

    def test_custom_buffer(self):
        """Test custom buffer configuration."""
        config = NewsFilterConfig(minutes_before=60, minutes_after=45)
        assert config.minutes_before == 60
        assert config.minutes_after == 45

    def test_strict_mode_default(self):
        """Test strict mode default."""
        config = NewsFilterConfig()
        assert config.strict_mode is True

    def test_post_news_config_defaults(self):
        """Test post-news volatility check config defaults."""
        config = NewsFilterConfig()
        assert config.post_news_check_interval == 60
        assert config.post_news_volatility_threshold == 2.0
        assert config.post_news_min_wait == 2


# ─────────────────────────────────────────────
#  SHOULD_BLOCK_TRADING TESTS (POST-NEWS VOLATILITY)
# ─────────────────────────────────────────────
class TestShouldBlockTrading:
    """Tests for should_block_trading method with post-news volatility check."""

    def test_blocks_during_pre_event_window(self, filter_with_events, sample_calendar):
        """Test that should_block_trading blocks during pre-event window."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 10 minutes before NFP
        check_time = nfp_time - timedelta(minutes=10)
        result = filter_with_events.should_block_trading(
            current_atr=5.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is True
        assert result["state"] == "pre_news"
        assert "Pre-news" in result["reason"]

    def test_blocks_during_post_news_min_wait(self, filter_with_events, sample_calendar):
        """Test that should_block_trading blocks during post_news_min_wait (2 min after event)."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 1 minute after NFP (within 2 min wait)
        check_time = nfp_time + timedelta(minutes=1)
        result = filter_with_events.should_block_trading(
            current_atr=5.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is True
        assert result["state"] == "post_news_min_wait"

    def test_allows_after_min_wait_low_volatility(self, filter_with_events, sample_calendar):
        """Test that should_block_trading allows trading after min_wait if volatility is low."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 5 minutes after NFP (past 2 min wait)
        check_time = nfp_time + timedelta(minutes=5)
        # current_atr=3.0, normal_atr=3.0 -> ratio=1.0 < threshold 2.0
        result = filter_with_events.should_block_trading(
            current_atr=3.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is False
        assert result["state"] == "post_news_safe"
        assert "low volatility" in result["reason"]

    def test_blocks_after_min_wait_high_volatility(self, filter_with_events, sample_calendar):
        """Test that should_block_trading keeps blocking if volatility is high after min_wait."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 5 minutes after NFP (past 2 min wait)
        check_time = nfp_time + timedelta(minutes=5)
        # current_atr=8.0, normal_atr=3.0 -> ratio=2.67 >= threshold 2.0
        result = filter_with_events.should_block_trading(
            current_atr=8.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is True
        assert result["state"] == "post_news_high_vol"
        assert "checking volatility" in result["reason"]

    def test_allows_outside_any_window(self, filter_with_events, sample_calendar):
        """Test that should_block_trading allows trading outside any event window."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 2 hours before NFP (well outside window)
        check_time = nfp_time - timedelta(hours=2)
        result = filter_with_events.should_block_trading(
            current_atr=5.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is False
        assert result["state"] == "clear"

    def test_allows_after_min_wait_no_atr_data(self, filter_with_events, sample_calendar):
        """Test that should_block_trading allows after min_wait when no ATR data provided."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        # 5 minutes after NFP (past 2 min wait)
        check_time = nfp_time + timedelta(minutes=5)
        # No ATR data provided
        result = filter_with_events.should_block_trading(
            current_atr=None, normal_atr=None, check_time=check_time
        )
        assert result["blocked"] is False
        assert result["state"] == "post_news_safe"

    def test_blocks_at_exact_event_time_within_min_wait(self, filter_with_events, sample_calendar):
        """Test at exact event time (within min wait period)."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        result = filter_with_events.should_block_trading(
            current_atr=5.0, normal_atr=3.0, check_time=nfp_time
        )
        assert result["blocked"] is True
        assert result["state"] == "post_news_min_wait"

    def test_volatility_threshold_boundary(self, filter_with_events, sample_calendar):
        """Test at exact volatility threshold boundary."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        check_time = nfp_time + timedelta(minutes=5)
        # ratio = 6.0/3.0 = 2.0, which equals threshold (not below)
        result = filter_with_events.should_block_trading(
            current_atr=6.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is True
        assert result["state"] == "post_news_high_vol"

    def test_volatility_just_below_threshold(self, filter_with_events, sample_calendar):
        """Test just below volatility threshold allows trading."""
        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])
        check_time = nfp_time + timedelta(minutes=5)
        # ratio = 5.9/3.0 = 1.967, which is below threshold 2.0
        result = filter_with_events.should_block_trading(
            current_atr=5.9, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is False
        assert result["state"] == "post_news_safe"

    def test_custom_min_wait_config(self, sample_calendar):
        """Test with custom post_news_min_wait configuration."""
        config = NewsFilterConfig(
            minutes_before=30,
            minutes_after=30,
            strict_mode=False,
            post_news_min_wait=5,  # 5 minutes min wait
        )
        nf = NewsCalendarFilter(config=config)
        nf._calendar = sample_calendar
        nf._last_refresh = datetime.now(timezone.utc)

        nfp_time = datetime.fromisoformat(sample_calendar[0]["date"])

        # 3 minutes after event (within 5 min wait)
        check_time = nfp_time + timedelta(minutes=3)
        result = nf.should_block_trading(
            current_atr=3.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is True
        assert result["state"] == "post_news_min_wait"

        # 6 minutes after event (past 5 min wait, low vol)
        check_time = nfp_time + timedelta(minutes=6)
        result = nf.should_block_trading(
            current_atr=3.0, normal_atr=3.0, check_time=check_time
        )
        assert result["blocked"] is False
        assert result["state"] == "post_news_safe"


# ─────────────────────────────────────────────
#  IS_POST_NEWS_SAFE TESTS
# ─────────────────────────────────────────────
class TestIsPostNewsSafe:
    """Tests for is_post_news_safe method."""

    def test_safe_when_low_volatility(self, filter_with_events):
        """Test returns True when volatility ratio is below threshold."""
        # ratio = 3.0/3.0 = 1.0 < 2.0 threshold
        assert filter_with_events.is_post_news_safe(3.0, 3.0) is True

    def test_not_safe_when_high_volatility(self, filter_with_events):
        """Test returns False when volatility ratio is above threshold."""
        # ratio = 8.0/3.0 = 2.67 >= 2.0 threshold
        assert filter_with_events.is_post_news_safe(8.0, 3.0) is False

    def test_not_safe_when_at_threshold(self, filter_with_events):
        """Test returns False when volatility ratio equals threshold."""
        # ratio = 6.0/3.0 = 2.0, not < 2.0
        assert filter_with_events.is_post_news_safe(6.0, 3.0) is False

    def test_not_safe_when_normal_atr_zero(self, filter_with_events):
        """Test returns False when normal_atr is zero (avoid division)."""
        assert filter_with_events.is_post_news_safe(5.0, 0.0) is False

    def test_not_safe_when_normal_atr_negative(self, filter_with_events):
        """Test returns False when normal_atr is negative."""
        assert filter_with_events.is_post_news_safe(5.0, -1.0) is False

    def test_safe_when_volatility_very_low(self, filter_with_events):
        """Test returns True when current volatility is much lower than normal."""
        # ratio = 1.0/5.0 = 0.2, well below threshold
        assert filter_with_events.is_post_news_safe(1.0, 5.0) is True
