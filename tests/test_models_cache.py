"""Tests for models.dev fetch/cache and pricing integration."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

import claude_analysis.models_cache as mc
from claude_analysis.pricing import ModelPricing, claude_cost, claude_pricing, codex_pricing


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_API = {
    "anthropic": {
        "models": {
            "claude-test-model": {
                "cost": {
                    "input": 3.0,
                    "output": 15.0,
                    "cache_read": 0.3,
                    "cache_write": 3.75,
                }
            }
        }
    },
    "openai": {
        "models": {
            "gpt-test-model": {
                "cost": {
                    "input": 2.0,
                    "output": 8.0,
                    "cache_read": 0.5,
                }
            }
        }
    },
}


def _make_http_mock(data: dict, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# _cache_load
# ---------------------------------------------------------------------------

class TestCacheLoad:
    def test_missing_file(self, tmp_path):
        with patch.object(mc, "CACHE_PATH", tmp_path / "missing.json"):
            data, age = mc._cache_load()
        assert data is None
        assert age == float("inf")

    def test_fresh_cache(self, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time(), "data": SAMPLE_API}))
        with patch.object(mc, "CACHE_PATH", cache_file):
            data, age = mc._cache_load()
        assert data == SAMPLE_API
        assert age < 5

    def test_stale_cache_returns_data_and_large_age(self, tmp_path):
        cache_file = tmp_path / "models.json"
        old_ts = time.time() - 90_000
        cache_file.write_text(json.dumps({"_ts": old_ts, "data": SAMPLE_API}))
        with patch.object(mc, "CACHE_PATH", cache_file):
            data, age = mc._cache_load()
        assert data == SAMPLE_API
        assert age > mc.TTL_SECONDS

    def test_corrupted_file(self, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text("not valid json{{")
        with patch.object(mc, "CACHE_PATH", cache_file):
            data, age = mc._cache_load()
        assert data is None
        assert age == float("inf")


# ---------------------------------------------------------------------------
# _cache_write
# ---------------------------------------------------------------------------

class TestCacheWrite:
    def test_writes_file_with_timestamp(self, tmp_path):
        cache_file = tmp_path / "ducktrace" / "models.json"
        with patch.object(mc, "CACHE_PATH", cache_file):
            mc._cache_write(SAMPLE_API)
        assert cache_file.exists()
        content = json.loads(cache_file.read_text())
        assert content["data"] == SAMPLE_API
        assert "_ts" in content
        assert abs(content["_ts"] - time.time()) < 5

    def test_creates_parent_directories(self, tmp_path):
        cache_file = tmp_path / "a" / "b" / "c" / "models.json"
        with patch.object(mc, "CACHE_PATH", cache_file):
            mc._cache_write({})
        assert cache_file.exists()


# ---------------------------------------------------------------------------
# get_models: cache hit / miss / failure
# ---------------------------------------------------------------------------

class TestGetModels:
    def test_fresh_cache_no_network_call(self, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time(), "data": SAMPLE_API}))
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen") as mock_open:
            result = mc.get_models()
        mock_open.assert_not_called()
        assert result == SAMPLE_API

    def test_stale_cache_fetches_network_and_overwrites(self, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time() - 90_000, "data": {"old": True}}))
        fresh_data = {"anthropic": {"models": {}}}
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", return_value=_make_http_mock(fresh_data)):
            result = mc.get_models()
        assert result == fresh_data
        # Verify disk was updated
        assert json.loads(cache_file.read_text())["data"] == fresh_data

    def test_no_cache_fetches_network(self, tmp_path):
        cache_file = tmp_path / "missing.json"
        fresh_data = {"anthropic": {"models": {}}}
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", return_value=_make_http_mock(fresh_data)):
            result = mc.get_models()
        assert result == fresh_data

    def test_fetch_failure_returns_stale_cache(self, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time() - 90_000, "data": SAMPLE_API}))
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", side_effect=OSError("network down")):
            result = mc.get_models()
        assert result == SAMPLE_API

    def test_fetch_failure_no_cache_returns_empty(self, tmp_path):
        cache_file = tmp_path / "missing.json"
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", side_effect=OSError("network down")):
            result = mc.get_models()
        assert result == {}

    def test_writes_cache_on_successful_fetch(self, tmp_path):
        cache_file = tmp_path / "models.json"
        fresh_data = {"anthropic": {"models": {}}}
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", return_value=_make_http_mock(fresh_data)):
            mc.get_models()
        assert cache_file.exists()

    def test_non_200_response_falls_back_to_stale_cache(self, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time() - 90_000, "data": SAMPLE_API}))
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", return_value=_make_http_mock({}, status=500)):
            result = mc.get_models()
        assert result == SAMPLE_API

    def test_non_200_no_cache_returns_empty(self, tmp_path):
        cache_file = tmp_path / "missing.json"
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen", return_value=_make_http_mock({}, status=503)):
            result = mc.get_models()
        assert result == {}


# ---------------------------------------------------------------------------
# get_models: offline mode
# ---------------------------------------------------------------------------

class TestGetModelsOfflineMode:
    def test_offline_uses_stale_cache_without_network(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time() - 90_000, "data": SAMPLE_API}))
        monkeypatch.setenv("DUCKTRACE_MODELS_CACHE_OFFLINE", "1")
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen") as mock_open:
            result = mc.get_models()
        mock_open.assert_not_called()
        assert result == SAMPLE_API

    def test_offline_no_cache_returns_empty(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "missing.json"
        monkeypatch.setenv("DUCKTRACE_MODELS_CACHE_OFFLINE", "1")
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen") as mock_open:
            result = mc.get_models()
        mock_open.assert_not_called()
        assert result == {}

    def test_offline_fresh_cache_still_used(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({"_ts": time.time(), "data": SAMPLE_API}))
        monkeypatch.setenv("DUCKTRACE_MODELS_CACHE_OFFLINE", "1")
        with patch.object(mc, "CACHE_PATH", cache_file), \
             patch("urllib.request.urlopen") as mock_open:
            result = mc.get_models()
        mock_open.assert_not_called()
        assert result == SAMPLE_API


# ---------------------------------------------------------------------------
# find_model_cost
# ---------------------------------------------------------------------------

class TestFindModelCost:
    def test_finds_claude_model_in_anthropic_provider(self):
        with patch.object(mc, "get_models", return_value=SAMPLE_API):
            cost = mc.find_model_cost("claude-test-model")
        assert cost == {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75}

    def test_finds_openai_model(self):
        with patch.object(mc, "get_models", return_value=SAMPLE_API):
            cost = mc.find_model_cost("gpt-test-model")
        assert cost is not None
        assert cost["input"] == 2.0
        assert cost["output"] == 8.0

    def test_unknown_model_returns_none(self):
        with patch.object(mc, "get_models", return_value=SAMPLE_API):
            cost = mc.find_model_cost("unknown-model-xyz")
        assert cost is None

    def test_empty_api_data_returns_none(self):
        with patch.object(mc, "get_models", return_value={}):
            cost = mc.find_model_cost("any-model")
        assert cost is None

    def test_model_missing_cost_field_skipped(self):
        data = {"anthropic": {"models": {"no-cost-model": {"name": "No Cost"}}}}
        with patch.object(mc, "get_models", return_value=data):
            cost = mc.find_model_cost("no-cost-model")
        assert cost is None

    def test_falls_back_to_non_preferred_provider(self):
        # Model lives under a third-party provider, not anthropic/openai
        data = {
            "anthropic": {"models": {}},
            "bedrock": {
                "models": {
                    "claude-bedrock-model": {
                        "cost": {"input": 4.0, "output": 20.0}
                    }
                }
            },
        }
        with patch.object(mc, "get_models", return_value=data):
            cost = mc.find_model_cost("claude-bedrock-model")
        assert cost is not None
        assert cost["input"] == 4.0

    def test_zero_input_price_treated_as_missing(self):
        data = {
            "anthropic": {
                "models": {
                    "zero-input-model": {
                        "cost": {"input": 0, "output": 15.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=data):
            cost = mc.find_model_cost("zero-input-model")
        assert cost is None

    def test_zero_output_price_treated_as_missing(self):
        data = {
            "anthropic": {
                "models": {
                    "zero-output-model": {
                        "cost": {"input": 3.0, "output": 0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=data):
            cost = mc.find_model_cost("zero-output-model")
        assert cost is None

    def test_non_numeric_price_treated_as_missing(self):
        data = {
            "anthropic": {
                "models": {
                    "bad-price-model": {
                        "cost": {"input": "not-a-number", "output": 15.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=data):
            cost = mc.find_model_cost("bad-price-model")
        assert cost is None


# ---------------------------------------------------------------------------
# pricing.py integration: claude_pricing / codex_pricing
# ---------------------------------------------------------------------------

class TestClaudePricingFromApi:
    def test_api_pricing_returned_when_model_found(self):
        api_data = {
            "anthropic": {
                "models": {
                    "claude-api-model": {
                        "cost": {"input": 6.0, "output": 30.0, "cache_read": 0.6, "cache_write": 7.5}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = claude_pricing("claude-api-model")
        assert p is not None
        assert p.input == pytest.approx(6.0 / 1e6)
        assert p.output == pytest.approx(30.0 / 1e6)
        assert p.cache_read == pytest.approx(0.6 / 1e6)
        assert p.cache_write == pytest.approx(7.5 / 1e6)

    def test_fallback_to_hardcoded_when_api_empty(self):
        with patch.object(mc, "get_models", return_value={}):
            p = claude_pricing("claude-3-haiku-20240307")
        assert p is not None
        assert p.input == pytest.approx(0.25 / 1e6)
        assert p.output == pytest.approx(1.25 / 1e6)

    def test_date_suffix_stripped_for_api_lookup(self):
        # models.dev uses dateless IDs; our canonical form has a date suffix
        api_data = {
            "anthropic": {
                "models": {
                    "claude-future-model": {
                        "cost": {"input": 10.0, "output": 50.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = claude_pricing("claude-future-model-20270101")
        assert p is not None
        assert p.input == pytest.approx(10.0 / 1e6)
        assert p.output == pytest.approx(50.0 / 1e6)

    def test_anthropic_prefix_stripped(self):
        api_data = {
            "anthropic": {
                "models": {
                    "claude-prefix-model": {
                        "cost": {"input": 1.0, "output": 5.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = claude_pricing("anthropic/claude-prefix-model")
        assert p is not None
        assert p.input == pytest.approx(1.0 / 1e6)

    def test_unknown_model_returns_none(self):
        with patch.object(mc, "get_models", return_value={}):
            p = claude_pricing("claude-totally-unknown-xyz")
        assert p is None

    def test_api_missing_cache_rates_fall_back_to_hardcoded(self):
        # models.dev returns input/output but omits cache_read / cache_write;
        # hardcoded table has correct cache rates — those should be used.
        api_data = {
            "anthropic": {
                "models": {
                    "claude-3-haiku-20240307": {
                        "cost": {"input": 0.25, "output": 1.25}
                        # cache_read and cache_write intentionally absent
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = claude_pricing("claude-3-haiku-20240307")
        assert p is not None
        assert p.input == pytest.approx(0.25 / 1e6)
        assert p.output == pytest.approx(1.25 / 1e6)
        # Hardcoded cache rates for claude-3-haiku-20240307: read=0.03/M, write=0.3/M
        assert p.cache_read == pytest.approx(0.03 / 1e6)
        assert p.cache_write == pytest.approx(0.3 / 1e6)

    def test_api_with_full_cache_rates_not_overridden_by_hardcoded(self):
        api_data = {
            "anthropic": {
                "models": {
                    "claude-3-haiku-20240307": {
                        "cost": {"input": 0.25, "output": 1.25, "cache_read": 0.1, "cache_write": 0.5}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = claude_pricing("claude-3-haiku-20240307")
        assert p is not None
        assert p.cache_read == pytest.approx(0.1 / 1e6)
        assert p.cache_write == pytest.approx(0.5 / 1e6)


class TestCodexPricingFromApi:
    def test_api_pricing_returned_for_gpt_model(self):
        api_data = {
            "openai": {
                "models": {
                    "gpt-api-model": {
                        "cost": {"input": 5.0, "output": 20.0, "cache_read": 1.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = codex_pricing("gpt-api-model")
        assert p is not None
        assert p.input == pytest.approx(5.0 / 1e6)
        assert p.output == pytest.approx(20.0 / 1e6)
        assert p.cache_read == pytest.approx(1.0 / 1e6)

    def test_fallback_to_hardcoded_for_known_codex_model(self):
        with patch.object(mc, "get_models", return_value={}):
            p = codex_pricing("gpt-4o")
        assert p is not None
        assert p.input == pytest.approx(2.5 / 1e6)

    def test_openai_prefix_stripped(self):
        api_data = {
            "openai": {
                "models": {
                    "gpt-openai-prefixed": {
                        "cost": {"input": 2.0, "output": 8.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            p = codex_pricing("openai/gpt-openai-prefixed")
        assert p is not None
        assert p.output == pytest.approx(8.0 / 1e6)


# ---------------------------------------------------------------------------
# Cost calculation with mocked API data
# ---------------------------------------------------------------------------

class TestCostCalculationWithApiData:
    def test_claude_cost_uses_api_pricing(self):
        api_data = {
            "anthropic": {
                "models": {
                    "claude-cost-test": {
                        "cost": {"input": 10.0, "output": 40.0, "cache_read": 1.0, "cache_write": 12.5}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            # 1M input tokens @ $10/M = $10
            # 500k output tokens @ $40/M = $20
            # total = $30
            cost = claude_cost("claude-cost-test", 1_000_000, 500_000, 0, 0)
        assert cost == pytest.approx(30.0)

    def test_claude_cost_cache_tokens_billed_correctly(self):
        api_data = {
            "anthropic": {
                "models": {
                    "claude-cache-test": {
                        "cost": {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write": 5.0}
                    }
                }
            }
        }
        with patch.object(mc, "get_models", return_value=api_data):
            # 100k cache_creation tokens @ $5/M = $0.50
            # 200k cache_read tokens @ $0.4/M = $0.08
            cost = claude_cost("claude-cache-test", 0, 0, 100_000, 200_000)
        assert cost == pytest.approx(0.50 + 0.08)

    def test_claude_cost_falls_back_to_hardcoded_when_api_empty(self):
        with patch.object(mc, "get_models", return_value={}):
            # claude-3-haiku-20240307: input=0.25/M, output=1.25/M
            cost = claude_cost("claude-3-haiku-20240307", 1_000_000, 1_000_000, 0, 0)
        assert cost == pytest.approx(0.25 + 1.25)

    def test_unknown_model_cost_is_zero(self):
        with patch.object(mc, "get_models", return_value={}):
            cost = claude_cost("claude-no-such-model", 1_000_000, 1_000_000, 0, 0)
        assert cost == 0.0
