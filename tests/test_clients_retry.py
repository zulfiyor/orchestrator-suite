# Program: Orchestrator Client Retry Tests
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Ensure BaseClient retries transient failures."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.clients import BaseClient


def test_base_client_retries() -> None:
    client = BaseClient("http://service", retries=2, backoff_s=0)
    response = MagicMock()
    with patch("requests.post", side_effect=[Exception("boom"), response]) as post:
        result = client._post("/path")  # noqa: SLF001
    assert result is response
    assert post.call_count == 2


def test_base_client_raises_after_retries() -> None:
    client = BaseClient("http://service", retries=1, backoff_s=0)
    with patch("requests.post", side_effect=Exception("boom")):
        with pytest.raises(Exception):
            client._post("/path")  # noqa: SLF001


# Created by Dr. Z. Bakhtiyorov
