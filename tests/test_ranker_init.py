"""Lightweight tests that don't require model weights or torch at import time."""

import sys
from unittest.mock import patch

# Patch heavy imports for environments without torch (CI smoke)
with patch.dict("sys.modules", {"torch": object(), "transformers": object(), "peft": object()}):
    # Reimport inside patch may still fail on real torch usage, so we only test construction
    pass


def test_config_defaults():
    # Import after ensuring we don't execute top level heavy code
    from blockrank_rag.ranker import BlockRankerConfig
    cfg = BlockRankerConfig()
    assert cfg.attn_layer == 20
    assert "blockrank" in cfg.model_name.lower() or "mistral" in cfg.model_name.lower()


def test_ranker_can_be_instantiated():
    from blockrank_rag.ranker import BlockRanker
    r = BlockRanker()
    assert r is not None
    assert not r._loaded   # should not auto-load
