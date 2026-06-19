"""Basic tests for formatting utilities."""

from blockrank_rag.core.formatting import build_ranking_prompt, parse_ranked_ids


def test_build_prompt_returns_messages():
    msgs = build_ranking_prompt("test query", ["doc one", "doc two"], use_chat=True)
    assert isinstance(msgs, list)
    assert len(msgs) >= 1
    assert "role" in msgs[0]


def test_parse_ranked_ids():
    assert parse_ranked_ids("Final Answer: [3]") == [3]
    assert parse_ranked_ids("The ids are 1 and 4") == [1, 4]
    assert parse_ranked_ids("[]") == []
    assert parse_ranked_ids("Answer [10, 2, 99]") == [10, 2, 99]
