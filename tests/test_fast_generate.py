"""Unit tests for _Voxtral._consume_tokens, the token-stream bookkeeping of the
fast greedy path.

_fast_generate delegates generation to the maintained mlx_lm.generate_step and
keeps only the consumption logic here: stop at the first stop token. It
deliberately does NOT replicate generate_stream's 10-identical-token backstop --
a repetition loop must run so the engine's text-level _looks_degenerate detector
can catch it and retry. These tests drive that logic with plain int streams --
no weights, no mlx, no real Voxtral.
"""
from noScribe.voxtral_engine import _Voxtral


def _consume(stream):
    v = _Voxtral.__new__(_Voxtral)  # no __init__: we only exercise _consume_tokens
    return v._consume_tokens(iter(stream))


def test_stops_at_first_stop_token_excluded():
    # 2 is a stop token; it and everything after are dropped.
    assert _consume([5, 6, 7, 2, 9, 9, 9]) == [5, 6, 7]


def test_each_stop_token_ends_generation():
    for stop in (2, 4, 32000):
        assert _consume([8, 9, stop, 8, 8]) == [8, 9], f"stop={stop}"


def test_repetition_loop_passes_through_for_the_degeneracy_detector():
    # No token backstop: a long single-token run is kept in full (up to the stop),
    # so _looks_degenerate can see it and trigger a retry instead of it being
    # truncated to 10 tokens and slipping through.
    assert _consume([3] * 50 + [2]) == [3] * 50


def test_consumes_until_stop_regardless_of_repeats():
    assert _consume([9] * 9 + [1] + [9] * 9 + [2]) == [9] * 9 + [1] + [9] * 9


def test_empty_stream():
    assert _consume([]) == []


def test_immediate_stop_gives_empty():
    assert _consume([2, 5, 6]) == []


def test_stop_tokens_are_the_library_defaults():
    assert set(_Voxtral._STOP_TOKENS) == {2, 4, 32000}
