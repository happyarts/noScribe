"""Regression tests for the speaker-names feature (PR #323).

Covers name parsing/sanitization and the diarization-label -> user-name mapping,
including the identity-vs-display separation that keeps two speakers with the
same name in separate paragraphs.
"""
from types import SimpleNamespace

import noScribe.main as m
from noScribe.main import parse_speaker_names


def test_parse_splits_and_trims():
    assert parse_speaker_names("Mona, Lena") == ["Mona", "Lena"]
    assert parse_speaker_names("Mona; Lena") == ["Mona", "Lena"]
    assert parse_speaker_names(["Mona", "Lena"]) == ["Mona", "Lena"]


def test_parse_none_and_blank():
    assert parse_speaker_names(None) == []
    assert parse_speaker_names("") == []
    assert parse_speaker_names("   ") == []
    assert parse_speaker_names(",;") == []


def test_parse_sanitizes_anchor_breaking_chars():
    # underscore -> space (underscore is the anchor field separator)
    assert parse_speaker_names("Anna_Lena") == ["Anna Lena"]
    # <, >, ", & are removed (they break the HTML anchor attribute)
    assert parse_speaker_names('<Boss>, AT&T, "Q"') == ["Boss", "ATT", "Q"]
    # colon is removed (it is the "Name: text" label separator)
    assert parse_speaker_names("Dr: Smith") == ["Dr Smith"]


def _stub_app():
    """Minimal App-like object for calling _apply_speaker_name unbound."""
    logs = []
    return SimpleNamespace(
        _speaker_name_map={},
        _speaker_name_overflow_warned=False,
        logn=lambda *a, **k: logs.append(a),
        _logs=logs,
    )


def _job(names):
    return SimpleNamespace(speaker_names=parse_speaker_names(names))


def test_maps_in_first_appearance_order():
    app, job = _stub_app(), _job("Mona, Markus")
    # first label heard gets the first name, regardless of S00/S01 numbering
    assert m.App._apply_speaker_name(app, "S01", job) == "Mona"
    assert m.App._apply_speaker_name(app, "S02", job) == "Markus"
    # stable on repeat
    assert m.App._apply_speaker_name(app, "S01", job) == "Mona"


def test_overlap_prefix_preserved():
    app, job = _stub_app(), _job("Mona, Markus")
    assert m.App._apply_speaker_name(app, "S01", job) == "Mona"
    # an overlapping turn keeps the // marker on the mapped name
    assert m.App._apply_speaker_name(app, "//S01", job) == "//Mona"


def test_overflow_keeps_base_label_and_warns_once():
    app, job = _stub_app(), _job("OnlyOne")
    assert m.App._apply_speaker_name(app, "S01", job) == "OnlyOne"
    # more speakers than names -> extra speaker keeps its raw label, not a name
    assert m.App._apply_speaker_name(app, "S02", job) == "S02"
    assert m.App._apply_speaker_name(app, "S03", job) == "S03"
    assert app._speaker_name_overflow_warned is True
    # warned exactly once across all overflow speakers
    assert sum(1 for a in app._logs if a) == 1


def test_empty_names_returns_label_unchanged():
    app, job = _stub_app(), _job("")
    assert m.App._apply_speaker_name(app, "S01", job) == "S01"


def test_duplicate_names_map_to_distinct_keys():
    """Two speakers given the same name still occupy distinct map entries, so the
    on_segment paragraph logic (which compares the raw labels) can keep them
    apart even though their display name is identical."""
    app, job = _stub_app(), _job("Anna, Anna")
    assert m.App._apply_speaker_name(app, "S01", job) == "Anna"
    assert m.App._apply_speaker_name(app, "S02", job) == "Anna"
    assert set(app._speaker_name_map) == {"S01", "S02"}
