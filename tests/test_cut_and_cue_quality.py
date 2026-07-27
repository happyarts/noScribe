"""Wo ein Chunk geschnitten wird, und wie lang ein Untertitel-Cue werden darf.

Beides sind Stellen, an denen ein falscher Wert kein Fehlerbild erzeugt, sondern
still ein schlechteres Transkript: ein Schnitt mitten im Wort, oder ein Cue, der
eine halbe Minute stehen bleibt. Deshalb sind sie hier festgehalten.
"""
import numpy as np

from noScribe.voxtral_engine import (
    QUIET_LEVEL,
    SAMPLE_RATE,
    SUB_MAX_SEC,
    _chunk_boundaries,
    _frame_energy,
    _segments_from_words,
)


def _noise(rng, n, amp):
    return (rng.standard_normal(n) * amp).astype(np.float32)


def _speech(rng, seconds, amp=0.30):
    return _noise(rng, int(seconds * SAMPLE_RATE), amp)


def _fill(arr, t0, t1, amp, rng):
    i0, i1 = int(t0 * SAMPLE_RATE), int(t1 * SAMPLE_RATE)
    arr[i0:i1] = _noise(rng, i1 - i0, amp)


# --------------------------------------------------------------------------- #
# Schnittsuche
# --------------------------------------------------------------------------- #
def test_quiet_speech_is_not_mistaken_for_a_pause():
    """_frame_energy liefert Leistung (Amplitude^2). Wurde QUIET_LEVEL darauf
    ungequadratet angewendet, lag die Schwelle bei -8.2 dB statt -16.5 dB --
    also bei leiser Sprache, nicht bei einer Pause. Jede 400-ms-Passage
    unbetonter Silben zählte damit als "klare Sprecherpause" und schlug, weil
    näher am Ziel, die echte Pause: der Schnitt landete mitten im Wort."""
    rng = np.random.default_rng(0)
    audio = _speech(rng, 1200)
    _fill(audio, 589.0, 590.5, 0.0002, rng)                  # echte Pause
    _fill(audio, 598.2, 598.8, 0.30 * 10 ** (-12 / 20), rng)  # leise Sprache

    bounds = _chunk_boundaries(audio, 600 * SAMPLE_RATE, 90 * SAMPLE_RATE,
                               20 * SAMPLE_RATE, 600 * SAMPLE_RATE)
    cut = bounds[1] / SAMPLE_RATE
    assert 589.0 <= cut <= 590.5, f"Schnitt bei {cut:.2f}s statt in der Pause"


def test_noisy_recording_still_finds_its_best_pause():
    """Gegenprobe zur schärferen Schwelle: eine Aufnahme mit hohem Grundrauschen
    (Raumton, Brummen) hat nirgends echte Stille. Sie darf deshalb nicht auf den
    ungesnappten Zielwert zurückfallen, sondern muss weiterhin das relative
    Minimum nehmen."""
    rng = np.random.default_rng(1)
    audio = _speech(rng, 1200)
    _fill(audio, 589.0, 590.5, 0.30 * 10 ** (-10 / 20), rng)   # nur ein Einbruch

    bounds = _chunk_boundaries(audio, 600 * SAMPLE_RATE, 90 * SAMPLE_RATE,
                               20 * SAMPLE_RATE, 600 * SAMPLE_RATE)
    cut = bounds[1] / SAMPLE_RATE
    assert 589.0 <= cut <= 590.5, f"Schnitt bei {cut:.2f}s, Einbruch verfehlt"


def test_quiet_level_is_compared_in_the_power_domain():
    """Die Konstante ist ein Amplitudenanteil; _frame_energy ist Leistung. Der
    Unterschied ist 2x in dB und genau der Fehler, der oben behoben wurde."""
    rng = np.random.default_rng(2)
    loud = _frame_energy(_speech(rng, 1.0, 0.30), 800)
    quiet = _frame_energy(_speech(rng, 1.0, 0.30 * QUIET_LEVEL), 800)
    # Leise Passage liegt bei QUIET_LEVEL**2 der Leistung, nicht bei QUIET_LEVEL.
    assert np.median(quiet) < np.median(loud) * QUIET_LEVEL ** 2 * 2
    assert np.median(quiet) > np.median(loud) * QUIET_LEVEL ** 2 * 0.5


# --------------------------------------------------------------------------- #
# Cue-Länge
# --------------------------------------------------------------------------- #
def _cues(stamps):
    return [(round(s["start"], 2), round(s["end"], 2), s["text"].strip())
            for s in _segments_from_words(stamps)]


def _w(word, start, end, prob=0.9):
    return {"word": word, "start": start, "end": end, "prob": prob}


def test_a_long_gap_does_not_produce_an_overlong_cue():
    """SUB_MAX_SEC wurde erst geprüft, nachdem das Wort schon im Cue lag, und
    flush() kann einen Cue nur einschließlich dieses Wortes beenden -- jede
    Sprechpause länger als die Obergrenze erzeugte also einen Cue von genau
    dieser Länge (gemessen: 31 s für zwei Wörter um ein Musikstück herum)."""
    cues = _cues([_w("Musik", 0.50, 1.00), _w("beginnt.", 31.00, 31.50)])
    assert len(cues) == 2, cues
    assert all(end - start <= SUB_MAX_SEC for start, end, _ in cues), cues


def test_an_ordinary_thinking_pause_also_splits_the_cue():
    """Kein Sonderfall Musik: eine 9-Sekunden-Denkpause mitten im Satz reichte
    schon für einen 10.4-s-Cue."""
    cues = _cues([_w("Ich", 0.0, 0.3), _w("glaube", 0.4, 0.9),
                  _w("dass", 10.0, 10.4), _w("es.", 10.5, 10.8)])
    assert len(cues) == 2, cues
    assert all(end - start <= SUB_MAX_SEC for start, end, _ in cues), cues


def test_normal_speech_still_lands_in_one_cue():
    """Gegenprobe: ohne große Lücke darf die neue Vorabprüfung nichts zerhacken."""
    stamps = [_w(w, i * 0.4, i * 0.4 + 0.35)
              for i, w in enumerate("wir haben das gestern kurz besprochen".split())]
    assert len(_cues(stamps)) == 1


def test_no_cue_has_zero_duration():
    """Ein Cue mit start == end ist laut WebVTT-Spezifikation ungültig; der
    Writer in utils prüft das nicht nach, also muss es hier stimmen."""
    stamps = [_w("Das", 0.0, 0.3), _w("war", 0.4, 0.7),
              _w("1990.", 0.7, 0.7, prob=0.0),      # OOV, interpoliert
              _w("Genau.", 3.0, 3.4)]
    for start, end, text in _cues(stamps):
        assert end > start, f"Cue ohne Dauer: {text!r}"
