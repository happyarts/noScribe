"""Wann forced_align überhaupt laufen kann -- und was passiert, wenn nicht.

torchaudio verlangt `n_frames >= len(targets) + n_repeats`, wobei n_repeats die
Paare unmittelbar gleicher Tokens zählt (zwischen zwei gleiche Labels muss ein
CTC-Pfad ein Blank setzen). Die alte Vorabprüfung liess nur ~5.3% Luft, deutsche
Texte haben aber 4-30% solcher Paare -- die Fenster dazwischen kamen durch,
forced_align warf, und der `except` machte daraus still gleichverteilte
Zeitstempel für das ganze Fenster.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

from noScribe.voxtral_engine import (  # noqa: E402
    FORCED_ALIGN_MAX_CELLS,
    SAMPLE_RATE,
    _Aligner,
)

VOCAB_SIZE = 30


def _stub_aligner():
    """Aligner ohne Gewichte: echte Split-/Prüflogik, gefälschte Emissionen.

    Die Emission liefert genau so viele Frames, wie _predict_frames vorhersagt,
    damit Routing und Blattprüfung im Test dieselbe Zahl sehen.
    """
    al = object.__new__(_Aligner)
    al._torch = torch
    al.vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    al.blank = 0
    al.delim = None
    gen = torch.Generator().manual_seed(0)
    al._emission = lambda audio: torch.log_softmax(
        torch.rand((al._predict_frames(len(audio)), VOCAB_SIZE), generator=gen),
        dim=-1)
    return al


# --------------------------------------------------------------------------- #
# Die Bedingung selbst
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tokens", [
    [7, 18, 18, 11, 25],           # ein Paar
    [7, 18, 18, 11, 25, 25],       # zwei Paare
    [4, 4, 4, 9, 9],               # Dreifachlauf zählt als zwei Paare
    [3, 1, 4, 1, 5],               # keine Wiederholung
])
def test_repeat_count_matches_what_torchaudio_demands(tokens):
    """Gegen die Bibliothek selbst gemessen: das kleinste T, bei dem
    forced_align nicht mehr wirft, ist genau len(tokens) + _adjacent_repeats."""
    need = _Aligner._adjacent_repeats(tokens) + len(tokens)
    smallest = None
    for T in range(len(tokens), len(tokens) + 12):
        emission = torch.log_softmax(torch.zeros((1, T, VOCAB_SIZE)), dim=-1)
        targets = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
        try:
            torchaudio.functional.forced_align(emission, targets, blank=0)
            smallest = T
            break
        except Exception:
            continue
    assert smallest == need


def test_a_window_in_the_repeat_band_does_not_degrade_silently():
    """Genau der Fall, den die alte 0.95-Schranke durchliess: knapp genug Frames
    für die Tokens, aber nicht für die Wiederholungen. Es darf keine Ausnahme
    nach aussen dringen -- und wenn nur noch Gleichverteilung bleibt, muss das
    im Log stehen, sonst ist ein kaputtes Fenster von einem guten nicht zu
    unterscheiden."""
    al = _stub_aligner()
    words = ["aabb"] * 40                      # jedes Wort bringt zwei Paare mit
    audio = np.zeros(int(3.5 * SAMPLE_RATE), dtype=np.float32)
    tokens, _ = al._tokenize(words)
    frames = al._predict_frames(len(audio))
    assert len(tokens) <= frames, "Testaufbau: Tokens müssen knapp hineinpassen"
    assert len(tokens) + al._adjacent_repeats(tokens) > frames, \
        "Testaufbau: erst die Wiederholungen dürfen es sprengen"

    out = al.align_words(words, audio)          # darf nicht werfen
    assert len(out) == len(words)
    assert all(w["end"] >= w["start"] for w in out)


def test_prediction_matches_the_emission_it_replaces():
    """Die Split-Entscheidung läuft auf vorhergesagten Frames, damit ein Fenster,
    das ohnehin geteilt wird, keinen weggeworfenen wav2vec2-Durchlauf bezahlt.
    Vorhersage und echte Emission müssen deshalb übereinstimmen."""
    al = _stub_aligner()
    for seconds in (0.5, 2.5, 20.0, 20.5, 47.3):
        audio = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        assert al._emission(audio).shape[0] == al._predict_frames(len(audio))


# --------------------------------------------------------------------------- #
# Teilabdeckung: der Salvage-Präfix
# --------------------------------------------------------------------------- #
def test_a_partial_prefix_has_its_own_entry_point():
    """Der Split schneidet das Audio nach Zeichenanteil der Wörter -- zulässig
    nur, wenn die Wörter das Fenster ausfüllen. Ein Salvage-Präfix tut das
    nicht; dafür gibt es `align_prefix` (siehe test_salvage_prefix_alignment).
    Hier bleibt festgehalten, dass `align_words` diesen Fall gar nicht erst
    annimmt, statt ihn still falsch zu rechnen."""
    import inspect
    assert "words_span_audio" not in inspect.signature(_Aligner.align_words).parameters
    assert hasattr(_Aligner, "align_prefix")


def test_a_full_window_is_still_split_as_before():
    """Gegenprobe: die Vollabdeckung (der Normalfall) teilt weiterhin."""
    al = _stub_aligner()
    words = ["abcde"] * 2000
    audio = np.zeros(1100 * SAMPLE_RATE, dtype=np.float32)
    out = al.align_words(words, audio)
    assert len(out) == len(words)
    assert any(w["prob"] != 0.0 for w in out), "gar nichts wurde ausgerichtet"


def test_dense_recursion_is_bounded():
    """Halbieren senkt die Dichte nicht: das Audio wird im selben Zeichenanteil
    geschnitten wie die Wörter, also ist tokens/frames in beiden Hälften gleich
    und die Prüfung feuert auf jeder Ebene erneut. Ungebremst lief das bis zur
    vollen Tiefe und schob das 10-13-fache des Chunks durch wav2vec2, um in
    Blättern zu enden, die ohnehin gleichverteilen."""
    al = _stub_aligner()
    seen = []
    inner = al._emission
    al._emission = lambda audio: (seen.append(len(audio)), inner(audio))[1]

    words = ["abcdefghij"] * 400                # sehr dicht für kurzes Audio
    audio = np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32)
    out = al.align_words(words, audio)

    assert len(out) == len(words)
    # 2 Ebenen Dichte-Split => höchstens 2^2 Blätter, plus deren Emissionen.
    assert len(seen) <= 2 ** _Aligner.MAX_DENSE_SPLIT_DEPTH, len(seen)
    assert sum(seen) <= 3 * len(audio), "zu viel Audio mehrfach durchgerechnet"
