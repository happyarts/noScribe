"""Die Audiozeit, an der ein sauberer Präfix endet -- zuverlässig bestimmt.

Der Präfix-Erhalt der Loop-Leiter braucht genau eine Zahl: wo im Audio der
saubere Teil aufhört. Zwei Dinge stehen dem im Weg, beide an echten Emissionen
nachgemessen (300 s Deutsch, Grundwahrheit aus dem Greedy-Decode desselben
Aligners, also frame-genau bekannt):

1. **Reine Ausrichtung kann nicht früh aufhören.** Ein CTC-Pfad muss jeden
   Frame belegen, und über 215 s echter Sprache auf Blank zu parken kostet mehr
   (-21905), als die letzten Präfix-Wörter darüber zu ziehen. Also tut der
   wahrscheinlichste Pfad genau das: ein 85-s-Präfix endete bei 299.98 s eines
   300-s-Fensters -- mit tadellosem Score (-0.000), weshalb kein nachgelagerter
   Wächter das je hätte sehen können. Es ist keine Willkür des Tie-Breaks, der
   verschobene Pfad gewinnt um 802 nat. Gegenmittel ist das Star-Token aus dem
   MMS-/torchaudio-Rezept für Teiltranskripte.
2. **Ein grosses Fenster reisst FORCED_ALIGN_MAX_CELLS.** Die Antwort von
   `align_words` darauf -- Wörter halbieren, Audio nach ihrem Zeichenanteil
   schneiden -- ist für einen Präfix unzulässig (gemessen: Wiederaufsetzpunkt
   152 s zu spät, diese Sprache fiel aus dem Transkript). `align_prefix`
   schneidet deshalb die WÖRTER und gibt jedem Stück das gesamte Restaudio.

Die Sprosse war dadurch nicht nur begrenzt (1500-s-Fenster: Präfixe bis
~440 s), sie schnitt in dem Band, in dem sie feuerte, an der falschen Stelle.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

from noScribe import voxtral_engine as v  # noqa: E402
from noScribe.voxtral_engine import (  # noqa: E402
    FORCED_ALIGN_MAX_CELLS,
    SAMPLE_RATE,
    _Aligner,
)

VOCAB_SIZE = 30
FPS = 50.0                      # 16 kHz / 320 samples je Frame


def _stub_aligner(emission_frames):
    """Aligner ohne Gewichte: echte Auswahl-/Ketten-Logik, gestellte Emission."""
    al = object.__new__(_Aligner)
    al._torch = torch
    al.vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    al.blank = 0
    al.delim = None
    al._emission = lambda audio: emission_frames
    return al


_ALPHA = "abcdefghijklmnopqrstuvwxy"      # 'z' bleibt dem Fremdmaterial


def _words(n):
    """`n` möglichst unterschiedliche Wörter.

    Wichtig für die Beweisführung: wiederholt sich der Wortlauf, hat die
    Ausrichtung mehrere gleich gute Pfade und der "wahre" Zeitpunkt ist gar
    nicht bestimmt -- der Test misst dann nur noch die Willkür des Tie-Breaks.
    """
    return [_ALPHA[i % 25] + _ALPHA[(i // 25) % 25] + _ALPHA[(i * 7) % 25]
            + _ALPHA[(i * 11 + 3) % 25] + _ALPHA[(i * 17 + 5) % 25]
            for i in range(n)]


def _planted(al, words, n_frames, frames_per_char=2):
    """Emission, in der `words` wirklich am Anfang des Audios stehen -- und in
    der DANACH weitergesprochen wird.

    Der zweite Teil ist der entscheidende: hinter dem Präfix liegt im echten
    Fall keine Stille, sondern der Rest des Fensters. Auf Blank zu parken ist
    dort teuer, und genau deshalb zieht die reine Ausrichtung die letzten
    Präfix-Wörter über das Restaudio (an echten Emissionen nachgerechnet: der
    verschobene Pfad gewinnt um 802 nat, das ist keine Willkür des
    Tie-Breaks). Rückgabe: (Emission, wahres Ende des letzten Wortes).
    """
    path = [al.blank] * n_frames
    f = int(1.0 * FPS)                       # ein bisschen Vorlauf
    for w in words:
        for ch in w:
            for _ in range(frames_per_char):
                path[f] = al.vocab[ch]
                f += 1
            path[f] = al.blank               # Blank zwischen gleichen Labels
            f += 1
        f += 2                               # Wortlücke
    true_end = (f - 3) / FPS
    # Restaudio: durchgehend Sprache, kein Blank -- sonst kostet das
    # Nicht-Abdecken nichts und der Fehler tritt gar nicht erst auf. Und
    # abwechslungsreich, denn genau daher bezieht der verschobene Pfad seinen
    # Vorteil: er findet für jedes Wort irgendwo hinten passende Buchstaben.
    for n, g in enumerate(range(f + int(1.0 * FPS), n_frames)):
        path[g] = al.vocab[_ALPHA[(n // frames_per_char) % len(_ALPHA)]]
    em = torch.full((n_frames, VOCAB_SIZE), -12.0)
    em[torch.arange(n_frames), torch.tensor(path)] = 0.0
    return torch.log_softmax(em, dim=-1), true_end


def _audio(n_frames):
    return np.zeros(int(n_frames * 320), dtype=np.float32)


def _count_cells(monkeypatch):
    cells = []
    real = torchaudio.functional.forced_align

    def counting(emission, targets, blank=0):
        cells.append(emission.shape[1] * (2 * targets.shape[1] + 1))
        return real(emission, targets, blank=blank)

    monkeypatch.setattr(torchaudio.functional, "forced_align", counting)
    return cells


# --------------------------------------------------------------------------- #
# Der Kern: ein Präfix, der eine einzelne forced_align-Rechnung sprengt
# --------------------------------------------------------------------------- #
def test_a_prefix_too_big_for_one_call_still_gets_a_real_end_time(monkeypatch):
    """Bisher endete dieser Fall in `_spread` (prob 0.0) und die Sprosse lehnte
    ab -- ein 1500-s-Fenster liess nur Präfixe bis ~440 s durch. Jetzt muss die
    Kette eine echte Zeit liefern, und zwar die richtige."""
    n_frames = 6000                                     # 120 s
    words = _words(100)
    al = _stub_aligner(None)
    emission, true_end = _planted(al, words, n_frames)
    al._emission = lambda audio: emission

    # Die Zellgrenze so setzen, dass der ganze Präfix in einem Rutsch nicht
    # ausgerichtet werden kann -- derselbe Zustand wie ein 900-s-Präfix in
    # einem 1500-s-Fenster, nur klein genug für einen Test.
    tokens, _ = al._tokenize(words)
    cap = n_frames * (2 * len(tokens) + 1) // 3
    monkeypatch.setattr(v, "FORCED_ALIGN_MAX_CELLS", cap)
    cells = _count_cells(monkeypatch)

    out = al.align_prefix(words, _audio(n_frames))

    assert len(cells) >= 2, "der Präfix passte doch in eine Rechnung"
    assert all(c <= cap for c in cells), cells
    assert len(out) == len(words)
    # Das ist die Zahl, aus der der Schnitt entsteht.
    assert out[-1]["prob"] != 0.0, "das Präfixende wurde nicht wirklich ausgerichtet"
    assert abs(out[-1]["end"] - true_end) < 0.5, (out[-1]["end"], true_end)
    # ...und der Präfix bleibt vorn: er darf nicht ins Fremdmaterial laufen.
    assert out[-1]["end"] < n_frames / FPS * 0.6
    starts = [w["start"] for w in out]
    assert starts == sorted(starts)
    assert all(w["end"] >= w["start"] for w in out)


def test_the_audio_is_never_cut_by_character_share(monkeypatch):
    """Der eigentliche Defekt: geteilt werden die WÖRTER, nie das Audio nach
    Zeichenanteil. Jedes Stück muss deshalb gegen das gesamte Restaudio laufen
    -- erkennbar daran, dass die Framezahl jeder Rechnung dem Rest entspricht
    und nicht einem Bruchteil, der aus der Textlänge geschätzt wurde."""
    n_frames = 6000
    words = _words(100)
    al = _stub_aligner(None)
    emission, true_end = _planted(al, words, n_frames)
    al._emission = lambda audio: emission

    tokens, _ = al._tokenize(words)
    monkeypatch.setattr(v, "FORCED_ALIGN_MAX_CELLS",
                        n_frames * (2 * len(tokens) + 1) // 3)

    seen = []
    real = torchaudio.functional.forced_align
    monkeypatch.setattr(
        torchaudio.functional, "forced_align",
        lambda e, t, blank=0: (seen.append(e.shape[1]), real(e, t, blank=blank))[1])

    al.align_prefix(words, _audio(n_frames))

    assert seen[0] == n_frames, "das erste Stück sah nicht das ganze Fenster"
    # Jedes Folgestück beginnt am Ende des vorigen und behält alles danach:
    # die Restlänge sinkt, endet aber immer am Fensterende.
    assert seen == sorted(seen, reverse=True)
    # Der Rest ist echtes Restaudio, nicht der geschätzte Zeichenanteil des
    # Präfix: nach dem letzten Wort liegt noch Fremdmaterial im Fenster.
    assert seen[-1] > (n_frames - true_end * FPS) * 0.8


def test_a_small_window_is_still_one_single_alignment(monkeypatch):
    """Gegenprobe: wo eine Rechnung reicht, darf nichts gestückelt werden --
    die Kette ist die Ausnahme, nicht der neue Normalfall."""
    n_frames = 3000
    words = _words(30)
    al = _stub_aligner(None)
    emission, true_end = _planted(al, words, n_frames)
    al._emission = lambda audio: emission
    cells = _count_cells(monkeypatch)

    out = al.align_prefix(words, _audio(n_frames))

    assert len(cells) == 1 and cells[0] <= FORCED_ALIGN_MAX_CELLS
    assert abs(out[-1]["end"] - true_end) < 0.5
    assert out[-1]["prob"] != 0.0


# --------------------------------------------------------------------------- #
# Was passieren muss, wenn ein Stück nicht wirklich ausgerichtet werden kann
# --------------------------------------------------------------------------- #
def test_one_failed_piece_invalidates_the_whole_chain(monkeypatch):
    """Eine Kette ist nur so gut wie ihr schwächstes Glied: wäre eine
    Stückgrenze geraten, lägen alle folgenden Stücke daneben -- und kämen mit
    tadellosen Scores zurück, also unbemerkbar. Deshalb muss ein gescheitertes
    Stück das ganze Ergebnis auf gleichverteilte Zeiten (prob 0.0) werfen, auf
    die `_salvage_prefix` per Konstruktion nicht schneidet."""
    n_frames = 6000
    words = _words(100)
    al = _stub_aligner(None)
    emission, _ = _planted(al, words, n_frames)
    al._emission = lambda audio: emission

    tokens, _ = al._tokenize(words)
    monkeypatch.setattr(v, "FORCED_ALIGN_MAX_CELLS",
                        n_frames * (2 * len(tokens) + 1) // 3)

    calls = []
    real = torchaudio.functional.forced_align

    def flaky(emission, targets, blank=0):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulierter Ausfall im zweiten Stück")
        return real(emission, targets, blank=blank)

    monkeypatch.setattr(torchaudio.functional, "forced_align", flaky)

    out = al.align_prefix(words, _audio(n_frames))

    assert len(out) == len(words)
    assert all(w["prob"] == 0.0 for w in out), \
        "ein Teilergebnis der Kette wurde als echte Zeit ausgegeben"
    # ...und _salvage_prefix lehnt darauf ab.
    text = " ".join(words[:len(words) // 2]) + ". " + "dass das, " * 40
    got, why = v._salvage_prefix(text, _audio(n_frames),
                                 lambda w, a: al.align_prefix(w, a))
    assert got is None and why


def test_the_ladder_asks_for_a_prefix_alignment(monkeypatch):
    """Verdrahtung: die Sprosse muss `align_prefix` bekommen. `align_words` mit
    `words_span_audio=False` degradiert bei grossen Fenstern sichtbar zu
    `_spread` -- genau die Obergrenze, die hier fallen soll."""
    from noScribe.voxtral_engine import _AlignerPool

    used = []

    class _Stub:
        def align_prefix(self, words, audio, t_offset=0.0):
            used.append("align_prefix")
            return [{"word": w, "start": i, "end": i + 1, "prob": -0.1}
                    for i, w in enumerate(words)]

        def align_words(self, words, audio, t_offset=0.0, words_span_audio=True):
            used.append("align_words")
            return []

    pool = _AlignerPool("de", None)
    pool.aligner_for = lambda text, remember=True: _Stub()
    out = pool.align_for_salvage(["Guten", "Morgen."], "AUDIO")

    assert used == ["align_prefix"], used
    assert out[-1]["end"] == 2
