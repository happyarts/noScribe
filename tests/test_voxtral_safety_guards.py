"""Die Schutzgeländer vor dem Modell-Laden -- und die des Salvage-Schnitts.

Gemeinsam ist ihnen, dass ein Fehler hier nicht abstürzt, sondern die Maschine
in den Swap schickt, den Job nie fertig werden lässt, oder Sprache aus dem
Transkript entfernt, ohne dass es im Log anders aussieht als ein guter Lauf.
"""
import numpy as np
import pytest

from noScribe import voxtral_engine as v
from noScribe.voxtral_engine import (
    HARD_MIN_CHUNK_SEC,
    SAMPLE_RATE,
    _model_kind,
    _salvage_prefix,
    _transcribe_guarded,
)


class _StopRun(Exception):
    """Der gestubbte Modell-Konstruktor wirft das: beweist, dass der Lauf genau
    bis zum Modell-Laden kam und keinen Schritt weiter."""


def _tiny_wav(tmp_path):
    import soundfile as sf
    p = tmp_path / "t.wav"
    sf.write(p, np.zeros(2 * 16000, dtype="float32"), 16000)
    return str(p)


def _sized(monkeypatch, tmp_path, ram=32.0, **kw):
    """transcribe() bis zum Modell-Laden treiben und die Logzeilen einsammeln."""
    def stop(repo):
        raise _StopRun

    monkeypatch.setattr(v, "_Voxtral", stop)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: ram)
    logs = []
    with pytest.raises(_StopRun):
        v.transcribe(_tiny_wav(tmp_path), need_timestamps=False,
                     log_cb=lambda lvl, msg: logs.append((lvl, msg)), **kw)
    return logs


# --------------------------------------------------------------------------- #
# Speicher-Profil eines Builds
# --------------------------------------------------------------------------- #
def test_a_24b_build_without_a_bit_width_is_metered_conservatively():
    """`small` ist das 4-Bit-Profil und damit das billigste der drei 24B-
    Einträge. Ein unquantisierter oder 6-Bit-Ordner ohne Bitbreite im Namen
    landete darauf und bekam mehrere hundert Sekunden lange Durchgänge für
    Gewichte, die 25-48 GB brauchen -- der Lauf swappt dann endlos, statt
    abgelehnt zu werden."""
    assert _model_kind("models/Voxtral-Small-24B-2507") == "small8"
    assert _model_kind("my-voxtral-24b") == "small8"
    # Mit ausdrücklicher Bitbreite bleibt alles wie gehabt.
    assert _model_kind("voxtral-small-4bit") == "small"
    assert _model_kind("voxtral-small-6bit") == "small6"
    assert _model_kind("voxtral-small-8bit") == "small8"
    # Die mini-Seite war nie betroffen: ohne Bitbreite trifft sie das teurere
    # bf16-Profil, also die sichere Richtung.
    assert _model_kind("models/Voxtral-Mini-3B-2507") == "mini"
    assert _model_kind("voxtral-mini-8bit") == "mini8"


def test_an_unquantised_source_release_is_refused_by_path_too(monkeypatch, tmp_path):
    """Die Sperre verglich nur die vollständige Hub-Repo-ID. Eine lokale Kopie
    unter models/ ist aber ein Dateipfad -- also genau der Build, den die Sperre
    aufhalten soll, kam an ihr vorbei."""
    def stop(repo):
        raise _StopRun

    monkeypatch.setattr(v, "_Voxtral", stop)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 64.0)
    wav = _tiny_wav(tmp_path)
    for repo in ("mistralai/Voxtral-Small-24B-2507",
                 "models/Voxtral-Small-24B-2507",
                 "/srv/builds/Voxtral-Mini-3B-2507"):
        with pytest.raises(ValueError, match="unquantised source release"):
            v.transcribe(wav, voxtral_repo=repo, need_timestamps=False)


# --------------------------------------------------------------------------- #
# Angehefteter chunk_sec
# --------------------------------------------------------------------------- #
def test_a_sub_second_pinned_chunk_sec_cannot_collapse_the_passes(monkeypatch, tmp_path):
    """int() macht aus allem unter 1 s eine 0, und max(1, 0 * SAMPLE_RATE) macht
    daraus einen Durchgang von EINEM Sample: eine 2-Minuten-Datei zerfiel in
    2401 Voxtral-Dekodierungen à 50 ms, der Job wurde nie fertig. Die
    Untergrenze galt bisher nur im automatischen Pfad."""
    logs = _sized(monkeypatch, tmp_path, chunk_sec=0.5,
                  voxtral_repo="models/voxtral-mini-8bit")
    assert any(lvl == "warn" and "below the" in msg and "minimum" in msg
               for lvl, msg in logs), logs
    # ...und die alte, irreführende Begründung ist weg.
    assert not any("more context than the model has" in msg for _, msg in logs), logs


def test_a_pinned_chunk_sec_below_the_floor_is_lifted(monkeypatch, tmp_path):
    logs = _sized(monkeypatch, tmp_path, chunk_sec=30,
                  voxtral_repo="models/voxtral-mini-8bit")
    assert any(str(HARD_MIN_CHUNK_SEC) in msg for _, msg in logs), logs


def test_a_sane_pinned_chunk_sec_is_left_alone(monkeypatch, tmp_path):
    """Gegenprobe: ein brauchbarer Wert darf keine Warnung auslösen."""
    logs = _sized(monkeypatch, tmp_path, chunk_sec=600,
                  voxtral_repo="models/voxtral-mini-8bit")
    assert not any(lvl == "warn" and "voxtral_chunk_sec" in msg
                   for lvl, msg in logs), logs


# --------------------------------------------------------------------------- #
# Worauf der Salvage-Schnitt sich stützen darf
# --------------------------------------------------------------------------- #
LOOP_TEXT = ("Wir haben gestern lange über die eigentlichen Ziele gesprochen. "
             "Danach kam ziemlich unvermittelt die Frage nach den Werten auf. "
             "Ich fand diese Diskussion ausgesprochen aufschlussreich. "
             + "dass das, " * 40).strip()


def _stamps(words, real_tail, step=4.0):
    """Ein Wort alle `step` Sekunden -- weit genug, dass der Präfix über
    SALVAGE_MIN_PREFIX_SEC liegt und die Kürze-Schranke nicht zuerst greift.
    `real_tail` bestimmt, ob das LETZTE Wort wirklich ausgerichtet wurde oder
    nur eine Schätzung ist."""
    out = [{"word": w, "start": i * step, "end": (i + 1) * step, "prob": -0.1}
           for i, w in enumerate(words)]
    if not real_tail:
        out[-1] = dict(out[-1], prob=0.0)
    return out


def test_salvage_refuses_a_cut_taken_from_an_invented_timestamp():
    """Geprüft wurde mit any() über den ganzen Präfix -- ein einziges echt
    ausgerichtetes Wort genügte. Geschnitten wird aber ausschliesslich auf dem
    LETZTEN Zeitstempel. War dessen Zeit geraten (gleichverteilte oder
    interpolierte Wörter am Ende), landete der Wiedereinstieg gemessen 27 s zu
    spät, und die Sprache dazwischen fiel aus dem Transkript."""
    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)

    good, why = _salvage_prefix(
        LOOP_TEXT, audio, lambda w, a: _stamps(w, real_tail=True))
    assert good is not None and why is None

    bad, why = _salvage_prefix(
        LOOP_TEXT, audio, lambda w, a: _stamps(w, real_tail=False))
    assert bad is None
    assert "not really aligned" in why, why


def test_salvage_still_refuses_a_fully_spread_alignment():
    """Der ältere Fall bleibt erhalten, samt seiner Begründung im Log."""
    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)
    spread = lambda w, a: [{"word": x, "start": float(i), "end": float(i + 1),
                            "prob": 0.0} for i, x in enumerate(w)]
    out, why = _salvage_prefix(LOOP_TEXT, audio, spread)
    # Die Begründung nennt seit 2026-07-27 den strukturellen Grund: bei dieser
    # Fenstergröße muss die Ausrichtung splitten, und das ist für einen
    # Teil-Präfix unzulässig -- unterscheidbar von einer Einzelfall-Ablehnung.
    assert out is None and "stays inactive" in why


def test_a_degenerate_prefix_is_not_shipped_as_a_success():
    """Der `rest_sec < 1.0`-Ausgang war der einzige Erfolgsausgang der Leiter
    ohne Degenerations-Prüfung: ein Präfix, das selbst noch schleift, ging als
    fertiges Ergebnis durch und eskalierte nie."""
    audio = np.zeros(200 * SAMPLE_RATE, dtype=np.float32)
    # Ein Präfix, dessen Satz sich wortwörtlich wiederholt (Kompressionsarm des
    # Detektors), gefolgt von einer kurzen Zyklus-Schleife.
    sentence = "Und dann sagte sie genau dasselbe noch ein weiteres Mal dazu. "
    text = (sentence * 60 + "dass das, " * 40).strip()

    calls = []

    class _Vox:
        def transcribe_array(self, audio, language, max_new_tokens=4096,
                             repetition_penalty=1.0, token_cb=None,
                             temperature=0.0, seed=None, info=None):
            calls.append(round(len(audio) / SAMPLE_RATE))
            return text if len(calls) == 1 else "Ein sauberer zweiter Versuch."

    def _align_to_the_very_end(words, window):
        # Letztes Wort echt ausgerichtet, aber praktisch am Fensterende ->
        # rest_sec < 1.0, also der fragliche Ausgang.
        n = len(words)
        step = (len(window) / SAMPLE_RATE) / n
        return [{"word": w, "start": i * step, "end": (i + 1) * step,
                 "prob": -0.1} for i, w in enumerate(words)]

    logs = []
    out = _transcribe_guarded(_Vox(), audio, "de",
                              lambda lvl, msg: logs.append(msg), "Pass 1/1",
                              align_cb=_align_to_the_very_end)
    assert not v._looks_degenerate(out), out[:120]
    assert len(calls) > 1, "die Leiter hat nach dem Präfix nicht weitergemacht"
    assert any("still looks degenerate" in m for m in logs), logs
