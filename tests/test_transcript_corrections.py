"""
Tests for the `transcript_corrections.py` module.
"""

from noScribe import transcript_corrections as tc


def test_apply_corrections_is_literal():
    """A user-supplied replacement must never be read as a regex template."""
    rules = [(__import__("re").compile(r"(?<!\w)foobar(?!\w)", 2), r"Foo\Bar\1")]
    assert tc.apply_corrections("say foobar now", rules) == r"say Foo\Bar\1 now"


def test_load_corrections_skips_malformed(tmp_path, caplog):
    """Malformed entries are reported, not silently dropped."""
    f = tmp_path / "corrections.yml"
    f.write_text(
        "- to: VitaFlor\n"
        "  from: [vitaflor, 'vita flor']\n"
        "- 17\n"                       # not a mapping
        "- to: OnlyTo\n",              # missing 'from'
        encoding="utf-8",
    )
    rules = tc.load_corrections(str(f))
    assert len(rules) == 1
    assert tc.apply_corrections("ich nehme vita flor", rules) == "ich nehme VitaFlor"


def test_load_corrections_missing_file():
    assert tc.load_corrections(None) == []
    assert tc.load_corrections("/nonexistent/corrections.yml") == []


def test_default_template_has_no_active_rules(tmp_path):
    """The shipped file is a commented template: it must not rewrite anything."""
    path = tc.ensure_default_file(str(tmp_path))
    assert tc.load_corrections(path) == []


def test_koelner_phonetik_groups_same_sounding_names():
    kp = tc._koelner_phonetik
    assert kp("Markus") == kp("Marcus")          # c/k are indistinguishable by ear
    assert kp("Mona") == kp("Mohna") == kp("Muna")
    assert kp("Roman") != kp("Mona")


def test_apply_name_corrections_fixes_spelling():
    """Names the user provided win over same-sounding spellings."""
    names = ["Mona", "Markus"]
    out = tc.apply_name_corrections("Ich bin Muna Muster und heute mit Marcus hier.", names, "de")
    assert out == "Ich bin Mona Muster und heute mit Markus hier."
    # already correct -> untouched
    assert tc.apply_name_corrections("Hallo Markus, sagt Mona.", names, "de") == "Hallo Markus, sagt Mona."


def test_apply_name_corrections_only_for_german():
    """Cologne phonetics is German; other languages must not be touched by it.
    A language-neutral rule was tried and rewrote "Rome"/"Anna" -- see docstring."""
    names = ["Mona", "Markus"]
    for language in ("en", "fr", None, "", "auto"):
        assert tc.apply_name_corrections("We flew to Rome with Marcus.", names, language) \
            == "We flew to Rome with Marcus."


def test_apply_name_corrections_leaves_real_words_alone():
    """Cologne phonetics collapses vowels, so "Rom"/"Ruhm" share a code with
    "Mona". Equal length plus a small edit distance must keep them intact."""
    names = ["Mona", "Markus"]
    for sentence in [
        "Die Marke Markt in Rom ist rund.",
        "Sein Ruhm war groß, der Rum auch.",
        "Der Roman von Rosa lag im Sommer am Meer.",
    ]:
        assert tc.apply_name_corrections(sentence, names, "de") == sentence


def test_apply_name_corrections_ignores_empty_and_short_names():
    assert tc.apply_name_corrections("Muna kam.", [], "de") == "Muna kam."
    assert tc.apply_name_corrections("Muna kam.", ["Ro"], "de") == "Muna kam."


def test_degenerate_detector_separates_real_text_from_loops():
    """The repetition-loop detector must catch a runaway pass without ever
    flagging real speech.

    Calibrated on real transcripts (Whisper and Voxtral, German): the longest
    run of identical words is 2 and the compression ratio 2.59-2.63, while an
    observed loop ran to 690 identical words at a ratio of 5.75.
    """
    from noScribe.voxtral_engine import _looks_degenerate

    real = (
        "Und wenn du schon ein Produkt nimmst, hervorragend, dann hast du schon einen "
        "Schritt weiter gemacht als viele andere. Meine Empfehlung wäre trotzdem, es "
        "einmal auszuprobieren. Der Test zeigt dir nämlich ganz konkret, wo deine Werte "
        "wirklich liegen, und danach kannst du die Veränderung tatsächlich messen. Das "
        "Schöne daran ist, dass du überhaupt kein Risiko eingehst. Selbst wenn sich "
        "herausstellt, dass dein bisheriges Öl völlig in Ordnung war, hast du wenigstens "
        "Klarheit gewonnen. Für mich persönlich war genau das der entscheidende Punkt, "
        "weil ich vorher jahrelang im Dunkeln getappt bin. Wer Kinder hat, sollte sie "
        "unbedingt ebenfalls testen lassen, denn die brauchen anteilig sogar mehr."
    )
    assert not _looks_degenerate(real)
    # a pass that collapses into a loop, also when it starts out fine
    assert _looks_degenerate("Jetzt. " * 40)
    assert _looks_degenerate(real + " " + "Jetzt. " * 200)
    # short passes are never judged (a brief pass is legitimate)
    assert not _looks_degenerate("Ja ja ja ja ja.")
    assert not _looks_degenerate("")


def _fake_audio(seconds=200):
    import numpy as np
    from noScribe.voxtral_engine import SAMPLE_RATE
    a = (np.random.RandomState(0).randn(seconds * SAMPLE_RATE).astype(np.float32)) * 0.1
    mid = seconds // 2
    a[mid * SAMPLE_RATE:int((mid + 0.5) * SAMPLE_RATE)] = 0.0   # a clear pause to cut at
    return a


VARIED_TEXT = [
    "Ich kann es euch jetzt auch nicht nicht erzählen, weil die Sache zu wichtig ist. "
    "Wir haben lange überlegt und uns dann für den direkten Weg entschieden. "
    "Der Test zeigt dir schwarz auf weiß, wo du stehst, und danach entscheidest du.",
    "Was mich am meisten überzeugt hat, war die Klarheit der Werte nach acht Wochen. "
    "Vorher hätte ich behauptet, alles richtig zu machen, und lag damit daneben. "
    "Wer Kinder hat, sollte sie ebenfalls testen lassen, ihr Bedarf liegt höher.",
]


class _FakeVoxtral:
    """Loops on long passes at penalty 1.0, transcribes anything else fine."""

    def __init__(self, always_loop=False):
        self.calls = []
        self.always_loop = always_loop
        self._i = 0

    def transcribe_array(self, audio, language, max_new_tokens=0, repetition_penalty=1.0,
                         token_cb=None):
        from noScribe.voxtral_engine import SAMPLE_RATE
        dur = len(audio) / SAMPLE_RATE
        self.calls.append((round(dur), repetition_penalty))
        if repetition_penalty == 1.0 and (self.always_loop or dur > 150):
            return "Und das ist jetzt passiert. " + "Jetzt. " * 300
        text = VARIED_TEXT[self._i % len(VARIED_TEXT)]
        self._i += 1
        return text


def test_looping_pass_is_split_not_penalised():
    """Splitting must be tried first: a penalty deletes meaningful repetitions
    (it turned "nicht nicht erzählen" into "nicht erzählen", inverting it)."""
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    vox = _FakeVoxtral()
    out = _transcribe_guarded(vox, _fake_audio(), "de", None, "Pass 1/1")
    assert not _looks_degenerate(out)
    assert "nicht nicht" in out                       # meaningful repetition survives
    assert all(p == 1.0 for _, p in vox.calls)        # no penalty was needed
    assert [d for d, _ in vox.calls] == [200, 100, 100]


def test_penalty_is_the_last_resort_when_splitting_fails():
    from noScribe.voxtral_engine import _transcribe_guarded, _looks_degenerate

    vox = _FakeVoxtral(always_loop=True)
    out = _transcribe_guarded(vox, _fake_audio(), "de", None, "Pass 1/1")
    assert not _looks_degenerate(out)
    assert vox.calls[0] == (200, 1.0)                 # unpenalised attempt first
    penalties = [p for _, p in vox.calls if p != 1.0]
    assert penalties                                  # ...penalty only afterwards
    assert penalties[0] == 1.01                       # and the gentlest one first


def test_split_point_lands_in_the_pause():
    from noScribe.voxtral_engine import _quietest_split, SAMPLE_RATE

    cut = _quietest_split(_fake_audio(200)) / SAMPLE_RATE
    assert 99.9 <= cut <= 100.6


def test_model_ram_hint_round_trips():
    """The picker shows the RAM requirement, but the app must still get the
    plain model name back out of the decorated entry."""
    from types import SimpleNamespace
    import noScribe.main as m
    from noScribe import transcription, voxtral_engine as v

    stub = SimpleNamespace(whisper_models={}, MODEL_LABEL_SEP=m.App.MODEL_LABEL_SEP,
                           _model_label_to_name={})
    stub.whisper_models["voxtral-mini-8bit"] = transcription.WhisperModel(
        name="voxtral-mini-8bit", path=None, engine="voxtral",
        repo="models/voxtral-mini-8bit")
    stub.whisper_models["precise"] = transcription.WhisperModel(name="precise", path=None)

    label = m.App.model_label(stub, "voxtral-mini-8bit")
    assert "GB RAM" in label
    # As the dropdown does, remember the label -> name mapping.
    stub._model_label_to_name = {label: "voxtral-mini-8bit"}
    assert m.App.model_key(stub, label) == "voxtral-mini-8bit"
    # A decorated label not in the map still round-trips via the separator split.
    stub._model_label_to_name = {}
    assert m.App.model_key(stub, label) == "voxtral-mini-8bit"
    # non-Voxtral models are shown unchanged
    assert m.App.model_label(stub, "precise") == "precise"
    assert m.App.model_key(stub, "precise") == "precise"


def test_model_too_large_for_the_machine_is_refused():
    """Starting a run that cannot fit does not fail loudly -- it swaps until
    nothing progresses. Refuse up front instead."""
    import pytest
    from noScribe import voxtral_engine as v

    orig = v._total_ram_gb
    try:
        v._total_ram_gb = lambda: 16.0
        with pytest.raises(MemoryError):
            v._auto_chunk_sec("models/voxtral-small-8bit", None)
        # the small 3B build still works on the same machine
        assert v._auto_chunk_sec("models/voxtral-mini-8bit", None) > 0
    finally:
        v._total_ram_gb = orig


class _StopRun(Exception):
    """Sentinel raised by the stubbed model constructor: proves the run got
    exactly as far as the model load and no further."""


def _tiny_wav(tmp_path):
    import numpy as np
    import soundfile as sf
    p = tmp_path / "t.wav"
    sf.write(p, np.zeros(16000, dtype="float32"), 16000)
    return str(p)


def test_refusal_happens_before_the_model_is_loaded(monkeypatch, tmp_path):
    """The point of refusing an oversized model is to refuse *before* 20+ GB of
    weights push the machine into swap -- so transcribe() must size passes
    first and only then construct the model."""
    import pytest
    from noScribe import voxtral_engine as v

    wav = _tiny_wav(tmp_path)
    loaded = []
    monkeypatch.setattr(v, "_Voxtral", lambda repo: loaded.append(repo))
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 16.0)
    with pytest.raises(MemoryError):
        v.transcribe(wav, voxtral_repo="models/voxtral-small-8bit")
    assert loaded == []  # refused without ever touching the weights
    # ...but a missing FILE is a missing file, not a memory problem
    with pytest.raises(FileNotFoundError):
        v.transcribe(str(tmp_path / "missing.wav"),
                     voxtral_repo="models/voxtral-small-8bit")
    assert loaded == []


def test_pinned_chunk_sec_cannot_bypass_the_memory_ceiling(monkeypatch, tmp_path):
    """A voxtral_chunk_sec pinned while experimenting with mini must not let a
    hungrier model run passes whose working set cannot fit (that run would swap
    forever, not fail) -- nor exceed the model-context cap on a huge machine."""
    import pytest
    from noScribe import voxtral_engine as v

    wav = _tiny_wav(tmp_path)

    def stop(repo):
        raise _StopRun

    monkeypatch.setattr(v, "_Voxtral", stop)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 32.0)
    warnings = []
    with pytest.raises(_StopRun):  # got past sizing, stopped at model load
        v.transcribe(wav, voxtral_repo="models/voxtral-small-6bit",
                     chunk_sec=1500, need_timestamps=False,
                     log_cb=lambda lvl, msg: warnings.append((lvl, msg)))
    # 6-bit small on 32 GB: hard ceiling (32-6-20.9)/0.0135 = 377s
    assert any(lvl == "warn" and "voxtral_chunk_sec" in msg and "377" in msg
               for lvl, msg in warnings)
    # a model that cannot fit at all is refused even with a pinned length
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 16.0)
    with pytest.raises(MemoryError):
        v.transcribe(wav, voxtral_repo="models/voxtral-small-8bit", chunk_sec=60)
    # a huge machine must still respect the model-context cap (MAX_CHUNK_SEC)
    monkeypatch.setattr(v, "_total_ram_gb", lambda: 128.0)
    warnings.clear()
    with pytest.raises(_StopRun):
        v.transcribe(wav, voxtral_repo="models/voxtral-mini-8bit",
                     chunk_sec=2400, need_timestamps=False,
                     log_cb=lambda lvl, msg: warnings.append((lvl, msg)))
    assert any(lvl == "warn" and str(v.MAX_CHUNK_SEC) in msg
               for lvl, msg in warnings)


def test_low_reserve_is_not_a_refusal(monkeypatch):
    """voxtral_ram_reserve_gb below MIN_HEADROOM_GB is the documented way to
    use a freed-up machine; it must clamp to the hard ceiling, not refuse a
    model that fits (regression: est_peak ~= total - reserve tripped the
    refusal for every reserve < 6)."""
    from noScribe import voxtral_engine as v

    monkeypatch.setattr(v, "_total_ram_gb", lambda: 32.0)
    # mini bf16 profile on 32 GB with reserve 5 used to raise MemoryError
    chunk = v._auto_chunk_sec("models/voxtral-mini", None, ram_reserve_gb=5)
    assert chunk > 0
    # and never beyond the hard ceiling: peak stays under total - MIN_HEADROOM
    m = v.MEM_MODEL["mini"]
    assert m["fixed"] + m["slope"] * chunk <= 32.0 - v.MIN_HEADROOM_GB + 0.01


def test_unquantised_source_repo_is_refused(monkeypatch, tmp_path):
    """The raw mistralai releases are conversion *sources*; the engine must
    refuse them instead of downloading tens of GB it then meters with the wrong
    memory profile."""
    import pytest
    from noScribe import voxtral_engine as v

    loaded = []
    monkeypatch.setattr(v, "_Voxtral", lambda repo: loaded.append(repo))
    for src in v.SOURCE_REPOS:
        with pytest.raises(ValueError, match="quantize_voxtral"):
            v.transcribe(_tiny_wav(tmp_path), voxtral_repo=src)
    assert loaded == []


def test_dense_encoder_builds_classify_by_bit_width():
    """The shipped builds keep the encoder in bf16 ("dense-encoder"); that adds
    under a GB, so they must classify by their bit width -- both as a local dir
    and as the published hub repo id downloaded on first use."""
    from noScribe import voxtral_engine as v

    assert v._model_kind("models/voxtral-mini-8bit") == "mini8"
    assert v._model_kind("MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder") == "mini8"
    assert v._model_kind("MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder") == "small8"
    # older uniform names still map correctly
    assert v._model_kind("models/voxtral-small-6bit") == "small6"


def test_bare_api_default_model_exists():
    """transcribe() without voxtral_repo must fall back to a key that is
    actually in VOXTRAL_MODELS (the mini rename broke this silently once)."""
    from noScribe import voxtral_engine as v
    assert "voxtral-mini-8bit" in v.VOXTRAL_MODELS


def test_published_builds_are_offered(monkeypatch):
    """Both shipped builds download on first use, so has_local_build is true
    even with no local copy; an unknown name still needs a local build."""
    from noScribe import voxtral_engine as v

    monkeypatch.setattr(v, "_local_copy", lambda name: None)
    assert v.has_local_build("voxtral-mini-8bit")
    assert v.has_local_build("voxtral-small-8bit")
    assert not v.has_local_build("some-unbuilt-experiment")

    monkeypatch.setattr(v, "_local_copy", lambda name: f"models/{name}")
    assert v.has_local_build("some-unbuilt-experiment")


def test_split_sentences_keeps_unpunctuated_tail():
    """A pass whose text does not end in . ! ? must keep ALL its words, not just
    the last one. An earlier `\\S+$` fallback dropped everything between the last
    period and the final token (silent transcript loss on the .txt path)."""
    from noScribe.voxtral_engine import _split_sentences
    assert _split_sentences("das ist ein test ohne punkt") == [
        "das ist ein test ohne punkt"]
    assert _split_sentences("Hallo. Wie geht es dir") == [
        "Hallo.", "Wie geht es dir"]
    # normal punctuated text is unchanged
    assert _split_sentences("Ein Satz. Noch einer.") == ["Ein Satz.", "Noch einer."]


def test_koelner_phonetik_no_crash_on_uncoded_word():
    """h and j map to no Cologne code; a token of only those letters must return
    '' rather than IndexError on out[0] (which aborted the whole name-correction
    pass)."""
    from noScribe.transcript_corrections import _koelner_phonetik
    assert _koelner_phonetik("Jhh") == ""
    assert _koelner_phonetik("hj") == ""
    # a normal name still encodes
    assert _koelner_phonetik("Hallo")


def test_model_kind_unknown_is_conservative():
    """An unrecognised build (no mini/small/size token) must fall back to the
    most memory-hungry profile, never the cheap `mini` one -- under-sizing only
    runs slower, over-sizing swaps forever."""
    from noScribe.voxtral_engine import _model_kind
    assert _model_kind("voxtral-mini-8bit") == "mini8"
    assert _model_kind("voxtral-small-8bit") == "small8"
    assert _model_kind("voxtral-small-4bit") == "small"
    assert _model_kind("totally-unknown-build") == "small8"
    # a 24B build that forgot the "small" token is still metered as big
    assert _model_kind("my-voxtral-24b-8bit") == "small8"
