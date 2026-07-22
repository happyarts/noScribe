"""Tests for text-based aligner selection (_detect_language / _AlignerPool).

Auto jobs used to fall back to the romanised multilingual aligner even for
plain German audio -- measurably the weakest choice for German word
boundaries. The pool reads the language off each chunk's transcribed text
(which exists before that chunk is aligned) and picks the char-native
per-language model; mixed text without a dominant language keeps the
multilingual fallback, and an explicit-but-wrong language setting warns.
"""
import pytest

import noScribe.voxtral_engine as v
from noScribe.voxtral_engine import (
    ALIGN_MODELS,
    ALIGN_MODEL_MULTILINGUAL,
    _AlignerPool,
    _detect_language,
)

GERMAN = ("Und dann haben wir gesagt, dass wir das nicht einfach so machen, "
          "weil die Sache ja auch für die anderen wichtig ist. Aber wenn wir "
          "ehrlich sind, ist das schon ein großer Schritt, und ich glaube, "
          "dass wir jetzt auf einem guten Weg sind. ") * 4
ENGLISH = ("And then you know we just said that this is not what they wanted, "
           "but if you think about it, they would have been fine with the "
           "idea, because it was just like the other things we did. ") * 4
# Denglisch: German matrix with English phrases mixed in -- the German
# function words still dominate, so the German model must win.
DENGLISCH = ("Und dann haben wir das Mindset komplett geändert, you know, "
             "weil die Journey ja auch ein Commitment ist. Aber wenn wir das "
             "nicht committen, dann ist das eben not the end of the world, "
             "und ich glaube, dass wir da jetzt all in gehen sollten. ") * 4
RUSSIAN = "Мы посмотрели на это и решили, что так будет лучше для всех. " * 6
JAPANESE = "それでは、今日はこのテーマについて話しましょう。よろしくお願いします。" * 6


def test_detects_the_major_languages():
    assert _detect_language(GERMAN)[0] == "de"
    assert _detect_language(ENGLISH)[0] == "en"
    assert _detect_language(RUSSIAN)[0] == "ru"
    assert _detect_language(JAPANESE)[0] == "ja"


def test_denglisch_resolves_to_the_majority_language():
    assert _detect_language(DENGLISCH)[0] == "de"


def test_balanced_mix_and_thin_evidence_stay_undetected():
    assert _detect_language(GERMAN[:200] + " " + ENGLISH[:200] +
                            GERMAN[200:400] + ENGLISH[200:400])[0] is None
    assert _detect_language("Hallo und danke.")[0] is None
    assert _detect_language("")[0] is None


# --------------------------------------------------------------------------- #
# _AlignerPool
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_aligner(monkeypatch):
    loads = []

    class _Fake:
        def __init__(self, model):
            loads.append(model)
            self.model = model

    monkeypatch.setattr(v, "_Aligner", _Fake)
    return loads


def test_auto_picks_the_char_native_model(fake_aligner):
    logs = []
    pool = _AlignerPool(None, lambda lvl, m: logs.append((lvl, m)))
    a = pool.aligner_for(GERMAN)
    assert a.model == ALIGN_MODELS["de"]
    assert fake_aligner == [ALIGN_MODELS["de"]]
    assert any("Detected language 'de'" in m for _, m in logs)


def test_auto_without_dominant_language_uses_multilingual(fake_aligner):
    pool = _AlignerPool(None, None)
    a = pool.aligner_for("Hallo und danke.")   # thin evidence
    assert a.model == ALIGN_MODEL_MULTILINGUAL


def test_language_change_between_chunks_switches_and_caches(fake_aligner):
    pool = _AlignerPool(None, None)
    assert pool.aligner_for(GERMAN).model == ALIGN_MODELS["de"]
    assert pool.aligner_for(ENGLISH).model == ALIGN_MODELS["en"]
    assert pool.aligner_for(GERMAN).model == ALIGN_MODELS["de"]  # cached
    assert fake_aligner == [ALIGN_MODELS["de"], ALIGN_MODELS["en"]]  # 2 loads only


def test_cache_is_bounded(fake_aligner):
    pool = _AlignerPool(None, None)
    pool.aligner_for(GERMAN)
    pool.aligner_for(ENGLISH)
    pool.aligner_for(RUSSIAN)
    assert len(pool._cache) == _AlignerPool.MAX_CACHED


def test_undetected_chunk_keeps_the_previous_choice(fake_aligner):
    pool = _AlignerPool(None, None)
    pool.aligner_for(GERMAN)
    a = pool.aligner_for("Hm. Ja. Okay.")      # no evidence -> stick with de
    assert a.model == ALIGN_MODELS["de"]
    assert fake_aligner == [ALIGN_MODELS["de"]]


def test_explicit_language_stays_but_mismatch_warns_once(fake_aligner):
    logs = []
    pool = _AlignerPool("en", lambda lvl, m: logs.append((lvl, m)))
    a = pool.aligner_for(GERMAN)               # user picked English, audio is German
    assert a.model == ALIGN_MODELS["en"]       # explicit choice is respected
    warns = [m for lvl, m in logs if lvl == "warn"]
    assert len(warns) == 1 and "looks like 'de'" in warns[0]
    pool.aligner_for(GERMAN)                   # second chunk: no repeat warning
    assert len([m for lvl, m in logs if lvl == "warn"]) == 1


def test_explicit_matching_language_does_not_warn(fake_aligner):
    logs = []
    pool = _AlignerPool("de", lambda lvl, m: logs.append((lvl, m)))
    assert pool.aligner_for(GERMAN).model == ALIGN_MODELS["de"]
    assert not [m for lvl, m in logs if lvl == "warn"]
