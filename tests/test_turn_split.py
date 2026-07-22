"""Tests for the speaker-turn-aware loop split and the escalation diagnosis.

The cleanest place to cut a looping chunk is where the speaker changes:
context within a turn stays intact, across the change it matters least.
These tests pin the boundary selection (different-speaker only, no overlap
cuts, middle half, nearest-to-middle wins), the window clipping used by the
recursion, and that the ladder actually prefers the turn boundary over the
quietest-frame fallback -- plus the diarization profile line once a loop
resists the gentle repairs.
"""
import numpy as np

from noScribe.voxtral_engine import (
    SAMPLE_RATE,
    _clip_turns,
    _transcribe_guarded,
    _turn_gap_split,
    _turn_profile,
)


def _audio(seconds, silences=()):
    """Noisy audio with 1 s of silence centred on each given second mark."""
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32)
    for at in silences:
        a[int((at - 0.5) * SAMPLE_RATE):int((at + 0.5) * SAMPLE_RATE)] = 0.0
    return a


# --------------------------------------------------------------------------- #
# _turn_gap_split
# --------------------------------------------------------------------------- #
def test_cut_lands_on_the_speaker_change():
    audio = _audio(200, silences=(100,))
    turns = [(0.0, 99.5, "A"), (100.5, 200.0, "B")]
    cut = _turn_gap_split(audio, turns)
    assert cut is not None
    assert 99.0 <= cut / SAMPLE_RATE <= 101.0

def test_same_speaker_pause_is_not_a_turn_boundary():
    audio = _audio(200, silences=(100,))
    turns = [(0.0, 99.5, "A"), (100.5, 200.0, "A")]
    assert _turn_gap_split(audio, turns) is None

def test_overlapping_speech_boundary_is_skipped():
    audio = _audio(200, silences=(100,))
    turns = [(0.0, 105.0, "A"), (95.0, 200.0, "B")]  # 10 s overlap
    assert _turn_gap_split(audio, turns) is None

def test_boundary_outside_the_middle_half_is_ignored():
    audio = _audio(200, silences=(20.5,))
    turns = [(0.0, 20.0, "A"), (21.0, 200.0, "B")]  # change at 10% of the window
    assert _turn_gap_split(audio, turns) is None

def test_nearest_boundary_to_the_middle_wins():
    audio = _audio(200, silences=(60.5, 100.0))
    turns = [(0.0, 60.0, "A"), (61.0, 99.5, "B"), (100.5, 200.0, "C")]
    cut = _turn_gap_split(audio, turns)
    assert 99.0 <= cut / SAMPLE_RATE <= 101.0

def test_no_turns_means_no_turn_split():
    assert _turn_gap_split(_audio(10), None) is None
    assert _turn_gap_split(_audio(10), []) is None


# --------------------------------------------------------------------------- #
# _clip_turns / _turn_profile
# --------------------------------------------------------------------------- #
def test_clip_turns_shifts_to_window_relative_times():
    turns = [(10.0, 30.0, "A"), (40.0, 80.0, "B")]
    assert _clip_turns(turns, 20.0, 60.0) == [(0.0, 10.0, "A"), (20.0, 40.0, "B")]
    assert _clip_turns(turns, 90.0, 120.0) is None
    assert _clip_turns(None, 0.0, 10.0) is None

def test_turn_profile_counts_speakers_changes_and_coverage():
    turns = [(0.0, 30.0, "A"), (30.0, 60.0, "B")]
    p = _turn_profile(turns, 60.0)
    assert "2 speaker(s)" in p and "1 turn change(s)" in p and "100% speech" in p


# --------------------------------------------------------------------------- #
# Ladder integration
# --------------------------------------------------------------------------- #
VARIED = ("Der Test zeigt dir schwarz auf weiß, wo du stehst. "
          "Wir haben lange überlegt und uns dann entschieden.")


class _FakeVox:
    """Loops on windows > 150 s (any temperature); short windows come out fine."""

    def __init__(self):
        self.calls = []

    def transcribe_array(self, audio, language, max_new_tokens=0,
                         repetition_penalty=1.0, token_cb=None,
                         temperature=0.0, seed=None, info=None):
        dur = len(audio) / SAMPLE_RATE
        self.calls.append(round(dur))
        if repetition_penalty == 1.0 and dur > 150:
            return "Und dann. " + "Jetzt. " * 300
        return VARIED


def test_ladder_prefers_the_turn_boundary_over_the_middle():
    # Turn change at 130 s, well away from the middle. The quietest-frame
    # fallback only searches +/-5 s around 100 s, so 130/70 halves prove the
    # turn boundary was used.
    audio = _audio(200, silences=(100.0, 130.0))
    turns = [(0.0, 129.5, "A"), (130.5, 200.0, "B")]
    vox = _FakeVox()
    out = _transcribe_guarded(vox, audio, "de", None, "t", turns=turns)
    assert VARIED in out
    assert vox.calls == [200, 200, 130, 70]  # greedy, T=0.2, then the two halves

def test_diagnosis_line_appears_when_gentle_repairs_fail():
    audio = _audio(60)  # too short to split
    turns = [(0.0, 30.0, "A"), (30.0, 60.0, "B")]

    class _AlwaysLoops(_FakeVox):
        def transcribe_array(self, *a, **kw):
            super().transcribe_array(*a, **kw)
            return "Und dann. " + "Jetzt. " * 300

    logs = []
    _transcribe_guarded(_AlwaysLoops(), audio, "de",
                        lambda lvl, msg: logs.append(msg), "t", turns=turns)
    assert any("diarization profile" in m and "2 speaker(s)" in m for m in logs)
