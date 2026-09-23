"""noScribe.nemotron_mp_worker without the model: the turns it hands to main.py,
and the voice check's evidence built from its speaker probabilities."""
import numpy as np
import pytest

from noScribe import nemotron_mp_worker as nw
from noScribe import voice_check as vc


def test_labels_and_columns_agree():
    """main.py maps columns to speakers by these labels; the turns must use the same."""
    assert nw.turns([{'Start': 0.0, 'End': 1.0, 'Speaker': 3}])[0]['label'] == nw.label(3) == 'SPEAKER_03'


def test_turns_are_noscribe_turns_sorted_by_start():
    """main.py's overlap assignment stops scanning at the first turn that starts
    after a segment, so the turns must come sorted -- and a turn of no length
    would only be noise in the log."""
    got = nw.turns([{'Start': 3.2, 'End': 4.0, 'Speaker': 1}, {'Start': 0.0, 'End': 2.5, 'Speaker': 0},
                    {'Start': 5.0, 'End': 5.0, 'Speaker': 1}])
    assert got == [{'start': 0, 'end': 2500, 'label': 'SPEAKER_00'},
                   {'start': 3200, 'end': 4000, 'label': 'SPEAKER_01'}]


def test_without_a_shipped_copy_the_model_comes_from_the_hub():
    source = nw.model_source()
    assert source == nw.MODEL_REPO or source.endswith(nw.MODEL_DIR)


def probabilities(*rows_per_second):
    """One row per 10 ms: each argument is (seconds, [p per speaker])."""
    return np.array([p for seconds, p in rows_per_second for _ in range(round(seconds * 100))], dtype=np.float32)


def test_a_unit_scores_the_mean_probability_of_each_named_speaker():
    probs = probabilities((1.0, [0.9, 0.1, 0.0]), (1.0, [0.1, 0.8, 0.0]))
    evidence = vc.Probabilities(probs, {'S00': 0, 'S01': 1}, 0.01)
    (first, second, both) = evidence.scores([[0.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    assert first == pytest.approx({'S00': 0.9, 'S01': 0.1})
    assert second['S01'] > second['S00'] and abs(both['S00'] - 0.5) < 1e-6
    assert evidence.margins([]) == (0.0, 0.0)


def test_a_span_beyond_the_probabilities_gives_nothing_to_go_by():
    evidence = vc.Probabilities(probabilities((1.0, [0.9, 0.1])), {'S00': 0, 'S01': 1}, 0.01)
    assert evidence.scores([[5.0, 6.0]]) == [None]


def test_only_the_columns_it_is_given_take_part():
    """main.py passes the speakers the diarization named: a column that never
    reached the threshold must not win a unit."""
    evidence = vc.Probabilities(probabilities((1.0, [0.2, 0.1, 0.6])), {'S00': 0, 'S01': 1}, 0.01)
    assert evidence.labels == {'S00', 'S01'} and set(evidence.scores([[0.0, 1.0]])[0]) == {'S00', 'S01'}


def test_float16_rows_are_averaged_without_overflow():
    """The worker saves float16; a mean over thousands of rows must not be
    accumulated in float16."""
    probs = np.full((200000, 2), 0.75, dtype=np.float16)
    assert vc.Probabilities(probs, {'S00': 0, 'S01': 1}, 0.01).scores([[0.0, 2000.0]])[0]['S00'] == pytest.approx(0.75)


def test_a_unit_goes_to_the_speaker_the_model_hears_most_in_it():
    """Margin 0: the second sentence of a segment the diarization gave to S00
    moves to S01 as soon as the model hears S01 more there."""
    ws = [{'word': ' Right?', 'start': 0.0, 'end': 0.8}, {'word': ' Sure.', 'start': 1.2, 'end': 1.9}]
    seg = {'start': 0.0, 'end': 1.9, 'text': ' Right? Sure.', 'words': ws}
    probs = probabilities((1.0, [0.9, 0.1]), (1.0, [0.45, 0.55]))
    turns = [{'start': 0, 'end': 1900, 'label': 'S00'}]
    passages, moves = vc.relabel([(seg, 'S00', False)], turns, vc.Probabilities(probs, {'S00': 0, 'S01': 1}, 0.01))
    assert [(p['text'], s) for p, s in passages] == [(' Right?', 'S00'), (' Sure.', 'S01')] and len(moves) == 1
