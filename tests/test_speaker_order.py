"""Speakers are numbered in the order they are first heard, not by pyannote's cluster number."""
from noScribe.pyannote_mp_worker import in_order_of_appearance


def seg(start, label):
    return {'start': start, 'end': start + 1000, 'label': label}


def test_the_first_voice_heard_becomes_speaker_00():
    order = in_order_of_appearance([seg(0, 'SPEAKER_01'), seg(1500, 'SPEAKER_01'), seg(3000, 'SPEAKER_00')])
    assert order == {'SPEAKER_01': 'SPEAKER_00', 'SPEAKER_00': 'SPEAKER_01'}


def test_labels_already_in_order_stay_as_they_are():
    turns = [seg(0, 'SPEAKER_00'), seg(900, 'SPEAKER_01'), seg(2000, 'SPEAKER_02'), seg(2500, 'SPEAKER_00')]
    assert in_order_of_appearance(turns) == {label: label for label in ('SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02')}


def test_the_order_follows_the_clock_not_the_list():
    """pyannote hands the turns over chronologically, but nothing here relies on it."""
    order = in_order_of_appearance([seg(5000, 'SPEAKER_00'), seg(0, 'SPEAKER_02'), seg(2000, 'SPEAKER_01')])
    assert order == {'SPEAKER_02': 'SPEAKER_00', 'SPEAKER_01': 'SPEAKER_01', 'SPEAKER_00': 'SPEAKER_02'}


def test_no_speech_means_no_labels():
    assert in_order_of_appearance([]) == {}
