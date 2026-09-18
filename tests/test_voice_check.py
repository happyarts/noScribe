"""noScribe.voice_check: units, the two margins, and the passages a rewrite is made of.

The thresholds themselves are justified in the module's docstring (word-level
speaker error against ground truth). These tests pin the *mechanics* the
measurement relied on, so a refactor cannot quietly change what was measured:
where a unit ends, which margin applies when, and that an untouched segment is
handed back untouched.
"""
import pytest

from noScribe import voice_check as vc


def words(*items):
    """('text', start, end) -> faster-whisper style words (leading space)."""
    return [{'word': ' ' + text, 'start': start, 'end': end} for text, start, end in items]


A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
CENTROIDS = {'S00': A, 'S01': B}


def voice(towards_b):
    """An embedding whose cosine margin for S01 over S00 is about `towards_b`."""
    import math
    # cos to B minus cos to A = sin(x) - cos(x); solve numerically, good to 1e-3
    best = min((abs(math.sin(x / 1000) - math.cos(x / 1000) - towards_b), x / 1000) for x in range(0, 1571))[1]
    return [math.cos(best), math.sin(best), 0.0]


def test_a_unit_ends_at_a_sentence_end_and_at_a_long_pause():
    ws = words(('Is', 0.0, 0.2), ('that', 0.2, 0.4), ('so?', 0.4, 0.7),
               ('Yes.', 0.8, 1.0), ('and', 1.1, 1.3), ('then', 1.9, 2.1))
    units = vc.split_units(ws)
    assert [[w['word'].strip() for w in u] for u in units] == [['Is', 'that', 'so?'], ['Yes.'], ['and'], ['then']]


def test_a_short_pause_does_not_end_a_unit():
    ws = words(('and', 0.0, 0.2), ('then', 0.2 + vc.UNIT_PAUSE_S - 0.01, 1.0))
    assert len(vc.split_units(ws)) == 1


def test_an_abbreviation_only_offers_a_cut_the_voice_still_has_to_take_it():
    """"Dr." ends a unit, which is harmless: without a voice that says otherwise
    both units keep the segment's speaker and are written as one passage."""
    ws = words(('Frau', 0.0, 0.3), ('Dr.', 0.3, 0.6), ('Muster', 0.6, 1.0), ('kommt.', 1.0, 1.4))
    seg = {'start': 0.0, 'end': 1.4, 'text': ' Frau Dr. Muster kommt.', 'words': ws}
    passages, changed = vc.relabel([(seg, 'S00')], [], CENTROIDS, lambda spans: [voice(-0.9)] * len(spans))
    assert changed == 0 and passages == [(seg, None, 'S00')]


def test_words_without_stamps_keep_the_segment_whole():
    ws = [{'word': ' Yes.', 'start': None, 'end': None}, {'word': ' No.', 'start': 1.0, 'end': 1.2}]
    assert vc.split_units(ws) == [ws]


def test_the_diarization_agreeing_needs_the_small_margin_only():
    turns = {'S01': 400}
    assert vc.decide('S00', turns, voice(vc.MARGIN_AGREE + 0.05), CENTROIDS) == 'S01'
    assert vc.decide('S00', turns, voice(vc.MARGIN_AGREE - 0.05), CENTROIDS) == 'S00'


def test_the_voice_alone_needs_the_large_margin():
    turns = {'S00': 400}
    assert vc.decide('S00', turns, voice(vc.MARGIN_OVERRULE + 0.05), CENTROIDS) == 'S01'
    assert vc.decide('S00', turns, voice(vc.MARGIN_OVERRULE - 0.05), CENTROIDS) == 'S00'


def test_a_whole_segment_is_never_moved_on_the_small_margin():
    """Its diarization majority is what gave it its speaker in the first place --
    a different 'named' label here would be the overlap rule's doing, not news."""
    assert vc.decide('S00', {'S01': 400}, voice(0.2), CENTROIDS, single_unit=True) == 'S00'


@pytest.mark.parametrize('embedding, centroids', [
    (None, CENTROIDS),                    # no audio to embed
    (voice(0.9), {'S01': B}),             # the current speaker has no centroid
    (voice(0.9), {'S00': A, 'S01': [0.0, 0.0, 0.0]}),  # a padded, empty centroid
])
def test_without_something_to_compare_nothing_moves(embedding, centroids):
    assert vc.decide('S00', {'S01': 400}, embedding, centroids) == 'S00'


def test_a_turn_final_answer_becomes_its_own_passage():
    ws = words(('Does', 0.0, 0.2), ('it', 0.2, 0.3), ('help?', 0.3, 0.8), ('Yes,', 0.9, 1.1), ('sure.', 1.1, 1.5))
    seg = {'start': 0.0, 'end': 1.5, 'text': ' Does it help? Yes, sure.', 'words': ws}
    turns = [{'start': 0, 'end': 850, 'label': 'S00'}, {'start': 880, 'end': 1500, 'label': 'S01'}]
    asked = []

    def embed(spans):
        asked.extend(spans)
        return [voice(-0.9), voice(0.5)]

    passages, changed = vc.relabel([(seg, 'S00')], turns, CENTROIDS, embed)
    assert asked == [[0.0, 0.8], [0.9, 1.5]]
    assert changed == 1
    assert [(p['text'], label, given) for p, label, given in passages] == [
        (' Does it help?', 'S00', 'S00'), (' Yes, sure.', 'S01', 'S00')]
    assert passages[1][0]['start'] == 0.9 and passages[1][0]['words'] == ws[3:]


def test_a_segment_moved_as_a_whole_keeps_its_dict():
    ws = words(('Exactly.', 0.0, 0.6))
    seg = {'start': 0.0, 'end': 0.6, 'text': ' Exactly.', 'words': ws}
    passages, changed = vc.relabel([(seg, 'S00')], [], CENTROIDS, lambda spans: [voice(0.9)])
    assert changed == 1 and passages == [(seg, 'S01', 'S00')]


def test_bare_tokens_are_joined_with_spaces():
    ws = [{'word': 'Yes.', 'start': 0.0, 'end': 0.3}, {'word': 'No', 'start': 0.4, 'end': 0.6},
          {'word': 'way.', 'start': 0.6, 'end': 0.9}]
    seg = {'start': 0.0, 'end': 0.9, 'text': ' Yes. No way.', 'words': ws}
    passages, _ = vc.relabel([(seg, 'S00')], [], CENTROIDS, lambda spans: [voice(-0.9), voice(0.9)])
    assert [p['text'] for p, _, _ in passages] == [' Yes.', ' No way.']


def test_one_speaker_means_no_embedding_call_at_all():
    seg = {'start': 0.0, 'end': 0.6, 'text': ' Exactly.', 'words': words(('Exactly.', 0.0, 0.6))}

    def embed(spans):
        raise AssertionError('the worker must not be started for nothing')

    assert vc.relabel([(seg, 'S00')], [], {'S00': A}, embed) == ([(seg, None, 'S00')], 0)


@pytest.mark.parametrize('value, expected', [('0', False), ('off', False), ('1', True), ('', True)])
def test_the_environment_switch(monkeypatch, value, expected):
    monkeypatch.setenv('NOSCRIBE_VOICE_CHECK', value)
    assert vc.enabled() is expected
