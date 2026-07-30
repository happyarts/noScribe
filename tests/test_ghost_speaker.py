"""Wann ein Diarisierungs-Label einen zweiten Blick wert ist.

Die Schwellen sind gegen Ground Truth gemessen, nicht geraten: VoxConverse v0.3,
216 dev- plus 232 test-Aufnahmen mit 1 bis 21 Sprechern und RTTM-Referenz. Auf
Label-Ebene liegt die Genauigkeit bei rund zwei Dritteln (dev 83 %, test 69 %),
die Trefferquote bei etwa einem Achtel. Beide Zahlen sind hier festgehalten,
weil sie die Aussage der Meldung begrenzen -- und weil eine engere Schwelle
nachweislich NICHT hilft: 0,01 traf auf dev 100 % und auf test 67 %.

Die Gegenprobe, die die zweite Bedingung rechtfertigt: auf einer 25-minütigen
Fragerunde mit 14 echten Sprechern liegen 13 unter 5 % Sprechzeit -- alle
sprechen aber in Sätzen, und über VoxConverse dev bleiben nur 2 von 860 echten
Sprechern unter einem 2-s-Beitrag.
"""
import yaml
from pathlib import Path

from noScribe.main import (
    GHOST_SPEAKER_MAX_SHARE,
    GHOST_SPEAKER_MAX_TURN_MS,
    find_ghost_speakers,
)


def _seg(start, end, label):
    return {'start': int(start * 1000), 'end': int(end * 1000), 'label': label}


def _spread(label, n, length, step=20.0, offset=0.0):
    """n kurze Beiträge über die Datei verteilt -- das Bild eines Geistes."""
    return [_seg(offset + i * step, offset + i * step + length, label)
            for i in range(n)]


def _floor(label, n, length, step=60.0, offset=5.0):
    """n echte Redebeiträge."""
    return [_seg(offset + i * step, offset + i * step + length, label)
            for i in range(n)]


def test_the_measured_ghost_is_reported():
    """Die echte Stelle: zwei Sprecher mit je ~50 % und Beiträgen bis 29 s, dazu
    ein Label mit 1,7 % und keinem Beitrag über 1,52 s -- beides unter den Schwellen."""
    diarization = (_floor('SPEAKER_00', 20, 28.0)
                   + _floor('SPEAKER_01', 20, 26.0, offset=35.0)
                   + _spread('SPEAKER_02', 56, 0.35))
    ghosts = find_ghost_speakers(diarization)
    assert [g[0] for g in ghosts] == ['SPEAKER_02'], ghosts
    label, share, longest = ghosts[0]
    assert share < 0.05 and longest <= 1520


def test_fourteen_real_speakers_are_left_alone():
    """Die Falsch-Positiv-Kontrolle, und der Grund für die zweite Bedingung:
    13 der 14 gemessenen Sprecher liegen unter 5 % Sprechzeit. Ihr längster
    Beitrag lag zwischen 5,86 s und 42,8 s, also weit über der Schwelle."""
    longest_turns = [5.855, 6.109, 7.830, 8.522, 10.125, 10.851, 11.104,
                     11.813, 15.086, 16.301, 22.950, 24.401, 28.620, 42.778]
    diarization = []
    for i, longest in enumerate(longest_turns):
        # ein langer Beitrag plus Kleinkram -- der Anteil bleibt klein
        diarization.append(_seg(i * 100, i * 100 + longest, f'SPEAKER_{i:02d}'))
        diarization += _spread(f'SPEAKER_{i:02d}', 3, 0.4, offset=i * 100 + 50)
    diarization += _floor('SPEAKER_99', 12, 42.0, offset=2000.0)   # der Trainer
    assert find_ghost_speakers(diarization) == []


def test_two_speakers_are_never_judged():
    """Mit zwei Labels ist "eines davon ist falsch" keine Aussage, die hier
    getroffen werden kann -- das verbleibende müsste alle sein. Auch dann nicht,
    wenn eines der beiden winzig ist."""
    diarization = _floor('SPEAKER_00', 20, 28.0) + _spread('SPEAKER_01', 40, 0.3)
    assert find_ghost_speakers(diarization) == []


def test_not_every_label_can_be_a_ghost():
    """Eine Datei ohne jeden Sprecher ist nie die nützliche Lesart, wie schief
    die Verteilung auch ist."""
    diarization = (_spread('SPEAKER_00', 30, 0.4)
                   + _spread('SPEAKER_01', 30, 0.4, offset=1.0)
                   + _spread('SPEAKER_02', 30, 0.4, offset=2.0))
    assert find_ghost_speakers(diarization) == []


def test_a_brief_but_real_third_speaker_is_left_alone():
    """Der Grenzfall, den die Dauerbedingung schützt: jemand antwortet einmal
    auf eine Frage und schweigt sonst. Wenig Sprechzeit, aber ein ganzer Satz."""
    diarization = (_floor('SPEAKER_00', 20, 28.0)
                   + _floor('SPEAKER_01', 20, 26.0, offset=35.0)
                   + [_seg(500.0, 504.5, 'SPEAKER_02')])
    assert find_ghost_speakers(diarization) == []


def test_empty_and_degenerate_input():
    assert find_ghost_speakers(None) == []
    assert find_ghost_speakers([]) == []
    # Alle Segmente ohne Länge: keine Sprechzeit, also nichts zu beurteilen.
    assert find_ghost_speakers([_seg(1.0, 1.0, f'SPEAKER_0{i}') for i in range(3)]) == []


def test_thresholds_are_the_documented_ones():
    """Die Werte tragen die Aussage der Meldung; ein stiller Dreh daran macht aus
    einer gemessenen Schwelle eine geratene. 0,02 ist der einzige Wert, der auf
    dev UND test besser war als der ursprüngliche 0,05."""
    assert GHOST_SPEAKER_MAX_SHARE == 0.02
    assert GHOST_SPEAKER_MAX_TURN_MS == 2000


def test_a_quiet_real_speaker_can_trip_this_and_that_is_known():
    """Die Grenze des Verfahrens, absichtlich festgehalten. Auf VoxConverse test
    war etwa ein Drittel der Meldungen ein echter Sprecher, der einfach fast
    nichts sagte -- gemessen z. B. 0,32 % Sprechzeit mit 1266 ms als längstem
    Beitrag, extremer als der Geist, der das Ganze ausgelöst hat (1,7 % /
    1519 ms). Auf diesen zwei Achsen ist diese Verwechslung nicht auflösbar;
    deshalb meldet die Funktion und entscheidet nicht."""
    diarization = (_floor('SPEAKER_00', 20, 30.0)
                   + _floor('SPEAKER_01', 20, 28.0, offset=35.0)
                   + [_seg(400.0, 401.27, 'SPEAKER_02'),
                      _seg(700.0, 701.1, 'SPEAKER_02')])
    ghosts = find_ghost_speakers(diarization)
    assert [g[0] for g in ghosts] == ['SPEAKER_02'], ghosts


def test_many_short_turns_are_not_required():
    """Die naheliegende dritte Achse ist gemessen und verworfen: der auslösende
    Geist hatte 56 kurze Beiträge, aber auf VoxConverse haben überzählige Labels
    typisch wenige -- eine Mindestzahl an Beiträgen brach die Treffer von 5 auf 1
    ein. Ein Label mit nur zwei kurzen Beiträgen muss also gemeldet werden."""
    diarization = (_floor('SPEAKER_00', 20, 30.0)
                   + _floor('SPEAKER_01', 20, 28.0, offset=35.0)
                   + [_seg(400.0, 400.9, 'SPEAKER_02'),
                      _seg(900.0, 901.4, 'SPEAKER_02')])
    assert [g[0] for g in find_ghost_speakers(diarization)] == ['SPEAKER_02']


def test_warning_string_exists_in_every_locale_or_falls_back_to_english():
    """i18n fällt auf "en" zurück, also muss der Schlüssel dort da sein --
    sonst erscheint im Log der Schlüsselname statt eines Satzes."""
    trans = Path(__file__).resolve().parent.parent / 'trans'
    en = yaml.safe_load((trans / 'noScribe.en.yml').read_text(encoding='utf-8'))
    assert 'warn_ghost_speaker' in en['en']
    for placeholder in ('%{speaker}', '%{share}', '%{longest}'):
        assert placeholder in en['en']['warn_ghost_speaker']
