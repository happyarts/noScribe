"""Guards a fork-only addition, `local_build_stamp` in `noScribe/main.py`.
It does not ship upstream.
"""


def test_build_stamp_extends_the_version_without_replacing_it():
    """Der Transkript-Kopf soll zeigen, welcher Build ihn erzeugt hat.

    Die reine app_version ist zwischen zwei Releases identisch und sagt bei
    einem Arbeitsbaum, der sich mehrmals täglich ändert, nichts aus. Wichtig:
    app_version selbst bleibt unangetastet -- sie wird im Update-Check gegen
    die veröffentlichte Version verglichen."""
    import noScribe.main as m

    stamp = m.local_build_stamp()
    assert stamp.startswith(m.app_version)      # Version zuerst, dann Herkunft
    assert 'MK' in stamp
    assert m.app_version == '0.7.2'             # unverändert für version_higher
