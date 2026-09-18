"""Guard: the diarization worker finds the bundled pipeline on every Python
version the project supports.

``pyannote`` is a namespace shared by the repository's ``pyannote/`` folder
(config.yaml and the models) and the installed library, so
``impres.files("pyannote")`` is a view over both. The worker used to hand that
view to ``impres.as_file``, which only accepts a directory from Python 3.12
on: on 3.10 and 3.11 every diarization failed at once with
``FileNotFoundError: MultiplexedPath(...) is not a file``. CI runs 3.10, so
this test fails there if the lookup goes back to ``as_file``.
"""
from pathlib import Path

from noScribe.pyannote_mp_worker import bundled_pipeline_dir

REPO = Path(__file__).resolve().parents[1]


def test_bundled_pipeline_dir_is_the_repository_folder():
    path = bundled_pipeline_dir()
    assert isinstance(path, Path)
    assert path.resolve() == (REPO / "pyannote").resolve()
    assert (path / "config.yaml").is_file()
