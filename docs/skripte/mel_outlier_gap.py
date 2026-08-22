"""Wie weit liegt die lauteste Mel-Zelle über dem Rest? Auf echtem Material.

Der Clamp-Boden ist `log_max - 8`, und `log_max` ist **eine** Zelle von
Millionen. Ob das fragil ist, hängt allein daran, wie weit diese eine Zelle
über dem übrigen Material liegt: fällt sie mit einem hohen Perzentil zusammen,
sind beide Böden identisch und die ganze Frage ist gegenstandslos. Liegt sie
weit darüber, wird der Boden von einem Ausreißer gesetzt.

Gemessen wird deshalb der **Abstand in dB zwischen dem Maximum und mehreren
Perzentilen** desselben log-Mel, pro Datei und zusätzlich über gleitende
Fenster von Passlänge -- denn was zählt, ist der Ausreißer *innerhalb* dessen,
was in einem Durchgang zusammen normalisiert wird.

Kein Modell, nur Signal. Ergebnis in dB: 10 dB Abstand heißt, der Boden liegt
10 dB höher, als ein robustes Maß ihn setzen würde.

    python docs/skripte/mel_outlier_gap.py <audio> [audio ...]
"""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mlx_voxtral.audio_processing import log_mel_spectrogram   # noqa: E402

SR = 16000
PCTS = (99.99, 99.9, 99.0)
WINDOW_SEC = 600.0          # eine typische Passlänge des Chunkers


def load16k(path):
    """Beliebiges Format -> 16 kHz mono float32, über ffmpeg wie die Pipeline."""
    if str(path).lower().endswith(".wav"):
        a, sr = sf.read(str(path), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr == SR:
            return np.ascontiguousarray(a)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        tmp = fh.name
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", str(path),
                    "-ar", str(SR), "-ac", "1", tmp], check=True)
    a, _ = sf.read(tmp, dtype="float32")
    pathlib.Path(tmp).unlink(missing_ok=True)
    return np.ascontiguousarray(a)


def raw_log_mel(a):
    """Ungeklemmtes log-Mel: der `global_max`-Hebel, Affinität invertiert."""
    cs = 30 * SR
    n = int(np.ceil(len(a) / cs))
    a = np.pad(a, (0, n * cs - len(a)))
    return np.array(log_mel_spectrogram(mx.array(a), global_max=-1e6)) * 4.0 - 4.0


def gaps(mel):
    """Abstand Maximum - Perzentil, in dB (eine log10-Einheit = 10 dB)."""
    mx_ = float(mel.max())
    return {p: (mx_ - float(np.percentile(mel, p))) * 10.0 for p in PCTS}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("    python")[1].strip())
    hdr = "  ".join(f"max-p{p:g}" for p in PCTS)
    print(f"{'Datei':34s} {'Dauer':>7s}  {hdr}   schlimmstes {WINDOW_SEC:.0f}s-Fenster")
    for path in sys.argv[1:]:
        a = load16k(path)
        mel = raw_log_mel(a)
        g = gaps(mel)
        # gleitende Fenster: der Ausreißer zählt nur innerhalb eines Passes
        step = int(WINDOW_SEC * 100)          # 100 Mel-Frames je Sekunde
        worst = 0.0
        for s in range(0, max(1, mel.shape[1] - step + 1), step):
            w = mel[:, s:s + step]
            if w.shape[1] < step // 2:
                continue
            worst = max(worst, gaps(w)[99.9])
        name = pathlib.Path(path).name[:33]
        cells = "  ".join(f"{g[p]:7.1f}" for p in PCTS)
        print(f"{name:34s} {len(a)/SR:6.0f}s  {cells}   {worst:20.1f} dB", flush=True)


if __name__ == "__main__":
    main()
