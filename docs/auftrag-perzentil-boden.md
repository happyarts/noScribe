# Auftrag: robuster Clamp-Boden für den log-Mel-Pfad

Diese Datei ist für eine frische Sitzung geschrieben. Sie ist ein **Arbeitsauftrag,
keine Dokumentation**: alles darin wurde am 2026-08-22 gemessen, und bei jeder Zahl
steht, womit — damit nachgeprüft statt geglaubt wird.

**Stand: noch nicht eingebaut.** Die Messlage trägt den Einbau, die Gegenargumente
sind ernst und stehen vollständig in Abschnitt 4. Wer diesen Auftrag ausführt, soll
Abschnitt 4 gelesen haben, bevor er Abschnitt 5 anfasst.

## 1. Der Mechanismus, konkret

Das log-Mel ist eine Tabelle aus 128 Frequenzbändern × 100 Zeilen je Sekunde. Weil
`log10` verwendet wird, ist **eine Einheit = 10 dB**. Die Klemmung lautet
(`mlx_voxtral/audio_processing.py:330–342`, zeichengleich in transformers,
faster-whisper und transcribe.cpp):

```python
log_spec = log10(maximum(mel_spec, 1e-10))
log_spec = maximum(log_spec, log_spec.max() - 8.0)   # <- hier
log_spec = (log_spec + 4.0) / 4.0
```

Also: **ein Noise-Gate 80 dB unter der lautesten Zelle** der ganzen Eingabe, gefolgt
von einer *festen* Affinität. Zwei Folgerungen, die den ganzen Auftrag tragen:

1. Wo das Gate liegt, entscheidet **eine Zelle von Millionen** (bei 300 s: eine von
   3,8 Mio.). Ein einzelner lauter Ton hebt den Boden für den gesamten Durchgang.
2. Weil die Affinität fest ist, wird der Pegel **nicht** normalisiert — er verschiebt
   das Bild nur, und dagegen ist das Modell unempfindlich
   (`voxtral-audio-vorverarbeitung.md` §2, §4).

Gemessen: ein 0,1-s-Knall hebt den Boden um **9,6 dB** und verändert **100 %** der
Frames im unberührten Audio davor.

## 2. Was das kostet

Dosis-Wirkung auf beiden Handreferenzen. Das Audio wird gedämpft, der Knall bleibt
konstant, verglichen wird immer *innerhalb* eines Bodens (sauber gegen Knall) —
damit fällt jede Verzerrung eines Referenztranskripts heraus.

| Knall über der Sprachspitze | `hart` max | `hart` Perz. | `zoom` max | `zoom` Perz. |
|---|---|---|---|---|
| ~ +2,6 / +0,3 dB | +1,66 | +0,00 | −0,12 | +0,00 |
| ~ +14,7 / +12,4 dB | +2,61 | +0,24 | +1,86 | +0,00 |
| ~ +27,1 / +24,8 dB | **+5,21** | +0,00 | **+11,99** | +0,00 |

dWER in Punkten. Monoton auf beiden Passagen, der Perzentil-Boden praktisch flach.
+11,99 Punkte sind 103 Wörter von 859 — bei einer leise ausgesteuerten Aufnahme mit
einer zugeschlagenen Tür.

**Echtheitsmerkmal, das anfangs wie ein Verdachtsmoment aussah:** auf `hart` kostet
der Knall an drei verschiedenen Positionen (5 %, 45 %, 85 %) **exakt gleich viel**
(+1,66). Das ist genau, was der Mechanismus vorhersagt — der Boden steigt
positionsunabhängig — und deshalb ein Beleg, keine Auffälligkeit.

**Der Ausreißer ist kein Laborartefakt.** Abstand zwischen der lautesten Mel-Zelle
und dem 99,9-Perzentil in echtem Material (`mel_outlier_gap.py`):

| Material | max − p99,9 |
|---|---|
| Interview A, roh | 9,4 dB |
| Interview B, roh | 9,1 dB |
| Podcast, roh | 13,2 dB |
| Podcast, gemastert | 9,3 dB |
| Podcast, unbearbeitete Rohspur | **21,9 dB** |
| `zoom` (Videokonferenz) | 14,4 dB |
| VoxPopuli-Ströme, Median | 13,0 dB |

In **jeder** Aufnahme wird der Boden 7–22 dB höher gesetzt, als ein robustes Maß ihn
setzen würde, am stärksten bei rohem Material — also genau bei dem, was Nutzer bringen.

## 3. Was der Perzentil-Boden auf sauberem Material kostet: nichts nachweisbares

Gemessen auf **VoxPopuli de** (spontane Parlamentsrede mit Goldtranskript, 11,4 % WER
gegen FLEURS' 4,8 % — der härtere externe Korpus, siehe `voxpopuli_floor.py`),
10 Ströme à 300 s, 50,6 min, gepaart auf bitgleichem Audio:

> max 11,39 % / 7,32 % · Perzentil 99,9 11,29 % / 7,20 %
> **dWER −0,10 [−0,83, +0,47] · dCER −0,12 [−0,77, +0,38]**

Beide Intervalle enthalten die Null. Wichtig für die Auslegung: diese Ströme haben mit
13,0 dB Median denselben Ausreißer-Abstand wie unser echtes Material — der Boden wurde
dort also real um 13 dB gesenkt, ohne Wirkung.

**Die Asymmetrie, die beide Befunde zusammenbringt:** den Boden zu *senken* zeigt mehr
Rauschdetail, das das Modell ignoriert (VoxPopuli: null). Den Boden zu *heben* zerstört
leise Sprachdetails (Knalltests: bis +11,99). Der Perzentil-Boden liegt **immer** unter
oder gleich dem Maximum-Boden — er kann also nur verhindern, dass gehoben wird.

## 4. Was dagegen spricht — vor dem Einbau lesen

1. **Alle vier Referenzimplementierungen nehmen das nackte Maximum**: transformers,
   mlx-audio (delegiert an transformers), transcribe.cpp (`per_utterance`) und
   mlx-voxtral 0.0.6. Damit wurde das Modell trainiert.
2. **Die Eingabe verlässt den trainierten Wertebereich.** Der Ist-Zustand liefert per
   Konstruktion **exakt 2,000** Einheiten Spannweite. Mit Perzentil-Boden gemessen:
   2,182 (`hart`), 2,235 (Interview), 2,360 (`zoom`), 2,547 (Rohspur) — bis **27 %
   breiter**, nach unten. In [openai/whisper#269](https://github.com/openai/whisper/discussions/269)
   warnt der Whisper-Autor genau davor: ohne die Normalisierung werde die Eingabe
   *out-of-distribution* und die Leistung falle deutlich ab. Unsere Messung sagt, dass
   die hier auftretende Verbreiterung folgenlos ist — aber sie ist der Grund, warum das
   nicht ungemessen eingebaut werden darf.
3. **Es ist eine Eigenentwicklung.** Eine Websuche findet keinen Bericht dieses
   Mechanismus; Abwesenheit von Treffern ist kein Beweis, heißt aber: es gibt keine
   fremde Erfahrung, auf die wir uns stützen können.

## 5. Der Einbau

**Wo.** `noScribe/voxtral_engine.py`, `_Voxtral.transcribe_array` holt die Features
über `self.proc.apply_transcrition_request(...)`. Der Eingriff ist ein Ersatz für
`vox.proc.feature_extractor`, genau wie ihn die Messskripte vornehmen — die Bibliothek
selbst wird nicht angefasst.

**Wie — Einpass-Variante.** Der `global_max`-Parameter der Bibliothek ist der Hebel:
ein sehr negativer Wert klemmt nichts, liefert also das rohe log-Mel; die Affinität ist
invertierbar. Danach wird **in numpy** geklemmt, statt die FFT ein zweites Mal zu
rechnen. Nachgemessen: bitgleich zur Zwei-Pass-Fassung und **1,58× schneller**
(38,2 ms gegen 60,3 ms auf 300 s).

```python
raw = np.array(log_mel_spectrogram(arr, global_max=-1e6)) * 4.0 - 4.0
floor = np.percentile(raw, 99.9) - 8.0
mel = (np.maximum(raw, floor) + 4.0) / 4.0
```

**Perzentil 99,9.** Auf `hart` sauber gemessen: 99,9 → 4,50 %/3,10 %, 99,0 → 5,45 %/3,29 %,
99,99 → 4,98 %/3,34 %, Maximum → 4,27 %/3,39 %. 99,0 ist erkennbar schlechter, 99,9 der
beste Kompromiss. **Ein Sweep über mehrere Perzentile auf VoxPopuli fehlt und wäre die
sinnvollste zusätzliche Messung.**

## 6. Abnahmekriterien

1. **Perzentil-100-Kontrolle**: mit `pct=100` muss die Ausgabe **bitgleich** zum
   eingebauten Pfad sein. Das beweist, dass nur der Boden geändert wurde. Diese
   Kontrolle gehört als Test ins Repo.
2. Ein Test, der den Knall-Fall festnagelt: dieselbe Passage mit und ohne eingesetzten
   Transienten muss unter dem Perzentil-Boden **denselben Text** liefern.
3. `venv/bin/python3 -m pytest tests/ -q` grün (aktuell 372).
4. Die Zahlen aus Abschnitt 3 auf VoxPopuli reproduzieren, bevor die Doku umgeschrieben
   wird.

## 7. Was NICHT zu tun ist

* **Kein Limiter auf dem Audio.** Er greift dieselbe Physik an, bezahlt aber im Signal,
  und dort gibt es Zahlen: §4 des Messdokuments hat drei Pegelwerkzeuge auf echtem
  Material gemessen — `dynaudnorm` neutral, `loudnorm16` und `speechnorm` mit
  Intervallen, die die Null ausschließen, also schlechter. Zusätzlich ist er irreversibel.
* **Keine Lautheitsnormalisierung.** Sie multipliziert die Wellenform mit einer
  Konstanten, was im Log-Raum ein konstanter Summand auf **jede** Zelle ist, den
  Ausreißer eingeschlossen. Der Abstand Knall-zu-Sprache bleibt, der Boden hängt weiter
  am Knall. Sie verschiebt genau das, was das Modell ohnehin ignoriert. Hilft hier **nicht**.
* **Kein fester Boden.** Voxtral Realtime macht das (`global_log_mel_max` in
  transcribe.cpp), es wäre auch chunk-unabhängig — erzwingt dann aber die
  Lautheitsnormalisierung, die eben verworfen wurde. Größerer Umbau, kein Mehrwert hier.
* **Die 8.0 nicht anfassen.** Sie entspricht librosas `top_db=80` und ist der Bereich,
  auf dem das Modell trainiert wurde.

## 8. Offene Fäden

* **Der Whisper-Pfad trägt dieselbe Zeile** (`faster_whisper/feature_extractor.py:227`),
  und noScribe übergibt Whisper die ganze Datei — es beträfe also die Standard-Engine
  aller Nutzer. Ein Effekt ließ sich **nicht zeigen**: Whispers eigene Streuung auf
  `hart` (9,00 % mit VAD gegen 21,09 % ohne) ist größer als der gesuchte Unterschied.
  Offen, nicht widerlegt. Eine eigene Untersuchung wert, weil die Tragweite größer wäre
  als bei Voxtral.
* **Chunk-Kopplung.** Der Boden wird über das berechnet, was in einem Durchgang übergeben
  wird — bei uns also über die vom Chunker gewählte Länge. Gemessen: dieselben 150 s in
  zwei Chunkungen verschieben den Boden um 1,14 dB und verändern 100 % der Frames. Der
  Perzentil-Boden ist gegen diese Kopplung stabiler, aber nicht immun; gemessen ist es
  nicht.

## 9. Skripte

| Skript | Wofür |
|---|---|
| `docs/skripte/voxpopuli_floor.py` | gepaarter Vergleich der Böden auf sauberem VoxPopuli; enthält `PercentileFloor` und `load_clips` |
| `docs/skripte/voxpopuli_spike.py` | Transientenschaden über viele Ströme, gepaart je Boden |
| `docs/skripte/mel_outlier_gap.py` | Abstand Maximum zu Perzentil in echtem Material, ohne Modell |

Alle mit `venv/bin/python3` starten (es gibt kein `python`).
