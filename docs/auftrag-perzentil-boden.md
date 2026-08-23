# Auftrag: robuster Clamp-Boden für den log-Mel-Pfad

Arbeitsauftrag für eine frische Sitzung. **Die Belege stehen nicht hier**, sondern
in `voxtral-audio-vorverarbeitung.md` **§6b** — dort ist die Messlage vollständig,
mit Intervallen und Skripten. Diese Datei sagt nur, was zu tun ist, was dagegen
spricht und woran der Einbau scheitert.

**Stand: umgesetzt am 2026-08-23** — `_PercentileFloorFeatures` und
`clamp_log_mel` in `noScribe/voxtral_engine.py`, Konstante `MEL_FLOOR_PERCENTILE`,
Tests in `tests/test_mel_floor.py`. Was beim Einbau anders kam als hier gedacht,
steht am Ende unter „Ergebnis des Einbaus"; die Messlage dazu in §6b.

## Worum es geht, in fünf Zeilen

Der log-Mel-Boden liegt bei `log_max - 8`, wobei `log_max` das Maximum über die
**ganze Eingabe** ist. Eine einzige laute Zelle setzt damit den Boden für den
gesamten Durchgang. Ein tieffrequenter Türknall hebt ihn um rund 27 dB und kostet
gemessen **+1,52 WER-Punkte** [+0,78, +2,26]. Ein Perzentil statt des Maximums
beseitigt das vollständig (+0,04) und kostet auf sauberem Material nichts
(−0,10 [−0,83, +0,47] über 50 min VoxPopuli).

## Was dagegen spricht — vor dem Einbau lesen

1. **Alle vier Referenzimplementierungen nehmen das nackte Maximum**: transformers,
   mlx-audio (delegiert an transformers), transcribe.cpp (`per_utterance`) und
   mlx-voxtral 0.0.6. Damit wurde das Modell trainiert.
2. **Die Eingabe verlässt den trainierten Wertebereich** — bis 27 % breiter nach
   unten, gemessen. In [openai/whisper#269](https://github.com/openai/whisper/discussions/269)
   warnt der Whisper-Autor genau davor. Bei den real auftretenden Breiten ist es
   folgenlos, aber es ist der Grund, warum hier nichts ungemessen eingebaut wird.
3. **Es ist eine Eigenentwicklung.** Eine Websuche findet keinen Bericht dieses
   Mechanismus — es gibt also keine fremde Erfahrung, auf die man sich stützen kann.

## Der Einbau

**Wo.** `noScribe/voxtral_engine.py`, `_Voxtral.transcribe_array` holt die Features
über `self.proc.apply_transcrition_request(...)`. Der Eingriff ist ein Ersatz für
`vox.proc.feature_extractor`, wie ihn die Messskripte vornehmen — die Bibliothek
selbst wird nicht angefasst.

**Der Fix bleibt Voxtral-only, und zwar von selbst.** Der Whisper-Pfad trägt
dieselbe Zeile, ist aber nachweislich nicht betroffen (§6b). Es braucht **keine**
Option im gemeinsamen Teil, die nur für Voxtral aktiv wäre, und **keinen** PR an
faster-whisper. Der Code liegt ohnehin in einer Datei, die es upstream nicht gibt,
und geht mit dem Voxtral-Feature hoch.

**Wie — Einpass-Variante.** Der `global_max`-Parameter der Bibliothek ist der Hebel:
ein sehr negativer Wert klemmt nichts, liefert also das rohe log-Mel; die Affinität
ist invertierbar. Danach wird **in numpy** geklemmt, statt die FFT ein zweites Mal
zu rechnen — bitgleich zur Zwei-Pass-Fassung und **1,58× schneller** (38,2 ms gegen
60,3 ms auf 300 s).

```python
raw = np.array(log_mel_spectrogram(arr, global_max=-1e6)) * 4.0 - 4.0
floor = np.percentile(raw, 99.9) - 8.0
mel = (np.maximum(raw, floor) + 4.0) / 4.0
```

**Perzentil 99,9**, weil die Schadensläufe damit gefahren wurden. Die Wahl ist
unkritisch: 99, 99,9 und 99,99 sind auf sauberem Material ununterscheidbar vom
Maximum und voneinander (§6b).

## Abnahmekriterien

1. **Perzentil-100-Kontrolle**: mit `pct=100` muss die Ausgabe **bitgleich** zum
   eingebauten Pfad sein. Das beweist, dass nur der Boden geändert wurde, und
   gehört als Test ins Repo.
2. Ein Test, der den Knall-Fall festnagelt: dieselbe Passage mit und ohne
   eingesetzten Transienten muss unter dem Perzentil-Boden **denselben Text**
   liefern.
3. `venv/bin/python3 -m pytest tests/ -q` grün (aktuell 372).
4. Die Zahlen aus §6b reproduzieren, bevor die Doku umgeschrieben wird.

## Was NICHT zu tun ist

* **Kein Limiter auf dem Audio.** Er greift dieselbe Physik an, bezahlt aber im
  Signal, und dort gibt es Zahlen: §4 hat drei Pegelwerkzeuge auf echtem Material
  gemessen — `dynaudnorm` neutral, `loudnorm16` und `speechnorm` mit Intervallen,
  die die Null ausschließen, also schlechter. Zusätzlich ist er irreversibel.
* **Keine Lautheitsnormalisierung.** Sie multipliziert die Wellenform mit einer
  Konstanten, was im Log-Raum ein konstanter Summand auf **jede** Zelle ist, den
  Ausreißer eingeschlossen. Der Abstand Knall-zu-Sprache bleibt, der Boden hängt
  weiter am Knall. Hilft hier **nicht**.
* **Kein fester Boden.** Voxtral Realtime macht das (`global_log_mel_max` in
  transcribe.cpp) und wäre auch chunk-unabhängig — erzwingt dann aber die
  Lautheitsnormalisierung, die eben verworfen wurde.
* **Die 8.0 nicht anfassen.** Sie entspricht librosas `top_db=80` und ist der
  Bereich, auf dem das Modell trainiert wurde.

## Skripte

| Skript | Wofür |
|---|---|
| `docs/skripte/voxpopuli_floor.py` | Böden auf sauberem Material, gepaart; enthält `load_clips` (mit `VOXPOPULI_PARQUET` aus der lokalen Datei) |
| `docs/skripte/voxpopuli_spike.py` | Transientenschaden über viele Ströme; `pair()` garantiert die Dosis; Perzentil-Liste als viertes Argument |
| `docs/skripte/voxpopuli_sparse.py` | dasselbe auf Strömen, die überwiegend still sind (Sprachanteil wählbar) |
| `docs/skripte/mel_outlier_gap.py` | Abstand Maximum zu Perzentil in echtem Material, ohne Modell |
| `docs/skripte/whisper_spike_streams.py` | derselbe Test auf der Whisper-Standardengine |

Alle mit `venv/bin/python3` starten (es gibt kein `python`). Die Perzentil-Arme
kommen seit dem Einbau aus `_PercentileFloorFeatures`, der Max-Arm explizit aus
`VoxtralFeatureExtractor()` — `vox.proc.feature_extractor` ist jetzt der
Perzentil-Pfad und taugt nicht mehr als Stock-Arm.

## Zwei Messfallen, beide selbst hineingetappt

* **Ein über die Sprachspitze skalierter Impuls wird gekappt.** Echtes Material ist
  bereits bis 1,0 ausgesteuert; verlangte +12 dB kamen als 5,2 dB Bodenanstieg an,
  und das Ergebnis sah aus wie ein sauberer Nulleffekt. Richtig: Audio dämpfen,
  Impuls auf Vollausschlag. `voxpopuli_spike.py` bricht jetzt ab, wenn die Dosis
  nicht ankommt.
* **Ein Punktschätzer ohne tragfähiges Intervall zeigt in die falsche Richtung.**
  Whisper stand auf 8 Strömen bei +0,60/+0,95, auf 24 bei −0,20/−0,51. Und dass der
  Schaden auf einer Passage an drei Knallpositionen „exakt gleich groß" war, belegte
  Determinismus, nicht Allgemeinheit.

## Ergebnis des Einbaus (2026-08-23)

Die Belege mit Zahlen stehen wieder in §6b; hier nur, was vom Plan oben abwich.

* **Nicht über den `global_max`-Hebel.** Die Einpass-Variante ist nicht per
  Konstruktion bitgleich: die Affinität `(x+4)/4` verliert für `x > −2`
  Mantissenbits, und auf leisem Material (`log_max` in [−2, 0)) bekommt jede
  achte Eingabe daraus einen um ein Bit falschen Boden. Die float64-Fassung des
  Snippets oben (`np.percentile` hebt unter NumPy 2 das ganze Array auf float64)
  trifft noch öfter daneben. Der Einbau rechnet das ungeklemmte Spektrogramm
  aus den Primitiven der Bibliothek und klemmt in float32; Abnahmekriterium 1
  gilt damit auf jedem Pegel, auch dem gedämpften `hart`-Fall.
* **Perzentil 99, nicht 99,9.** Die Wahl ist nur auf sauberem Material
  unkritisch. Unter einem Transienten verdrängen dessen Zellen die Statistik um
  ihre Anzahl; ein breitbandiger 100-ms-Rauschburst (~1300 Zellen) lässt bei
  99,9 auf einem 60-s-Pass 7 dB Bodenanstieg stehen, bei 99 noch 0,75 dB. Auf
  den Strömen: +0,04 [+0,00, +0,10] gegen +0,01 [−0,03, +0,06].
* **Abnahmekriterium 2 hält nicht wörtlich.** Der Text ist mit und ohne Knall
  nicht derselbe, sondern fast derselbe (3 statt 15 geänderte Stellen auf 120 s
  Interview), weil der Boden um einen Bruchteil eines dB wandert und nicht um
  null. Der Test verlangt deshalb eine kleine Differenz, deutlich unter der des
  Max-Bodens, statt Gleichheit.
* **Polsterfrei.** Das Perzentil über den nullgepolsterten 30-s-Block hätte
  bei kurzen Eingaben einen bis 18 dB tieferen Boden ergeben; die Statistik
  nimmt nur die echten Frames (Review-Fund).
* **Stille Pässe gemessen, nicht abgesichert.** Bei 50 % Raumton zwischen
  den Äußerungen kostet der Perzentil-Boden +0,24 [−0,03, +0,55], bei 20 %
  +0,07 [−0,72, +1,10] — kein Nachweis, aber ein dünner Rand. Eine Kappe nach
  unten ist als Option in §6b beschrieben und ungemessen.
* **Kriterium 4 erfüllt:** +1,52 [+0,78, +2,26] und +0,04 [+0,00, +0,10] auf
  die zweite Stelle reproduziert, durch den Produktionscode.
