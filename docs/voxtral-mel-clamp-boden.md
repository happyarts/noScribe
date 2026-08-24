# Der Clamp-Boden des log-Mel

Stand: 24. August 2026 · M1 Max · Build `voxtral-mini-8bit`, mlx-voxtral 0.0.6.
Alle Zahlen gemessen, keine geschätzt.

Das log-Mel-Spektrogramm wird nach unten geklemmt, bevor das Modell es sieht.
Wo dieser Boden liegt, entschied die Bibliothek über das **Maximum der ganzen
Eingabe** — womit eine einzige laute Zelle ihn für den gesamten Durchgang
festlegt. Das kostet auf echtem Material messbar Erkennungsqualität, und seit
dem 23.08.2026 nimmt die Engine stattdessen ein **Perzentil**. Diese Datei ist
die Messlage dazu: was der alte Boden kostet, was der neue kostet, was dagegen
spricht, und wo seine Grenze liegt.

Der Mechanismus selbst steht im Code, beim Konstanten-Block über
`MEL_FLOOR_PERCENTILE` in `noScribe/voxtral_engine.py`; die Implementierungs-
Feinheiten in den Docstrings von `clamp_log_mel` und `_PercentileFloorFeatures`
und in `tests/test_mel_floor.py`. Hier steht, was dort keinen Platz hat.

---

## 1. Warum der Boden wandert

Der Boden liegt bei `log_max − 8`, `log_max` ist das Maximum über die ganze
Eingabe. Ein 0,1-s-Knall hebt ihn um **9,6 dB** und verändert **100 %** der
Frames im unberührten Audio davor.

**Welche echten Geräusche welche Dosis erzeugen** (gleiche Spitzenamplitude,
leise Aufnahme): Klatschen 10,3 dB, Rauschburst 17,3 dB, **tieffrequenter
Türknall 27,1 dB**, Rechteckblock 31,5 dB. Der teuerste Fall ist der
alltäglichste.

**Der Ausreißer ist kein Laborartefakt.** Abstand zwischen der lautesten
Mel-Zelle und dem 99,9-Perzentil desselben Spektrogramms, ohne Modell gemessen
(`mel_outlier_gap.py`): Interview roh 9,4 und 9,1 dB, Podcast roh 13,2 dB,
Podcast gemastert 9,3 dB, unbearbeitete Rohspur **21,9 dB**, `zoom` 14,4 dB,
VoxPopuli-Ströme im Median 13,0 dB. In **jeder** Aufnahme wird der Boden
7–22 dB höher gesetzt, als ein robustes Maß ihn setzen würde, am stärksten bei
rohem Material — also genau bei dem, was Nutzer bringen.

## 2. Was der Max-Boden kostet

**Die belastbare Zahl steht auf Strömen.** Zehn VoxPopuli-Ströme à 300 s, jeder
einmal sauber und einmal mit eingesetztem Transienten, gepaart je Boden
(`voxpopuli_spike.py`):

| Bodenanstieg (Median) | max-Boden | Perzentil 99,9 |
|---|---|---|
| 5,2 dB | −0,07 [−0,29, +0,15] | +0,04 [+0,00, +0,09] |
| 16,2 dB | **+0,35 [+0,03, +0,66]** \* | +0,03 [+0,00, +0,07] |
| 28,2 dB | **+1,52 [+0,78, +2,26]** \* | +0,04 [+0,00, +0,10] |

\* = Intervall schließt die Null aus. Monoton, bei den beiden oberen Dosen
nachweisbar, der Perzentil-Boden bei allen dreien flach. **+1,52 bei 28 dB ist
die Zahl, die trägt.**

Die Passagenmessungen davor sagten +1,66 bis **+5,21** (`hart`, gedämpft, Knall
konstant über +2,6 / +14,7 / +27,1 dB) und einmal +11,99 — sie **überschätzen um
das Vier- bis Achtfache**. Unter Greedy-Decoding kippt auf einer einzelnen
Passage ein Token und der Rest zieht nach; die Strommessung mittelt das heraus.
Aus demselben Grund war die Beobachtung, der Schaden sei an drei
Knallpositionen „exakt gleich groß", kein Beleg für Allgemeinheit, sondern nur
für Determinismus — mechanisch richtig ist sie trotzdem, der Boden steigt ja
positionsunabhängig.

## 3. Der Perzentil-Boden: was er kostet und welches Perzentil

**Auf sauberem Material: nichts nachweisbares.** Gemessen auf **VoxPopuli de** —
spontane Parlamentsrede mit Goldtranskript, der Korpus, der uns gegenüber dem
vorgelesenen FLEURS fehlte, und mit 11,4 % WER deutlich härter — 10 Ströme à
300 s, 50,6 min, gepaart auf bitgleichem Audio (`voxpopuli_floor.py`). Sweep über
die Kandidaten, alle gegen dieselbe Max-Basis: 99 → **−0,12** [−0,85, +0,49],
99,9 → −0,10 [−0,83, +0,47], 99,99 → −0,25 [−0,99, +0,31]. Alle drei
ununterscheidbar vom Maximum und voneinander.

**Für die Robustheit ist das Perzentil sehr wohl eine Wahl.** Die Zellen des
Knalls sitzen alle an der Spitze der Verteilung und verdrängen die
Ordnungsstatistik um ihre Anzahl — um so mehr, je kürzer der Pass und je
breitbandiger der Knall. Bodenverschiebung in dB auf `tests/data/interview.mp3`
(−20 dB ausgesteuert, Transient auf Vollausschlag, ohne Modell):

| Pass | Transient | max | 99 | 99,9 | 99,99 |
|---|---|---|---|---|---|
| 60 s | Rechteckblock 100 ms | 29,6 | 0,15 | 0,50 | 4,38 |
| 60 s | Türknall 60 Hz 100 ms | 25,5 | 0,05 | 0,22 | 1,81 |
| 60 s | **Rauschburst 100 ms** | 9,7 | **0,75** | **7,10** | 9,20 |
| 60 s | Klatschen 5 ms | 4,2 | 0,13 | 0,45 | 2,03 |
| 120 s | Rechteckblock 100 ms | 29,6 | 0,07 | 0,27 | 1,45 |
| 120 s | Rauschburst 100 ms | 9,9 | 0,33 | 2,90 | 8,33 |
| 300 s | Rechteckblock 100 ms | 30,5 | 0,03 | 0,12 | 0,54 |
| 300 s | Rauschburst 100 ms | 10,0 | 0,14 | 0,93 | 7,57 |

Der Rauschburst belegt rund 1300 Zellen über der Schwelle; das oberste Promille
eines 60-s-Passes sind 768 Zellen, das oberste Prozent 7680. **99,99 scheidet
damit ganz aus, und 99 schlägt 99,9 um eine Größenordnung** — die beiden
Aussagen „unkritisch" und „sehr wohl eine Wahl" widersprechen sich nicht, sie
gelten für verschiedene Fragen: die Kosten auf sauberem Material sind es nicht,
die Robustheit ist es.

**Der Schutz ist mengenbegrenzt.** Breitbandiges Rauschen, das mehr als das
oberste Prozent eines Passes füllt — grob 0,6 s auf 60 s —, gilt als Signal und
hebt den Boden wieder wie zuvor.

## 4. Was dagegen spricht

**Die Eingabe verlässt den trainierten Wertebereich.** Der Ist-Zustand liefert
per Konstruktion **exakt 2,000** Einheiten Spannweite. Mit Perzentil-Boden:
2,182 (`hart`), 2,235 (Interview), 2,360 (`zoom`), 2,547 (Rohspur) — bis **27 %
breiter**, nach unten. In
[openai/whisper#269](https://github.com/openai/whisper/discussions/269) warnt der
Whisper-Autor genau davor: ohne diese Normalisierung liege die Eingabe
*out-of-distribution*. Bei den Breiten, die echtes Material erzeugt, ist es
gemessen folgenlos — aber es ist der Grund, warum das nicht ungemessen eingebaut
werden durfte. Mit 99 wird die Spannweite etwas breiter als mit 99,9 (Interview
2,39–2,45 gegen 2,22–2,27), bleibt aber unter den 2,547, die die Rohspur mit
99,9 folgenlos erreicht.

**Architekturtreue.** Alle vier Referenzimplementierungen nehmen das nackte
Maximum — transformers, mlx-audio (delegiert an transformers), transcribe.cpp
(`per_utterance`) und mlx-voxtral 0.0.6. Damit wurde das Modell trainiert. Und
eine Websuche findet keinen Bericht dieses Mechanismus: es gibt keine fremde
Erfahrung, auf die man sich stützen könnte.

**Warum der Eingriff trotzdem hier sitzt und nicht im Signal.** Das Maximum wird
*ausschließlich* für den Boden benutzt — danach folgt eine feste Affinität —,
also ist eine robuste Statistik genau dort der richtige Ort. Die Alternativen,
alle verworfen:

* **Limiter auf dem Audio** — greift dieselbe Physik an, bezahlt aber im Signal
  und ist irreversibel. `voxtral-audio-vorverarbeitung.md` §4 hat drei
  Pegelwerkzeuge auf echtem Material gemessen: neutral bis schädlich.
* **Lautheitsnormalisierung** — hilft gar nicht. Sie multipliziert die
  Wellenform mit einer Konstanten, im Log-Raum ein konstanter Summand auf
  *jede* Zelle, den Ausreißer eingeschlossen. Der Abstand Knall-zu-Sprache
  bleibt, der Boden hängt weiter am Knall.
* **Fester Boden** — Voxtral Realtime macht das (`global_log_mel_max` in
  transcribe.cpp) und wäre auch chunk-unabhängig, erzwingt dann aber genau die
  eben verworfene Lautheitsnormalisierung.
* **Die 8.0** bleibt unangetastet: sie entspricht librosas `top_db=80` und ist
  der Bereich, auf dem das Modell trainiert wurde.

## 5. Die Grenze: stille Pässe, und die verworfene Kappe

Alle Ströme oben sind dichte Rede; dort liegt das 99. Perzentil 19,5–26,8 dB
unter dem Maximum (Median 22,0 — auf dem redigierten Interview nur 15,6–17,8).
In einem Pass, der vor allem aus Pausen besteht — die 60-s-Kopfprobe auf einer
Aufnahme, die mit Stille beginnt —, rutscht das Perzentil innerhalb der
Sprachzellen nach unten und der Boden liegt tiefer als je gemessen.
`voxpopuli_sparse.py` misst das: dieselben Clips, dazwischen weißes Rauschen bei
−60 dBFS als Raumton, bis der Sprachanteil stimmt.

| Sprachanteil | Abstand max → p99 | max | Perzentil 99 | gepaart |
|---|---|---|---|---|
| ~100 % | 22,0 dB (19,5–26,8) | 11,39 % / 7,32 % | 11,27 % / 7,14 % | −0,12 [−0,85, +0,49] · CER −0,18 [−0,79, +0,28] |
| 50 % | 23,6 dB (20,7–28,9) | 10,14 % / 6,35 % | 10,38 % / 6,67 % | **+0,24 [−0,03, +0,55]** · CER +0,32 [−0,01, +0,77] |
| 20 % | 28,5 dB (26,1–30,0) | 12,30 % / 8,25 % | 12,36 % / 8,54 % | +0,07 [−0,72, +1,10] · CER +0,28 [−0,25, +1,02] |

Kein Intervall schließt die Null aus, aber bei 50 % fehlt dazu ein Hundertstel,
und beide CER-Intervalle stehen knapp darunter. Der tiefere Boden zeigt in den
Pausen Rauschstruktur, die die dichten Ströme nie hatten — die Asymmetrie an
ihrer Grenze. Der Raumton ist hier synthetisch und weiß, echter wäre
tieffrequenter; und der 20-%-Lauf trägt nur 10 min Sprache.

**Die Kappe, gemessen und verworfen (2026-08-24).** Der naheliegende Ausweg ist
ein Boden, der nie mehr als 20 oder 25 dB unter dem Maximum liegt:
`max(p99, log_max − Kappe) − 8`. `clamp_log_mel` trägt dafür einen
`cap`-Parameter (Voreinstellung `MEL_FLOOR_CAP = None`), die Skripte die
Arm-Kurzform `99c20`. Alle vier Regime, gleiche Ströme:

| Regime | max | p99 | p99 Kappe 20 dB | p99 Kappe 25 dB |
|---|---|---|---|---|
| Knall 28 dB, Kosten sauber→Knall | **+1,52 [+0,78, +2,26]** \* | +0,01 [−0,03, +0,06] | **+0,81 [+0,18, +1,57]** \* | **+0,75 [+0,10, +1,53]** \* |
| dicht sauber, gegen max | — | −0,12 [−0,85, +0,49] | −0,07 [−0,81, +0,54] | −0,10 [−0,83, +0,51] |
| 50 % still, gegen max | — | +0,24 [−0,03, +0,55] | **+0,05 [−0,17, +0,37]** | +0,24 [−0,03, +0,55] |
| 20 % still, gegen max | — | +0,07 [−0,72, +1,10] | +0,07 [−0,67, +1,07] | +0,07 [−0,67, +1,07] |

Die Kappe tut auf den stillen Strömen, was sie soll (+0,24 → +0,05 bei 50 %),
aber unterm Knall verliert sie nachweisbar: der Knall hebt das Maximum, die
Kappe hängt am Maximum, also steigt der Boden wieder mit. Die +0,81 bestehen
dabei zu etwa drei Vierteln aus einem zurückgegebenen Gewinn — auf dem um 24 dB
gedämpften Knall-Korpus ist der tiefe Perzentil-Boden dem Max-Boden um rund 0,6
Punkte voraus (11,36 gegen 10,76 sauber, in allen vier Armen wiederholt), weil
leise ausgesteuertes Material genau der Fall ist, in dem schon die Sprachspitze
den Boden zu hoch setzt. 25 dB ist beidseitig dominiert. Abwägung: die Kappe
kauft +0,19 im 50-%-Regime, wo kein Intervall die Null ausschließt, und bezahlt
+0,8 im Knall-Regime, wo eines es tut — dem Anwendungsfall, für den der Boden
gebaut wurde. **Es bleibt beim ungekappten Perzentil 99.**

## 6. Der Einbau (2026-08-23)

`_PercentileFloorFeatures` und `clamp_log_mel` in `noScribe/voxtral_engine.py`,
Konstante `MEL_FLOOR_PERCENTILE = 99`, Tests in `tests/test_mel_floor.py`. Die
Engine tauscht den Feature-Extractor auf dem Prozessor aus; die Bibliothek
selbst wird nicht angefasst.

Zwei Implementierungsfallen, beide im Code dokumentiert und hier nur benannt,
damit sie nicht neu entdeckt werden: der `global_max`-Hebel der Bibliothek
**taugt nicht** für die Bitgleichheits-Kontrolle (er liefert nur das affine
Spektrogramm, das auf leisem Material Mantissenbits verliert — 12 % der
Eingaben bekommen einen um ein Bit falschen Boden), weshalb das ungeklemmte
Spektrogramm aus den Primitiven der Bibliothek nachgerechnet wird; und das
Perzentil läuft **nur über die echten Frames**, weil die Nullpolsterung auf
30 s es sonst mit dem Polsteranteil nach unten zöge (5-s-Clip: 18 dB). Beides
steht ausführlich in den Docstrings von `_PercentileFloorFeatures` und
`clamp_log_mel`.

**Reproduktion durch den Produktionscode** (`voxpopuli_spike.py 10 300 24
99.9,99`, dieselben zehn Ströme, Bodenanstieg median 28,2 dB):

| Boden | sauber | mit Knall | Kosten des Knalls, gepaart |
|---|---|---|---|
| max (Bibliothek) | 11,36 % / 7,36 % | 12,88 % / 7,90 % | **+1,52 [+0,78, +2,26]** \* |
| Perzentil 99,9 | 10,82 % / 6,93 % | 10,86 % / 6,88 % | +0,04 [+0,00, +0,10] |
| Perzentil 99 | 10,76 % / 6,88 % | 10,77 % / 6,88 % | +0,01 [−0,03, +0,06] |

Max und 99,9 reproduzieren die Zahlen oben auf die zweite Stelle; 99 ist die
einzige Variante, deren Intervall die Null enthält.

**Am Transkript ist der Effekt klein und nicht null.** Auf 120 s Interview,
−20 dB, 60-Hz-Knall bei 45 %: unter dem Max-Boden ändert der Knall 15 Stellen
über die ganze Passage, unter dem Perzentil-Boden 3, mindestens zwei davon weit
weg vom Knall — die Restverschiebung von 0,1 dB bewegt jede geklemmte Zelle um
ein Bit, und Greedy-Decoding lässt irgendwo ein knappes Token kippen. Die
Bitgleichheit auf `hart` war ein Passagenbefund, kein Verhalten.

**Der Stack-Benchmark, neu aufgezeichnet (2026-08-24).** `bench_stack.py` (nur
in dieser Arbeitskopie) hat die Baseline `macos27-p99floor`. Auf der
60-s-Benchdatei ändert der Boden **genau ein Wort von 167**. Die Gegenprobe
sitzt tiefer: der heutige Stack mit dem Bibliotheks-Max-Boden reproduziert den
Juli-Text **bitgleich** — über die Sprünge mlx-voxtral 0.1.0 → 0.0.6, mlx
0.32.0 → 0.32.1 und den numpy-Viterbi hinweg war der Decode-Pfad also bit-stabil,
und der Boden ist die einzige Textänderung. Laufzeit und Speicher unverändert.

## 7. Nebenbefund: der Whisper-Pfad ist nicht betroffen

`faster_whisper/feature_extractor.py:227` trägt dieselbe Zeile, und noScribe
übergibt Whisper die ganze Datei — die Wirkung bleibt dort aber aus. Auf 24
Strömen (122,3 min, 28,7 dB Bodenanstieg) kostet der Transient **−0,20
[−1,85, +1,19]** mit VAD und **−0,51 [−1,98, +0,83]** ohne. Beide Intervalle
enthalten die Null **und schließen Voxtrals +1,52 aus**; ein Effekt dieser Größe
ist dort also nicht bloß ungezeigt, sondern ausgeschlossen. Der Fix bleibt damit
Voxtral-only, liegt ohnehin in einer Datei, die es upstream nicht gibt, und
braucht keinen PR an faster-whisper. Skript: `whisper_spike_streams.py`.

## 8. Vorgeschichte: je Block gegen ganze Datei

mlx-voxtral berechnete das log-Mel bis 0.0.5 **je 30-s-Block** und normalisierte
jeden Block gegen sein eigenes Maximum. Voxtral spezifiziert ein Spektrogramm
über die ganze Eingabe, das erst danach geteilt wird (arXiv:2507.13264 §2.1) —
beim Clamp landet ein Block mit eigenem Maximum also auf einem anderen Boden.
Auf 90 s Sprache waren die Blockmaxima 1,897 / 1,721 / 1,526 bei einem
Dateimaximum von 1,897.

Eine echte Abweichung, gemeldet und mit Patch
([mzbac/mlx.voxtral#3](https://github.com/mzbac/mlx.voxtral/issues/3),
[PR #5](https://github.com/mzbac/mlx.voxtral/pull/5)) — aber am Transkript
ändert sie nichts. 18 FLEURS-Ströme à 300 s, 92 min, beide Fassungen auf
bitgleichem Audio, gepaart: dWER **+0,02** [−0,19, +0,24] · dCER **−0,03**
[−0,13, +0,04]. Enge Intervalle: ein Effekt über ±0,24 WER-Punkten ist
ausgeschlossen. Skript: `mel_stream.py`.

**Also nicht selbst gepatcht** — ohne Qualitätsgewinn wiegt ein Patch auf einer
Fremdbibliothek schwerer als die Architekturtreue. Upstream war die Änderung
trotzdem richtig, und mzbac hat PR #5 am 2026-08-22 gemergt und mit **0.0.6**
veröffentlicht.

**Seit dem Einbau des Perzentil-Bodens rechnet die Engine das Spektrogramm
ohnehin selbst** — über die ganze Datei, aus den Primitiven der Bibliothek. Die
Ganze-Datei-Eigenschaft hängt also nicht mehr am Pin; was der Pin auf 0.0.6 noch
leistet, ist, den Bitgleichheits-Test aussagekräftig zu halten, der die eigene
Rechnung gegen `VoxtralFeatureExtractor` prüft. Ein Rückfall auf 0.0.5 zeigte
sich als **fehlender Test**, nicht als geändertes Transkript.

## 9. Werkzeuge und eine Zitierfalle

| Skript | wofür |
|---|---|
| `voxpopuli_floor.py` | Böden auf sauberem Material, gepaart; `load_clips` (VoxPopuli, `VOXPOPULI_PARQUET` für die lokale Datei) und die Arm-Kurzform `floor_arm` (`max`, `99`, `99c20`) |
| `voxpopuli_spike.py` | Transientenschaden über viele Ströme; `pair()` garantiert die Dosis |
| `voxpopuli_sparse.py` | dasselbe auf überwiegend stillen Strömen (Sprachanteil wählbar) |
| `mel_outlier_gap.py` | Abstand Maximum zu Perzentil in echtem Material, ohne Modell |
| `mel_stream.py` | Per-Block gegen ganze Datei, gepaart über lange Ströme |
| `whisper_spike_streams.py` | derselbe Transienten-Test auf der Whisper-Standardengine |

Alle in `docs/skripte/`, alle mit `venv/bin/python3` starten. Die Perzentil-Arme
kommen aus `_PercentileFloorFeatures`, der Max-Arm explizit aus
`VoxtralFeatureExtractor()` — `vox.proc.feature_extractor` ist seit dem Einbau
der Perzentil-Pfad und taugt nicht mehr als Stock-Arm.

**Eine Falle beim Zitieren absoluter VoxPopuli-Zahlen.** Rund **8,7 %** der
Gold-Referenzen schreiben Ziffern („60 Jahre", „21. Februar"), der Sprecher sagt
aber Wörter, und `wer.py`s `norm()` behält Ziffern. Das kostet **1,22
WER-Punkte**: derselbe Lauf ergibt 11,39 % mit und **10,17 %** ohne diese
Äußerungen (`load_clips(..., skip_digits=True)`). Gepaarte Differenzen sind
nicht betroffen, weil beide Arme dieselbe Referenz sehen.
