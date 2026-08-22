# Auftrag: robuster Clamp-Boden für den log-Mel-Pfad

Diese Datei ist für eine frische Sitzung geschrieben. Sie ist ein **Arbeitsauftrag,
keine Dokumentation**: alles darin wurde am 2026-08-22 gemessen, und bei jeder Zahl
steht, womit — damit nachgeprüft statt geglaubt wird.

**Stand: noch nicht eingebaut, aber begründet.** Der Schaden ist auf einem
unabhängigen Korpus mit Intervallen nachgewiesen, die die Null ausschließen; die
Abhilfe beseitigt ihn vollständig und kostet auf sauberem Material nichts. Die
Gegenargumente sind ernst und stehen vollständig in Abschnitt 4. Wer diesen Auftrag ausführt, soll
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

Das Audio wird gedämpft, der Impuls bleibt auf Vollausschlag — so kommt die Dosis
garantiert an. (Umgekehrt herum, den Impuls über die Sprachspitze zu verstärken,
scheiterte still: echtes Material ist bereits bis 1,0 ausgesteuert, der Impuls
wurde gekappt, und aus verlangten +12 dB wurden 5,2 dB Bodenanstieg. Der Lauf sah
aus wie ein sauberer Nulleffekt. `voxpopuli_spike.py` bricht deshalb jetzt ab,
wenn die Dosis nicht ankommt.) Verglichen wird immer *innerhalb* eines Bodens
(sauber gegen Knall) — damit fällt jede Verzerrung einer Referenz heraus.

**Die belastbare Messung** — zehn VoxPopuli-Ströme à 300 s (50,6 min), jeder
einmal sauber und einmal mit Transient, gepaart je Boden:

| Bodenanstieg (Median) | max-Boden | Perzentil 99,9 |
|---|---|---|
| 5,2 dB | −0,07 [−0,29, +0,15] | +0,04 [+0,00, +0,09] |
| 16,2 dB | **+0,35 [+0,03, +0,66]** \* | +0,03 [+0,00, +0,07] |
| 28,2 dB | **+1,52 [+0,78, +2,26]** \* | +0,04 [+0,00, +0,10] |

\* = Intervall schließt die Null aus. Monoton; der Perzentil-Boden ist flach.

**Die Passagenzahlen überschätzen — nicht zitieren.** Auf den beiden
handkorrigierten Passagen ergab dieselbe Behandlung +5,21 (`hart`) und +11,99
(`zoom`), also das Vier- bis Achtfache. Unter Greedy-Decoding kippt auf einer
einzelnen Passage ein Token und der Rest zieht nach. **+1,52 bei 28 dB ist die
Zahl, die trägt.**

Aus demselben Grund ist eine frühere Beobachtung *kein* Beleg: dass der Schaden
auf `hart` an drei Knallpositionen exakt gleich groß war, zeigt Determinismus
(derselbe Boden, dieselbe eine Divergenz), nicht Allgemeinheit.

**Welche echten Geräusche welche Dosis erzeugen** (gleiche Spitzenamplitude,
leise Aufnahme): Klatschen 10,3 dB, Rauschburst 17,3 dB, **tieffrequenter
Türknall 27,1 dB**, Rechteckblock 31,5 dB. Der teuerste Fall ist der
alltäglichste.

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

Gemessen auf **VoxPopuli de** (spontane Parlamentsrede mit Goldtranskript, 10,2 % WER
gegen FLEURS' 4,8 % — der härtere externe Korpus, siehe `voxpopuli_floor.py`;
die 10,2 % gelten mit `skip_digits=True`, siehe unten),
10 Ströme à 300 s, 50,6 min, gepaart auf bitgleichem Audio:

> max 11,39 % / 7,32 % · Perzentil 99,9 11,29 % / 7,20 %
> **dWER −0,10 [−0,83, +0,47] · dCER −0,12 [−0,77, +0,38]**

Beide Intervalle enthalten die Null. Wichtig für die Auslegung: diese Ströme haben mit
13,0 dB Median denselben Ausreißer-Abstand wie unser echtes Material — der Boden wurde
dort also real um 13 dB gesenkt, ohne Wirkung.

**Die Asymmetrie, die beide Befunde zusammenbringt:** den Boden zu *senken* zeigt mehr
Rauschdetail, das das Modell ignoriert (VoxPopuli: null). Den Boden zu *heben* zerstört
leise Sprachdetails (Knalltests: +1,52 auf Strömen). Der Perzentil-Boden liegt **immer** unter
oder gleich dem Maximum-Boden — er kann also nur verhindern, dass gehoben wird.

**Eine Falle beim Zitieren absoluter VoxPopuli-Zahlen.** Rund **8,7 %** der
Gold-Referenzen enthalten Ziffern („60 Jahre", „21. Februar"), der Sprecher sagt
aber Wörter, und `wer.py`s `norm()` behält Ziffern. Gemessen kostet das
**1,22 WER-Punkte**: derselbe Lauf ergibt 11,39 % mit und **10,17 %** ohne diese
Äußerungen (`load_clips(..., skip_digits=True)`). Mehr als ein Zehntel der
gemeldeten Fehlerrate ist also Schreibweise, nicht Erkennung. **Gepaarte
Differenzen sind davon nicht betroffen** — beide Arme sehen dieselbe Referenz —,
weshalb keine Entscheidung in diesem Dokument davon abhängt. Wer absolute Zahlen
veröffentlicht, setzt `skip_digits=True`.

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

**Perzentil 99,9 — die Wahl ist unkritisch.** Sweep auf denselben zehn Strömen,
alle gegen dieselbe Max-Basis gepaart:

| Boden | WER | gepaart gegen max |
|---|---|---|
| max (Ist) | 11,39 % | — |
| Perzentil 99 | 11,27 % | dWER −0,12 [−0,85, +0,49] |
| Perzentil 99,9 | 11,29 % | dWER −0,10 [−0,83, +0,47] |
| Perzentil 99,99 | 11,14 % | dWER −0,25 [−0,99, +0,31] |

Alle drei ununterscheidbar vom Maximum und voneinander, alle drei nominell
minimal besser. 99,9 nehmen, weil es in den Schadensläufen verwendet wurde.

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

* **Der Whisper-Pfad ist nicht betroffen — geprüft, nicht angenommen.**
  `faster_whisper/feature_extractor.py:227` trägt dieselbe Zeile, und noScribe
  übergibt Whisper die ganze Datei. Auf 24 Strömen (122,3 min, 28,7 dB
  Bodenanstieg) kostet der Transient dort **−0,20 [−1,85, +1,19]** mit VAD und
  **−0,51 [−1,98, +0,83]** ohne — beide Intervalle enthalten die Null und
  schließen Voxtrals +1,52 aus.

  **Das entscheidet, wo der Fix hingehört.** Er bleibt vollständig in
  `voxtral_engine.py` und geht mit dem Voxtral-Feature hoch. Es braucht **keine**
  Option im gemeinsamen Teil, die nur für Voxtral aktiv wäre, und **keinen** PR
  an faster-whisper — dort ist nichts zu reparieren. (Was ohnehin ungünstig
  gewesen wäre: faster-whisper hatte seinen letzten Push im November 2025,
  während seine Engine CTranslate2 aktiv weiterentwickelt wird.)
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
| `docs/skripte/whisper_spike_streams.py` | derselbe Test auf der Whisper-Standardengine, beide VAD-Zustände |

Alle mit `venv/bin/python3` starten (es gibt kein `python`).
