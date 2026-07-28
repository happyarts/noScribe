# Was beim Herunterrechnen der Audiodatei wirklich passiert

Stand: 28. Juli 2026 · M1 Max · PyAV 12.3.0 (libswresample 4.12.100), mlx-voxtral 0.0.4,
mistral-common 1.11.6

Alle Zahlen unten sind gemessen, nicht geschätzt. Gemessen wurde gegen den
tatsächlichen Code-Pfad, nicht gegen eine Nachbildung: `noScribe/audio/convert.py`
in einem Wegwerf-Skript aufgerufen, Ergebnis mit `soundfile` zurückgelesen,
Mel-Features mit `mlx_voxtral.audio_processing.process_audio_chunk` berechnet —
also mit genau dem Feature-Extractor, den die Engine benutzt. Testmaterial waren
vier reale Interview-/Podcast-Aufnahmen (AAC, 44,1 kHz) plus synthetische Signale
für die Kennlinien.

---

## 1. Es wird alles drei gleichzeitig heruntergerechnet

Eine einzige Zeile macht die komplette Reduktion — `noScribe/audio/convert.py:49`:

```python
self.stream_output = self.container_output.add_stream(
    "pcm_s16le", rate=16000, layout="mono"
)
```

PyAV schiebt beim `encode()` automatisch einen Resampler davor, wenn das
eingehende Frame nicht schon Format, Layout und Rate des Encoders hat. Damit
passiert an dieser Stelle:

| Schritt | typische Quelle | Ziel | verifiziert |
|---|---|---|---|
| **Samplerate** | 44 100 / 48 000 Hz | 16 000 Hz | ✔ |
| **Kanäle** | 2 | 1, als `(L+R)/2` | ✔ |
| **Sample-Format** | float32 (AAC/MP3-Dekoder) bzw. 24 Bit | int16 | ✔ |

Zurückgelesen wird in `noScribe/voxtral_engine.py:2388` mit
`sf.read(..., dtype="float32")` — libsndfile teilt int16 durch 32768, wir landen
also wieder bei float32, aber nur noch mit 65 536 möglichen Werten.

**Nicht Voxtral-lokal:** dieselbe temporäre WAV füttert auch pyannote
(`main.py:2861`) und die VAD-Pausenkorrektur (`main.py:3002`). Eine Änderung am
Zwischenformat trifft die Diarisierung mit.

## 2. Was der Resampler genau tut (gemessen)

* **Anti-Aliasing:** −3 dB bei ≈ 7,6 kHz, Sperrdämpfung ≈ **−74 dB**.
  Zum Vergleich derselbe Sweep über `soxr` VHQ: −158 dB. Das ist der FFmpeg-swr-
  Default (Kaiser, `filter_size=32`, `cutoff=0.97` → 7760 Hz).
* **Kein Dither.** Digitale Stille kommt als exakt 0 heraus, ein −100-dBFS-Ton
  wird vollständig zu Null gerundet. (Die FFmpeg-Resampler-Doku nennt für
  `dither_method` einen Default „rectangular", der Enum-Wert 0 in
  `swresample.h` ist aber `SWR_DITHER_NONE` — die Messung entscheidet: es wird
  nicht gedithert.)
* **Gerundet, nicht abgeschnitten.** Über eine Rampe zwischen zwei int16-Stufen
  stimmt das Ergebnis exakt mit `round()` überein, nie mit `trunc()`.
* **Länge bleibt sample-genau.** 8,000 s bei 48 kHz rein → 128 000 Samples bei
  16 kHz raus. Weder Dekoder- noch Resampler-Schwanz geht verloren.
* **Downmix ist `(L+R)/2`.** Nachgewiesen mit L = −R: das Ergebnis ist exakt
  Stille. Zwei Konsequenzen: eine phasenverdrehte Kanalpaarung löscht sich
  lautlos aus, und ein Signal, das nur auf einem Kanal liegt (einzelnes Lavalier
  an einem Stereo-Interface), verliert **6 dB** Pegel.
  Im Testmaterial ist das unkritisch: die Stereo-Dateien sind praktisch
  Dual-Mono (Kanalkorrelation +0,98 bis +1,00).

## 3. Die Rangordnung der Fehlerquellen

Das ist der eigentliche Befund. Verglichen wird immer die **Mel-Feature-Matrix**,
also das, was der Encoder tatsächlich sieht — nicht die Wellenform. Die Features
spannen einen Bereich von etwa −0,55 bis 1,45, eine Differenz von 1,0 wäre also
eine Verwüstung.

| Eingriff | mittlere Abweichung | p99,9 | schlimmster Bin |
|---|---|---|---|
| **16-Bit-Quantisierung** (bei 0 dBFS Spitze) | **1,9 · 10⁻⁴** | 5,6 · 10⁻³ | 0,013 |
| **Resampler swr statt soxr VHQ** | 1,2 · 10⁻³ | 0,27 | 0,66 |
| **Clipping** durch AAC-Überschwinger | siehe Abschnitt 5 — betrifft einzelne Samples | | |
| **Pegeländerung um 6 dB** | 1,5 · 10⁻¹ | — | 0,15 |
| **Pegeländerung um 20 dB** | 5,0 · 10⁻¹ | — | 0,50 |

Auf der Wellenform gelesen: der 16-Bit-Schritt allein kostet **82 dB** SNR, der
komplette bestehende Pfad gegen einen float64/soxr-Pfad **40 dB**. Der Resampler
ist also numerisch rund 40 dB „teurer" als die Bittiefe.

**Aber:** diese 40 dB sind keine Verzerrung, sondern ein Formunterschied zweier
legitimer Anti-Aliasing-Filter, und er sitzt fast vollständig an der Bandkante.
Spektral aufgeschlüsselt beträgt der lokale Abstand zwischen beiden Pfaden

| Band | lokaler SNR zwischen den Pfaden |
|---|---|
| 0 – 1 kHz | +89 dB |
| 1 – 4 kHz | +78 dB |
| 4 – 7 kHz | +42 dB |
| 7 – 7,6 kHz | +17 dB |
| 7,6 – 8 kHz | +0,7 dB |

Dasselbe im Mel-Raum: über die Bins 0–111 liegt die Abweichung bei 2 · 10⁻⁴ —
also auf demselben Niveau wie reines 16-Bit-Runden. Nur die **obersten 16 der 128
Mel-Bins** (≈ 6,5–8 kHz) tragen den Unterschied, dort ist er 40-mal größer und
erreicht 0,66.

Ob das etwas ausmacht, ist damit eine sehr scharf gestellte Frage: *hört Voxtral
Zischlaute (s / f / sch) schlechter, wenn die obersten 12 % der Mel-Bins anders
geformt sind?* Das ist nicht dasselbe wie „der Pfad ist schlechter" — und
Abschnitt 11 beantwortet es am Transkript: nein.

> Zur Einordnung der Zahlen: diese Tabelle nennt **mittlere Beträge**,
> Abschnitt 8 nennt **RMS**. Bei einer so schwanzlastigen Verteilung
> (p99,9 = 0,27 bei einem Mittel von 1,2 · 10⁻³) liegen die beiden Maße um eine
> Größenordnung auseinander, ohne sich zu widersprechen.

## 4. Bittiefe: wo die Grenze wirklich liegt

Die Frage nach dem Rauschen im letzten Bit lässt sich exakt beantworten, indem
man denselben 30-s-Ausschnitt vor der Quantisierung im Pegel verschiebt:

| Spitzenpegel | mittlere Mel-Abweichung float ↔ 16 Bit | schlimmster Bin |
|---|---|---|
| 0 dBFS | 1,2 · 10⁻⁴ | 0,012 |
| −12 dBFS | 4,8 · 10⁻⁴ | 0,051 |
| −24 dBFS | 2,0 · 10⁻³ | 0,210 |
| −36 dBFS | 1,9 · 10⁻² | 0,511 |
| −48 dBFS | 3,4 · 10⁻² | 0,545 |
| −60 dBFS | 4,4 · 10⁻² | 0,534 |

Mit 24 Bit als Zwischenformat fällt die Abweichung selbst bei −72 dBFS auf
7,7 · 10⁻⁵ zurück, verschwindet also.

**Lesart:** oberhalb von etwa **−24 dBFS Spitzenpegel ist die Bittiefe
irrelevant** — sie liegt zwei Größenordnungen unter allem anderen. Zwischen −24
und −36 dBFS wird sie sichtbar, darunter dominiert sie. Real heißt das: nur
pathologisch leise Aufnahmen (Diktiergerät in der Tasche, Fernfeldmikrofon ohne
Aussteuerung) sind betroffen, und die haben dann ein akustisches
Grundrauschen, das ohnehin 40–60 dB über dem Quantisierungsboden liegt.

Der Grund, warum es überhaupt eine Grenze gibt: Whisper/Voxtral setzen im
Log-Mel einen **relativen** Boden bei `log_max − 8,0`, also 80 dB unter dem
lautesten Mel-Bin des jeweiligen 30-s-Fensters
(`mlx_voxtral/audio_processing.py:341`). Der 16-Bit-Quantisierungsboden liegt fest
bei −96 dBFS, der Mel-Boden wandert mit dem Pegel: bei einer laut ausgesteuerten
Aufnahme liegt das Quantisierungsrauschen darunter und wird weggeklemmt, bei einer
leisen sinkt der Boden mit und holt es herein. Die Messreihe oben zeigt, wo der
Übergang praktisch stattfindet — zwischen −24 und −36 dBFS.

## 5. Clipping: real, aber viel seltener als zunächst gemessen

> **Korrektur.** Eine frühere Fassung dieses Abschnitts nannte 0,02–0,04 % geklippte
> Samples. Das war falsch. Die Referenz, gegen die gemessen wurde, benutzte
> `av.AudioResampler(layout="mono")` — und der downmixt mit `0,7071·(L+R)` statt
> mit `(L+R)/2`, liegt bei Dual-Mono-Material also **3 dB zu heiß**. Die Zahlen
> unten stammen aus `docs/skripte/audio_audit.py`, das denselben Downmix wie
> `convert.py` benutzt; gegengeprüft, indem der echte `ToWav` über die ganze Datei
> lief.

Lossy-Dekoder liefern Samples jenseits von ±1,0 — normal, kein Fehler, in float
verlustfrei. Der Schritt nach int16 kappt sie hart. Über die vollen Dateien:

| Datei | Spitze | True Peak | Samples > 1,0 | nach 16 kHz | LUFS |
|---|---|---|---|---|---|
| A (mono, AAC) | 1,242 | 1,251 | 255 von 162 M | 94 | −16,6 |
| B | 1,013 | 1,027 | 4 von 239 M | 1 | −18,8 |
| C | 0,965 | 0,965 | 0 | 0 | −18,6 |
| D | 1,064 | 1,064 | 10 von 72 M | 4 | −18,8 |
| Referenz-Podcast | 0,765 | 0,768 | 0 | 0 | −19,4 |

Gegenprobe mit dem echten Konverter über Datei B: **genau ein einziges Sample**
landet auf dem int16-Anschlag. Der True Peak liegt praktisch auf dem Sample-Peak,
es gibt also auch kein verstecktes Zwischensample-Problem.

Damit ist Clipping zwar der einzige Punkt im Pfad, an dem Information *hart*
verloren geht — aber es trifft eine Handvoll Samples pro Datei, nicht Promille.
Ein Headroom-Umbau kauft entsprechend wenig, und er verschiebt nebenbei den Pegel
(siehe Abschnitt 3). Was das wirklich kostet, misst Abschnitt 10: nichts.

## 6. Der Bandschnitt bei 8 kHz ist kein Verlust gegenüber dem Modell

Über 8 kHz liegen in einer realen Aufnahme −29 dB der Gesamtenergie — nicht
nichts. Aber oberhalb 11 kHz sind es schon −90 dB: **der AAC-Encoder hat dort
längst tiefpassgefiltert**, bevor unser Resampler ansetzt. Und Voxtrals Encoder
ist ein Whisper-large-v3-Encoder, der nie etwas über 8 kHz gesehen hat. Die
16 kHz sind hier kein Kompromiss, sondern die Spezifikation.

## 7. Zwei Referenzimplementierungen, die einander widersprechen

Das ist der interessanteste Fund, und er ist kein Messwert, sondern ein Blick in
zwei Quelltexte:

* **OpenAI Whisper** (`whisper/audio.py`) ruft
  `ffmpeg -f s16le -ac 1 -acodec pcm_s16le -ar 16000` auf und teilt anschließend
  durch 32768. Das ist **bit-für-bit unser Pfad**: swr-Default-Resampler,
  `(L+R)/2`-Downmix, int16-Zwischenstufe.
* **Mistrals eigenes `mistral_common`** (`tokens/tokenizers/audio.py:260`) liest
  mit soundfile als **float32** und resampelt mit **`soxr` quality="HQ"** — ohne
  jede int16-Stufe und damit ohne Clipping bei Überschwingern.

Voxtral erbt seinen Audio-Encoder von Whisper large-v3, die Vorverarbeitung ihres
Referenz-Stacks stammt aber von Mistral. Wir sitzen auf der Whisper-Seite dieser
Gabelung. `mlx_voxtral.load_audio` läge übrigens auch auf der Mistral-Seite
(soundfile + soxr) — wir umgehen es, weil wir dem Prozessor ein fertiges Array
übergeben.

## 8. Pegel ist ein reiner Offset — und das ist beweisbar

Der Verdacht aus Abschnitt 3 lässt sich exakt prüfen, indem man die Differenz
zweier Mel-Matrizen in zwei Teile zerlegt (`docs/skripte/preproc_mel.py`):

* **Offset** — die beste konstante Verschiebung. Im Prinzip ein Regler.
* **Shape** — was danach übrig bleibt. Das kann keine Verstärkung rückgängig machen.

Referenz ist `p2-soxr`, also float + soxr VHQ beim Originalpegel:

| Variante | Offset | Shape RMS | Shape p99,9 | Shape max | oberste 16 Bins |
|---|---|---|---|---|---|
| `p0-ist` (Produktionspfad) | +0,0003 | 1,58 · 10⁻² | 2,90 · 10⁻¹ | 0,81 | 1,8 · 10⁻³ |
| `p1-headroom3` (−3 dB, soxr, int16) | −0,0750 | 7,34 · 10⁻⁴ | 7,3 · 10⁻³ | 0,024 | 2,6 · 10⁻⁴ |
| `g-12` (−12 dB) | −0,3000 | **7,1 · 10⁻⁷** | 6,2 · 10⁻⁶ | 0,000 | 5,6 · 10⁻⁷ |
| `p3-peak1` (Peak auf −1 dBFS) | +0,0426 | **5,1 · 10⁻⁷** | 6,1 · 10⁻⁶ | 0,000 | 1,7 · 10⁻⁷ |
| `p4-lufs23` (−23 LUFS) | −0,0956 | **5,0 · 10⁻⁷** | 6,1 · 10⁻⁶ | 0,000 | 1,6 · 10⁻⁷ |
| `hot-clip` (auf Peak 1,24 gefahren, geklippt) | +0,1155 | 2,00 · 10⁻² | 3,66 · 10⁻¹ | 0,80 | 2,7 · 10⁻³ |
| `hot-headroom` (gleicher Pegel, −3 dB statt Kappen) | +0,0393 | 4,31 · 10⁻⁴ | 4,3 · 10⁻³ | 0,014 | 1,5 · 10⁻⁴ |

Drei Dinge stehen damit fest:

1. **Eine Pegeländerung ist bis auf sieben Nachkommastellen nichts als ein
   Offset.** Peak-Normalisierung, Lautheitsnormalisierung und eine glatte
   Dämpfung richten *null* Formschaden an. Normalisieren ist also risikolos —
   die einzige Frage ist, ob es etwas bringt.
2. **Der Produktionspfad hat den größten Formfehler von allen** (1,58 · 10⁻²),
   und der stammt vom Resampler, nicht von der Bittiefe: derselbe int16-Schritt
   auf soxr-Basis liegt bei 7,34 · 10⁻⁴, also 20-mal darunter.
3. **Clipping richtet echten Formschaden an, wenn es auftritt** (2,00 · 10⁻²) —
   aber Abschnitt 5 zeigt, dass es das in echtem Material fast nie tut. Mit
   3 dB Headroom fällt derselbe Fall auf 4,31 · 10⁻⁴.

## 9. Gemessen am Transkript: die Pegelhypothese hält nicht

Mel-Abstände sind kein Selbstzweck. Also gegen die handkorrigierte
Referenzpassage transkribiert (`docs/skripte/preproc_cer.py`, Build
`voxtral-mini-8bit`, 422 Wörter / 2034 Zeichen, greedy und damit deterministisch —
ein wiederholter Lauf kam zeichengleich zurück):

| Variante | LUFS | WER | CER | Sub | Del | Ins |
|---|---|---|---|---|---|---|
| `p0-ist` | −19,2 | 4,27 % | 3,39 % | 10 | 8 | 0 |
| `p2-soxr` | −19,2 | 5,21 % | 3,88 % | 11 | 9 | 2 |
| `g-3` | −22,2 | 4,98 % | 3,83 % | 10 | 10 | 1 |
| `g-6` | −25,2 | 5,45 % | 3,93 % | 11 | 10 | 2 |
| `g-9` | −28,2 | 4,50 % | **2,16 %** | 12 | 4 | 3 |
| `g-12` | −31,2 | 4,98 % | **2,16 %** | 13 | 4 | 4 |
| `g-15` | −34,2 | 4,98 % | **2,16 %** | 13 | 4 | 4 |
| `g-18` | −37,2 | 4,74 % | **2,21 %** | 13 | 4 | 3 |
| `g-24` | −43,2 | 4,74 % | **2,31 %** | 14 | 3 | 3 |
| `g-30` | −49,2 | 4,98 % | **2,31 %** | 15 | 2 | 4 |
| `g-36` | −55,2 | 4,50 % | **2,21 %** | 14 | 2 | 3 |

Das sieht nach einer sauberen Stufe zwischen −6 und −9 dB aus, mit einem
Plateau über 27 dB hinweg. Es ist trotzdem **kein Befund**, aus zwei Gründen:

**Erstens sagt der Bootstrap nein.** `docs/skripte/bootstrap_cer.py` legt um jede
gepaarte Differenz ein 95-%-Intervall, und alle enthalten die Null —
`p2-soxr − g-12` etwa liegt bei dCER +1,72 [−0,33, +4,90]. Auf 422 Wörtern ist
ein Unterschied dieser Größe schlicht nicht auflösbar.

**Zweitens hängt der ganze Unterschied an einer einzigen Stelle.** Der Diff
zwischen dem lauten und dem leisen Transkript zeigt sieben Abweichungen, sechs
davon rein orthografisch (`Hokuspokus` / `Hocus Pocus`, `reinmachst` /
`rein, machst`). Die siebte ist ein **acht Wörter langer, leiser Einschub**, den
die laute Fassung komplett verschluckt und die leise wiederfindet. Genau daher
kommt der Sprung von 9 auf 4 Auslassungen — und ein einzelnes gerettetes
Satzfragment unter Greedy-Decoding ist eine Anekdote, kein Effekt.

**Die Gegenprobe mit Trennschärfe** läuft deshalb über FLEURS German, wo
180 unabhängige Aufnahmen (43 min) einen gepaarten Test über Aufnahmen erlauben
(`docs/skripte/fleurs_gain.py`):

| Gain | Peak | WER | CER |
|---|---|---|---|
| 0 dB | 0,673 | 4,41 % | 1,29 % |
| −12 dB | 0,169 | 4,48 % | 1,27 % |

> dWER −0,07 [−0,36, +0,17] · dCER +0,02 [−0,18, +0,18]

**Nichts.** Und die Intervalle sind eng genug, um einen Pegeleffekt größer als
etwa 0,2 CER-Punkte auszuschließen. Die Log-Mel-Features verschieben sich beim
Pegelwechsel messbar um 0,3 — dem Modell ist das erkennbar egal.

Einschränkung, die dazugehört: FLEURS ist vorgelesene, saubere Einzelsprecher-
Aufnahme. Der Ausfallmodus, um den es beim Podcast ging — ein *leiser Einschub
neben lauter Sprache* — kommt darin gar nicht vor. FLEURS schließt einen
generellen Pegeleffekt aus, nicht einen auf genau diesen Fall. Deshalb misst
Abschnitt 10 diesen Fall gezielt und mit bekannter Wahrheit.

## 10. Was 3 dB Headroom kosten — und was sie bringen

Die Frage lässt sich mit derselben Trennschärfe beantworten, indem man jede der
180 FLEURS-Aufnahmen absichtlich zu heiß fährt. Beide Arme bekommen denselben
Pegel; einer läuft in den int16-Anschlag, der andere bekommt die 3 dB Luft
(`docs/skripte/fleurs_gain.py`, Spezifikation mit `c` klippt):

| Arm | Peak | WER | CER |
|---|---|---|---|
| `n1.24c` — auf Peak 1,24 gefahren und geklippt | 1,000 | 4,36 % | 1,30 % |
| `n0.877c` — gleicher Pegel, 3 dB Headroom | 0,877 | 4,41 % | 1,29 % |

> dWER −0,05 [−0,24, +0,10] · dCER +0,01 [−0,07, +0,11]

**Kein Unterschied**, und das Intervall ist auf ±0,1 CER-Punkte eng. Dabei ist
das ein weit brutaleres Szenario als die Wirklichkeit: hier klippt *jede*
Aufnahme, während in echtem Material eine Handvoll Samples pro Datei betroffen
sind (Abschnitt 5) und der rohe Zoom-Mitschnitt aus Abschnitt 13 **gar nicht**
klippt.

Die Kosten der 3 dB sind damit ebenfalls beziffert: 0,5 Bit Auflösung
(Rauschboden −96 → −93 dBFS, im Mel-Raum weiterhin ~10⁻⁴) und ein Offset von
−0,075 auf jedem Feature, gegen den das Modell nachweislich unempfindlich ist.
**Headroom kostet nichts und bringt nichts.** Es ist kein Argument gegen den
Umbau, aber auch keines dafür.

Dieselbe Gegenprobe auf der Podcast-Passage bestätigt das bis auf die Stelle:
`hot-clip` und `hot-headroom` liefern dort ein **zeichengleiches** Transkript
(beide WER 5,21 %, CER 3,88 %, 11 Substitutionen / 9 Auslassungen / 2 Einfügungen,
415 Wörter). Auch `p3-peak1` landet auf demselben Ergebnis. Der einzige Ausreißer
ist `p0-ist` mit CER 3,39 % — und der ist als einziger auf swr statt soxr
gerechnet, liegt aber innerhalb dessen, was diese Passage laut Abschnitt 9
überhaupt auflösen kann.

## 11. Der „bessere Pfad" existiert nicht

Die Resampler-Frage lässt sich auf FLEURS nicht direkt stellen — das Material
liegt schon in 16 kHz vor, es wird also gar nichts resampelt. Der Ausweg: jede
Aufnahme zuerst mit soxr VHQ auf 44,1 kHz hochsetzen, für alle Arme identisch,
und sie dann auf dem jeweiligen Kandidatenweg zurückholen. Was übrig bleibt, ist
genau der Unterschied zwischen den Wegen (`docs/skripte/fleurs_resample.py`):

| Arm | WER | CER |
|---|---|---|
| `swr-s16` — der Produktionspfad | 4,38 % | 1,21 % |
| `soxr-s16` — soxr VHQ, gleiche int16-Stufe | 4,33 % | 1,20 % |
| `soxr-flt` — soxr VHQ, gar keine int16-Stufe | 4,33 % | 1,20 % |

> `swr-s16` − `soxr-s16`: dWER +0,05 [+0,00, +0,15] · dCER +0,01 [+0,00, +0,03]
> `swr-s16` − `soxr-flt`: dWER +0,05 [+0,00, +0,15] · dCER +0,01 [+0,00, +0,03]

Zwei Ablesungen:

1. **`soxr-s16` und `soxr-flt` sind identisch** — gleiche WER, gleiche CER,
   gleiche Differenz zum dritten Arm. Die int16-Zwischenstufe ändert am
   Transkript buchstäblich nichts. Damit ist die Ausgangsfrage nach dem Rauschen
   im letzten Bit endgültig beantwortet, nicht mehr nur im Mel-Raum.
2. **Der Resampler ist am Rand der Nachweisbarkeit** und der Effekt beträgt
   0,01 CER-Punkte. Selbst wenn er echt ist, ist er bedeutungslos: die
   Referenzpassage hat 2034 Zeichen, 0,01 Punkte sind ein Fünftel eines Zeichens.

Der komplette Umbau auf Mistrals Vorverarbeitung — float statt int16, soxr statt
swr — kauft also **höchstens 0,03 CER-Punkte**. Der Formfehler aus Abschnitt 8
war real, aber er sitzt in den obersten 16 Mel-Bins, und dort hört das Modell
offenbar nichts, worauf es ankommt.

## 12. Ein Hebel, der auf konstruiertem Material viel bringt: Dynamik

> **Achtung, hier steht ein Zwischenstand.** Was dieser Abschnitt und
> Abschnitt 18 zeigen, hat sich an echtem Material **nicht bestätigt** — dort
> schadet der Leveller (Abschnitt 22), und Abschnitt 23 erklärt, warum die
> Konstruktion hier zu optimistisch ist. Die Abschnitte bleiben stehen, weil die
> Messungen richtig sind und der Weg zur Auflösung durch sie hindurchführt.

Abschnitt 9 hat den Pegel erledigt, aber eine Lücke gelassen: FLEURS enthält
keine *leisen Einschübe neben lauter Sprache* — genau den Fall, an dem die
Podcast-Passage hing. Also wird er gebaut. Aus je zwei FLEURS-Aufnahmen entsteht

    LAUT | 0,5 s Pause | LEISE (−25 dB) | 0,5 s Pause

und beide Hälften werden **getrennt** bewertet (`docs/skripte/fleurs_quiet.py`,
90 Paare). Getrennt, weil eine Behandlung, die leise Sprache auf Kosten der
lauten rettet, sich hinter einer flachen Gesamt-CER verstecken könnte. Das Maß
ist Wort-Recall: hat es die Passage überhaupt ins Transkript geschafft.

| Bedingung | Recall laut | Recall leise | WER | CER |
|---|---|---|---|---|
| `raw` | 96,5 % | 90,5 % | 7,53 % | 3,91 % |
| `-12` (nur leiser gemacht) | 96,2 % | **84,7 %** | 10,78 % | 7,23 % |
| **`dynaudnorm`** | 96,3 % | **96,3 %** | **4,60 %** | **1,22 %** |
| `loudnorm16` | 96,3 % | 95,6 % | 4,92 % | 1,39 % |
| `speechnorm` | 96,4 % | 95,4 % | 4,90 % | 1,58 % |
| `acompressor` | 96,4 % | 93,9 % | 5,83 % | 2,33 % |

Gepaart über die 90 Paare, gegen `raw` (positiv = besser):

| Behandlung | Δ Recall leise | Δ Recall laut |
|---|---|---|
| `-12` | **−5,86** [−10,42, −2,03] * | −0,24 [−0,74, +0,10] |
| `dynaudnorm` | **+5,81** [+3,19, +9,10] * | −0,19 [−0,69, +0,33] |
| `loudnorm16` | **+5,11** [+2,46, +8,42] * | −0,15 [−0,59, +0,32] |
| `speechnorm` | **+4,86** [+2,32, +8,10] * | −0,05 [−0,32, +0,25] |
| `acompressor` | **+3,36** [+1,14, +6,28] * | −0,10 [−0,51, +0,29] |

\* = Intervall schließt die Null aus

Das ist die mit Abstand größte Wirkung, die in diesem ganzen Dokument gemessen
wurde — alles andere lag bei ≤ 0,03 CER-Punkten, hier sind es **2,7 Punkte**,
eine Reduktion um 69 % relativ. Und anders als der Pegeleffekt aus Abschnitt 9
übersteht dieser den Bootstrap mühelos.

Die Signatur ist eindeutig und in allen vier Behandlungen dieselbe: **die leise
Hälfte gewinnt signifikant, die laute bleibt unangetastet.** Kein einziges
Intervall für den lauten Teil schließt die Null aus — es wird nichts geopfert.
`dynaudnorm` hebt die leise Hälfte von 90,5 auf **96,3 %**, also exakt auf das
Niveau der lauten: der Nachteil leiser Sprache verschwindet vollständig.

Das ist kein Pegeleffekt, sondern ein Dynamikeffekt. Dieselbe Aufnahme nur leiser
zu machen (`-12`) macht die leise Hälfte **signifikant schlechter** — die
Podcast-Anekdote aus Abschnitt 9 zeigte also nicht nur in eine nicht belegbare,
sondern in die *falsche* Richtung.

Damit fügt sich auch zusammen, warum die Referenzpassage so gute Zahlen liefert:
sie stammt aus einem Podcast, der mit Auphonic geleveled wurde. Der Leveller war
längst gelaufen, bevor unsere Pipeline das Material gesehen hat.

`dynaudnorm=f=150:g=9:p=0.9:m=20` verkleinert an konstruierten Sprachpaaren den
Abstand zwischen lauter und leiser Hälfte von 24,5 dB auf 10,9 dB; `speechnorm`
auf 16,8 dB (`docs/skripte/audio_filters.py`). Beide Filter stecken schon in
FFmpeg, das noScribe ohnehin mitbringt — es kostet keine neue Abhängigkeit.

### Und bei stärkerem Abfall wird es dramatisch

Dieselbe Anlage mit 35 statt 25 dB Abstand zwischen den Hälften:

| Bedingung | Recall laut | Recall leise | WER | CER |
|---|---|---|---|---|
| `raw` | 96,5 % | **62,4 %** | 22,28 % | 18,73 % |
| **`dynaudnorm`** | 96,4 % | **95,2 %** | 5,12 % | **1,44 %** |
| `speechnorm` | 96,8 % | 88,8 % | 8,22 % | 4,64 % |

> `raw` → `dynaudnorm`: leise **+32,87** [+24,12, +41,83] * · laut −0,05 [−0,47, +0,36]
> `raw` → `speechnorm`: leise **+26,40** [+18,53, +34,85] * · laut +0,29 [+0,00, +0,63]

Unbehandelt verliert der Pfad **mehr als ein Drittel** der leisen Sprache. Mit
`dynaudnorm` kommen 95,2 % zurück, die CER fällt von 18,73 % auf 1,44 % — eine
Reduktion um 92 % relativ, und die laute Hälfte bleibt wieder unberührt.

Der Effekt wächst also mit der Dynamik, und das ist genau die Größe, in der sich
rohes von gemastertem Material unterscheidet (Abschnitt 13).

## 13. Rohes Material sieht anders aus als gemastertes

Die Referenzpassage stammt aus einem Podcast, der mit Auphonic auf −16 LUFS
getrimmt wurde — professionell geleveltes Material, also gerade nicht der
Normalfall. Zum Vergleich ein roher Zoom-Mitschnitt, 4,8 h, 48 kHz Stereo
(`docs/skripte/audit_long.py`, Fenster von 5 Minuten):

| | gemasterter Podcast | roher Zoom-Mitschnitt |
|---|---|---|
| Lautheit | −19,4 LUFS (Datei) | Median **−27,0 LUFS** |
| Spitze | 0,765 | 0,9985 |
| Samples > 1,0 | 0 | **0** |
| Spreizung p95−p5 über die Fenster | — | **14,2 dB** |
| leisestes / lautestes Fenster | — | −38,6 / −17,6 LUFS |

Zwei Dinge daran sind wichtig. Erstens **klippt auch das rohe Material nicht** —
Zoom limitiert selbst, die Spitze liegt bei 0,9985 und kein einziges Sample
darüber. Die Headroom-Frage ist für diese Klasse von Aufnahmen gegenstandslos.
Zweitens liegt der Unterschied zum gemasterten Material nicht nur 8 dB tiefer im
Mittel, sondern vor allem in der **Spreizung**: zwischen dem leisesten und dem
lautesten Fünf-Minuten-Fenster liegen 21 dB.

Genau diese Spreizung ist der einzige Kandidat, den Abschnitt 9 nicht erledigt
hat — und Abschnitt 12 zeigt, dass er der richtige ist. Eine globale Verstärkung
verschiebt die Spreizung nicht, sie verschiebt alles gleich. Ein Leveller
verschiebt sie, und genau das hat Auphonic auf dem Podcast-Material getan.

**Der praktische Schluss:** unser Testbestand ist in dieser Hinsicht
unrepräsentativ. Die Datei, an der die meisten Voxtral-Messungen dieses Projekts
hängen, kommt bereits geleveled bei uns an. Der Effekt aus Abschnitt 12 ist auf
ihr deshalb gar nicht sichtbar — auf rohem Material aber sehr wohl.

## 14. Was schon vom Tisch ist

* **Dither** — nein. Es würde Rauschen bei −96 dBFS hinzufügen, wo der Mel-Boden
  ohnehin klemmt.
* **Hochpass zur Headroom-Gewinnung** — gemessen und tot. Unter 60 Hz liegen je
  nach Datei −28,5 bis −55 dB der Gesamtenergie, aber ein 60-Hz-Hochpass *senkt*
  die Spitze nicht verlässlich: gemessen zwischen −0,58 dB und **+1,21 dB**. Als
  Headroom-Werkzeug taugt er nicht.
* **DC-Offset** — zwischen 2 · 10⁻⁶ und 1 · 10⁻⁴, also höchstens −80 dBFS.
  Kein Thema.
* **True Peak** — liegt bei allen Dateien praktisch auf dem Sample-Peak
  (1,242 / 1,251 im Extremfall). Kein verstecktes Zwischensample-Problem.
* **Bittiefe** — bei normal ausgesteuertem Material 10⁻⁴ im Mel-Raum und damit
  unter allem anderen. Erst unter −24 dBFS Spitzenpegel wird sie sichtbar.
* **Headroom vor dem int16-Schritt** — gemessen, Abschnitt 10: kostet nichts,
  bringt nichts.
* **Der Stereo-Downmix** — jede Stereodatei im Testbestand ist praktisch
  Dual-Mono. Kanalkorrelation +0,98 bis +1,00 bei den Podcast-Dateien, und beim
  rohen Zoom-Mitschnitt über fünf Stichproben hinweg **+1,0000** bei einem
  Seiten-zu-Mitte-Verhältnis von −47 dB. `(L+R)/2` wirft hier nichts weg.

  Der Vorbehalt bleibt aber stehen und ist billig zu prüfen: eine Aufnahme mit
  *einem Sprecher pro Kanal* — bei Zoom oder Riverside eine Einstellungssache —
  würde von unserem Downmix zusammengerührt. Getrennt transkribiert wäre sie
  besser erkannt *und* nebenbei diarisiert. Die Erkennung dafür ist eine einzige
  Korrelation über ein paar Sekunden. Das ist allerdings ein Feature, keine
  Vorverarbeitung.

## 16. Nachtrag: die Position im 30-s-Fenster kostet messbar

Der Feature-Extractor zerlegt jede Eingabe in 30-Sekunden-Fenster und füllt das
letzte mit Nullen auf. Unser Chunker schneidet an Sprechpausen, nicht am
30-s-Raster — eine Äußerung kann also irgendwo im Fenster beginnen und über eine
Fenstergrenze hinweglaufen. Ob das etwas ausmacht, lässt sich prüfen, indem man
jeder FLEURS-Aufnahme k Sekunden Stille voranstellt
(`docs/skripte/fleurs_window.py`, 180 Aufnahmen, Median 12,5 s lang):

| Vorlauf | Aufnahmen über einer Fenstergrenze | WER | CER |
|---|---|---|---|
| 0 s | 0 | 4,41 % | 1,29 % |
| 6 s | 15 | 4,87 % | 1,39 % |
| 14 s | 58 | 5,07 % | 1,44 % |
| 22 s | 154 | 5,10 % | 1,46 % |
| 27 s | 180 | 5,47 % | **1,69 %** |

Gepaart gegen Vorlauf 0:

| | dWER | dCER |
|---|---|---|
| 6 s | −0,47 [−0,92, −0,05] * | −0,11 [−0,35, +0,12] |
| 14 s | −0,66 [−1,22, −0,13] * | −0,16 [−0,41, +0,08] |
| 22 s | −0,69 [−1,29, −0,07] * | −0,18 [−0,47, +0,11] |
| 27 s | **−1,06** [−1,61, −0,52] * | **−0,40** [−0,72, −0,11] * |

\* = Intervall schließt die Null aus

**Jeder Vorlauf verschlechtert die WER signifikant**, und beim Vorlauf von 27 s —
wo jede einzelne Aufnahme über eine Fenstergrenze läuft — wird auch die CER
signifikant schlechter: 1,29 → 1,69 %, also 31 % relativ.

Zwei Einschränkungen gehören dazu. Erstens sind die Vorläufe 22 s und 27 s
zusätzlich dadurch belastet, dass die Aufnahme dann in **zwei** 30-s-Fenster
fällt statt in eines — mehr Audio-Tokens, mehr Nullfüllung. Der Effekt bei 6 s
und 14 s ist davon frei und zeigt trotzdem schon eine signifikante
WER-Verschlechterung. Zweitens ist das hier künstlich vorangestellte Stille;
ob echte Raumstille dasselbe tut, ist nicht dasselbe Experiment.

Aber die Richtung ist eindeutig und die Konsequenz billig: **Sprache sollte
möglichst früh im Fenster beginnen und möglichst nicht über eine Fenstergrenze
laufen.** Beides liegt in der Hand unseres Chunkers, der seine Schnitte heute
allein nach Sprechpausen setzt. Führende Stille zu kürzen und Schnitte zusätzlich
am 30-s-Raster auszurichten kostet nichts und ist der zweite messbare Hebel
neben dem Leveller.

Der Größenordnung nach bleibt es aber klar nachgeordnet: 0,4 CER-Punkte gegen
2,7 bis 17 Punkte beim Leveller (Abschnitt 12).

## 17. Nachtrag: Entrauschen bringt nichts — und der Leveller schadet im Rauschen nicht

Der naheliegende Einwand gegen einen Leveller: er hebt auch das Grundrauschen in
Sprechpausen an. Und die naheliegende Alternative wäre ein Entrauscher. Beides
lässt sich prüfen, indem man FLEURS kontrolliert verrauscht — weisses Rauschen
bei 10 dB SNR — und die Behandlungen gegen „gar nichts tun" hält
(`docs/skripte/fleurs_noise.py`, 150 Aufnahmen):

| Bedingung | WER | CER | gegen `raw` (dCER) |
|---|---|---|---|
| `clean` (unverrauscht, zum Vergleich) | 4,57 % | 1,33 % | **+5,08** [+4,18, +6,05] * |
| `raw` (verrauscht, unbehandelt) | 13,45 % | 6,41 % | — |
| `afftdn` | 13,27 % | 6,96 % | −0,55 [−1,54, +0,35] |
| `anlmdn` | 14,03 % | 7,11 % | −0,70 [−1,47, −0,02] * |
| `dynaudnorm` | 13,60 % | 6,70 % | −0,29 [−0,81, +0,18] |

\* = Intervall schließt die Null aus

**Entrauschen lohnt sich nicht.** Das Rauschen selbst kostet 5,08 CER-Punkte;
kein Entrauscher holt davon nennenswert etwas zurück. `afftdn` bewirkt gar nichts
Nachweisbares, `anlmdn` kratzt mit 0,70 Punkten am Rand der Signifikanz — und
verschlechtert dabei die WER. Das passt zur Erwartung: Entrauscher sind aufs Ohr
optimiert und hinterlassen Artefakte, die das Modell im Training nie gesehen hat.

**Und der entscheidende Punkt für Abschnitt 12: `dynaudnorm` schadet auf
verrauschtem Material nicht.** Das Intervall liegt bei −0,29 [−0,81, +0,18], also
neutral bis leicht positiv. Die Sorge, ein Leveller könnte durch angehobenes
Pausenrauschen Halluzinationen provozieren, bestätigt sich hier nicht.

## 18. Ab welcher Dynamik der Leveller sich lohnt

Dieselbe Anlage wie in Abschnitt 12, über den Abstand zwischen lauter und leiser
Hälfte durchgefahren. Alles `dynaudnorm`, 90 Paare je Punkt:

| Abstand | Recall leise, roh | Recall leise, geleveled | Δ leise | Δ laut |
|---|---|---|---|---|
| 10 dB | 96,3 % | 96,6 % | +0,25 [−0,23, +0,71] | +0,00 [−0,28, +0,37] |
| 15 dB | 96,1 % | 96,6 % | +0,50 [−0,05, +1,06] | −0,15 [−0,40, +0,10] |
| 20 dB | 94,8 % | 96,5 % | **+1,75** [+0,73, +3,03] * | +0,24 [−0,20, +0,78] |
| 25 dB | 90,5 % | 96,3 % | **+5,81** [+3,19, +9,10] * | −0,19 [−0,69, +0,33] |
| 30 dB | 82,4 % | 96,0 % | **+13,58** [+8,77, +19,24] * | −0,24 [−0,69, +0,19] |
| 35 dB | 62,4 % | 95,2 % | **+32,87** [+24,12, +41,83] * | −0,05 [−0,47, +0,36] |

\* = Intervall schließt die Null aus

Zwei Ablesungen, und die zweite ist die bemerkenswerte:

1. **Die Schwelle liegt zwischen 15 und 20 dB.** Darunter passiert nichts —
   weder Gutes noch Schlechtes. Bei 10 dB ist der Effekt null, bei 15 dB
   streift er die Signifikanz, ab 20 dB ist er da und wächst dann steil.
2. **`dynaudnorm` hält den Recall der leisen Hälfte über den gesamten Bereich
   konstant bei 95,2 bis 96,6 %** — also immer auf dem Niveau der lauten Sprache,
   egal ob der Abstand 10 oder 35 dB beträgt. Der Leveller neutralisiert die
   Dynamik vollständig. Und in **keiner** Zeile leidet die laute Hälfte.

Das ist die eigentliche Empfehlung: nicht „Leveller bringt X Prozent", sondern
**der Leveller macht die Erkennung unabhängig von der Dynamik der Aufnahme.**

## 19. Gegenprobe auf bereits gemastertem Material

Dieselben Filter auf der Auphonic-getrimmten Referenzpassage
(`docs/skripte/preproc_cer.py` mit `f-<chain>`-Varianten):

| Variante | LUFS | WER | CER |
|---|---|---|---|
| `p0-ist` | −19,2 | 4,27 % | 3,39 % |
| `f-dynaudnorm` | −16,3 | 5,21 % | 3,88 % |
| `f-speechnorm` | −13,1 | 4,74 % | 2,16 % |

Diese Passage kann Unterschiede dieser Größe nicht auflösen — Abschnitt 9 hat
gezeigt, dass ihr Bootstrap-Intervall rund ±2,5 CER-Punkte breit ist, und die
Spanne hier beträgt 1,7. Die Zahlen streuen also in beide Richtungen, ohne etwas
zu belegen.

Das ist trotzdem die Antwort, die gebraucht wurde: **auf bereits geleveltem
Material ist kein Schaden nachweisbar.** Es passt zur 10-dB-Zeile aus
Abschnitt 18 — unterhalb der Schwelle tut der Leveller schlicht nichts. Ein
Material, das Auphonic schon durchlaufen hat, liegt genau dort.

---

## 20. Nachtrag: wie viel Dynamik hat echtes Material überhaupt?

Abschnitt 13 nennt für den rohen Zoom-Mitschnitt eine Spreizung von 14,2 dB.
Diese Zahl ist über **Fünf-Minuten-Fenster** gemessen — also langsame Drift über
die Aufnahme hinweg. Für die Leveller-Frage ist das die falsche Größe. Was zählt,
ist der Abstand zwischen lauten und leisen Stellen **innerhalb** einer Passage,
denn nur der steht im selben Encoder-Fenster.

Langsame Drift ist nachweislich harmlos: sie wirkt wie eine Pegeländerung, und
die ist im Mel-Raum ein reiner Offset, gegen den das Modell unempfindlich ist
(Abschnitte 8 und 9).

Also nachgemessen, Sekundenpegel über die ganzen 4,8 Stunden, Spreizung als
p90 − p10 über die Sprachsekunden je Fenster (`docs/skripte/find_passage.py`):

| Fensterlänge | Median | p90 | Maximum |
|---|---|---|---|
| 120 s (726 Fenster) | 5,9 dB | 8,3 dB | **18,5 dB** |
| 300 s (710 Fenster) | 6,0 dB | 7,6 dB | **13,8 dB** |

**Das dämpft die Empfehlung aus Abschnitt 18 erheblich.** Der Leveller wirkt dort
ab etwa 20 dB Spreizung; bei 15 dB streift der Effekt gerade die Signifikanz. Die
typische Passage dieser Aufnahme liegt bei 6 dB, die extremste bei 18,5 dB. Auf
diesem Material ist also **kein großer Gewinn zu erwarten** — die konstruierten
Paare aus Abschnitt 12 mit 25 und 35 dB Abstand sind härter als das, was hier
vorliegt.

Was bleibt trotzdem für den Leveller:

* Er **schadet nicht** — nicht unterhalb der Schwelle (Abschnitt 18, 10-dB-Zeile),
  nicht auf verrauschtem Material (Abschnitt 17), nie der lauten Sprache.
* Er macht die Erkennung **unabhängig** von der Dynamik. Für einen Einzelfall mit
  einem sehr leisen Gesprächspartner ist das der Unterschied zwischen 62 % und
  95 % Recall — und solche Aufnahmen liegen nicht in unserem Testbestand, wohl
  aber im Posteingang der Nutzer.

Die ehrliche Formulierung ist damit nicht „der Leveller bringt X", sondern:
**er ist eine Versicherung gegen Aufnahmen, die wir nicht getestet haben, und
kostet auf denen, die wir getestet haben, nichts.** Ob dieser Handel es wert
ist, ist eine Produktentscheidung, keine Messfrage.

Um wenigstens die Schadensfreiheit auf echtem, rohem Material zu belegen, liegt
eine Passage zur Handkorrektur bereit — die dynamikreichste 5-Minuten-Stelle der
Aufnahme, 848 Wörter, mit markierten leisen Strecken. Aufbau und Grenzen stehen
in `Audiotest2/referenz/zoom_9890-10190_LIESMICH.md`.

---


---

## 22. Der Leveller auf echtem Material: kein Nutzen, teils Schaden

Abschnitt 12 stützt sich auf **konstruierte** Paare. Inzwischen liegen zwei
Prüfungen an echtem Material mit handkorrigierter Referenz vor. In keiner ist ein
Nutzen zu sehen; in einer schadet der Leveller deutlich.

### Zoom-Passage, 859 Wörter

Die dynamikreichste Fünf-Minuten-Stelle des rohen Zoom-Mitschnitts
(`Audiotest2/referenz/zoom_9890-10190`, 13,8 dB Spreizung):

| Variante | LUFS | WER | CER |
|---|---|---|---|
| `p0-ist` (Produktionspfad, **Quelle des Entwurfs**) | −19,7 | 0,81 % | 0,64 % |
| `p2-soxr` (float/soxr, unbehandelt) | −19,7 | 1,05 % | 0,89 % |
| `f-dynaudnorm` | −16,5 | 1,51 % | 0,81 % |
| `f-loudnorm16` | −16,4 | 2,33 % | 1,26 % |
| `f-speechnorm` | −13,4 | 2,68 % | 1,46 % |

**Der Vergleich muss gegen `p2-soxr` laufen, nicht gegen `p0-ist`** — und das ist
kein Detail. Diese Referenz ist durch Korrektur des `p0-ist`-Entwurfs entstanden;
der Abstand zwischen Entwurf und Referenz beträgt 0,81 % WER, es wurden also rund
sieben Wörter geändert. Übersehene Fehler stehen jetzt als Wahrheit darin und
bevorzugen genau diesen Arm. Wie stark, lässt sich beziffern: `p2-soxr` ist
dieselbe Aufnahme durch denselben float-Pfad wie die Leveller, nur ohne Filter —
und verliert allein durch den anderen Resampler 0,25 CER-Punkte gegenüber
`p0-ist`. Das ist die Verzerrung, und sie ist größer als der ganze Abstand von
`dynaudnorm`.

Gegen die faire Basis gerechnet:

| | dWER | dCER |
|---|---|---|
| `p2-soxr` − `f-dynaudnorm` | −0,47 [−1,17, +0,23] | **+0,07** [−0,37, +0,59] |
| `p2-soxr` − `f-loudnorm16` | **−1,28** [−2,56, −0,35] * | −0,37 [−0,81, +0,10] |
| `p2-soxr` − `f-speechnorm` | **−1,63** [−3,60, −0,23] * | −0,57 [−1,65, +0,18] |
| `p0-ist` − `p2-soxr` (die Verzerrung selbst) | −0,23 [−0,58, +0,00] | −0,25 [−0,64, +0,00] |

\* = Intervall schließt die Null aus

**`dynaudnorm` ist auf dieser Passage neutral** — kein nachweisbarer Unterschied
in beide Richtungen. `speechnorm` und `loudnorm16` sind signifikant schlechter.
Ein Nutzen ist bei keinem zu sehen, und das passt: die Passage hat 13,8 dB
Spreizung und liegt damit unter der Wirkschwelle aus Abschnitt 18.

Die wenigen Streitstellen lassen sich einzeln nachhören
(`docs/skripte/adjudicate.py` listet sie mit Zeitmarke). Zwischen `p0-ist` und
`f-dynaudnorm` sind es sieben. Bei einer davon **rettet der Leveller ein Wort,
das der andere Pfad verschluckt** — der vorhergesagte Ausfallmodus, einmal
beobachtet. Bei vier anderen verliert er Funktionswörter oder verhört sich.

### Natürliches Experiment: dieselbe Episode roh und mit Auphonic

Die stärkere Prüfung, weil sie ohne neue Handarbeit auskommt. Von einer Episode
liegen eine rohe Fassung (26 kbps AAC) und die Auphonic-Fassung (verlustfrei)
vor; die bestehende Referenz `hart_780-900` ist in beiden auffindbar
(`docs/skripte/locate_passage.py`, Fenster und Versätze in
`Audiotest2/MATERIAL.md`). Gleiche Wörter, verschiedene Vorverarbeitung, eine
Wahrheit:

| Arm | LUFS | WER | CER |
|---|---|---|---|
| Auphonic, verlustfrei | −16,0 | 6,40 % | **2,70 %** |
| Auphonic + Intro (`Podcast 323`) | −19,2 | 5,21 % | 3,88 % |
| **roh** | −25,1 | 10,19 % | **5,06 %** |
| roh + `speechnorm` | −13,9 | 9,00 % | 6,34 % |
| roh + `dynaudnorm` | −16,9 | 11,85 % | 6,93 % |

Zwei Dinge stehen damit fest, und sie ziehen in verschiedene Richtungen:

1. **Die Auphonic-Bearbeitung ist real und groß**: 2,70 % gegen 5,06 % CER,
   also fast eine Halbierung.
2. **Unsere Leveller reproduzieren davon nichts.** Auf dieselbe rohe Datei
   angewandt machen sie es *schlechter* — `dynaudnorm` von 5,06 auf 6,93 %.

### Woher kommt Auphonics Vorsprung dann?

Zum Teil aus der Quelle. Die verlustfreie Fassung auf die Bitrate der rohen
gebracht und neu gemessen:

| Arm | WER | CER |
|---|---|---|
| Auphonic, verlustfrei | 6,40 % | 2,70 % |
| dieselbe Datei auf 64 kbps | 7,11 % | 2,80 % |
| dieselbe Datei auf 25 kbps | 8,06 % | **3,24 %** |
| dieselbe Datei auf 25 kbps + `dynaudnorm` | 7,82 % | 3,69 % |

Die Bitrate kostet also **0,54 CER-Punkte** — sie erklärt rund ein Viertel des
Abstands von 2,36 Punkten zwischen roh und Auphonic. Die übrigen 1,8 Punkte
stammen aus dem, was Auphonic mit dem Signal macht. Nur besteht das eben nicht
bloß aus Levelling: Auphonics Kette enthält auch Rausch- und Hallreduktion sowie
Filterung. Ein reiner Dynamikregler bildet sie nicht nach — und ist, wie die
letzte Zeile zeigt, auch auf der bitratenreduzierten Fassung schädlich.

## 23. Warum der konstruierte Test das Gegenteil zeigte

Zwei Fehler in der Konstruktion, beide erst durch das echte Material sichtbar.

### Erstens: gedämpfte saubere Sprache behält ihren Störabstand

In Abschnitt 12 wurde die leise Hälfte durch Dämpfung einer *sauberen* Aufnahme
erzeugt. Damit sinkt der Pegel, der Störabstand bleibt aber perfekt — eine
Situation, die es in einer echten Aufnahme nicht gibt: wer leiser spricht, hat
denselben Raum und dasselbe Grundrauschen wie vorher, sein Störabstand sinkt
mit. Ein Leveller kann die Dämpfung rückgängig machen, aber er hebt das Rauschen
mit an und gibt keinen Störabstand zurück, den es nie gab.

Dieselbe Anlage mit einem gleichmäßigen Rauschteppich 45 dB unter der lauten
Hälfte — die leise liegt damit bei +20 dB Störabstand:

| Bedingung | Recall laut | Recall leise | WER | CER |
|---|---|---|---|---|
| `raw` | 96,7 % | 89,6 % | 8,30 % | 4,10 % |
| `dynaudnorm` | 96,3 % | 92,5 % | 6,79 % | 2,44 % |
| `speechnorm` | 96,2 % | 92,0 % | 7,07 % | 2,54 % |

> `raw` → `dynaudnorm`: leise **+2,86** [+0,70, +5,64] * · laut −0,39 [−0,82, +0,00]
> `raw` → `speechnorm`: leise **+2,35** [+0,20, +5,15] * · laut **−0,48** [−0,90, −0,10] *

Der Gewinn halbiert sich (+2,86 statt +5,81), und — das ist neu — **die laute
Hälfte leidet jetzt**. Ohne Rauschen war sie in keiner Bedingung betroffen; mit
Rauschen ist der Verlust bei `speechnorm` signifikant und bei `dynaudnorm`
grenzwertig. Der Leveller hebt eben auch das Rauschen in den Pausen an.

### Zweitens: echte Passagen erreichen die Schwelle gar nicht

Abschnitt 18 nennt die Wirkschwelle bei etwa 20 dB Spreizung. Wie viel haben die
Passagen, an denen tatsächlich gemessen wurde?

| Passage | RMS | Spreizung p90−p10 |
|---|---|---|
| Zoom-Referenzpassage | −21,5 dB | **13,8 dB** |
| Black Week, roh | −24,9 dB | **4,9 dB** |
| Black Week, Auphonic | −15,9 dB | 2,9 dB |
| Podcast 323 | −19,1 dB | 2,9 dB |

**Keine davon erreicht 20 dB.** Die rohe Black-Week-Passage hat 4,9 dB — sie ist
nicht dynamisch, sondern einfach nur leise, und leise allein ist nachweislich
egal (Abschnitt 9). Die dynamischste Stelle des gesamten Zoom-Mitschnitts kam auf
18,5 dB in einem Zwei-Minuten-Fenster (Abschnitt 20).

Damit ist der Widerspruch aufgelöst: der Leveller wirkt nur oberhalb einer
Schwelle, die unser echtes Material nirgends erreicht. Unterhalb davon tut er
nicht nichts — er kostet, weil er Rauschen und Codec-Artefakte mit anhebt und
den Pegel innerhalb von Äußerungen moduliert.

---

## 24. Fazit: eine Antwort auf jede Ausgangsfrage

**Welches Herunterrechnen benutzen wir genau?** Alle drei Reduktionen auf einmal,
in einer Zeile: 44,1/48 kHz → 16 kHz mit libswresample (Kaiser, −3 dB bei
7,6 kHz, Sperrdämpfung −74 dB), Stereo → Mono als `(L+R)/2`, float → int16 mit
Rundung und ohne Dither. Das ist bit-für-bit OpenAI Whispers Referenzpfad.

**Gibt es Qualitätsunterschiede, Stichwort Quantisierung und Rauschen im letzten
Bit?** Nein — und das ist jetzt nicht mehr nur eine Mel-Abschätzung, sondern am
Transkript gemessen: `soxr-s16` und `soxr-flt` liefern auf 180 FLEURS-Aufnahmen
**identische** Werte (Abschnitt 11). Die int16-Stufe ändert nichts. Sichtbar wird
die Bittiefe erst unter −24 dBFS Spitzenpegel, und dort liegt das akustische
Grundrauschen jeder realen Aufnahme längst darüber.

**Wie teuer wären 3 dB Headroom?** 0,5 Bit Auflösung und ein Feature-Offset von
−0,075 — beides nachweislich wirkungslos. Am Transkript: dCER +0,01
[−0,07, +0,11], und das in einem Szenario, in dem *jede* Aufnahme klippt
(Abschnitt 10). Headroom kostet nichts und bringt nichts.

**Welchen qualitätsverbessernden Pfad hätten wir?** Im Resampler und in der
Bittiefe: keinen. Der komplette Wechsel auf Mistrals Vorverarbeitung — float
statt int16, soxr statt swr — kauft höchstens 0,03 CER-Punkte (Abschnitt 11).
Der Formfehler des Produktionspfads ist real, sitzt aber in den obersten 16 der
128 Mel-Bins, und dort hört das Modell nichts, worauf es ankommt.

**Würde mehr Headroom oder Normalisieren der Lautheit etwas bringen?** Headroom
nein, Lautheitsnormalisierung nein. Eine Pegeländerung ist im Mel-Raum ein reiner
Offset ohne jeden Formschaden (Shape-RMS 5 · 10⁻⁷, Abschnitt 8) — und das Modell
ist gegen diesen Offset unempfindlich: −12 dB auf 180 FLEURS-Aufnahmen ergeben
dCER +0,02 [−0,18, +0,18] (Abschnitt 9). Auf leise Passagen wirkt sie sogar
**schädlich** (−5,86 Punkte Recall, signifikant).

**Was bringt wirklich etwas?** Auf konstruiertem Material: ein Dynamik-Leveller,
ab etwa 20 dB Abstand zwischen lauter und leiser Sprache (Abschnitte 12 und 18).
**Auf echtem Material: nichts von dem, was hier geprüft wurde** — und der
Leveller schadet sogar (Abschnitt 22). Der Widerspruch ist aufgeklärt
(Abschnitt 23): der konstruierte Test dämpfte saubere Sprache, was ihren
Störabstand künstlich erhält, und keine unserer echten Passagen erreicht die
Wirkschwelle. Sie liegen bei 2,9 bis 13,8 dB.

**Und ein Nutzen ist auf echtem Material nirgends nachweisbar.** Auf der
Zoom-Passage ist `dynaudnorm` gegen eine faire Basis neutral, `speechnorm` und
`loudnorm16` sind signifikant schlechter; auf der rohen Podcast-Passage
verschlechtert `dynaudnorm` von 5,06 auf 6,93 % CER.

**Was auf echtem Material tatsächlich hilft, ist die Aufnahme selbst.** Dieselbe
Episode roh gegen Auphonic-bearbeitet: CER 5,06 % gegen 2,70 %, fast eine
Halbierung. Davon gehen 0,54 Punkte auf die Bitrate (26 kbps) und rund 1,8 Punkte
auf Auphonics Bearbeitung — die aber mehr enthält als Levelling, nämlich auch
Rausch- und Hallreduktion. Nachbauen lässt sich das mit einem Dynamikregler
nicht: derselbe Regler auf dieselbe Datei angewandt verschlechtert von 5,06 auf
6,93 %.

### Was daraus für den Umbau folgt

1. **Nichts am Resampler, nichts an der Bittiefe, nichts am Headroom ändern.**
   Alle drei sind am Transkript gemessen und alle drei sind wirkungslos. Wer den
   Pfad trotzdem umbauen will, braucht ein anderes Argument als Qualität — etwa
   die Nähe zu Mistrals Referenzimplementierung.
2. **Keinen Leveller einbauen.** Auf jeder echten Passage, die geprüft wurde,
   ist er messbar schlechter als nichts zu tun. Er wirkt erst ab etwa 20 dB
   Spreizung innerhalb einer Passage, und so etwas kam im gesamten Testbestand —
   4,8 Stunden Zoom, mehrere Podcast-Episoden, Interviews — kein einziges Mal
   vor. Falls er je gebaut wird, dann **bedingt**: Spreizung messen, und nur
   oberhalb der Schwelle eingreifen. Die Messung dafür steht in
   `docs/skripte/find_passage.py`.
3. **Den Nutzern Vorverarbeitung empfehlen, statt sie nachzubauen.** Der
   Auphonic-Vorsprung ist mit fast einer Halbierung der CER der größte Effekt in
   diesem ganzen Dokument. Er stammt aber aus einer Kette, die wir nicht haben,
   und der Teil davon, den wir nachbilden können, ist der, der schadet.
4. **Den Chunker die 30-s-Fenstergrenze mitbedenken lassen** (Abschnitt 16).
   Führende Stille kürzen, Schnitte nach Möglichkeit nicht mitten durch eine
   Äußerung legen. 0,4 CER-Punkte, kostenlos — und damit nach diesem Dokument
   der einzige verbliebene Hebel im Code.

### Was offen bleibt

* Wie sich Auphonics Vorsprung aufteilt. Rausch- und Hallreduktion sind die
  naheliegenden Kandidaten; unser Versuch mit `afftdn` und `anlmdn` half nicht
  (Abschnitt 17), war aber gegen weisses Rauschen geprüft, nicht gegen Raumhall.
* Ob es überhaupt reales Material mit mehr als 20 dB Spreizung innerhalb einer
  Passage gibt. Im Testbestand nicht. Eine Aufnahme mit einem sehr leisen
  Gesprächspartner neben einem lauten wäre der Fall — sie fehlt uns.
* Getrennte Transkription echter Zweikanal-Aufnahmen (Abschnitt 14).
* Die Zoom-Referenz ist durch Korrektur des `p0-ist`-Entwurfs entstanden und
  damit an diesen Arm angelehnt (Abschnitt 22). Die Streitstellen sind wenige und
  mit `docs/skripte/adjudicate.py` gezielt nachhörbar.
