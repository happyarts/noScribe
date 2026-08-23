# Die Audio-Vorverarbeitung im Voxtral-Pfad

Stand: 28. Juli 2026 · M1 Max · PyAV 12.3.0, mlx-voxtral 0.0.4, Build
`voxtral-mini-8bit`. Alle Zahlen gemessen, keine geschätzt.

**Ergebnis in einem Satz: am Audiopfad ist nichts zu verbessern.** Jeder
geprüfte Eingriff ist entweder wirkungslos oder schädlich. Was die Erkennung
wirklich bewegt, liegt vor unserer Pipeline — in der Qualität, in der die
Aufnahme bei uns ankommt.

---

## 1. Was der Pfad tut

Eine Zeile in `noScribe/audio/convert.py:49` macht die komplette Reduktion:

```python
self.container_output.add_stream("pcm_s16le", rate=16000, layout="mono")
```

PyAV schiebt beim `encode()` automatisch einen Resampler davor. Damit passiert
dort alles drei gleichzeitig:

| Schritt | von | nach |
|---|---|---|
| Samplerate | 44 100 / 48 000 Hz | 16 000 Hz (libswresample, Kaiser, −3 dB bei 7,6 kHz, Sperrdämpfung −74 dB) |
| Kanäle | 2 | 1, als `(L+R)/2` |
| Format | float32 / 24 Bit | int16, gerundet, **ohne** Dither |

Zurückgelesen wird in `noScribe/voxtral_engine.py:2388` als float32 (`/32768`).
Das ist **bit-für-bit OpenAI Whispers Referenzpfad**. Mistrals eigenes
`mistral_common` läge auf der anderen Seite dieser Gabelung: float32 plus `soxr`
HQ, ohne int16-Stufe. Gemessen macht das keinen Unterschied (Abschnitt 2).

Dieselbe temporäre WAV füttert auch pyannote (`main.py:2861`) und die
VAD-Pausenkorrektur (`main.py:3002`) — eine Änderung am Zwischenformat wäre
nicht Voxtral-lokal.

## 2. Was geprüft wurde, und was es gebracht hat

Bewertet wird immer am **Transkript**, nie am Spektrum. Alle Intervalle sind
gepaarte 95-%-Bootstrap-Intervalle.

| Eingriff | Ergebnis | Beleg |
|---|---|---|
| **int16 weglassen** (float durchreichen) | **nichts.** `soxr-s16` und `soxr-flt` liefern über 180 FLEURS-Aufnahmen *identische* Werte | `fleurs_resample.py` |
| **Resampler soxr statt swr** | **nichts.** dCER +0,01 [+0,00, +0,03] | `fleurs_resample.py` |
| **3 dB Headroom vor int16** | **nichts.** dCER +0,01 [−0,07, +0,11] — obwohl in diesem Test *jede* Aufnahme in den Anschlag gefahren wurde | `fleurs_gain.py` |
| **Lautheit normalisieren** | **nichts.** −12 dB über 180 Aufnahmen: dCER +0,02 [−0,18, +0,18] | `fleurs_gain.py` |
| **Dynamik-Leveller** | **neutral bis schädlich** auf echtem Material | Abschnitt 4 |
| **Entrauschen** | **nichts**, `anlmdn` verschlechtert die WER | `fleurs_noise.py` |
| **Hochpass** (Headroom gewinnen) | untauglich: hebt die Spitze eher an (−0,58 bis **+1,21** dB) | — |
| **Chunks am 30-s-Fenster ausrichten** | **nichts.** dWER +0,00 [−0,75, +0,52] | Abschnitt 5 |
| **log-Mel über die ganze Datei statt je 30-s-Block** (seit 0.0.6 der Produktionspfad) | **nichts.** dWER +0,02 [−0,19, +0,24] | Abschnitt 6 |
| **Kanäle getrennt transkribieren** | gegenstandslos: alles Testmaterial ist Dual-Mono (Korrelation +0,98 bis +1,0000) | `audio_audit.py` |

### Warum die Bittiefe egal ist

Whisper/Voxtral klemmen das Log-Mel bei `log_max − 8,0`, also 80 dB unter dem
lautesten Mel-Bin der ganzen Eingabe
(`mlx_voxtral/audio_processing.py:341`; bis 0.0.5 war es der lauteste Bin des
jeweiligen 30-s-Fensters — siehe Abschnitt 6). Der 16-Bit-Rauschboden liegt bei
−96 dBFS und damit darunter. Erst wenn eine Aufnahme unter etwa **−24 dBFS
Spitzenpegel** liegt, wandert der Mel-Boden so weit mit, dass die Quantisierung
sichtbar wird — und dort liegt das akustische Grundrauschen jeder realen
Aufnahme längst darüber. Im Mel-Raum: 1,2 · 10⁻⁴ bei 0 dBFS, 2,0 · 10⁻³ bei
−24 dBFS, 1,9 · 10⁻² bei −36 dBFS.

### Warum Clipping kein Thema ist

Über die vollen Testdateien landen zwischen **0 und 255 Samples** auf dem
int16-Anschlag — bei 72 bis 239 Millionen Samples pro Datei. Der rohe
Zoom-Mitschnitt klippt **gar nicht** (Spitze 0,9985, kein Sample darüber; Zoom
limitiert selbst). Der True Peak liegt überall praktisch auf dem Sample-Peak.

### Warum Pegel nichts ändert

Eine Pegeländerung ist im Mel-Raum ein **reiner Offset**: nach Abzug der besten
Konstanten bleibt ein Formfehler von 5 · 10⁻⁷. Peak-Normalisierung,
Lautheitsnormalisierung und Dämpfung richten also nachweislich keinen Schaden
an — sie bewirken aber auch nichts, weil das Modell gegen diesen Offset
unempfindlich ist.

## 3. Was tatsächlich zählt: die Quelle

Von einer Podcast-Episode liegen eine rohe Fassung (26 kbps AAC) und die
Auphonic-Fassung (verlustfrei) vor. Die bestehende Referenz ist in beiden
auffindbar, also gleiche Wörter, eine Wahrheit:

| Fassung | LUFS | WER | CER |
|---|---|---|---|
| Auphonic, verlustfrei | −16,0 | 6,40 % | **2,70 %** |
| roh, 26 kbps | −25,1 | 10,19 % | **5,06 %** |
| roh + `dynaudnorm` | −16,9 | 11,85 % | 6,93 % |

Fast eine Halbierung der Fehlerrate — **der größte Effekt in diesem ganzen
Dokument.** Davon gehen 0,54 Punkte auf die Bitrate (dieselbe verlustfreie Datei
auf 26 kbps gebracht: 2,70 → 3,24 %).

Der Rest ist **nicht** durch Levelling nachzubauen: derselbe Regler auf dieselbe
rohe Datei angewandt verschlechtert von 5,06 auf 6,93 %. Und Auphonics übrige
Kette enthält Rausch- und Hallreduktion — von der die Literatur sagt, dass sie
ASR schadet (Abschnitt 7). Damit bleibt als wahrscheinlichste Erklärung die
**Quellenqualität**: die rohe Datei ist ein 26-kbps-Export, die bearbeitete
stammt aus einer besseren Kette.

> **Die Empfehlung an Nutzer lautet deshalb nicht „lasst Auphonic drüberlaufen",
> sondern „exportiert in ordentlicher Qualität".** Was davon Bearbeitung und was
> Bitrate ist, lässt sich mit dem vorhandenen Material nicht weiter trennen —
> dafür bräuchte es dieselbe Aufnahme unbearbeitet in guter Qualität.

## 4. Warum kein Leveller eingebaut wird

Der Kandidat, der am längsten überzeugend aussah. Wer ihn wieder vorschlägt,
findet hier die Zahlen.

**Auf konstruiertem Material wirkt er stark.** Aus FLEURS-Paaren gebaut —
`LAUT | Pause | LEISE(−N dB)`, beide Hälften getrennt bewertet — hebt
`dynaudnorm` den Recall der leisen Hälfte auf das Niveau der lauten:

| Abstand | Recall leise, roh | mit `dynaudnorm` |
|---|---|---|
| 10 dB | 96,3 % | 96,6 % |
| 20 dB | 94,8 % | 96,5 % |
| 25 dB | 90,5 % | 96,3 % |
| 35 dB | **62,4 %** | **95,2 %** |

Die Wirkschwelle liegt bei etwa **20 dB Spreizung innerhalb einer Passage**.

**Auf echtem Material bringt er nichts.** Zwei Gründe, beide gemessen:

1. **Keine echte Passage erreicht die Schwelle.** Spreizung p90−p10 über die
   Sprachsekunden: rohe Podcast-Passage 4,9 dB, Zoom-Referenzpassage 13,8 dB,
   Auphonic-Fassung 2,9 dB. Über den ganzen 4,8-Stunden-Zoom-Mitschnitt liegt
   der Median bei 6 dB, das Maximum bei 18,5 dB. Rohes Material ist nicht
   dynamischer als gemastertes — es ist nur **leiser**, und leise allein ist
   nachweislich egal.
2. **Der konstruierte Test war zu optimistisch.** Eine gedämpfte *saubere*
   Aufnahme behält ihren Störabstand; eine wirklich leise Stimme im selben Raum
   nicht. Mit realistischem Rauschteppich halbiert sich der Gewinn (+2,86 statt
   +5,81 Punkte), und die **laute** Hälfte fängt an zu leiden — ohne Rauschen
   war sie in keiner Bedingung betroffen.

Am Transkript, gegen eine faire Basis (`p2-soxr` — derselbe float-Pfad wie die
Leveller, nur ohne Filter):

| | dWER | dCER |
|---|---|---|
| `dynaudnorm` | −0,47 [−1,17, +0,23] | +0,07 [−0,37, +0,59] |
| `loudnorm16` | **−1,28** [−2,56, −0,35] * | −0,37 [−0,81, +0,10] |
| `speechnorm` | **−1,63** [−3,60, −0,23] * | −0,57 [−1,65, +0,18] |

\* = Intervall schließt die Null aus. `dynaudnorm` ist neutral, die anderen sind
schlechter. Kein Nutzen, teils Schaden.

## 5. Warum der Chunker unverändert bleibt

Einzelne kurze Äußerungen reagieren messbar darauf, wo sie im 30-s-Encoderfenster
liegen. Auf **durchgehender Rede** — und nur die schneidet unser Chunker —
verschwindet der Effekt: 18 Ströme von je ~300 s aus FLEURS-Aufnahmen mit
bekanntem Transkript, 92 min Audio, jeder einmal mit vollem und einmal mit
angebrochenem letztem Fenster, gepaart ausgewertet:

> dWER **+0,00** [−0,75, +0,52] · dCER **−0,32** [−1,16, +0,16]

Ein Umbau wurde gebaut, gemessen und wieder entfernt. Auf einer einzelnen
Passage sah der Effekt groß und sauber aus (CER 0,89 gegen 3,04 %) und war
zweimal reproduzierbar — auf der zweiten Passage kehrte sich das Vorzeichen um.

**Nur für Voxtral relevant.** Der Whisper-Pfad übergibt die ganze Datei an
`faster_whisper.transcribe()` mit `vad_filter=True`
(`noScribe/whisper_mp_worker.py:140`); Segmentierung und Fensterung passieren
dort intern, wir haben keinen Hebel.

## 6. Der Mel-Pfad: gemessen, dann von upstream übernommen

mlx-voxtral berechnete das log-Mel bis 0.0.5 **je 30-s-Block** und normalisierte
jeden Block gegen sein eigenes Maximum. Voxtral spezifiziert ein Spektrogramm
über die ganze Eingabe, das erst danach geteilt wird (arXiv:2507.13264 §2.1) —
beim Clamp auf `log_max − 8` landet ein Block mit eigenem Maximum also auf einem
anderen Boden. Auf 90 s Sprache waren die Blockmaxima 1,897 / 1,721 / 1,526 bei
einem Dateimaximum von 1,897.

Eine echte Abweichung, gemeldet und mit Patch
([mzbac/mlx.voxtral#3](https://github.com/mzbac/mlx.voxtral/issues/3),
[PR #5](https://github.com/mzbac/mlx.voxtral/pull/5)) — aber am Transkript ändert
sie nichts. 18 FLEURS-Ströme à 300 s, 92 min, beide Fassungen auf bitgleichem
Audio, gepaart:

> dWER **+0,02** [−0,19, +0,24] · dCER **−0,03** [−0,13, +0,04]

Enge Intervalle: ein Effekt über ±0,24 WER-Punkten ist ausgeschlossen. Der Test
ist dabei schärfer als der Anwendungsfall — die Blockpegel-Spanne der Ströme
liegt bei 1,87 log10 gegen 0,94 und 0,12 auf den beiden handkorrigierten
Passagen, weil dort verschiedene Aufnahmen aneinanderhängen.

Dieselbe Falle wie in Abschnitt 5: `hart_780-900` (Spanne 0,12) lieferte
wortgleiche Ergebnisse, `zoom_9890-10190` einen scheinbar signifikanten
Unterschied — jene Referenz ist aber um 0,25 CER-Punkte zugunsten des Builds
verzerrt, aus dessen Entwurf sie korrigiert wurde. Zwei Passagen entscheiden das
nicht. Skript: `mel_stream.py`.

**Also nicht selbst gepatcht** — ohne Qualitätsgewinn wiegt ein Patch auf einer
Fremdbibliothek schwerer als die Architekturtreue. Upstream war die Änderung
trotzdem richtig, und dort ist sie inzwischen drin: mzbac hat PR #5 am 2026-08-22
gemergt und mit **0.0.6** veröffentlicht, worauf `requirements_voxtral_macOS_arm64.txt`
jetzt zeigt. Der Produktionspfad rechnet damit ab sofort referenztreu.

**Die beiden Handreferenzen dazu, gemessen am 2026-08-22** (gleicher Build,
nur der Feature-Pfad getauscht, `mel_ab.py`-Aufbau):

| Passage | Spreizung | 0.0.6 ganze Datei | 0.0.5 je Block |
|---|---|---|---|
| `hart_780-900` (422 W.) | 0,12 log10 | 4,27 % / 3,39 % | **bitgleicher Text** |
| `zoom_9890-10190` (859 W.) | 0,94 log10 | 1,98 % / 1,09 % | 0,81 % / 0,64 % |

Auf `hart` ist die Änderung buchstäblich unsichtbar — identischer Text, nicht nur
identische Rate. Auf `zoom` sieht die alte Fassung besser aus, und **das ist
genau die Falle, vor der das LIESMICH dieser Referenz warnt**: sie entstand durch
Korrektur des Entwurfs von `p0-ist`, also des damaligen Produktionspfads mit
Per-Block-Mel, und ihr eigener Abstand zu jenem Entwurf beträgt 0,81 % WER — die
Zahl, die die alte Fassung hier erreicht. Der Per-Block-Mel gewinnt dort gegen
sich selbst. 1,98 % ist der unverzerrte Wert; zwei Substitutionen auf 859 Wörtern
Gesprächsaudio waren nie plausibel. Für die Richtung entscheidend bleibt die
gepaarte Messung über 92 Minuten, die keine der beiden Referenzen berührt.

Der Bump ist keine Kosmetik — er ändert die Encoder-Eingabe. Auf 300 s echtem
Podcast-Material gegen die alte Fassung: max|diff| **1,30**, und **90 %** aller
Frames weichen um mehr als 1e-4 ab. Gegen die Referenz (transformers'
`VoxtralFeatureExtractor`, gleicher Aufbau: ganze Datei, dann `reshape` in
30-s-Blöcke) bleiben **0,00017** — Float32-Rauschen. Es ist genau die Differenz,
die oben gemessen wurde: der Versionssprung ist am Transkript folgenlos.

## 6b. Der Clamp-Boden und ein einzelner lauter Transient (2026-08-22)

Der Boden liegt bei `log_max - 8`, und `log_max` ist seit 0.0.6 das Maximum über
die **ganze Eingabe**. Damit hebt ein einzelner lauter Ton den Boden für den
gesamten Durchgang. Gemessen: ein 0,1-s-Knall hebt ihn um **9,6 dB** und
verändert **100 %** der Frames im unberührten Audio davor.

Das klingt schlimmer als es die ersten Zahlen nahelegten — und die ersten Zahlen
waren falsch. Ein Einzeltest auf `zoom` mit einer Knallposition ergab, der neue
Pfad sei robuster. Über beide Referenzen und drei Positionen kehrt sich das um:

| Knall bei | `hart` ganze Datei | `hart` je Block | `zoom` ganze Datei | `zoom` je Block |
|---|---|---|---|---|
| 5 % | +1,66 | +0,95 | −0,47 | +0,47 |
| 45 % | +1,66 | +0,71 | −0,12 | +0,12 |
| 85 % | +1,66 | **+0,00** | −0,12 | +0,23 |

dWER gegen den sauberen Lauf im selben Schema. Auf `hart` kostet der Knall den
Ganze-Datei-Pfad an **allen drei Positionen exakt gleich viel** — mechanisch
genau richtig, der Boden steigt ja positionsunabhängig — und das macht die Zahl
glaubwürdig, nicht verdächtig. Der Per-Block-Pfad kostet 0 bis 0,95, je nachdem
ob der Knall in einen ohnehin lauten Block fällt.

**Die Dosis-Wirkung, und sie ist monoton.** `hart` gedämpft, Knall konstant:

| Knall über der Sprachspitze | max-Boden | Perzentil-99,9-Boden |
|---|---|---|
| +2,6 dB | +1,66 | +0,00 |
| +14,7 dB | +2,61 | +0,24 |
| +27,1 dB | **+5,21** | +0,00 |

+5,21 Punkte sind 22 Wörter von 422, bei einer leise ausgesteuerten Aufnahme mit
einer zugeschlagenen Tür. Kein Laborfall.

**Der Eingriff, der es behebt, sitzt an genau einer Stelle.** Das Maximum wird
*ausschließlich* für den Boden benutzt — danach folgt eine feste Affinität —,
also ist eine robuste Statistik dort der richtige Ort, und nicht ein Limiter auf
dem Audio (§4: drei davon gemessen, neutral bis schädlich). Mit einem Perzentil
statt des Maximums ist der Text auf `hart` **mit und ohne Knall bitgleich**.
Kontrolle: mit Perzentil 100 reproduziert die Implementierung den Ist-Zustand
bitgleich, sie ändert also nachweislich nur den Boden.

**Was er auf sauberem Material kostet: nichts nachweisbares.** Gemessen auf
**VoxPopuli de** — spontane Parlamentsrede mit Goldtranskript, der Korpus, der
uns gegenüber dem vorgelesenen FLEURS fehlte, und mit 11,4 % WER deutlich härter
— 10 Ströme à 300 s, 50,6 min, gepaart auf bitgleichem Audio:

> max 11,39 % / 7,32 % · Perzentil 99,9 11,29 % / 7,20 %
> dWER **−0,10** [−0,83, +0,47] · dCER **−0,12** [−0,77, +0,38]

Skript: `voxpopuli_floor.py`.

**Auf Strömen bestätigt, und kleiner als die Passagen glauben machten.** Zehn
VoxPopuli-Ströme à 300 s, jeder einmal sauber und einmal mit Transient, gepaart
je Boden (`voxpopuli_spike.py`):

| Bodenanstieg (Median) | max-Boden | Perzentil 99,9 |
|---|---|---|
| 5,2 dB | −0,07 [−0,29, +0,15] | +0,04 [+0,00, +0,09] |
| 16,2 dB | **+0,35 [+0,03, +0,66]** \* | +0,03 [+0,00, +0,07] |
| 28,2 dB | **+1,52 [+0,78, +2,26]** \* | +0,04 [+0,00, +0,10] |

\* = Intervall schließt die Null aus. Monoton, bei den beiden oberen Dosen
nachweisbar, der Perzentil-Boden bei allen dreien flach.

**Die Passagenzahlen (+5,21 und +11,99) überschätzen um das Vier- bis
Achtfache.** Unter Greedy-Decoding kippt auf einer einzelnen Passage ein Token
und der Rest zieht nach; die Strommessung mittelt das heraus. **+1,52 bei 28 dB
ist die belastbare Zahl.** Aus demselben Grund war die Beobachtung, der Schaden
sei an drei Knallpositionen „exakt gleich groß", kein Beleg für Allgemeinheit,
sondern nur für Determinismus.

**Welche Perzentile in Frage kommen** — Sweep auf denselben Strömen, alle gegen
dieselbe Max-Basis: 99 → −0,12 [−0,85, +0,49], 99,9 → −0,10 [−0,83, +0,47],
99,99 → −0,25 [−0,99, +0,31]. Alle drei ununterscheidbar vom Maximum und
voneinander; die Wahl ist keine kritische Größe.

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
rohem Material — also genau bei dem, was Nutzer bringen. Das ordnet auch den
Nullbefund oben ein: die Ströme tragen denselben Abstand, der Boden wurde dort
real um 13 dB gesenkt, ohne Wirkung.

**Die Asymmetrie, die beide Befunde zusammenbringt:** den Boden zu *senken*
zeigt mehr Rauschdetail, das das Modell ignoriert; ihn zu *heben* zerstört leise
Sprachdetails. Der Perzentil-Boden liegt **immer** unter oder gleich dem
Maximum-Boden — er kann also nur verhindern, dass gehoben wird.

**Was dagegen spricht, beziffert.** Der Ist-Zustand liefert per Konstruktion
**exakt 2,000** Einheiten Spannweite. Mit Perzentil-Boden: 2,182 (`hart`), 2,235
(Interview), 2,360 (`zoom`), 2,547 (Rohspur) — bis **27 % breiter**, nach unten.
In [openai/whisper#269](https://github.com/openai/whisper/discussions/269) warnt
der Whisper-Autor genau davor: ohne diese Normalisierung liege die Eingabe
*out-of-distribution*. Bei den Breiten, die echtes Material erzeugt, ist es
gemessen folgenlos — aber es ist der Grund, warum das nicht ungemessen eingebaut
werden darf.

**Eine Falle beim Zitieren absoluter VoxPopuli-Zahlen.** Rund **8,7 %** der
Gold-Referenzen schreiben Ziffern („60 Jahre", „21. Februar"), der Sprecher sagt
aber Wörter, und `wer.py`s `norm()` behält Ziffern. Das kostet **1,22
WER-Punkte**: derselbe Lauf ergibt 11,39 % mit und **10,17 %** ohne diese
Äußerungen (`load_clips(..., skip_digits=True)`). Gepaarte Differenzen sind
nicht betroffen, weil beide Arme dieselbe Referenz sehen.

**Eingebaut am 2026-08-23**, mit **Perzentil 99** statt der 99,9 der ersten
Läufe — `_PercentileFloorFeatures` und `clamp_log_mel` in
`noScribe/voxtral_engine.py`, Konstante `MEL_FLOOR_PERCENTILE`, Tests in
`tests/test_mel_floor.py`. Dafür stand ein gemessener, dosisabhängiger Schaden
mit Intervallen, die die Null ausschließen, und eine Abhilfe, die ihn beseitigt
und auf sauberem Material nichts kostet; dagegen die Architekturtreue — alle
vier Implementierungen nehmen das nackte Maximum. Drei Dinge kamen beim Einbau
anders, als der Auftrag (`auftrag-perzentil-boden.md`) sie gedacht hatte.

**Der `global_max`-Hebel taugt nicht für die Bitgleichheits-Kontrolle.** Er
liefert nur das *affine* Spektrogramm `(x + 4) / 4`, und diese Summe verliert
für `x > −2` Mantissenbits; aus dem zurückgerechneten Wert lässt sich der
Referenzboden `log_max − 8` nicht mehr exakt bilden. Auf lautem Material
(`log_max` in [0, 4)) fällt das wegen gleicher Rundungsraster nie auf, auf
leisem (`log_max` in [−2, 0)) bekommt **jede achte Eingabe** einen um ein Bit
verschobenen Boden und damit jede geklemmte Zelle — gemessen auf 300
synthetischen Signalen: 12 %. Der Einbau rechnet das ungeklemmte Spektrogramm
deshalb aus den Primitiven der Bibliothek (`stft_mlx`, `hanning`,
`get_mel_filters`, dieselben Aufrufe wie `log_mel_spectrogram`) und klemmt in
float32. Mit Perzentil 100 ist das **bitgleich** zur Bibliothek — auf Vollpegel,
−40 dB, −60 dB, Stille, Rauschen und 300 leisen Signalen —, und genau dieser
Vergleich steht als Test im Repo; er fängt auch, wenn die Bibliothek ihren
Mel-Pfad ändert und unsere Kopie nicht mitzieht.

**Der Perzentil-Boden steht unter einem Transienten nicht exakt still.** Die
Zellen des Knalls sitzen alle an der Spitze der Verteilung und verdrängen die
Ordnungsstatistik um ihre Anzahl nach oben — um so mehr, je kürzer der Pass
und je breitbandiger der Knall. Bodenverschiebung in dB auf dem mitgelieferten
`tests/data/interview.mp3` (−20 dB ausgesteuert, Transient auf Vollausschlag,
ohne Modell):

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

Der Rauschburst belegt rund 1300 Zellen über der Schwelle; das oberste
Promille eines 60-s-Passes sind 768 Zellen, das oberste Prozent 7680. Der
Sweep oben hatte nur die *Kosten auf sauberem Material* als unkritisch
ausgewiesen — für die Robustheit ist das Perzentil sehr wohl eine Wahl, und
99,99 scheidet ganz aus. Die Spannweite der Eingabe wird mit 99 etwas breiter
als mit 99,9 (Interview: 2,39–2,45 gegen 2,22–2,27), bleibt aber unter den
2,547, die die Rohspur oben mit 99,9 folgenlos erreicht.

**Das Perzentil läuft nur über die echten Frames.** Der Block wird auf ein
Vielfaches von 30 s mit Nullen aufgefüllt, und Polsterzellen liegen auf dem
Minimum −10. Über den ganzen Block gerechnet rutscht das Perzentil mit dem
Polsteranteil nach unten: ein 5-s-Clip im 30-s-Block bekäme einen um 18 dB
tieferen Boden als derselbe Clip ungepolstert — ein Regime, das keine der
Messungen oben abdeckt (alle Ströme sind exakt 300 s) und das die Bibliothek
nicht kennt, weil ihr Maximum nie in einem Polsterframe liegt. Die Statistik
nimmt deshalb nur die Frames, die Eingabe enthalten; der Clamp trifft alle.
Bei 100 ändert das nichts, die Bitgleichheits-Kontrolle bleibt. (Gefunden im
Review, nicht im Entwurf.)

**Was in überwiegend stillen Pässen passiert: kein Nachweis, aber dünner
Rand.** Alle Ströme oben sind dichte Rede, dort liegt das 99. Perzentil
15–18 dB unter dem Maximum. In einem Pass, der vor allem aus Pausen besteht
(die 60-s-Kopfprobe auf einer Aufnahme, die mit Stille beginnt), rutscht das
Perzentil innerhalb der Sprachzellen nach unten und der Boden liegt tiefer als
je gemessen. `voxpopuli_sparse.py`: dieselben Clips, dazwischen weißes
Rauschen bei −60 dBFS als Raumton, bis der Sprachanteil stimmt; 10 Ströme à
300 s, beide Böden gepaart:

| Sprachanteil | Abstand max → p99 | max | Perzentil 99 | gepaart |
|---|---|---|---|---|
| ~100 % (oben) | 15–18 dB | 11,39 % / 7,32 % | 11,27 % / 7,14 % | −0,12 [−0,85, +0,49] · CER −0,18 [−0,79, +0,28] |
| 50 % | 23,6 dB (20,7–28,9) | 10,14 % / 6,35 % | 10,38 % / 6,67 % | **+0,24 [−0,03, +0,55]** · CER +0,32 [−0,01, +0,77] |
| 20 % | 28,5 dB (26,1–30,0) | 12,30 % / 8,25 % | 12,36 % / 8,54 % | +0,07 [−0,72, +1,10] · CER +0,28 [−0,25, +1,02] |

Kein Intervall schließt die Null aus, aber bei 50 % fehlt dazu ein
Hundertstel, und beide CER-Intervalle stehen knapp darunter. Der tiefere
Boden zeigt in den Pausen Rauschstruktur, die die dichten Ströme nie hatten —
das ist die Asymmetrie von oben an ihrer Grenze. Der Raumton ist hier
synthetisch und weiß, echter wäre tieffrequenter; und der 20 %-Lauf trägt nur
10 min Sprache. **Offen, nicht eingebaut:** den Boden nach unten zu begrenzen
(`max(p99, log_max − 20 dB) − 8`) hielte stille Pässe im Bereich der dichten
Messung, würde aber den Schutz vor einem 28-dB-Knall auf 20 dB kappen — die
Restdosis von 8 dB liegt nach der Tabelle oben zwischen „nichts" (5,2 dB) und
+0,35 (16,2 dB). Beides ist ungemessen; die Entscheidung braucht einen Lauf
mit der Kappe auf den stillen *und* den Knall-Strömen.

**Auf dem Interview ist der Text mit und ohne Knall nicht bitgleich.** 120 s,
−20 dB, 60-Hz-Knall bei 45 %: unter dem Max-Boden ändert der Knall 15 Stellen
über die ganze Passage, unter dem Perzentil-Boden 3, mindestens zwei davon
weit weg vom Knall: die Restverschiebung von 0,1 dB bewegt jede geklemmte Zelle
um ein Bit, und Greedy-Decoding lässt irgendwo ein knappes Token kippen. Die
Bitgleichheit auf `hart` war ein Passagenbefund, kein Verhalten. Der Test im
Repo verlangt darum *deutlich weniger* Änderungen als unter dem Max-Boden und
höchstens zwei Prozent der Wörter, nicht denselben Text.

**Reproduktion durch den Produktionscode** (`voxpopuli_spike.py 10 300 24
99.9,99`, dieselben zehn Ströme, Bodenanstieg median 28,2 dB):

| Boden | sauber | mit Knall | Kosten des Knalls, gepaart |
|---|---|---|---|
| max (Bibliothek) | 11,36 % / 7,36 % | 12,88 % / 7,90 % | **+1,52 [+0,78, +2,26]** \* |
| Perzentil 99,9 | 10,82 % / 6,93 % | 10,86 % / 6,88 % | +0,04 [+0,00, +0,10] |
| Perzentil 99 | 10,76 % / 6,88 % | 10,77 % / 6,88 % | +0,01 [−0,03, +0,06] |

Max und 99,9 reproduzieren die Zahlen oben auf die zweite Stelle; 99 ist die
einzige Variante, deren Intervall die Null *enthält*. Auf sauberem,
ungedämpftem Material (`voxpopuli_floor.py 10 300 99.9,99`, dieselben Ströme)
ebenso auf die zweite Stelle: max 11,39 % / 7,32 %, Perzentil 99,9 11,29 % /
7,20 % (dWER −0,10 [−0,83, +0,47]), Perzentil 99 11,27 % / 7,14 % (dWER
**−0,12 [−0,85, +0,49]**, dCER −0,18 [−0,79, +0,28]); Geschwindigkeit 7,8×
Echtzeit bei allen dreien. Dass die Perzentil-Böden im Knall-Lauf auch auf dem
sauberen Arm um ein halbes Prozent besser stehen, ist dort und nicht hier zu
sehen — `pair()` dämpft beide Arme um 24 dB; die naheliegende Erklärung ist,
dass auf so leise ausgesteuertem Material schon die Sprachspitze den Boden zu
hoch setzt. Gemessen ist das nicht.

**Nebenbefund, geklärt: der Whisper-Pfad ist nicht betroffen.**
`faster_whisper/feature_extractor.py:227` trägt dieselbe Zeile, und noScribe
übergibt Whisper die ganze Datei — die Wirkung bleibt dort aber aus. Auf 24
Strömen (122,3 min, 28,7 dB Bodenanstieg) kostet der Transient
**−0,20 [−1,85, +1,19]** mit VAD und **−0,51 [−1,98, +0,83]** ohne.

Beide Intervalle enthalten die Null **und schließen Voxtrals +1,52 aus**; ein
Effekt dieser Größe ist dort also nicht bloß ungezeigt, sondern ausgeschlossen.
Bemerkenswert am Weg dorthin: auf acht Strömen standen dieselben Zahlen noch bei
+0,60 und +0,95, also im Vorzeichen einig mit Voxtral. Der Vorzeichenwechsel bei
dreifacher Datenmenge ist das Verhalten von Rauschen, nicht von einem Effekt —
und eine Mahnung, wie wenig ein Punktschätzer ohne tragfähiges Intervall sagt.
Skript: `whisper_spike_streams.py`.

## 7. Was die Literatur bestätigt

Zwei Arbeiten, gegen die arXiv-Originale geprüft:

* **Chondhekar u. a. 2025**, [arXiv:2512.17562](https://arxiv.org/abs/2512.17562):
  MetricGAN+ auf 500 medizinischen Aufnahmen, vier ASR-Systeme — Aufbereitung
  verschlechtert in *allen* Bedingungen, um 1,1 bis 46,6 %. Unbehandeltes
  verrauschtes Audio schlägt durchweg das aufbereitete.
* **Islam, Nahar & Hamid 2026**, [arXiv:2603.04710](https://arxiv.org/abs/2603.04710):
  SAM-Audio hebt den PSNR von 32,28 auf 35,99 dB und verschlechtert dabei jede
  Konfiguration (Whisper large-v3 auf Bengali 65,83 → 77,35 % WER).

Beide finden genau die Trennung, die dieses Dokument durchhält: ein besseres
Signalmaß sagt nichts darüber, was das Modell hört.

## 8. Was beim Messen zu beachten ist

Vier Fallen, in alle einmal hineingetappt:

1. **Am Transkript messen, nicht am Spektrum.** Der Produktionspfad hat den
   größten Mel-Formfehler aller geprüften Varianten — und ändert am Transkript
   nichts, weil der Fehler in den obersten 16 von 128 Mel-Bins sitzt.
2. **Eine Passage von 422 bis 859 Wörtern trägt ±1,8 bis ±2,5 CER-Punkte.**
   Alles darunter ist nicht entscheidbar. Unter Greedy-Decoding macht ein
   einziger gekippter Token eine lange Divergenz — zwei kurze Passagen, die sich
   widersprechen, sind der Normalfall, kein Rätsel.
3. **Eine Referenz, die durch Korrektur eines Transkripts entstanden ist, ist an
   diesen Arm angelehnt.** Bei `zoom_9890-10190` sind es 0,25 CER-Punkte,
   gemessen, indem derselbe Pfad mit anderem Resampler gegen dieselbe Referenz
   antritt. Vergleiche laufen deshalb gegen `p2-soxr`, nicht gegen `p0-ist`;
   Streitstellen listet `adjudicate.py` mit Zeitmarke zum Nachhören.
4. **Reproduzierbarkeit ersetzt keine Stichprobe.** Die Messung, die den
   Chunker-Umbau rechtfertigte, war vollständig reproduzierbar und trotzdem
   falsch.

## 9. Werkzeuge

Alle in `docs/skripte/`. Herkunft, Qualität und Ausrichtung des Testmaterials
stehen in `Audiotest2/MATERIAL.md`.

| Skript | wofür |
|---|---|
| `audio_audit.py` · `audit_long.py` | Pegel, Spitze, Clipping, Lautheit, Spreizung einer Datei |
| `find_passage.py` · `build_reference.py` | eine Passage zur Handkorrektur auswählen und vorbereiten |
| `locate_passage.py` | dieselbe Passage in einer anderen Fassung wiederfinden |
| `preproc_variants.py` · `preproc_cer.py` · `preproc_mel.py` | Pfadvarianten bauen, am Transkript und im Mel-Raum vergleichen |
| `pair_cer.py` | Arme aus verschiedenen Quelldateien gegen eine Referenz |
| `audio_filters.py` | benannte libavfilter-Ketten (Leveller, Entrauscher) |
| `adjudicate.py` | Streitstellen zweier Arme mit Zeitmarke |
| `wer.py` · `bootstrap_cer.py` · `encoder_diff.py` | Bewertung, Intervalle, referenzfreier Vergleich |
| `fleurs_*.py` | die Einzelfragen auf FLEURS: Pegel, Resampler, leise Passagen, Rauschen, Fensterposition, lange Ströme |
| `mel_stream.py` | Per-Block- gegen Ganzdatei-Normalisierung des log-Mel, gepaart über lange Ströme |
