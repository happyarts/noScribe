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

**Stand der Entscheidung: nicht eingebaut, aber der erste Kandidat mit einem
echten Argument.** Dagegen steht Architekturtreue — alle vier Implementierungen
(transformers, mlx-audio, transcribe.cpp, mlx-voxtral 0.0.6) nehmen das nackte
Maximum, und damit wurde das Modell trainiert. Dafür steht ein gemessener,
monotoner Gewinn in einem realistischen Fehlerfall bei nicht nachweisbaren
Kosten. Was vor einem Einbau fehlt: die Dosis-Wirkung auf einer zweiten Passage
und auf VoxPopuli-Strömen mit eingesetztem Transienten — +5,21 steht bislang auf
einer Passage.

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
