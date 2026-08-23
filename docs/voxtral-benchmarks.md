# Voxtral in noScribe — Messergebnisse

Stand: 19./20. Juli 2026, im August nachgezogen · M1 Max, 32 GB · Testmaterial:
„Mona Podcast 323" (20,4 min, Deutsch, 2 Sprecher)

**Was im Code steht, steht hier nicht mehr.** Die Schwellen der
Schleifenerkennung, die Reparaturleiter, das Speichermodell, die Kopfreparatur
und die Namenskorrektur tragen ihre Messung im Kommentar neben der Konstante,
so wie es dieses Repository hält. Diese Datei behält, was dort keinen Platz
hat: Messungen zu Entscheidungen, die *nicht* getroffen wurden, Belege dafür,
dass ein Defekt nicht bei uns liegt, und Messfallen. Die Quantisierung hat ihr
eigenes Dokument, `voxtral-quantisierung.md`.

Referenzgröße für Lesbarkeit ist die Kommadichte pro 100 Wörter; Whisper liegt
auf diesem Material bei **9,58–10,62** (je nach Ausschnitt), was als „gut
lesbar" gilt.

---

## 1. Der wichtigste Fund: `repetition_penalty`

`mlx_voxtral.generate()` setzt den Wert per Default auf **1.2** — ein
Chat-Default, den wir nie überschrieben hatten. Warum das in gesprochener
Sprache gerade die Satzzeichen kostet, steht im Kommentar über dem Aufruf in
`voxtral_engine.transcribe_array`; die Sprossen der Strafenleiter stehen bei
`RETRY_REPETITION_PENALTIES`. Hier bleiben die beiden Messreihen, aus denen
diese Kommentare ihre Zahlen ziehen.

| `repetition_penalty` | Kommas/100 W (mini, 600 s) | doppeltes „nicht" |
|---|---|---|
| **1.0** | **10,87** | erhalten |
| 1.01 | 10,17 | erhalten |
| 1.05 | 10,01 | erhalten |
| 1.1 | 9,60 | **verschluckt** |
| 1.15 | 9,21 | **verschluckt** |
| 1.2 (alter Default) | 8,57 | verschluckt |

Am vollen Podcast durch die echte Pipeline:

| | Wörter | Kommas/100 W | Satzenden/100 W |
|---|---|---|---|
| Whisper (Referenz) | 4069 | 9,76 | 9,58 |
| mini **alt** (1.2) | 3954 | 7,89 | 7,54 |
| mini **neu** (1.0) | 4054 | **10,51** | 9,25 |

Aus dieser zweiten Tabelle stammt die Zahl „27 % aller Kommas": 3954 Wörter bei
7,89 sind 312 Kommas, 4054 bei 10,51 sind 426.

**Es gibt keinen belegten Standardwert.** Weder das Voxtral-Paper noch die
Modellkarte nennen einen; Mistrals Referenzaufruf ist
`TranscriptionRequest(model, audio, language, temperature=0.0)` ohne Strafe. Die
1.1 in `mzbac/mlx.voxtral` steht in einem illustrativen Beispiel. Die Literatur
erklärt den Konflikt: Die Strafe wirkt auf *alle* Tokens gleich und tauscht
Wiederholungsunterdrückung gegen grammatische Flüssigkeit.

**Weitere Parameter:** Bei `temperature=0.0` (greedy) sind `top_p`/`top_k`/`min_p`
wirkungslos — deshalb stört die Reparaturleiter den Decode über die Temperatur
und nicht über sie. `logit_bias` funktioniert als echte Hotword-Steuerung (Token
`'y'` +2 korrigierte „Mohnas" → „Mona's" ohne messbaren Kollateralschaden), ist
aber kontextfrei und tokenisierungsabhängig — `Markus` ist ein einzelnes Token,
`Mona` zerfällt in `[' Mon','a']`. Gebaut wurde stattdessen die
Korrekturliste, die ganze Phrasen trifft.

---

## 2. `max_new_tokens` — latenter Datenverlust

Deutsche Sprache erzeugt **4,64 Text-Tokens/s** (gemessen; das ist die Herkunft
des Faktors 20 im Token-Budget). Das feste Limit von 4096 hätte jeden Pass über
~15 min am Ende beschnitten:

| Pass | gebraucht | altes Limit 4096 |
|---|---|---|
| 600 s | 2782 | ok |
| 1006 s | ~4664 | **schneidet ab** |
| 1500 s | ~6955 | schneidet ab |

Die beiden langen Zeilen sind historisch: seit `TRUSTED_CHUNK_SEC` kappt die
Automatik jeden Pass bei 600 s, weil Voxtral darüber hinaus nicht gemessen ist.

Das Limit ist reine Schleifenbremse — es begrenzt nur die Generierungsschleife,
der KVCache wächst aus den *tatsächlich* erzeugten Tokens. Ein hoher Wert
reserviert **nichts**. Jetzt `min(32768, dauer*20 + 512)` = 4,3× Reserve. (Der
greedy-Pfad läuft inzwischen über `mlx_lm.generate_step`, das Argument gilt
dort genauso; nur die Strafen-Sprosse geht noch durch `mlx_voxtral.generate`.)
Das Kontextfenster von ~32k bindet nicht: Audio kostet 12,6 Tokens/s, ein
600-s-Pass also ~7,6k + ~2,8k Tokens.

---

## 3. Sprache

`language=None` (noScribes „Auto") fügt **keinen** `lang:`-Token ein — echtes
Auto-Detect, kein heimliches Englisch-Default.

**Achtung:** Eine *falsche* Sprachvorgabe lässt Voxtral **übersetzen** statt
transkribieren — `language='en'` auf deutschem Audio liefert flüssiges Englisch.
Deshalb wird die Sprache nur weitergegeben, wenn sie explizit gewählt wurde.

Auto und `de` sind allerdings auch bei klarem Deutsch nicht austauschbar: an
welcher Stelle der Unterschied auftaucht, zeigt §5 — dort kostet einmal die
Vorgabe `de` die ersten ~150 Wörter, einmal die Auto-Erkennung den Vorlauf.

*Das damals offene Restrisiko* — ein einzelner Pass kippt und übersetzt sich
selbst — ist inzwischen abgesichert, und es brauchte dafür weder einen
Whisper-Ladevorgang noch eine neue Abhängigkeit: ein Stoppwort- und
Schriftzeichen-Detektor bestimmt die Sprache aus dem *Text*, zwei Chunks müssen
sich einig sein, und ein übersetzter Pass wird zum Grund für die
Reparaturleiter. Der Mechanismus und seine Feldmessungen stehen bei
`_detect_language` / `_file_language` / `want_lang` in `voxtral_engine.py` und
in den Tests dazu.

---

## 4. Wiederholungsschleifen

Ohne Strafe kippte small auf einem 410-s-Pass in eine Schleife: **4099 identische
Wörter**. Das war der Anlass; Erkennungsschwellen, Kalibrierkorpus und die
Reihenfolge der Reparatur haben sich seither mehrfach geändert und stehen dort,
wo sie gelten — bei `DEGENERATE_CYCLE_REPEATS` und in `_transcribe_guarded`.

Was hier bleibt, weil es eine *nicht* getroffene Entscheidung begründet: eine
Strafe kann Schleife und Bedeutung nicht unterscheiden. 1.1 machte aus „wir
können es dir **nicht nicht** erzählen" ein „… nicht erzählen" — der Satz kippt
ins Gegenteil. Ein flüssig lesbarer, aber falscher Satz ist gefährlicher als
offensichtlich kaputter Text. Deshalb steht die Strafe am *Ende* der Leiter und
nicht am Anfang, und deshalb wird zuerst geteilt: der 410-s-Pass lieferte als
2×205 s sauberen Text, inklusive des doppelten „nicht" und des echten
vierfachen „Jetzt. Jetzt. Jetzt. Jetzt."

---

## 5. Der verlorene Anfang eines Passes

Ein langer Pass kommt manchmal **ohne seine ersten Sekunden Sprache** zurück —
ohne Schleife, ohne falsche Sprache, ohne Spur im Log. Der Defekt, seine
Häufigkeit und die Reparatur (`_recover_lost_head`) sind im Code beschrieben.
Hier stehen die drei Dinge, die dort keinen Platz haben.

**Erstens: es hängt am Ende des Fensters, nicht am Anfang.** Feiner Sweep bei
Auto auf einem 20-Minuten-Video, dessen erste 2,4 s eine Bemerkung vor dem Take
enthalten; der Anfang des Audios ist in allen sieben Läufen bitgleich, nur das
Ende wandert:

| Fenster | 1180 | 1195 | 1205 | 1215 | 1220 | 1223 | 1226 |
|---|---|---|---|---|---|---|---|
| Anfang | da | **weg** | da | da | da | da | **weg** |

Nicht monoton, keine Schwelle — dieselbe Signatur wie beim Sprach-Kippen.

**Zweitens: es liegt nicht an uns.** Damit sind die naheliegenden Gegenmittel
widerlegt: kürzere Fenster (1195 verliert, 1223 nicht), Sprache pinnen (kippt in
beide Richtungen — einmal kostete `lang:de` die ersten ~150 Wörter eines
1426-s-Fensters), Stille voranstellen (0 von 6, und löste den Verlust in einem
Test sogar aus). Auch referenzgenaue Audio-Features ändern nichts, und unsere
Prompt-Tokens sind bitgleich mit `mistral_common`.

**Drittens, die Messfalle:** FLEURS taugt für diese Frage nicht. Dort ist der
Anfang bei 120/300/450/600 s sauber — aber auch bei 1200 s, wo er es nicht sein
dürfte. Die Positivkontrolle fällt durch, also sagen die sauberen Zeilen nichts.
Wer das nachmisst, braucht Material, das den Defekt nachweislich zeigt; ein
roher Zoom-Mitschnitt (4,8 h) tut es zuverlässig — auf ihm verlieren 5 von 64
Fensterläufen (32 Fenster à 300 s und 600 s, je einmal auf Auto und auf `de`)
ihren Anfang, im schlimmsten Fall 18 Wörter zusammenhängender Rede. Das ist
die Zahl, die `VOXTRAL.md` zitiert; die Aufschlüsselung steht im Kommentar zu
`_recover_lost_head`.

---

## 6. Was 8-bit small auf 32 GB verhindert (geprüft, verworfen)

Der Sockel ist das Problem: **26,4 GB allein für die Gewichte**. Dazu ein
zweiter, unabhängiger K.-o.: **0,80× Echtzeit**, also langsamer als die
Aufnahme. Die Auswege sind in `voxtral-quantisierung.md` durchgemessen und
abgehakt; zwei Rechnungen von hier tragen noch, weil sie erklären, *warum* sie
so ausgingen.

**Der KV-Cache ist viel kleiner als die Steigung.** Aus der Architektur
gerechnet (`Schichten × 2 × KV-Köpfe × head_dim × Tokens`, Audio kostet
12,6 Tokens/s):

| | KV-Cache (bf16) | gemessene Steigung insgesamt |
|---|---|---|
| small, 300 s | 0,62 GB | 4,05 GB |
| small, 1500 s | 3,10 GB | 20,3 GB |
| mini, 600 s | 0,93 GB | 6,2 GB |

Der KV-Cache macht nur **~15 %** der Steigung aus; der Rest entfällt auf den
Audio-Encoder. Eine KV-Quantisierung kann den Peak deshalb kaum senken — die
spätere Direktmessung zeigt sogar, dass sie ihn *hebt*.

**Bringen längere Pässe überhaupt Qualität?** Gemessen: 600-s-Einzelpass 10,87
Kommas/100 W vs. 2×300 s 10,66 — praktisch gleich. Die Literatur stützt das: Es
gibt ein *Optimum* der Chunk-Länge, das vom Training abhängt (Whisper 30 s,
Distil-Whisper 15 s), nicht „je länger desto besser". Das reale Problem sind die
**Grenzen** (abgeschnittene Satzbezüge, Sprecherwechsel an der Kante) — dagegen
arbeiten pausen-ausgerichtete Schnitte plus Overlap bereits. Das ist der Beleg
dafür, dass die 600-s-Kappe nichts kostet.

---

## 7. Was aus den offenen Punkten wurde

Alle fünf sind entschieden; die Liste bleibt, damit niemand sie ein zweites Mal
aufmacht.

- **mini 8-bit als Standard** — gebaut und registriert als `voxtral-mini-8bit`,
  inzwischen die Empfehlung.
- **small 8-bit** — gebaut und registriert als `voxtral-small-8bit`. Die Frage
  „läuft es auf 32 GB?" ist beantwortet: nein, es braucht ~34 GB für seinen
  kürzesten Pass und wird darunter vor dem Start abgelehnt.
- **Sprach-Fixierung über mehrere Pässe** — gebaut, siehe §3. Rest-Lücke:
  Chunks, die schon geschrieben waren, bevor die Dateisprache feststand,
  bekommen nur eine Warnung mit ihrer Nummer.
- **Voxtral-Mini-4B-Realtime** — **verworfen**, an Mistrals eigenen
  FLEURS-Zahlen: Deutsch 6,19 % WER bei 480 ms, 4,15 % selbst bei 2,4 s
  Verzögerung, gegen 3,54 % beim offline Mini 3B — ein kleineres Modell schlägt
  es. Streaming tauscht Vorausschau gegen Latenz, und Latenz ist bei einer
  Datei-Transkription wertlos.
- **`voxtral-mini-2602` („Transcribe 2")** — weiterhin **API-only** und damit
  für vertrauliche Interviews ungeeignet.

---

## 8. Reproduzieren

Die Skripte liegen in `docs/skripte/`: `bitmatrix.py` (Tempo/Worttreue),
`decisive.py` (Satzzeichen je Strafe), `rep_matrix.py` (Strafensprossen gegen
das doppelte „nicht"), `split_retry.py` (Schleifenreparatur), `cli_check.py`
(Formatprüfung). Die Konvertierung liegt als `tools/quantize_voxtral.py` im
Repository.

Die Arbeit dieser Messreihe liegt auf `feature/voxtral-engine`: die
Decoding-Fixes (Strafe 1.0, Stop-Tokens, Token-Budget), die Namenskorrektur nur
für Deutsch, die Schleifenerkennung, Teilen-statt-Strafe mit Eskalation ab 1.01
und die konfigurierbare Speicherreserve. Frühere Fassungen dieser Datei nannten
dafür sieben Commit-Kurz-IDs; die zeigen auf Objekte, die von keinem Branch mehr
erreichbar sind, und sind deshalb ersatzlos entfernt.
