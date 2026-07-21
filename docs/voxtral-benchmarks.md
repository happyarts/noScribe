# Voxtral in noScribe — Messergebnisse

Stand: 19./20. Juli 2026 · M1 Max, 32 GB · Testmaterial: „Mona Podcast 323" (20,4 min, Deutsch, 2 Sprecher)

Alle Zahlen sind gemessen, nicht geschätzt. Referenzgröße für Lesbarkeit ist die
Kommadichte pro 100 Wörter; Whisper liegt auf diesem Material bei **9,58–10,62**
(je nach Ausschnitt), was als „gut lesbar" gilt.

---

## 1. Der wichtigste Fund: `repetition_penalty`

`mlx_voxtral.generate()` setzt den Wert per Default auf **1.2** — ein Chat-Default,
den wir nie überschrieben hatten. Er teilt den Logit jedes Tokens, das in den
letzten 20 Tokens vorkam. In gesprochener Sprache sind das vor allem **Satzzeichen
und Funktionswörter**, die dadurch systematisch verschwinden.

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

**Es gibt keinen belegten Standardwert.** Weder das Voxtral-Paper noch die
Modellkarte nennen einen; Mistrals Referenzaufruf ist
`TranscriptionRequest(model, audio, language, temperature=0.0)` ohne Strafe. Die
1.1 in `mzbac/mlx.voxtral` steht in einem illustrativen Beispiel. Die Literatur
erklärt den Konflikt: Die Strafe wirkt auf *alle* Tokens gleich und tauscht
Wiederholungsunterdrückung gegen grammatische Flüssigkeit.

**Weitere Parameter:** Bei `temperature=0.0` (greedy) sind `top_p`/`top_k`/`min_p`
wirkungslos. `logit_bias` funktioniert als echte Hotword-Steuerung (Token `'y'`
+2 korrigierte „Mohnas" → „Mona's" ohne messbaren Kollateralschaden), ist aber
kontextfrei und tokenisierungsabhängig — `Markus` ist ein einzelnes Token,
`Mona` zerfällt in `[' Rom','y']`.

---

## 2. `max_new_tokens` — latenter Datenverlust

Deutsche Sprache erzeugt **4,64 Text-Tokens/s**. Das feste Limit von 4096 hätte
jeden Pass über ~15 min am Ende beschnitten:

| Pass | gebraucht | altes Limit 4096 |
|---|---|---|
| 600 s | 2782 | ok |
| 1006 s (Auto-Wert auf 32 GB) | ~4664 | **schneidet ab** |
| 1500 s | ~6955 | schneidet ab |

Das Limit ist reine Schleifenbremse — `mlx_voxtral` nutzt es nur als
Schleifengrenze, der KVCache wächst in 256er-Schritten aus den *tatsächlich*
erzeugten Tokens. Ein hoher Wert reserviert **nichts**. Jetzt
`min(32768, dauer*20 + 512)` = 4,4× Reserve. Das Kontextfenster (131 072) bindet
nie: Audio kostet 12,6 Tokens/s, ein 1500-s-Pass also ~19k + ~7k Tokens.

---

## 3. Sprache

`language=None` (noScribes „Auto") fügt **keinen** `lang:`-Token ein — echtes
Auto-Detect, kein heimliches Englisch-Default. Bei klarem Deutsch messbar
gleichwertig zu `'de'`.

**Achtung:** Eine *falsche* Sprachvorgabe lässt Voxtral **übersetzen** statt
transkribieren — `language='en'` auf deutschem Audio liefert flüssiges Englisch
(„Especially for my metabolic health…"). Deshalb wird die Sprache nur
weitergegeben, wenn sie explizit gewählt wurde.

*Offenes Restrisiko:* Bei mehreren Pässen erkennt jeder Pass unabhängig; ein Pass
könnte theoretisch kippen und sich selbst übersetzen. Absicherung bräuchte einen
Spracherkenner (Whisper-Ladevorgang oder neue Abhängigkeit) — bewusst nicht gebaut.

---

## 4. Quantisierung — der größte Hebel

Konvertierung mit `/tmp/quantize6.py` (umgeht die 2/4/8-Beschränkung des
mitgelieferten Skripts). Laden und Quantisieren laufen **lazy** (0,4 GB RSS), die
Spitze entsteht erst beim Speichern.

### mini (3B), 150-s-Ausschnitt

| Variante | Größe | Tempo | Peak | wortgleich zu bf16 |
|---|---|---|---|---|
| bf16 (Original) | 8,7 GB | 1,53× | 13,5 GB | 100 % |
| **8-bit** | **5,1 GB** | **7,32×** | **7,6 GB** | **100,00 %** |
| 6-bit | 3,9 GB | 7,74× | 6,5 GB | 97,34 % |
| 4-bit | 2,7 GB | 8,23× | 5,4 GB | 94,12 % |

Auf 600 s: 8-bit = 99,61 % wortgleich, 10,81 Kommas/100 W, 3,73×, Peak 12,3 GB.

**8-bit ist der klare Sweetspot**: praktisch identische Ausgabe bei knapp
fünffachem Tempo und deutlich weniger Speicher. Die Bit-Breiten skalieren
sauber — es gibt *keinen* Kernel-Ausreißer bei 6 Bit. Der große Tempo-Sprung ist
bf16 → *irgendeine* Quantisierung (1,5× → 7,3×); zwischen 8 und 4 Bit liegen nur
12 % Tempo, aber 6 Prozentpunkte Worttreue.

**Oberhalb von 8 Bit ist nichts zu holen.** bf16, fp16 und 8-bit liefern auf
150 s denselben Text (je 414 Wörter, 100 % wortgleich); fp16 ist gegenüber bf16
weder schneller noch genauer (1,84× vs. 1,86×). Der Qualitätsverlust setzt erst
*unter* 8 Bit ein. „16-bit-Integer" gibt es in MLX nicht und wäre sinnlos —
quantisiert bräuchte es 128 KB + 8 KB Skalen pro 256×256-Block, also mehr als
fp16 mit 128 KB.

### small (24B)

| Variante | Größe | Tempo | Peak | Urteil auf 32 GB |
|---|---|---|---|---|
| 4-bit mixed | 14 GB | ~2× | 17 GB @120 s | nutzbar, 576-s-Pässe |
| **6-bit** | 20,3 GB | 1,69× | 26,4 GB @410 s | nutzbar, 303-s-Pässe |
| **8-bit** | 26,5 GB | **0,80×** | **27,2 GB @60 s** | **unpraktikabel** |
| bf16 | ~48,5 GB | — | — | passt nicht |

8-bit läuft zwar (Laden in 3 s, sauberer Text), braucht aber schon für **eine
Minute** Audio 27,2 GB und ist mit 0,80× langsamer als die Aufnahme selbst. Es
ist registriert, damit die Automatik warnt und auf den 60-s-Mindestpass
zurückfällt, statt den Speicher zu sprengen — brauchbar erst auf einer Maschine
mit deutlich mehr RAM.

### Lizenz und Weitergabe

Beide Originalmodelle stehen unter **Apache 2.0** — abgeleitete Gewichte dürfen
weitergegeben werden. Fertige MLX-Konvertierungen in 6/8 Bit gibt es auf HF
**nicht** (nur bf16-mini bei `mlx-community`, 4-bit-small und 8-bit-small bei
`VincentGOURBIN`). Zum Teilen wäre `mlx-community` die dauerhaftere Adresse als
ein Privatkonto; für mini ist auch „gar nicht hosten" eine Option, weil die
Konvertierung aus dem bf16-Modell nur 3 Sekunden dauert.

Konvertierungsquelle: `mistralai/Voxtral-Small-24B-2507`, 11 Shards, 48,5 GB
(die doppelte `consolidated.safetensors` wird übersprungen).

---

## 5. Speichermodell und Pass-Längen

`peak_GB ≈ fixed + slope · sekunden` — die Auto-Chunkung wählt die längste Pass-Dauer,
deren geschätzter Peak in `(RAM − Reserve)` passt.

| Modell | fixed | slope | Pass @ Reserve 7 GB |
|---|---|---|---|
| mini bf16 | 7,9 | 0,0160 | 1068 s |
| **mini 8-bit** | **6,0** | **0,0104** | **1816 s** |
| small 4-bit | 15,2 | 0,0170 | 576 s |
| small 6-bit | 20,9 | 0,0135 | 303 s |

Reserve per `voxtral_ram_reserve_gb` in der `config.yml` einstellbar
(6 GB / 5 GB → small 6-bit: 377 s / 451 s).

Belege für die Reserve: ein 21,8-GB-Pass lief mit 2,7 GB freiem RAM sauber durch,
ein 26,4-GB-Pass ebenfalls. **Wichtig:** Der Lade-Transient darf swappen (er wird
vor dem Generieren frei), der *Generate*-Working-Set nicht — MLX auf ausgelagerten
Puffern thrasht endlos.

---

## 6. Wiederholungsschleifen

Ohne Strafe kippte small auf einem 410-s-Pass in eine Schleife: **4099 identische
Wörter**. Erkennung, kalibriert an echten Transkripten:

| | echt | Schleife |
|---|---|---|
| längste Kette identischer Wörter | 2 | 690–4099 |
| Kompressionsrate | 2,59–2,63 | 5,75 |

Schwellen: Kette ≥ 12 (primär), Kompression > 4,0 (sekundär), Pässe unter 30
Wörtern werden nie beurteilt.

**Reparatur in dieser Reihenfolge:**
1. Pass an der leisesten Stelle **teilen**, Hälften ohne Strafe, rekursiv bis ~45 s.
   Gemessen: 410 s → 2×205 s, sauber, **inklusive „nicht nicht"** und des echten
   vierfachen „Jetzt. Jetzt. Jetzt. Jetzt."
2. Erst danach Strafe, eskalierend **1.01 → 1.1**.

Grund für diese Reihenfolge: Eine Strafe kann Schleife und Bedeutung nicht
unterscheiden. 1.1 machte aus „wir können es dir **nicht nicht** erzählen" ein
„… nicht erzählen" — der Satz kippt ins Gegenteil. Ein flüssig lesbarer, aber
falscher Satz ist gefährlicher als offensichtlich kaputter Text.

---

## 7. Eigennamen

Kein Decoding-Parameter behebt sie: „Monas Recovery VitalyTea" wird in *jeder*
Variante zu „Mohnas Recovery Reality", „Mona Muster" zu „Mohna/Muna Muster".

Zwei Mechanismen:
- **Korrekturliste** (`~/Library/Application Support/noScribe/voxtral_corrections.yml`) —
  ganze Phrasen, tokenisierungsunabhängig, vom Nutzer pflegbar.
  *Vorsicht bei zu allgemeinen Mustern:* `"balance all"` traf „die **Balance all**
  dieser Faktoren" → entfernt.
- **Phonetische Normalisierung** aus dem Sprecher-Namen-Feld — Kölner Phonetik,
  abgesichert durch *gleiche Wortlänge + Editierdistanz ≤ 2*. Fängt `Muna/Mohna →
  Mona` und `Marcus → Markus`. Über 4255 Wörter echtes Deutsch: genau 3 gewollte
  Änderungen, null Fehlalarme.
  **Nur für Deutsch** — eine sprachneutrale Variante schrieb „Rome" → „Mona" und
  „Anna" → „Anne" und wurde deshalb gestrichen.

---

## 8. Modellvergleich am vollen Podcast

| | Wörter | Kommas | Satzenden | Sonvita ✓/✗ |
|---|---|---|---|---|
| Whisper | 4144 | 9,58 | 9,41 | 0/3 |
| mini 2 (rep 1.0) | 4112 | **10,31** | 8,37 | 3/1 |
| small 3 (4-bit) | 4090 | 9,46 | 9,05 | 4/0 |
| small split+1.01 | 4105 | 9,67 | 9,09 | 4/0 |
| small 6-bit | 4131 | 10,12 | **9,34** | 4/0 |

6 Bit schlägt 4 Bit messbar (mehr Wörter, +7 % Kommas, bessere Satzgliederung) und
brauchte **keine** Strafe, wo 4-bit eine nötig hatte. Der Preis sind kurze Pässe
(303 s statt 576 s) und damit mehr Nähte.

---

## 8b. Was 8-bit small auf 32 GB verhindert (geprüft, verworfen)

Der Sockel ist das Problem: **26,4 GB allein für die Gewichte**. Dazu kommt ein
zweiter, unabhängiger K.-o.: **0,80× Echtzeit**, also langsamer als die Aufnahme.

Geprüfte Auswege:

| Ansatz | Wirkung auf 8-bit small | sonstiger Nutzen |
|---|---|---|
| Quantisierter KV-Cache | **keine** — betrifft nur die Steigung | **gering**, siehe unten |
| Neuere Quantisierer (AWQ-Prinzip) | gemischte Präzision senkt die Größe | ein **6/8-Bit-Mix** läge bei ~22–23 GB |
| QAT (z. B. Gemma-Ansatz) | scheidet aus — bräuchte Nachtraining durch Mistral | — |
| Aligner zeitlich trennen | ~2 GB — könnte knapp reichen | ~2 GB **für alle Modelle** |

MLX kennt nur den `affine`-Modus (`mxfp4`/`nf4` werden abgelehnt), hat aber
`QuantizedKVCache` — `mlx_voxtral` nutzt bisher den einfachen `KVCache`.
Auch vorhanden: `mx.set_wired_limit`.

### Der KV-Cache ist viel kleiner als die Steigung — Korrektur

Aus der Architektur gerechnet (`Schichten × 2 × KV-Köpfe × head_dim × Tokens`,
Audio kostet 12,6 Tokens/s):

| | KV-Cache (bf16) | gemessene Steigung insgesamt |
|---|---|---|
| small, 300 s | 0,62 GB | 4,05 GB |
| small, 1500 s | 3,10 GB | 20,3 GB |
| mini, 600 s | 0,93 GB | 6,2 GB |

Der KV-Cache macht nur **~15 %** der Steigung aus; der Rest entfällt auf den
Audio-Encoder, auf den wir über `mlx_voxtral` keinen Zugriff haben. Eine
KV-Quantisierung brächte daher nur **+8 % Pass-Länge** (small-6bit: 304 s → 328 s),
nicht die zunächst geschätzte Verdopplung. Der Aufwand lohnt nicht.

### Bringen längere Pässe überhaupt Qualität?

Gemessen: 600-s-Einzelpass 10,87 Kommas/100 W vs. 2×300 s 10,66 — praktisch
gleich. Die Literatur stützt das: Es gibt ein *Optimum* der Chunk-Länge, das vom
Training abhängt (Whisper 30 s, Distil-Whisper 15 s), nicht „je länger desto
besser". Das reale Problem sind die **Grenzen** (abgeschnittene Satzbezüge,
Sprecherwechsel an der Kante) — dagegen arbeiten pausen-ausgerichtete Schnitte
plus Overlap bereits.

**Fazit:** 8-bit small abhaken; KV-Quantisierung ebenfalls. Der einzige lohnende
Punkt bleibt die zeitliche Trennung von Voxtral und Aligner.

## 9. Offene Punkte

- **mini 8-bit als Standard** — identische Ausgabe, ~5× Tempo, 1816-s-Pässe statt
  1068 s. Liegt konvertiert unter `/tmp/mini8` (5,1 GB), noch nicht registriert.
- **small 8-bit** — 26,5 GB Gewichte; ob es auf 32 GB überhaupt läuft, ist offen.
  Quellgewichte liegen lokal, Konvertierung dauert ~25 s.
- **Sprach-Fixierung über mehrere Pässe** (siehe 3).
- **Voxtral-Mini-4B-Realtime-2602** — neuer, offen, Streaming mit rollendem
  KV-Cache (unbegrenzte Länge, konstanter Speicher). Eigenes Projekt.
- `voxtral-mini-2602` („Transcribe 2", native Timestamps) ist **API-only** und
  damit für vertrauliche Interviews ungeeignet.

---

## 10. Reproduzieren

Skripte liegen unter `/tmp` (flüchtig!):
`quantize6.py` (Konvertierung, beliebige Bit-Breite), `bitmatrix.py`
(Tempo/Worttreue), `decisive.py` (Satzzeichen je Strafe), `split_retry.py`
(Schleifenreparatur), `cli_check.py` (Formatprüfung).

Relevante Commits auf `feature/voxtral-engine`:
`3232ec8` (Decoding-Fixes) · `355be19` (Namen nur Deutsch) · `7031aed` (Loop-Erkennung) ·
`8e420a7` (Split statt Strafe) · `d6d1d5c` (Eskalation ab 1.01) ·
`0f9b977` (6-bit small) · `86b22cc` (Reserve 7 GB + konfigurierbar)
