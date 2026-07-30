# Diarisierung: Messbasis auf VoxConverse (29.-30.07.2026)

Anlass: ein Lauf mit automatischer Sprecherzaehlung erfand auf einer
Zwei-Personen-Aufnahme einen dritten Sprecher, und ein Wort mitten im Satz
landete darunter. Die Frage war, ob sich das im Programm reparieren laesst.

## Reproduziert der Aufbau die publizierten Zahlen?

VoxConverse v0.3 (frei, CC-BY-4.0; Audio robots.ox.ac.uk, RTTM
github.com/joonson/voxconverse). Gescort mit `pyannote.metrics` unter pyannotes
Bedingungen: `collar=0`, `skip_overlap=False`, aggregiert ueber die Summen aller
Dateien.

    Split   Dateien   Sprechzeit   unser DER   pyannote (community-1)
    dev         216       20.3 h      7.18 %   --
    test        232       40.2 h     11.14 %   11.2 %

Die Modellkarte nennt keinen Split; es ist **test**. Der Aufbau
(`pyannote/`, community-1 mit VBx+PLDA, MPS) ist damit validiert -- eigenen
Diarisierungsmessungen kann man ab hier glauben. Laufzeit 14 s/Datei auf dev,
~22 s auf test.

## Der Befund, der die Richtung vorgibt

    Sprecherzahl gegen RTTM-Wahrheit
    dev :  exakt 138 | zu viele 16 | zu wenige 62
    test:  exakt 109 | zu viele 30 | zu wenige 93

**Zu wenige Sprecher ist drei- bis viermal haeufiger als zu viele.** Alles, was
Sprecher zusammenfasst, drueckt also in die Richtung, in die die automatische
Zaehlung schon zu weit geht.

## Was daran geprueft und verworfen wurde

- **`clustering.Fb` von 0.8 auf 1.5.** Heilte die Ausloeser-Datei (3 -> 2
  Cluster) und liess V1/V3/V5 bitgleich, zog aber auf einer 25-minuetigen
  Fragerunde 14 erkannte Sprecher auf 12 zusammen. `Fb` ist der
  Sprecher-Regularisierer der VB-Schleife ("high values result in the VB
  inference dropping more speakers", Diez et al.); 0.07/0.8/0.6 sind pyannotes
  getunte Optima innerhalb der Suchraeume im Quelltext, nirgends dokumentiert.
  Falsche Fehlerrichtung, verworfen.
- **`clustering.threshold` hoeher.** 0.70 aenderte nichts, 0.80 (Obergrenze des
  Suchraums) liess den Geist stehen und benannte ihn nur um. Bei VBx
  initialisiert die Schwelle nur die Agglomeration.
- **Umlabeln des Geisters im Programm.** Gegen den 2-Sprecher-Lauf als Referenz
  traf ein Naechster-Nachbar-Fold der 56 Segmente 46 richtig und 10 falsch
  (82 %). Die 10 waeren stille Fehlzuordnungen an echte Sprecher anstelle eines
  sichtbaren Phantom-Sprechers -- die schlechtere Fehlerart. Ausserdem enthielt
  das Label Material *beider* Sprecher (24 Segmente des einen, 32 des anderen),
  es gibt also kein Verschmelzungsziel.
- **`AgglomerativeClustering` mit `min_cluster_size`.** Der Parameter zaehlt
  Einbettungen, nicht Laenge; der Geist verteilt sich ueber viele Chunks. Selbst
  bei 15 (Maximum 20) blieb die Datei bei 3 Clustern, und ohne VBx faellt die
  getunte 4.x-Pipeline weg.
- **Voxtrals Audio-Encoder als zweiter Identifikator.** Naechster-Centroid auf
  109 sicheren Segmenten >= 3 s: `audio_tower` 58.2 % +-11.1, Projektor 82.3 %
  +-7.8 bei zwei Klassen. cos(Sprecher A, Sprecher B) = 0.993 gegen
  cos(Segment, eigener Centroid) = 0.93/0.96 -- die Sprecherachse ist im Inhalt
  begraben, wie es fuer einen ASR-Encoder zu erwarten ist.

## Was blieb

Zwei Dinge. Der eigentliche Fix liegt nicht in der Diarisierung, sondern im
Cue-Bau: ein einzelnes gedehntes Wort durfte einen eigenen Cue behalten und
bekam seinen Sprecher von dem, was die Diarisierung darunter legte
(`voxtral_engine._segments_from_words`). Ueber alle sieben Aufnahmen der Serie
neu gerechnet: Wortzahlen unveraendert, eine 26-Minuten-Datei bitgleich, genau
zwei Woerter wechselten den Sprecher -- beides Reparaturen halbierter Saetze.

Dazu eine Meldung, wenn ein Label nie das Wort haelt. Gegen Ground Truth auf
Label-Ebene ("deckt ein laengeres Label denselben Referenzsprecher schon ab?"):

    Anteil < 0.05, Beitrag < 2000 ms   dev 71 %   test 67 %
    Anteil < 0.02, Beitrag < 2000 ms   dev 83 %   test 69 %   <- ausgeliefert
    Anteil < 0.01, Beitrag < 2000 ms   dev 100 %  test 67 %

0.02 ist der einzige Wert, der auf beiden Saetzen besser ist als der
urspruengliche; 0.01 ist Ueberanpassung an 32 Positive. Trefferquote 9 von 77
ueberzaehligen Labels auf test -- Schweigen bedeutet nichts. Eine dritte Achse
(Mindestzahl kurzer Beitraege) faellt aus: die Ausloeser-Datei hatte 56, auf
VoxConverse haben ueberzaehlige Labels typisch wenige, und eine Mindestzahl
brach die Treffer von 5 auf 1.

## Offen

`min_speakers` / `max_speakers` an die Oberflaeche bringen -- pyannote nimmt
beide, noScribe bietet nur "auto" oder eine exakte Zahl. Nach den Zahlen oben
ist die **Untergrenze** die wertvollere Haelfte. Caveat fuer den PR: eine feste
oder begrenzte Zahl schaltet `VBxClustering` bei Abweichung von der automatisch
gefundenen Zahl auf KMeans um, ist also ein Algorithmuswechsel und kein reiner
Gewinn.
