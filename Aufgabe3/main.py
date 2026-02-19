"""
Main file fuer Aufgabe 3

:author: Tamaro Walter, Simon Schoenhoeffer
:group: Praktikum Gruppe 14

Was ist die Aufgabe:
- Es soll eine Kanten/Konturerkennung mithilfe eines U-nets 

Genauer Wortlaut:
Trainieren Sie ein U-Net wie in der Vorlesung vorgestellt, das eine Segmentierung der Bilder
durch Erzeugen einer Kontur-Maske generiert.  ̈Ubernehmen Sie die gegebene Aufteilung des
Datensatzes in Trainings-, Validierungs- und Testdaten. Werten Sie die Qualit ̈at des U-Net
sinnvoll anhand der im Datensatz vorhandenen Konturmasken aus.

Idee der Implementierung:
- U-net mithilfe von pytorch bauen
- Bilder aus BSDS500/images/train zu kleineren bildern (256*256) transformieren
- Trainingsloop schreiben und fehler berechnen.
- Evaluation schreiben
"""