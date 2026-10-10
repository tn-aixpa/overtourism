# Fazzon: domande aperte per gli esperti di dominio

Ipotesi usate nell'analisi (`analyze_new_data.py`) che vanno ancora validate.
Quando una domanda viene chiusa, aggiornare la risposta qui e il codice corrispondente.

## Navetta / bus

1. **Bus 2026, persone in discesa > persone in salita.** In agosto 2026 la
   discesa supera la salita (es. picco 8-19 ago: 455 vs 404/giorno).
   *Ipotesi di lavoro:* molti visitatori salgono a piedi e scendono in navetta.
   *Da validare:* oppure il conteggio in salita è incompleto (es. biglietti
   acquistati online e non registrati, bambini gratis, corse non contate)?
2. **Bus 2025: biglietti o persone?** `Dati parcheggio Fazzon.xlsx`
   ("N° persone bus", 2025) riporta valori molto più bassi del file
   `Riepilogo dati bus estate 2025` ("Ticket venduti"): es. 1/7 15 vs 68, 13/7
   3 vs 22. Il Greenbus 2025 faceva un giro più ampio (Fucine, Ossana): il file
   Riepilogo conta forse tutte le salite, e il nuovo solo chi va a Fazzon?
   *Attualmente:* l'analisi usa ancora "Ticket venduti" (Riepilogo).
3. **Righe di agosto 2025 nel file Riepilogo bus** (6/13/15/20/27 ago):
   confermato che si riferiscono a un'altra area e sono scartate.

## Parcheggio

4. **Coefficienti di occupazione.** Il foglio usa 3 persone/auto; secondo il
   gestore nel 2025 erano ~2,5 (moto incluse) e nel 2026 ~3,5 ("si sale solo
   con l'auto piena"). Lo studio NetMobility 2021 dava 2,83. Esistono conteggi
   diretti a supporto dei valori 2025/2026?
5. **Auto 2026 non conteggiate.** Ingressi dopo le 17:00 (gratuiti) e auto nei
   parcheggi privati (Berry Farm, Olga Casanova, Rifugio Fazzon) non compaiono
   nei dati. Ordine di grandezza?
6. **Mezza giornata 2026.** La colonna "N° auto 1/2 gg" (ingresso dopo le 12,
   10 €) parte solo dal 18/7: prima di quella data gli ingressi pomeridiani
   erano inclusi nella giornata intera o non registrati?
7. **Colonna "N° persone 2026" (P) del foglio:** formule incoerenti (15-17/7,
   23/7 e 13/8 omettono moto o mezza giornata; 27/8 ripete il totale del
   26/8). L'analisi ricalcola i totali; segnalarlo a chi gestisce il foglio.

10. **Tariffa parcheggio 2026 fuori dal periodo regolato** (prima del 4/7 e
    dopo il 31/8): assunta 5 € come nel 2025 (`ASSUMPTIONS.md`, A7). È corretto?

## Dati da richiedere

8. **Conteggi traffico PAT 2026** (stessa esportazione, almeno stazioni 110
   Pellizzano, 155 Pracorno, 156 Celledizzo, 170 Revò): servono per stimare la
   domanda 2026 indipendentemente dal regime di parcheggio.
9. Presenze Vodafone 2026: **non** sono nel datalake (verificato il 10/10/2026:
   `vodafone_attendences_new.csv` copre 2023-01-01 .. 2025-12-31); da richiedere
   al fornitore. Anche i conteggi traffico nel datalake
   (`dati_transiti/dati_transito_veicolare.csv`) sono l'export 2022-2025.
   (Meteo 2026: risolto, scaricato dall'open data Meteotrentino, stazione di Malè T0074.)
