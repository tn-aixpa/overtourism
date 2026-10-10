"""Analisi dei nuovi dati AIxPA 2025 per Fazzon / Alta Val di Sole.

Script semplice con marcatori di cella ``# %%``, eseguibile sia da riga di
comando (``python analyze_new_data.py``) sia cella-per-cella in un REPL /
finestra interattiva stile Jupyter (VS Code, PyCharm, `jupytext`, ...).

Cosa fa, in ordine:

1. Costruisce la serie giornaliera target di Fazzon: auto (biglietti
   parcheggio, dato misurato) e biglietti navetta bus, per la stagione 2025.
2. Estrae le serie predittive giornaliere per la Val di Sole dai nuovi dati:
   presenze telefoniche Vodafone (granularita' POI/comune/apt/provincia),
   meteo (Meteotrentino, stazione di Male'), contatori bici/pedoni sulla
   ciclabile, utilizzo Guest Card e passaggi di autoveicoli rilevati dalle
   stazioni di conteggio del traffico della Provincia (Servizio Gestione
   Strade) in Val di Sole / Val di Non.
3. Calcola le correlazioni grezze e destagionalizzate (entro-mese) con il
   target, e una matrice di correlazione completa tra tutte le variabili.
4. Quantifica l'effetto di saturazione del canale auto (appiattimento della
   pendenza ad alta domanda) confrontandolo con il canale bici/pedoni.
5. Esegue l'"esperimento naturale" della navetta: confronta le auto per un
   dato livello di domanda/meteo tra il periodo in cui la navetta era
   realmente attiva (28/06/2025 -> 20/07/2025) e il resto della stagione.
6. Produce un grafico che rende esplicita l'assenza di dati: i dati sui
   biglietti navetta non coprono affatto i giorni di domanda più alta della
   stagione, quindi la domanda sullo spostamento modale non è oggi
   rispondibile con questo dataset.
7. Usa i conteggi stradali (auto *misurate* sulla SS42 a Pellizzano) come
   segnale di domanda alternativo alle presenze Vodafone: ripete l'analisi di
   saturazione e l'esperimento navetta, e confronta il periodo navetta 2025
   con lo stesso periodo del 2023/2024 (differenza-nelle-differenze rispetto a
   stazioni di controllo in valli laterali).
8. Confronta la stagione 2026 (parcheggio a pagamento e a numero chiuso,
   semaforo, navetta "Lakebus") con la stagione 2025 (parcheggio a 5 euro fino a
   esaurimento posti, navetta pilota Greenbus solo 28/6-20/7), a parita' di
   giorno della settimana.

Le ipotesi ancora da validare con gli esperti di dominio sono elencate in
``OPEN_QUESTIONS.md`` (stessa cartella).

Correzioni di qualita' dei dati incorporate qui (vedi memoria di progetto
``fazzon_bus_data_quality.md``): il file dei biglietti bus contiene cinque
righe sparse in agosto relative a un'altra area, non Fazzon, e la navetta
non ha funzionato al di fuori dell'intervallo 28/06/2025 -> 20/07/2025.
Entrambi i fatti sono applicati nella costruzione della serie target.

Nota sulla scelta delle variabili: dove possibile si usa il numero di auto
(biglietti di parcheggio venduti, dato *misurato*) invece dei "visitatori in
auto" (biglietti x 2.83 persone/auto, dato *inferito*) — vedi
``PERSONS_PER_CAR`` più sotto, calcolato comunque e disponibile nel pannello
per chi necessita della stima in persone.

Output (scritti in ``OUTPUT_DIR``): ``panel_daily.csv``, ``correlations.csv``,
``correlation_matrix.csv``, ``saturation_slopes.csv``, ``shuttle_road_did.csv``
e i grafici PNG numerati, con etichette in italiano.
"""

# %% Import e configurazione
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

FAZZON_DIR = Path(__file__).resolve().parent
DATA_DIR = FAZZON_DIR / "data"
NEW_DATA_DIR = DATA_DIR / "new"
OUTPUT_DIR = NEW_DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

PERSONS_PER_CAR = 2.83  # NetMobility 2021 field study, vedi fazzon_presence_stats.py

# La navetta ha funzionato solo dal 28/06/2025 al 20/07/2025. Le righe sparse
# successive nel file sorgente (6/13/15/20/27 agosto) si riferiscono a
# un'altra area e vanno scartate, non trattate come "nessun biglietto
# venduto quel giorno".
SHUTTLE_LAST_DAY = pd.Timestamp("2025-07-20")

# Foglio 2025 + 2026 affiancati (stessa riga = stesso giorno della settimana,
# 2026 = 2025 + 364 giorni). Vedi OPEN_QUESTIONS.md per le colonne dubbie.
PARKING_XLSX = DATA_DIR / "Dati parcheggio Fazzon.xlsx"
# Regime 2026: parcheggio a pagamento (20 euro/giorno, 10 euro dopo le 12) con
# semaforo e prenotazione dal 4/7 al 31/8, navetta Lakebus dal 4/7 al 30/8.
REGIME_2026_START = pd.Timestamp("2026-07-04")
REGIME_2026_SHUTTLE_END = pd.Timestamp("2026-08-30")
REGIME_2026_PARKING_END = pd.Timestamp("2026-08-31")
# Giorni 2026 esclusi dal confronto: evento "5days" (>2000 persone, prati
# aggiuntivi adibiti a parcheggio, auto non conteggiate) e 21/8 (bus non attivo).
EXCLUDED_DAYS_2026 = pd.to_datetime(["2026-07-02", "2026-07-03", "2026-08-21"])
# Persone per auto secondo il gestore (OPEN_QUESTIONS.md, punto 4): ~2,5 senza
# regime (2025, e 2026 fuori dal periodo regolato); ~3,5 con il regime 2026,
# perche' con parcheggio a 20 euro "si sale solo con l'auto piena".
OCCUPANCY_NO_REGIME = 2.5
OCCUPANCY_REGIME_2026 = 3.5

# Meteo misurato dalla stazione Meteotrentino di Male' (T0074, 720 m, stazione
# di riferimento dell'area "Sole, Peio e Rabbi"), scaricato dal servizio open
# data e salvato in cache in data/meteo/. A differenza di meteo_osservato.csv
# (valori arrotondati e descrizione testuale, probabilmente il bollettino per
# localita') sono misure di stazione, disponibili fino a ieri: coprono anche il 2026.
METEO_DIR = DATA_DIR / "meteo"
METEO_STATION = "T0074"
METEO_YEARS = range(2023, 2027)
RAINY_DAY_MM = 1.0  # soglia climatologica standard di "giorno piovoso"
METEO_SERVICE = "https://dati.meteotrentino.it/service.asmx/datiStoriciCaratteristiciSerie{measure}UnArea?codice={code}&anno={year}"

# Codici delle localita' della Val di Sole in vodafone_aree.csv
LOC_NAMES = {
    "LAGO_CAPR": "poi_caprioli",
    "8": "comune_ossana_pellizzano",
    "3": "comune_vermiglio",
    "104": "apt_valdisole",
    "22": "provincia_trentino",
}
VODAFONE_PROFILES = ["TOURIST", "VISITOR"]  # esclude INHABITANT / COMMUTER

BIKE_SENSORS_NEAR_FAZZON = ["TFXV3R06", "GLWB8Q97"]  # Salet->Pellizzano, Fucine di Ossana
BIKE_SENSOR_SHORT_NAMES = {  # da contamezzi_TM_info.csv
    "LLWW78M8": "Laghetti S. Leonardo", "GLWB8Q97": "Fucine di Ossana", "TFXV3R06": "Salèt (Pellizzano)",
    "RXSLG442": "Daolasa", "FFXV2R01": "Molini (Caldes)",
}

# Conteggi del traffico PAT (Servizio Gestione Strade), orari per classe di
# veicolo / velocita' / direzione, 2022-2025. ``query_passaggi_auto_*.xlsx`` e'
# solo un riepilogo mensile dello stesso dato (totali identici) e non viene letto.
VEHICLES_DIR = DATA_DIR / "vehicles"
ROAD_COUNTS_CSV = VEHICLES_DIR / "esportazione-002.csv"
ROAD_STATIONS = {
    "108": "croviana", "110": "pellizzano", "111": "tonale", "113": "dermulo",
    "155": "pracorno", "156": "celledizzo", "170": "revo", "175": "dimaro",
}
ROAD_CAR_CLASSES = [2, 3]  # autovetture e monovolumi, anche con rimorchio
# Stazione 110 (Pellizzano): sulla SS42 a valle dell'abitato, verso Mezzana,
# quindi a valle dell'innesto della strada per Fazzon. Sulla SS42 le
# chilometriche crescono verso valle (Ossana -> Pellizzano -> Mezzana), quindi
# direzione 2 (chilometriche decrescenti) = verso monte. Coerente con i
# profili orari (picco del mattino in direzione 2 ad agosto).
# Conseguenza: il sensore vede le auto dirette a Fazzon che salgono dalla
# bassa valle -- comprese quelle respinte a parcheggio pieno -- ma non quelle
# che arrivano da Ossana/Vermiglio/Tonale/Peio o che partono da Pellizzano.
ROAD_UP_VALLEY_DIR = 2
ROAD_MORNING_HOURS = range(7, 12)  # 7:00-11:59, la finestra di arrivo dei visitatori giornalieri
ROAD_MIN_DAILY_CARS = 50  # sotto questa soglia per direzione il sensore e' considerato spento
# Stazioni di controllo per il confronto anno-su-anno: valli laterali non
# attraversate dal traffico verso Fazzon (Val di Rabbi, Val di Non) e Val di
# Peio. Celledizzo non e' un controllo del tutto pulito: gli ospiti della Val
# di Peio diretti a Fazzon ci passano (scendendo verso Fucine/Ossana), ma
# sono al massimo poche decine di auto su ~3.800/giorno (~1%).
ROAD_CONTROL_STATIONS = ["pracorno", "celledizzo", "revo"]

# Posizione delle stazioni (fornita da PAT in UTM 32N / ETRS89, EPSG:25832),
# convertita in lat/lon WGS84 per non dipendere da pyproj.
ROAD_STATION_COORDS = {  # stazione: (lat, lon, descrizione)
    "croviana": (46.34606, 10.90902, "SS42 tra Croviana e Male'"),
    "pellizzano": (46.31227, 10.77262, "SS42 a valle di Pellizzano, verso Mezzana"),
    "tonale": (46.26344, 10.61173, "SS42 versante trentino, ~2 km a est del passo"),
    "dermulo": (46.33014, 11.06402, "Val di Non, Dermulo"),
    "pracorno": (46.36229, 10.90198, "Val di Rabbi, ~2 km a nord di Male'"),
    "celledizzo": (46.33704, 10.70649, "Val di Peio, tra Cusiano e Cogolo"),
    "revo": (46.39307, 11.04813, "Val di Non, Revo'"),
    "dimaro": (46.32093, 10.87225, "SS239 sopra Dimaro, verso Folgarida/Campiglio"),
}
# Posizione indicativa del parcheggio del Lago dei Caprioli (non fornita nei
# dati: valore approssimato, solo per la mappa).
FAZZON_COORDS = (46.2930, 10.7425)

VALDISOLE_COMUNI_RE = (
    r"(?i)OSSANA|PELLIZZANO|VERMIGLIO|DIMARO|MALE|RABBI|PEIO|COMMEZZADURA|CALDES|MEZZANA|FOLGARIDA"
)

# Etichette in italiano usate nei grafici (non nel codice, per leggibilita' del sorgente)
IT_LABELS = {
    "parking_tickets": "Auto/giorno (biglietti parcheggio)",
    "car_visitors": "Visitatori in auto (stimati)",
    "bus_tickets": "Biglietti navetta/giorno",
    "vf_pres_apt_valdisole": "Presenze Vodafone - Valli di Sole (APT)",
    "vf_pres_provincia_trentino": "Presenze Vodafone - Provincia di Trento",
    "vf_pres_comune_vermiglio": "Presenze Vodafone - Comune di Vermiglio",
    "vf_pres_comune_ossana_pellizzano": "Presenze Vodafone - Ossana+Pellizzano",
    "vf_pres_poi_caprioli": "Presenze Vodafone - Lago dei Caprioli (POI)",
    "bici_valley_total": "Bici/pedoni - intera Val di Sole",
    "bici_near_fazzon": "Bici/pedoni - vicino a Fazzon",
    "tgc_accessi_tonale": "Accessi Guest Card - Passo Tonale",
    "strada_pellizzano_salita_mattina": "Auto SS42 Pellizzano, verso monte 7-12",
    "strada_pellizzano_totale": "Auto SS42 Pellizzano, totale giornaliero",
    "strada_pracorno_totale": "Auto Pracorno (Val di Rabbi), totale",
    "strada_celledizzo_totale": "Auto Celledizzo (Val di Peio), totale",
    "strada_tonale_totale": "Auto Passo del Tonale, totale",
    "tmax": "Temperatura massima (C, stazione Male')",
    "tmin": "Temperatura minima (C, stazione Male')",
    "rain_mm": "Pioggia (mm, stazione Male')",
    "tmax_bollettino": "Temperatura massima (C, bollettino)",
    "rain_mm_bollettino": "Pioggia (mm, bollettino)",
    "is_rainy": "Giorno piovoso",
    "is_weekend": "Fine settimana",
}

# --- palette (palette di default dello skill dataviz del progetto) -------
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_VIOLET = "#4a3aa7"
C_RED = "#e34948"  # categorico slot 8, usato anche come polo del diverging
C_RED_STATUS = "#d03b3b"
C_GOOD_STATUS = "#0ca30c"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_NEUTRAL_MID = "#f0efec"
C_SURFACE_ORANGE = "#fbe1d6"  # riempimento chiaro per marcatori arancio secondari
SURFACE = "#fcfcfb"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_diverging", [C_BLUE, C_NEUTRAL_MID, C_RED]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": C_AXIS,
    "axes.labelcolor": C_INK2,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "text.color": C_INK,
    "xtick.color": C_INK2,
    "ytick.color": C_INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "figure.dpi": 130,
})


# %% 1. Serie target: auto e biglietti navetta a Fazzon
def build_target() -> pd.DataFrame:
    park = pd.read_csv(DATA_DIR / "Dati parcheggio Fazzon 2025 per IA_Parcheggio.csv")
    park["date"] = pd.to_datetime(park["DATA"])
    park = park.rename(columns={"N° auto": "parking_tickets"})[["date", "parking_tickets"]]
    park["car_visitors"] = park["parking_tickets"] * PERSONS_PER_CAR

    bus_raw = pd.read_csv(DATA_DIR / "Riepilogo dati bus estate 2025_Foglio1.csv")
    bus = bus_raw.iloc[:, [1, 2]].copy()
    bus.columns = ["date", "bus_tickets_raw"]
    bus = bus.dropna(subset=["date"])
    bus = bus[bus["date"].str.match(r"^\d{4}-\d{2}-\d{2}")]
    bus["date"] = pd.to_datetime(bus["date"])
    bus["bus_tickets_raw"] = pd.to_numeric(bus["bus_tickets_raw"], errors="coerce")

    target = park.merge(bus, on="date", how="outer").sort_values("date").reset_index(drop=True)
    target["shuttle_running"] = target["date"] <= SHUTTLE_LAST_DAY
    # Scarta le righe sparse dopo il 20/07/2025: confermato che si riferiscono
    # a un'altra area, non Fazzon (memoria di progetto: fazzon_bus_data_quality.md).
    target["bus_tickets"] = target["bus_tickets_raw"].where(target["shuttle_running"])
    target["dow"] = target["date"].dt.day_name()
    target["month"] = target["date"].dt.month
    target["is_weekend"] = (target["date"].dt.dayofweek >= 5).astype(int)
    return target.drop(columns=["bus_tickets_raw"])


# %% 2. Serie predittive
def _load_vodafone_presences() -> pd.DataFrame:
    """Presenze Vodafone giornaliere a granularita' POI/comune/apt/provincia.

    Somma la riga nazionale italiana (``userCountry == "ITA"``) con il totale
    estero pre-aggregato (``*_STR.csv``) invece di ogni singola nazionalita',
    per aggirare la soglia di privacy (i valori sotto 15 sono riportati come
    0 mascherato a livello di singola nazionalita').
    """
    loc_filter = list(LOC_NAMES.keys())

    att = (
        pl.scan_csv(NEW_DATA_DIR / "vodafone_attendences.csv", schema_overrides={"locId": pl.Utf8})
        .filter(pl.col("locId").is_in(loc_filter))
        .select(["date", "locId", "userCountry", "userProfile", "value"])
        .collect()
        .to_pandas()
    )
    att_str = (
        pl.scan_csv(NEW_DATA_DIR / "vodafone_attendences_STR.csv", schema_overrides={"locId": pl.Utf8})
        .filter(pl.col("locId").is_in(loc_filter))
        .select(["date", "locId", "userProfile", "value"])
        .collect()
        .to_pandas()
    )

    ita = (
        att[(att["userCountry"] == "ITA") & (att["userProfile"].isin(VODAFONE_PROFILES))]
        .groupby(["date", "locId"])["value"].sum().rename("value_ita")
    )
    strv = (
        att_str[att_str["userProfile"].isin(VODAFONE_PROFILES)]
        .groupby(["date", "locId"])["value"].sum().rename("value_str")
    )
    pres = pd.concat([ita, strv], axis=1).fillna(0)
    pres["presences_tourist_visitor"] = pres["value_ita"] + pres["value_str"]
    pres = pres.reset_index()

    wide = pres.pivot(index="date", columns="locId", values="presences_tourist_visitor")
    wide.columns = [f"vf_pres_{LOC_NAMES.get(c, c)}" for c in wide.columns]
    wide.index = pd.to_datetime(wide.index)
    return wide.reset_index().rename(columns={"index": "date"})


def _load_meteo() -> pd.DataFrame:
    meteo = pd.read_csv(NEW_DATA_DIR / "meteo_osservato.csv")
    meteo["date"] = pd.to_datetime(meteo["data"]).dt.normalize()
    male = meteo[meteo["comune"] == "MALE'"][["date", "meteo", "tmin", "tmax", "pioggiamm"]].copy()
    return male.rename(columns={
        "meteo": "weather_desc", "tmin": "tmin_bollettino", "tmax": "tmax_bollettino", "pioggiamm": "rain_mm_bollettino",
    })


def fetch_meteo_station(refresh: bool = False) -> pd.DataFrame:
    """Temperatura min/max e pioggia giornaliere della stazione ``METEO_STATION``.

    Scarica un file JSON per anno e misura in ``METEO_DIR`` solo se manca (o
    con ``refresh=True``; l'anno in corso va riscaricato per avere i giorni
    nuovi). I codici di qualita' non sono documentati dal servizio: 145 dal
    2024, 1 nel 2023 (e 81 su 4 giorni del 2023); sono riportati cosi' come
    arrivano in ``meteo_quality``.
    """
    METEO_DIR.mkdir(exist_ok=True)
    frames = {}
    for measure in ("Temperature", "Precipitazioni"):
        parts = []
        for year in METEO_YEARS:
            path = METEO_DIR / f"{METEO_STATION}_{measure}_{year}.json"
            if refresh or not path.exists():
                url = METEO_SERVICE.format(measure=measure, code=METEO_STATION, year=year)
                with urllib.request.urlopen(url, timeout=60) as resp:
                    path.write_bytes(resp.read())
            parts.append(pd.DataFrame(json.loads(path.read_text())["dati"]))
        frames[measure] = pd.concat(parts, ignore_index=True)

    temp = frames["Temperature"].rename(columns={"Min": "tmin", "Max": "tmax", "QualityMax": "meteo_quality"})
    rain = frames["Precipitazioni"].rename(columns={"Valore": "rain_mm"})
    out = temp.merge(rain[["giorno", "rain_mm"]], on="giorno", how="outer")
    out["date"] = pd.to_datetime(out["giorno"])  # ISO yyyy-mm-dd (le piogge arrivano ordinate per giorno, poi mese)
    return out[["date", "tmin", "tmax", "rain_mm", "meteo_quality"]].sort_values("date").reset_index(drop=True)


def _load_bici() -> pd.DataFrame:
    bici = pd.read_csv(NEW_DATA_DIR / "contabici_VALDISOLE.csv")
    bici["date"] = pd.to_datetime(bici["data"]).dt.normalize()
    bici["bikes_peds"] = bici[["cl8", "cl10"]].sum(axis=1)
    near = (
        bici[bici["sensore"].isin(BIKE_SENSORS_NEAR_FAZZON)]
        .groupby("date")["bikes_peds"].sum().rename("bici_near_fazzon")
    )
    valley = bici.groupby("date")["bikes_peds"].sum().rename("bici_valley_total")
    return pd.concat([near, valley], axis=1).reset_index()


def _load_tgc_accessi_tonale() -> pd.DataFrame:
    accessi = (
        pl.scan_csv(NEW_DATA_DIR / "TGC_Accessi.csv", separator=";", encoding="utf8-lossy")
        .filter(pl.col("Luogo").str.contains(VALDISOLE_COMUNI_RE))
        .select(["Data Marcata", "NomeServizio", "Entranti"])
        .collect()
        .to_pandas()
    )
    accessi["date"] = pd.to_datetime(
        accessi["Data Marcata"], format="%d/%m/%Y %H:%M", errors="coerce"
    ).dt.normalize()
    tonale = accessi["NomeServizio"].str.contains("Tonale", case=False, na=False)
    return accessi[tonale].groupby("date")["Entranti"].sum().rename("tgc_accessi_tonale").reset_index()


def load_road_counts_hourly() -> pd.DataFrame:
    """Auto (classi 2+3) per stazione / giorno / direzione / ora.

    Il CSV sorgente e' ~2.3 GB (una riga per classe x fascia di velocita'):
    polars lo aggrega in streaming a ~0.5 M righe.
    """
    hourly = (
        pl.scan_csv(ROAD_COUNTS_CSV, schema_overrides={"PUNTO": pl.Utf8, "ORA": pl.Utf8})
        .filter(pl.col("CLASSE").is_in(ROAD_CAR_CLASSES))
        .with_columns(
            pl.col("DATA").str.strptime(pl.Date, "%d/%m/%Y").alias("date"),
            pl.col("ORA").cast(pl.Int32).alias("hour"),
            pl.col("PUNTO").replace_strict(ROAD_STATIONS).alias("station"),
        )
        .group_by(["station", "date", "DIREZIONE", "hour"])
        .agg(pl.col("N_VEICOLI").sum().alias("cars"))
        .rename({"DIREZIONE": "direction"})
        .collect()
        .to_pandas()
    )
    hourly["date"] = pd.to_datetime(hourly["date"])
    return hourly


def road_daily_validity(hourly: pd.DataFrame) -> pd.DataFrame:
    """Una riga per stazione / giorno / direzione, con flag ``valid``.

    Il dato e' valido se sono presenti almeno 23 ore e almeno
    ``ROAD_MIN_DAILY_CARS`` auto: diversi sensori restano "accesi" ma
    riportano zero per mesi (Dimaro da meta' 2024, Tonale nel 2022, la
    direzione 2 di Croviana quasi sempre), e quegli zeri vanno trattati come
    dato mancante, non come strada vuota.
    """
    daily = (
        hourly.groupby(["station", "date", "direction"])
        .agg(cars=("cars", "sum"), hours=("hour", "nunique"))
        .reset_index()
    )
    daily["valid"] = (daily["hours"] >= 23) & (daily["cars"] >= ROAD_MIN_DAILY_CARS)
    return daily


def road_daily_series(hourly: pd.DataFrame) -> pd.DataFrame:
    """Serie giornaliere di auto usate come predittori (NaN dove il sensore
    non e' affidabile)."""
    daily = road_daily_validity(hourly)
    daily["cars"] = daily["cars"].where(daily["valid"])
    by_dir = daily.pivot_table(index="date", columns=["station", "direction"], values="cars", dropna=False)

    up = hourly[(hourly["station"] == "pellizzano") & (hourly["direction"] == ROAD_UP_VALLEY_DIR)
                & hourly["hour"].isin(ROAD_MORNING_HOURS)]
    morning_up = up.groupby("date")["cars"].sum().where(by_dir[("pellizzano", ROAD_UP_VALLEY_DIR)].notna())

    def both_dirs(station: str) -> pd.Series:
        # sum(min_count=2): NaN se una delle due direzioni manca
        return by_dir[station].sum(axis=1, min_count=2)

    out = pd.DataFrame({
        "strada_pellizzano_salita_mattina": morning_up,
        "strada_pellizzano_totale": both_dirs("pellizzano"),
        "strada_pracorno_totale": both_dirs("pracorno"),
        "strada_celledizzo_totale": both_dirs("celledizzo"),
        "strada_tonale_totale": both_dirs("tonale"),
        "strada_revo_totale": both_dirs("revo"),
    })
    out.index.name = "date"
    return out.reset_index()


def build_predictors(road_hourly: pd.DataFrame) -> pd.DataFrame:
    vodafone_daily = _load_vodafone_presences()
    meteo_daily = fetch_meteo_station().merge(_load_meteo(), on="date", how="left")
    bici_daily = _load_bici()
    tgc_daily = _load_tgc_accessi_tonale()
    road_daily = road_daily_series(road_hourly).drop(columns=["strada_revo_totale"])

    predictors = vodafone_daily
    for df in (meteo_daily, bici_daily, tgc_daily, road_daily):
        predictors = predictors.merge(df, on="date", how="left")
    return predictors


# %% 3. Assemblaggio del pannello giornaliero
def build_panel(road_hourly: pd.DataFrame) -> pd.DataFrame:
    target = build_target()
    predictors = build_predictors(road_hourly)
    panel = target.merge(predictors, on="date", how="left")
    panel["is_rainy"] = (panel["rain_mm"] >= RAINY_DAY_MM).astype(int)
    return panel


# %% 4. Tabella di correlazione (grezza + entro-mese) e matrice di correlazione
TARGET_COL = "parking_tickets"  # dato misurato; car_visitors ne è un multiplo costante


def correlation_table(panel: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        c for c in panel.columns
        if c.startswith(("vf_pres_", "bici_", "tgc_", "strada_"))
        or c in ("tmax", "tmin", "rain_mm", "is_rainy", "is_weekend", "tmax_bollettino", "rain_mm_bollettino")
    ]

    def within_month(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return s - s.groupby(panel["month"]).transform("mean")

    y_raw = panel[TARGET_COL].astype(float)
    y_resid = within_month(panel[TARGET_COL])

    rows = []
    for col in candidate_cols:
        x_raw = panel[col].astype(float)
        mask = x_raw.notna() & y_raw.notna()
        if mask.sum() < 10:
            continue
        r_raw, p_raw = stats.pearsonr(x_raw[mask], y_raw[mask])

        x_resid = within_month(panel[col])
        r_wm, p_wm = stats.pearsonr(x_resid[mask], y_resid[mask])

        rows.append({
            "predictor": col, "n": mask.sum(),
            "raw_r": r_raw, "raw_r2": r_raw ** 2, "raw_p": p_raw,
            "within_month_r": r_wm, "within_month_r2": r_wm ** 2, "within_month_p": p_wm,
        })
    return pd.DataFrame(rows).sort_values("within_month_r2", ascending=False).reset_index(drop=True)


CORRELATION_MATRIX_COLS = [
    "parking_tickets", "bus_tickets",
    "vf_pres_apt_valdisole", "vf_pres_provincia_trentino", "vf_pres_comune_vermiglio",
    "vf_pres_comune_ossana_pellizzano", "vf_pres_poi_caprioli",
    "bici_valley_total", "bici_near_fazzon",
    "strada_pellizzano_salita_mattina", "strada_pellizzano_totale", "strada_pracorno_totale",
    "tgc_accessi_tonale", "tmax", "tmin", "rain_mm", "is_weekend",
]


def correlation_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Matrice di correlazione di Pearson a coppie complete tra tutte le
    variabili di interesse (utile per relazioni fuori dal target, ad es.
    is_weekend vs bici_*, non catturate dalla tabella di correlazione sopra).
    """
    return panel[CORRELATION_MATRIX_COLS].corr(method="pearson")


def correlation_road_vs_presences(panel: pd.DataFrame) -> float:
    """Correlazione entro-mese tra le due misure di domanda (auto SS42 vs
    presenze Vodafone in valle): quanto si sovrappongono."""
    a = panel["strada_pellizzano_salita_mattina"] - panel.groupby("month")["strada_pellizzano_salita_mattina"].transform("mean")
    b = panel["vf_pres_apt_valdisole"] - panel.groupby("month")["vf_pres_apt_valdisole"].transform("mean")
    return a.corr(b)


# %% 5. Analisi di saturazione: canale auto vs canale bici/pedoni
def _find_optimal_breakpoint(x: pd.Series, y: pd.Series, min_frac: float = 0.2) -> dict:
    """Cerca (per griglia) il punto di rottura che minimizza la somma degli
    errori quadratici di due rette OLS indipendenti (una per lato).

    Non e' una regressione segmentata continua (le due rette non sono
    vincolate a incontrarsi nel breakpoint): stesso impianto "due rette
    indipendenti" usato per la soglia mediana, solo con la soglia scelta per
    minimizzare l'errore totale invece che fissata a priori.

    ``min_frac`` impone almeno questa frazione di punti per lato, per evitare
    rotture agli estremi con rette stimate su pochissimi punti.
    """
    x = x.to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    min_n = max(5, int(np.ceil(min_frac * n)))

    best = None
    for c in np.unique(xs):
        lo_mask, hi_mask = xs <= c, xs > c
        if lo_mask.sum() < min_n or hi_mask.sum() < min_n:
            continue
        fit_lo = stats.linregress(xs[lo_mask], ys[lo_mask])
        fit_hi = stats.linregress(xs[hi_mask], ys[hi_mask])
        sse = (
            np.sum((ys[lo_mask] - (fit_lo.intercept + fit_lo.slope * xs[lo_mask])) ** 2)
            + np.sum((ys[hi_mask] - (fit_hi.intercept + fit_hi.slope * xs[hi_mask])) ** 2)
        )
        if best is None or sse < best["sse"]:
            best = {
                "breakpoint": c, "sse": sse, "fit_lo": fit_lo, "fit_hi": fit_hi,
                "n_lo": int(lo_mask.sum()), "n_hi": int(hi_mask.sum()),
            }

    sst = np.sum((ys - ys.mean()) ** 2)
    single_fit = stats.linregress(xs, ys)
    sse_single = np.sum((ys - (single_fit.intercept + single_fit.slope * xs)) ** 2)
    best["r2_single_line"] = 1 - sse_single / sst
    best["r2_two_segments"] = 1 - best["sse"] / sst
    return best


def saturation_slopes(
    panel: pd.DataFrame, demand_col: str = "vf_pres_apt_valdisole", mode: str = "median", min_frac: float = 0.2,
) -> pd.DataFrame:
    """Confronta la pendenza a domanda bassa/alta per auto e traffico bici/pedoni.

    ``demand_col`` deve essere un segnale di domanda esterno alla capacita'
    fisica di Fazzon stessa (default: presenze Vodafone nell'intera valle),
    perché il confronto non sia circolare.

    ``mode="median"``: soglia fissa alla mediana della domanda (stesso numero
    di giorni per lato, non ottimizzata).
    ``mode="optimal"``: soglia scelta per canale, per griglia, minimizzando
    l'errore quadratico totale delle due rette (vedi ``_find_optimal_breakpoint``).
    Con n=82 e una soglia scelta sui dati stessi, il punto di rottura va letto
    come indicativo, non come stima precisa del limite fisico di capacita'.
    """
    D = panel[demand_col].astype(float)
    rows = []
    for y_col in ["parking_tickets", "bici_near_fazzon", "bici_valley_total"]:
        if mode == "median":
            bp = D.median()
            lo, hi = panel[D <= bp], panel[D > bp]
            fit_lo = stats.linregress(lo[demand_col], lo[y_col])
            fit_hi = stats.linregress(hi[demand_col], hi[y_col])
            r2_single = r2_two = np.nan
        elif mode == "optimal":
            best = _find_optimal_breakpoint(D, panel[y_col], min_frac=min_frac)
            bp, fit_lo, fit_hi = best["breakpoint"], best["fit_lo"], best["fit_hi"]
            r2_single, r2_two = best["r2_single_line"], best["r2_two_segments"]
        else:
            raise ValueError(f"mode sconosciuta: {mode!r}")

        rows.append({
            "channel": y_col, "mode": mode, "breakpoint": bp,
            "slope_low_demand": fit_lo.slope, "slope_high_demand": fit_hi.slope,
            "pct_change": (fit_hi.slope - fit_lo.slope) / fit_lo.slope * 100,
            "r2_single_line": r2_single, "r2_two_segments": r2_two,
        })
    return pd.DataFrame(rows)


# %% 6. "Esperimento naturale" della navetta
def shuttle_natural_experiment(panel: pd.DataFrame, demand_col: str = "vf_pres_apt_valdisole") -> dict:
    """Confronta le auto per un dato livello di domanda/meteo tra il periodo
    in cui la navetta era realmente attiva (28/06/2025 -> 20/07/2025) e il
    resto della stagione, limitandosi all'intervallo di domanda in cui i due
    periodi si sovrappongono davvero.
    """
    era = panel["shuttle_running"]
    lo_overlap = max(panel.loc[era, demand_col].min(), panel.loc[~era, demand_col].min())
    hi_overlap = min(panel.loc[era, demand_col].max(), panel.loc[~era, demand_col].max())
    in_overlap = panel[demand_col].between(lo_overlap, hi_overlap)

    overlap = panel[in_overlap].copy()
    shuttle_days = overlap[overlap["shuttle_running"]]
    no_shuttle_days = overlap[~overlap["shuttle_running"]]

    # Regressione multipla sull'intera stagione: auto ~ domanda + temp.max + pioggia + navetta_attiva
    X = panel[[demand_col, "tmax", "rain_mm"]].astype(float).copy()
    X["shuttle_running"] = panel["shuttle_running"].astype(float)
    X.insert(0, "const", 1.0)
    y = panel[TARGET_COL].astype(float)
    mask = X.notna().all(axis=1) & y.notna()
    coef, residuals, *_ = np.linalg.lstsq(X[mask].values, y[mask].values, rcond=None)
    yhat = X[mask].values @ coef
    resid = y[mask].values - yhat
    n, k = mask.sum(), X.shape[1]
    dof = n - k
    mse = (resid ** 2).sum() / dof
    XtX_inv = np.linalg.inv(X[mask].values.T @ X[mask].values)
    se = np.sqrt(np.diag(mse * XtX_inv))
    coef_names = list(X.columns)
    shuttle_idx = coef_names.index("shuttle_running")
    shuttle_coef = coef[shuttle_idx]
    shuttle_se = se[shuttle_idx]
    t_stat = shuttle_coef / shuttle_se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    return {
        "overlap_demand_range": (lo_overlap, hi_overlap),
        "n_shuttle_days_in_overlap": len(shuttle_days),
        "n_no_shuttle_days_in_overlap": len(no_shuttle_days),
        "mean_parking_tickets_shuttle_running": shuttle_days[TARGET_COL].mean(),
        "mean_parking_tickets_no_shuttle": no_shuttle_days[TARGET_COL].mean(),
        "regression_coef_names": coef_names,
        "regression_coef": coef,
        "shuttle_running_coef": shuttle_coef,
        "shuttle_running_se": shuttle_se,
        "shuttle_running_p": p_val,
        "n_obs_regression": n,
    }


def shuttle_road_did(road_hourly: pd.DataFrame, years=(2023, 2024, 2025)) -> pd.DataFrame:
    """Confronto anno-su-anno sulla SS42 a Pellizzano.

    Per ogni anno: rapporto medio fra auto verso monte al mattino a
    Pellizzano e auto totali nelle stazioni di controllo, nella finestra della
    navetta (28/06-20/07) e nel periodo successivo (21/07-31/08). Se la
    navetta avesse tolto auto dalla strada per Fazzon, nel 2025 il rapporto
    finestra/dopo dovrebbe essere piu' basso che nel 2023/2024. Il 2022 e'
    escluso: la direzione 2 a Pellizzano manca per gran parte dell'estate.
    """
    daily = road_daily_series(road_hourly).set_index("date")
    controls = daily[[f"strada_{s}_totale" for s in ROAD_CONTROL_STATIONS]].sum(axis=1, min_count=len(ROAD_CONTROL_STATIONS))
    ratio = daily["strada_pellizzano_salita_mattina"] / controls

    rows = []
    for year in years:
        window = ratio[f"{year}-06-28":f"{year}-07-20"].dropna()
        later = ratio[f"{year}-07-21":f"{year}-08-31"].dropna()
        rows.append({
            "year": year, "shuttle_pilot": year == 2025,
            "ratio_window": window.mean(), "ratio_later": later.mean(),
            "window_over_later": window.mean() / later.mean(),
            "n_window": len(window), "n_later": len(later),
        })
    return pd.DataFrame(rows)


# %% 8. Confronto stagione 2025 vs 2026
def load_parking_2025_2026() -> pd.DataFrame:
    """Legge ``Dati parcheggio Fazzon.xlsx`` in formato lungo: una riga per
    giorno, con ``year`` e la data 2025 corrispondente (stesso giorno della
    settimana) in ``date_2025``.

    I totali di persone sono ricalcolati qui: la colonna \"N° persone 2026\"
    del foglio ha formule incoerenti (OPEN_QUESTIONS.md, punto 7).
    \"N° persone bus\" e' trattato come persone in salita; le persone in
    discesa (2026, dal 1/8) sono annotate come testo nella colonna note.
    """
    raw = pd.read_excel(PARKING_XLSX, header=None, skiprows=3).iloc[:, :18]
    raw.columns = [
        "date_2025", "cars_2025", "_p_auto", "_p_auto_corr", "bus_up_2025", "moto_persons_2025", "_p_tot", "_p_tot_corr",
        "date_2026", "cars_full_2026", "_p_full", "cars_half_2026", "_p_half", "bus_up_2026", "moto_persons_2026",
        "_p_tot_2026", "note_2026", "weather_note_2026",
    ]
    raw = raw.dropna(subset=["date_2025"])
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")
    notes = raw["note_2026"].astype("string")
    bus_down = notes.str.extract(r"(\d+)")[0].astype(float).where(notes.str.contains("discesa", na=False))

    y25 = pd.DataFrame({
        "date": pd.to_datetime(raw["date_2025"]), "year": 2025,
        "cars_full": num("cars_2025"), "cars_half": 0.0,
        "bus_up": num("bus_up_2025"), "bus_down": np.nan, "moto_persons": num("moto_persons_2025"),
        "note": pd.NA,
    })
    y26 = pd.DataFrame({
        "date": pd.to_datetime(raw["date_2026"]), "year": 2026,
        "cars_full": num("cars_full_2026"), "cars_half": num("cars_half_2026").fillna(0),
        "bus_up": num("bus_up_2026"), "bus_down": bus_down, "moto_persons": num("moto_persons_2026"),
        "note": raw["weather_note_2026"].astype("string").fillna(notes.where(~notes.str.contains("discesa", na=False))),
    })
    y25["date_2025"] = y25["date"]
    y26["date_2025"] = y25["date"].to_numpy()
    out = pd.concat([y25, y26], ignore_index=True)
    out = out.merge(fetch_meteo_station()[["date", "tmax", "rain_mm"]], on="date", how="left")
    out["is_rainy"] = (out["rain_mm"] >= RAINY_DAY_MM).astype(int)
    out["cars"] = out["cars_full"] + out["cars_half"]
    out["excluded"] = out["date"].isin(EXCLUDED_DAYS_2026)
    in_regime = out["date"].between(REGIME_2026_START, REGIME_2026_PARKING_END)
    out["occupancy"] = np.where(in_regime, OCCUPANCY_REGIME_2026, OCCUPANCY_NO_REGIME)
    out["persons_cars"] = out["cars"] * out["occupancy"]
    out["persons_cars_2_83"] = out["cars"] * PERSONS_PER_CAR  # stessa occupazione nei due anni (sensibilita')
    extra = out["bus_up"].fillna(0) + out["moto_persons"].fillna(0)
    out["persons_total"] = out["persons_cars"] + extra
    out["persons_total_2_83"] = out["persons_cars_2_83"] + extra
    return out


def _period_2026(date_2026: pd.Timestamp) -> str:
    if date_2026 < REGIME_2026_START:
        return "1. fino al 3/7 (2026 senza regime)"
    if date_2026 <= REGIME_2026_SHUTTLE_END:
        return "2. 4/7-30/8 (2026: navetta + parcheggio a pagamento)"
    return "3. dal 31/8 (2026 senza navetta)"


def compare_2025_2026(park: pd.DataFrame) -> pd.DataFrame:
    """Medie giornaliere per periodo, stessi giorni della settimana nei due
    anni, esclusi gli eventi in ``EXCLUDED_DAYS_2026`` (e i corrispondenti
    giorni 2025)."""
    wide = park.pivot(index="date_2025", columns="year")
    excluded = wide[("excluded", 2026)].astype(bool)
    wide = wide[~excluded]
    period = wide[("date", 2026)].map(_period_2026)
    cols = ["cars", "bus_up", "bus_down", "persons_total", "persons_total_2_83", "tmax", "is_rainy"]
    rows = []
    for name, g in wide.groupby(period):
        row = {"period": name, "n_days": len(g)}
        for c in cols:
            for y in (2025, 2026):
                row[f"{c}_{y}"] = g[(c, y)].mean()
        row["cars_change_pct"] = (row["cars_2026"] / row["cars_2025"] - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


# %% Funzioni di supporto per i grafici
def _style_ax(ax):
    ax.grid(True, axis="y", zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


def plot_target_overview(panel: pd.DataFrame, path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    _style_ax(ax1)
    ax1.plot(panel["date"], panel["parking_tickets"], color=C_BLUE, linewidth=1.8, marker="o",
              markersize=3, label="Auto/giorno (biglietti parcheggio)")
    ax1.axvspan(panel["date"].min(), pd.Timestamp("2025-07-20"), color=C_GOOD_STATUS, alpha=0.07,
                 label="Navetta bus attiva")
    ax1.set_ylabel("Auto / giorno")
    ax1.set_title("Stagione Fazzon 2025: auto e copertura dei dati sui biglietti navetta",
                   color=C_INK, loc="left")
    ax1.legend(frameon=False, loc="upper left")

    _style_ax(ax2)
    valid = panel[panel["bus_tickets"].notna()]
    ax2.scatter(valid["date"], valid["bus_tickets"], color=C_GOOD_STATUS, s=28, zorder=3,
                 label="Biglietti navetta (validi, navetta attiva)")
    ax2.axvspan(panel["date"].min(), pd.Timestamp("2025-07-20"), color=C_GOOD_STATUS, alpha=0.07)
    ax2.set_ylabel("Biglietti navetta / giorno")
    ax2.set_xlabel("Data")
    ax2.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation_bars(corr: pd.DataFrame, path: Path):
    corr = corr.sort_values("within_month_r2")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    _style_ax(ax)
    labels = [IT_LABELS.get(c, c) for c in corr["predictor"]]
    ax.barh(labels, corr["within_month_r2"], color=C_BLUE, height=0.6, zorder=3)
    for y, v in enumerate(corr["within_month_r2"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=C_INK2)
    ax.set_xlabel("R² entro-mese vs auto/giorno (destagionalizzato)")
    ax.set_title("Quali segnali della Val di Sole seguono la domanda giornaliera a Fazzon?",
                  loc="left", color=C_INK)
    ax.set_xlim(0, corr["within_month_r2"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation_matrix(matrix: pd.DataFrame, path: Path):
    labels = [IT_LABELS.get(c, c) for c in matrix.columns]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3, 0.62 * n + 2))
    im = ax.imshow(matrix.values, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            v = matrix.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=C_INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Coefficiente di correlazione (r di Pearson)", color=C_INK2)
    cbar.ax.tick_params(labelcolor=C_INK2)

    ax.set_title(
        "Matrice di correlazione tra target e predittori\n"
        "(nota: fine settimana vs bici/pedoni ~ 0 -> nessun effetto di weekend)",
        loc="left", color=C_INK, fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_saturation(
    panel: pd.DataFrame, demand_col: str, path: Path, mode: str = "median", min_frac: float = 0.2,
    title: str = "La crescita delle auto rallenta all'aumentare della domanda in valle",
):
    """``mode="median"``: soglia fissa a meta' campione (Plot 3).
    ``mode="optimal"``: soglia scelta per griglia, minimizzando l'errore
    quadratico totale delle due rette (Plot 3b, 07).
    """
    D = panel[demand_col].astype(float)
    if mode == "median":
        bp = D.median()
        lo, hi = panel[D <= bp], panel[D > bp]
        fit_lo = stats.linregress(lo[demand_col], lo["parking_tickets"])
        fit_hi = stats.linregress(hi[demand_col], hi["parking_tickets"])
        subtitle = f"soglia = mediana della domanda ({bp:,.0f})".replace(",", ".")
    elif mode == "optimal":
        valid = panel[D.notna()]
        best = _find_optimal_breakpoint(valid[demand_col], valid["parking_tickets"], min_frac=min_frac)
        bp, fit_lo, fit_hi = best["breakpoint"], best["fit_lo"], best["fit_hi"]
        lo, hi = panel[D <= bp], panel[D > bp]
        subtitle = (
            f"soglia ottimizzata ({bp:,.0f}, n={best['n_lo']}+{best['n_hi']}) - "
            f"R² con 2 rette={best['r2_two_segments']:.2f} vs R² con 1 retta={best['r2_single_line']:.2f}"
        ).replace(",", ".")
    else:
        raise ValueError(f"mode sconosciuta: {mode!r}")

    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    _style_ax(ax)
    ax.scatter(panel[demand_col], panel["parking_tickets"], color=C_MUTED, s=22, alpha=0.7, zorder=2)
    xs_lo = np.array([lo[demand_col].min(), lo[demand_col].max()])
    xs_hi = np.array([hi[demand_col].min(), hi[demand_col].max()])
    ax.plot(xs_lo, fit_lo.intercept + fit_lo.slope * xs_lo, color=C_BLUE, linewidth=2.4,
             label=f"Retta a domanda bassa (pendenza={fit_lo.slope:.3f})", zorder=3)
    ax.plot(xs_hi, fit_hi.intercept + fit_hi.slope * xs_hi, color=C_ORANGE, linewidth=2.4,
             label=f"Retta a domanda alta (pendenza={fit_hi.slope:.3f}, {(fit_hi.slope/fit_lo.slope-1)*100:+.0f}%)",
             zorder=3)
    ax.axvline(bp, color=C_AXIS, linewidth=1, linestyle="--")
    ax.set_xlabel(IT_LABELS[demand_col])
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_title(
        f"{title}\n{subtitle}",
        loc="left", color=C_INK, fontsize=11,
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_car_vs_bici_divergence(
    panel: pd.DataFrame, demand_col: str, path: Path, mode: str = "median", min_frac: float = 0.2,
):
    """Stesso fenomeno di 03/03b (rallentamento auto vs accelerazione
    bici/pedoni), presentato come due pannelli affiancati -- uno per canale,
    ciascuno nella propria unita' di misura -- invece di un'unica curva
    indicizzata: indicizzare a un valore iniziale vicino allo zero (come nel
    caso delle auto) gonfia artificialmente le variazioni percentuali e
    capovolge la lettura del grafico. Qui la stessa codifica colore di 03/03b
    (blu = retta a domanda bassa, arancio = retta a domanda alta) e' condivisa
    fra i due pannelli, cosi' il confronto e' "quale retta si piega, e in che
    verso" invece di un numero indice.
    """
    D = panel[demand_col].astype(float)
    channels = [
        ("parking_tickets", IT_LABELS["parking_tickets"]),
        ("bici_near_fazzon", IT_LABELS["bici_near_fazzon"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (col, ylabel) in zip(axes, channels):
        _style_ax(ax)
        if mode == "median":
            bp = D.median()
            lo, hi = panel[D <= bp], panel[D > bp]
            fit_lo = stats.linregress(lo[demand_col], lo[col])
            fit_hi = stats.linregress(hi[demand_col], hi[col])
        else:
            best = _find_optimal_breakpoint(D, panel[col], min_frac=min_frac)
            bp, fit_lo, fit_hi = best["breakpoint"], best["fit_lo"], best["fit_hi"]

        ax.scatter(panel[demand_col], panel[col], color=C_MUTED, s=20, alpha=0.7, zorder=2)
        xs_lo = np.array([D.min(), bp])
        xs_hi = np.array([bp, D.max()])
        ax.plot(xs_lo, fit_lo.intercept + fit_lo.slope * xs_lo, color=C_BLUE, linewidth=2.4, zorder=3)
        ax.plot(xs_hi, fit_hi.intercept + fit_hi.slope * xs_hi, color=C_ORANGE, linewidth=2.4, zorder=3)
        ax.axvline(bp, color=C_AXIS, linewidth=1, linestyle="--")

        pct = (fit_hi.slope - fit_lo.slope) / fit_lo.slope * 100
        ax.set_title(
            f"{ylabel}\npendenza: {fit_lo.slope:.3g} -> {fit_hi.slope:.3g}  ({pct:+.0f}%),  soglia={bp:,.0f}".replace(",", "."),
            loc="left", color=C_INK, fontsize=10,
        )
        ax.set_xlabel(IT_LABELS[demand_col], fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)

    handles = [
        Line2D([0], [0], color=C_BLUE, linewidth=2.4, label="Retta a domanda bassa"),
        Line2D([0], [0], color=C_ORANGE, linewidth=2.4, label="Retta a domanda alta"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))

    subtitle = (
        "soglia = mediana della domanda (uguale per entrambi i canali)" if mode == "median"
        else "soglia ottimizzata per canale (minimizza l'errore totale delle due rette)"
    )
    fig.suptitle(
        f"Stessa domanda in valle, risposta opposta: le auto rallentano, bici/pedoni no\n{subtitle}",
        x=0.02, ha="left", fontsize=12, color=C_INK, y=1.03,
    )
    fig.subplots_adjust(top=0.72, wspace=0.28)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_shuttle_experiment(panel: pd.DataFrame, demand_col: str, result: dict, path: Path):
    lo_overlap, hi_overlap = result["overlap_demand_range"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    _style_ax(ax)

    shuttle = panel[panel["shuttle_running"]]
    no_shuttle = panel[~panel["shuttle_running"]]
    ax.scatter(shuttle[demand_col], shuttle["parking_tickets"], color=C_BLUE, s=32, zorder=3,
                label="Navetta attiva (28 giu - 20 lug)")
    ax.scatter(no_shuttle[demand_col], no_shuttle["parking_tickets"], color=C_ORANGE, s=32, zorder=3,
                label="Navetta non attiva (resto della stagione)")
    ax.axvspan(lo_overlap, hi_overlap, color=C_MUTED, alpha=0.08, zorder=1,
                label="Intervallo di domanda in cui i due periodi si sovrappongono")

    ax.set_xlabel(IT_LABELS[demand_col])
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_title(f"Giorni con/senza navetta a parità di domanda\n(domanda: {IT_LABELS[demand_col]})",
                 loc="left", color=C_INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    coef_txt = (
        f"Regressione: auto ~ domanda + temp.max + pioggia + navetta_attiva\n"
        f"coefficiente navetta_attiva = {result['shuttle_running_coef']:+.1f} auto/giorno "
        f"(p={result['shuttle_running_p']:.2f}, n={result['n_obs_regression']})\n"
        f"Non significativo -- coerente con 'nessuna differenza rilevabile nel numero di auto',\n"
        f"ma il periodo è confuso con il tempo solare (poche osservazioni, identificazione debole)."
    )
    ax.text(0.02, -0.20, coef_txt, transform=ax.transAxes, fontsize=8, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_bus_coverage_gap(panel: pd.DataFrame, path: Path):
    """Il grafico dell'"assenza di prove": mostra che i giorni senza dati
    navetta validi sono esattamente i giorni di domanda più alta (più
    "saturi") della stagione.
    """
    p = panel.sort_values("date").copy()
    saturated_threshold = p["parking_tickets"].quantile(0.75)
    p["saturated_day"] = p["parking_tickets"] >= saturated_threshold

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _style_ax(ax)
    ax.fill_between(p["date"], p["parking_tickets"], color=C_BLUE, alpha=0.15, zorder=1)
    ax.plot(p["date"], p["parking_tickets"], color=C_BLUE, linewidth=1.6, zorder=2, label="Auto / giorno")

    has_bus = p[p["bus_tickets"].notna()]
    no_bus = p[p["bus_tickets"].isna()]
    ax.scatter(has_bus["date"], has_bus["parking_tickets"], color=C_GOOD_STATUS, s=30, zorder=4,
                label="Dati navetta Fazzon validi in questo giorno")
    sat_no_bus = no_bus[no_bus["saturated_day"]]
    other_no_bus = no_bus[~no_bus["saturated_day"]]
    ax.scatter(sat_no_bus["date"], sat_no_bus["parking_tickets"], color=C_RED_STATUS, s=42, marker="X", zorder=5,
                label="Nessun dato navetta E domanda nel quartile superiore (giorno \"saturo\")")
    ax.scatter(other_no_bus["date"], other_no_bus["parking_tickets"], color=C_MUTED, s=16, zorder=3,
                label="Nessun dato navetta, domanda più bassa")

    ax.axhline(saturated_threshold, color=C_AXIS, linewidth=1, linestyle="--")
    ax.text(p["date"].max(), saturated_threshold, " soglia quartile superiore", va="bottom", ha="right",
             fontsize=8, color=C_INK2)
    ax.set_ylabel(IT_LABELS["parking_tickets"])
    ax.set_title(
        "Non esistono dati sulla navetta per i giorni di domanda più alta della stagione",
        loc="left", color=C_INK,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return saturated_threshold


def plot_shuttle_road_did(did: pd.DataFrame, path: Path):
    """Rapporto finestra navetta / periodo successivo, per anno: barra 2025
    (anno della navetta) evidenziata, 2023/2024 come riferimento."""
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    _style_ax(ax)
    colors = [C_BLUE if pilot else C_MUTED for pilot in did["shuttle_pilot"]]
    x = np.arange(len(did))
    ax.bar(x, did["window_over_later"], color=colors, width=0.55, zorder=3)
    for xi, v in zip(x, did["window_over_later"]):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C_INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n(navetta)" if p else str(y) for y, p in zip(did["year"], did["shuttle_pilot"])])
    lo = did["window_over_later"].min()
    ax.set_ylim(lo - 0.08, did["window_over_later"].max() + 0.04)
    ax.set_ylabel("Rapporto 28 giu-20 lug / 21 lug-31 ago")
    ax.set_title(
        "Auto verso monte a Pellizzano (7-12), rispetto alle valli laterali di controllo:\n"
        "nel 2025, con la navetta, la finestra 28/6-20/7 non è più bassa degli anni precedenti",
        loc="left", color=C_INK, fontsize=10.5,
    )
    note = (
        "Rapporto = auto SS42 Pellizzano verso monte 7-12 / auto totali a Pracorno + Celledizzo + Revò,\n"
        "media della finestra navetta diviso media del periodo successivo. Un effetto navetta abbasserebbe la barra 2025.\n"
        "Le auto dirette a Fazzon sono ~10-14% del flusso a Pellizzano: un calo di poche decine di auto/giorno (~2%)\n"
        "e' dello stesso ordine della variabilita' fra un anno e l'altro, quindi non sarebbe distinguibile."
    )
    ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=7.8, color=C_INK2, va="top")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_road_data_coverage(road_hourly: pd.DataFrame, panel: pd.DataFrame, path: Path):
    """Copertura dei sensori di traffico: giorni validi per stazione e
    direzione, 2022-2025, con la stagione Fazzon 2025 evidenziata."""
    daily = road_daily_validity(road_hourly)
    grid = daily.pivot_table(index=["station", "direction"], columns="date", values="valid", aggfunc="first")
    all_days = pd.date_range(grid.columns.min(), grid.columns.max(), freq="D")
    grid = grid.reindex(columns=all_days).fillna(False).astype(float)
    labels = [f"{s.capitalize()} dir.{d}" for s, d in grid.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = LinearSegmentedColormap.from_list("valid", [C_RED_STATUS, C_BLUE])
    ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=(0, len(all_days), len(labels), 0))
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=8)
    year_starts = [i for i, d in enumerate(all_days) if d.month == 1 and d.day == 1]
    ax.set_xticks(year_starts)
    ax.set_xticklabels([all_days[i].year for i in year_starts])
    ax.grid(False)
    s0 = (panel["date"].min() - all_days[0]).days
    s1 = (panel["date"].max() - all_days[0]).days + 1
    ax.axvline(s0, color=C_INK, linewidth=1)
    ax.axvline(s1, color=C_INK, linewidth=1)
    ax.text((s0 + s1) / 2, -0.3, "stagione Fazzon 2025", ha="center", va="bottom", fontsize=8, color=C_INK)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=C_BLUE, markersize=9, label="Dato valido"),
        Line2D([0], [0], marker="s", linestyle="", color=C_RED_STATUS, markersize=9,
               label="Mancante, ore incomplete o sensore a zero"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8.5)
    ax.set_title(
        "Copertura dei contatori di traffico PAT: Pellizzano completo dal 2023;\n"
        "Dimaro spento da metà 2024, Tonale nel 2022, Croviana dir.2 quasi sempre a zero",
        loc="left", color=C_INK, fontsize=10.5, pad=16,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_sensor_map(panel: pd.DataFrame, path: Path):
    """Mappa schematica (senza fondo cartografico) di stazioni di traffico,
    contatori bici/pedoni e Fazzon, colorati per ruolo nell'analisi."""
    bike = pd.read_csv(NEW_DATA_DIR / "contamezzi_TM_info.csv", sep=";", on_bad_lines="skip")
    bike = bike[bike["ambito"] == "VALDISOLE"].drop_duplicates("sensore")
    bike["label"] = bike["sensore"].map(BIKE_SENSOR_SHORT_NAMES)

    role = {
        "pellizzano": ("Predittore principale (auto verso monte 7-12)", C_BLUE),
        "pracorno": ("Stazione di controllo", C_AQUA),
        "celledizzo": ("Stazione di controllo", C_AQUA),
        "revo": ("Stazione di controllo", C_AQUA),
        "tonale": ("Predittore", C_VIOLET),
    }
    unused = ("Non usata (sensore guasto o fuori area)", C_MUTED)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.grid(False)
    for station, (lat, lon, _) in ROAD_STATION_COORDS.items():
        _, color = role.get(station, unused)
        ax.scatter(lon, lat, s=90, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.annotate(station.capitalize(), (lon, lat), xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=C_INK)
    near = bike["sensore"].isin(BIKE_SENSORS_NEAR_FAZZON)
    ax.scatter(bike["longitudine"], bike["latitudine"], marker="^", s=70,
               color=np.where(near, C_ORANGE, C_SURFACE_ORANGE), edgecolor=C_ORANGE, linewidth=1.2, zorder=3)
    for _, r in bike.iterrows():
        ax.annotate(r["label"], (r["longitudine"], r["latitudine"]), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=7.5, color=C_INK2)
    ax.scatter(*FAZZON_COORDS[::-1], marker="*", s=260, color=C_RED_STATUS, edgecolor=SURFACE, zorder=5)
    ax.annotate("Fazzon / Lago dei Caprioli", FAZZON_COORDS[::-1], xytext=(8, -4), textcoords="offset points",
                fontsize=9, color=C_RED_STATUS, fontweight="bold")

    ax.set_aspect(1 / np.cos(np.deg2rad(46.33)))
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, markersize=8, label=l)
               for l, c in dict(list(role.values()) + [unused]).items()]
    handles += [
        Line2D([0], [0], marker="^", linestyle="", color=C_ORANGE, markersize=8, label="Contatore bici/pedoni vicino a Fazzon"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor=C_SURFACE_ORANGE, markeredgecolor=C_ORANGE,
               markersize=8, label="Altro contatore bici/pedoni"),
        Line2D([0], [0], marker="*", linestyle="", color=C_RED_STATUS, markersize=12, label="Fazzon (posizione indicativa)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.set_title("Dove sono i sensori: stazioni di traffico PAT e contatori bici/pedoni in Val di Sole",
                 loc="left", color=C_INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_2025_vs_2026_daily(park: pd.DataFrame, path: Path):
    """Tre pannelli con asse x comune (data 2026; il 2025 e' allineato per
    giorno della settimana): auto, passeggeri navetta, persone totali."""
    wide = park.pivot(index="date_2025", columns="year")
    x = wide[("date", 2026)]
    excluded = wide[("excluded", 2026)].astype(bool)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1.2]})
    for ax in axes:
        _style_ax(ax)
        ax.axvspan(REGIME_2026_START, REGIME_2026_SHUTTLE_END, color=C_GOOD_STATUS, alpha=0.06, zorder=0)
        for d in EXCLUDED_DAYS_2026:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=C_MUTED, alpha=0.18, zorder=0)

    def line(ax, col, year, color, label, **kw):
        ax.plot(x, wide[(col, year)].where(~excluded), color=color, linewidth=1.8, label=label, **kw)

    ax = axes[0]
    line(ax, "cars", 2025, C_MUTED, "2025 (5 euro, fino a esaurimento posti)")
    line(ax, "cars", 2026, C_BLUE, "2026 (giornata intera + pomeriggio)")
    ax.set_ylabel("Auto / giorno")
    ax.set_title("Auto al parcheggio di Fazzon", loc="left", color=C_INK, fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    ax = axes[1]
    line(ax, "bus_up", 2025, C_MUTED, "2025 Greenbus pilota (persone in salita)")
    line(ax, "bus_up", 2026, C_BLUE, "2026 Lakebus, persone in salita")
    line(ax, "bus_down", 2026, C_ORANGE, "2026 Lakebus, persone in discesa (dal 1/8)")
    ax.set_ylabel("Persone / giorno")
    ax.set_title("Passeggeri navetta", loc="left", color=C_INK, fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    ax = axes[2]
    line(ax, "persons_total", 2025, C_MUTED, "2025: auto x 2,5 + bus")
    line(ax, "persons_total", 2026, C_BLUE, "2026: auto x 3,5 (x 2,5 fuori regime) + bus in salita + moto")
    line(ax, "persons_total_2_83", 2026, C_BLUE, "2026 con 2,83 persone/auto (sensibilità)", linestyle="--", alpha=0.7)
    ax.set_ylabel("Persone / giorno")
    ax.set_title("Persone stimate (coefficienti di occupazione del gestore, da validare)",
                 loc="left", color=C_INK, fontsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_xlabel("Data 2026 (2025 allineato per giorno della settimana)")

    fig.suptitle(
        "Fazzon 2025 vs 2026: con navetta e parcheggio a pagamento le auto calano di circa 2/3,\n"
        "le persone stimate restano simili (fascia verde: regime 2026; grigio: giorni esclusi)",
        x=0.01, ha="left", fontsize=12, color=C_INK,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_2025_vs_2026_periods(comp: pd.DataFrame, path: Path):
    """Medie per periodo: auto e persone stimate, 2025 vs 2026 affiancati."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    labels = [p.split(" (")[0].replace(". ", ".\n") for p in comp["period"]]
    xpos = np.arange(len(comp))
    w = 0.38
    for ax, col, title in [
        (axes[0], "cars", "Auto / giorno"),
        (axes[1], "persons_total", "Persone stimate / giorno"),
    ]:
        _style_ax(ax)
        for off, year, color in [(-w / 2, 2025, C_MUTED), (w / 2, 2026, C_BLUE)]:
            vals = comp[f"{col}_{year}"]
            ax.bar(xpos + off, vals, width=w, color=color, label=str(year), zorder=3)
            for xi, v in zip(xpos + off, vals):
                ax.text(xi, v + vals.max() * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=C_INK2)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, loc="left", color=C_INK, fontsize=10.5)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Medie per periodo, stessi giorni della settimana\n"
        "1 = fino al 3/7;  2 = 4/7-30/8, regime 2026 (navetta + parcheggio a pagamento);  3 = dal 31/8, senza navetta",
        x=0.01, ha="left", fontsize=11, color=C_INK,
    )
    note = ("Persone: 2025 = auto x 2,5 + bus; 2026 = auto x 3,5 nel regime (x 2,5 fuori) + bus in salita + moto "
            "(coefficienti del gestore, da validare).\nAuto 2026 non conteggiate: ingressi dopo le 17 e parcheggi privati.")
    fig.text(0.01, 0.01, note, fontsize=8, color=C_INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# %% Main
def main():
    print("Lettura dei conteggi di traffico PAT...")
    road_hourly = load_road_counts_hourly()
    print("Costruzione del pannello dati...")
    panel = build_panel(road_hourly)
    panel.to_csv(OUTPUT_DIR / "panel_daily.csv", index=False)
    print(f"  {len(panel)} giorni, {panel['date'].min().date()} -> {panel['date'].max().date()}")

    print("\nTabella di correlazione (prime 10 per R2 entro-mese):")
    corr = correlation_table(panel)
    corr.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
    print(corr.head(10).to_string(index=False))

    print("\nMatrice di correlazione (Pearson, coppie complete):")
    cmat = correlation_matrix(panel)
    cmat.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    print(f"  is_weekend vs bici_valley_total: r = {cmat.loc['is_weekend', 'bici_valley_total']:.3f}")
    print(f"  is_weekend vs bici_near_fazzon:  r = {cmat.loc['is_weekend', 'bici_near_fazzon']:.3f}")
    print(f"  auto SS42 verso monte vs presenze in valle (entro-mese): r = {correlation_road_vs_presences(panel):.3f}")

    print("\nPendenze di saturazione, soglia = mediana (auto vs bici/pedoni, vs presenze Vodafone in valle):")
    sat = saturation_slopes(panel, mode="median")
    print(sat.to_string(index=False))

    print("\nPendenze di saturazione, soglia ottimizzata per canale (minimizza l'errore totale):")
    sat_optimal = saturation_slopes(panel, mode="optimal")
    print(sat_optimal.to_string(index=False))

    print("\nPendenze di saturazione, soglia ottimizzata, domanda = auto SS42 Pellizzano verso monte (7-12):")
    road_panel = panel.dropna(subset=["strada_pellizzano_salita_mattina"])
    sat_road = saturation_slopes(road_panel, demand_col="strada_pellizzano_salita_mattina", mode="optimal")
    sat_road.insert(0, "demand", "strada_pellizzano_salita_mattina")
    print(sat_road.to_string(index=False))
    sat.insert(0, "demand", "vf_pres_apt_valdisole")
    sat_optimal.insert(0, "demand", "vf_pres_apt_valdisole")
    pd.concat([sat, sat_optimal, sat_road]).to_csv(OUTPUT_DIR / "saturation_slopes.csv", index=False)

    print("\nEsperimento naturale della navetta:")
    result = shuttle_natural_experiment(panel)
    for k, v in result.items():
        if k not in ("regression_coef_names", "regression_coef"):
            print(f"  {k}: {v}")

    print("\nEsperimento naturale della navetta, domanda = auto SS42 Pellizzano verso monte (7-12):")
    result_road = shuttle_natural_experiment(road_panel, demand_col="strada_pellizzano_salita_mattina")
    for k, v in result_road.items():
        if k not in ("regression_coef_names", "regression_coef"):
            print(f"  {k}: {v}")

    print("\nConfronto anno-su-anno sulla SS42 (Pellizzano vs valli di controllo):")
    did = shuttle_road_did(road_hourly)
    did.to_csv(OUTPUT_DIR / "shuttle_road_did.csv", index=False)
    print(did.to_string(index=False))

    print("\nGenerazione dei grafici...")
    plot_target_overview(panel, OUTPUT_DIR / "01_target_overview.png")
    plot_correlation_bars(corr, OUTPUT_DIR / "02_predictor_correlations.png")
    plot_correlation_matrix(cmat, OUTPUT_DIR / "02b_correlation_matrix.png")
    plot_saturation(panel, "vf_pres_apt_valdisole", OUTPUT_DIR / "03_car_saturation.png", mode="median")
    plot_saturation(panel, "vf_pres_apt_valdisole", OUTPUT_DIR / "03b_car_saturation_optimal_breakpoint.png",
                     mode="optimal")
    plot_car_vs_bici_divergence(panel, "vf_pres_apt_valdisole", OUTPUT_DIR / "04_car_vs_bici_divergence.png",
                                 mode="median")
    plot_car_vs_bici_divergence(panel, "vf_pres_apt_valdisole",
                                 OUTPUT_DIR / "04b_car_vs_bici_divergence_optimal_breakpoint.png", mode="optimal")
    plot_shuttle_experiment(panel, "vf_pres_apt_valdisole", result, OUTPUT_DIR / "05_shuttle_natural_experiment.png")
    threshold = plot_bus_coverage_gap(panel, OUTPUT_DIR / "06_bus_data_coverage_gap.png")
    print(f"  soglia auto/giorno per giorno 'saturo' (quartile superiore) = {threshold:.0f}")
    plot_saturation(road_panel, "strada_pellizzano_salita_mattina", OUTPUT_DIR / "07_car_vs_road_traffic.png",
                    mode="optimal", title="Auto a Fazzon vs auto in salita sulla SS42 a Pellizzano")
    plot_shuttle_experiment(road_panel, "strada_pellizzano_salita_mattina", result_road,
                            OUTPUT_DIR / "05b_shuttle_natural_experiment_road.png")
    plot_shuttle_road_did(did, OUTPUT_DIR / "08_shuttle_road_year_over_year.png")
    plot_road_data_coverage(road_hourly, panel, OUTPUT_DIR / "09_road_data_coverage.png")
    plot_sensor_map(panel, OUTPUT_DIR / "00_sensor_map.png")

    print("\nConfronto 2025 vs 2026 (medie giornaliere per periodo, stessi giorni della settimana):")
    park = load_parking_2025_2026()
    park.to_csv(OUTPUT_DIR / "parking_2025_2026_daily.csv", index=False)
    comp = compare_2025_2026(park)
    comp.to_csv(OUTPUT_DIR / "compare_2025_2026.csv", index=False)
    print(comp.round(1).T.to_string())
    plot_2025_vs_2026_daily(park, OUTPUT_DIR / "10_2025_vs_2026_daily.png")
    plot_2025_vs_2026_periods(comp, OUTPUT_DIR / "11_2025_vs_2026_periods.png")

    print(f"\nTutti gli output sono stati scritti in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
