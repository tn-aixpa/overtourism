"""Daily measured weather of a Meteotrentino station (open-data service), 2023 to today.

Writes Download/meteo_stazione_<code>.csv (date, tmin, tmax, rain_mm, quality), to be uploaded to
the data lake under fazzon/meteo/ (see model/fazzon/fazzon.yaml). Used by the Fazzon pipeline:
station T0074 = Male', reference station of the area "Sole, Peio e Rabbi".
"""

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://dati.meteotrentino.it/service.asmx/datiStoriciCaratteristiciSerie{measure}UnArea"
STATION = "T0074"
YEARS = range(2023, dt.date.today().year + 1)
SAVE_DIR = Path("Download")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def fetch(measure, year):
    r = requests.get(
        BASE.format(measure=measure),
        params={"codice": STATION, "anno": year},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    r.raise_for_status()
    return pd.DataFrame(r.json()["dati"])


temps, rains = [], []
for year in YEARS:
    temps.append(fetch("Temperature", year))
    rains.append(fetch("Precipitazioni", year))
    print(year, "ok")
    time.sleep(1)

temp = pd.concat(temps).rename(columns={"Min": "tmin", "Max": "tmax", "QualityMax": "quality"})
rain = pd.concat(rains).rename(columns={"Valore": "rain_mm"})
# dates are ISO yyyy-mm-dd (the rain series is listed by day, then month)
out = temp[["giorno", "tmin", "tmax", "quality"]].merge(rain[["giorno", "rain_mm"]], on="giorno", how="outer")
out = out.rename(columns={"giorno": "date"}).sort_values("date")[["date", "tmin", "tmax", "rain_mm", "quality"]]
path = SAVE_DIR / f"meteo_stazione_{STATION}.csv"
out.to_csv(path, index=False)
print(f"{path}: {len(out)} days, {out['date'].min()} .. {out['date'].max()}")
