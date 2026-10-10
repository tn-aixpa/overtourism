# Fazzon data preparation

Reference datasets for the Fazzon (Lago dei Caprioli, Val di Sole) model. Same structure as the
main pipeline (`data_preparation/README.md`), reusing its `utils` and `config/settings.yaml`
(directories, format, platform). Run from the `data_preparation` folder:

```bash
python -m model.fazzon.prepare                  # steps 0-2
python -m model.fazzon.prepare --skip-download  # reuse <normalized>/fazzon/ (steps 2 only)
python -m model.fazzon.prepare --refresh        # download the data-lake files again
python -m model.fazzon.prepare --upload         # also log the reference datasets to the platform
python -m model.fazzon.analyze_new_data         # exploratory analysis and plots (Italian labels)
```

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 Sources | `sources.py` | data lake or `local_source_dir` (`fazzon.yaml`) | raw files |
| 1 Standardize | `standardize.py` | raw files | `Output/data/normalized/fazzon/*.parquet` |
| 2 Reference datasets | `reference_datasets.py` | normalized tables | `Output/data/final_data/fazzon/*.parquet` |
| 3 Upload | `utils.remote.log_dataframe` | reference datasets | platform dataitems (`--upload`) |

* **Sources** (`fazzon.yaml`): each source has its data-lake key and its local path. With
  `local_source_dir` set (the default until the Fazzon files are uploaded, see `PLAN.md`) the
  files are read in place; otherwise they are streamed from the data lake to
  `Output/data/raw_data/fazzon/`. As in the main pipeline only the data lake is read: the
  measured weather (Male' station, T0074) is fetched from the Meteotrentino open-data service
  by `data_download/download_meteo_meteotrentino.py` and uploaded to `fazzon/meteo/`.
* **Standardize**: one table per source, cleaned and quality-flagged, no modelling choices:
  `fazzon_parking` (2025 + 2026), `fazzon_parking_2025`, `fazzon_greenbus_2025` (rows after the
  pilot flagged `other_area`), `road_counts_hourly` / `road_counts_daily` (cars, classes 2+3;
  `valid` = 23+ hours and 50+ cars), `meteo_station`, `meteo_bulletin`, `vodafone_presences`
  (tourist + visitor, Italian + foreign total), `bike_counts`, `tgc_accessi`, `stations`
  (from `reference/stations.csv`).
* **Reference datasets**:
  * `fazzon_daily`: one row per day from 2023-01-01: measured Fazzon data (cars, bus, motorbikes),
    access regime flags (`shuttle`, `regulated_parking`, `event_day`), weather and predictors
    (road counts, Vodafone, bikes, Guest Card).
  * `fazzon_presence_season_stats`, `fazzon_presence_day_type_stats`: the presence statistics of
    `executor-backend/src/model/fazzon/fazzon_presence_stats.py`, computed from `fazzon_daily`
    with the same method and assumptions (2025 parking x 2.83 people/car, non-car share 0.31).

The Fazzon model (`executor-backend/src/model/fazzon/`) reads the two statistics files from its
data folder (`FAZZON_DATA_DIR`, e.g. `<repo>/data_preparation/Output/data/final_data/fazzon` for a
local run), or downloads them from the platform at startup; it has no built-in values. See
`executor-backend/src/model/fazzon/fazzon_data.py` and `docs/howto/backend.md` (Model API data).

Assumptions and data questions still open with the domain experts: `OPEN_QUESTIONS.md`.
Tests: `tests/test_fazzon_pipeline.py` (synthetic fixtures, no data needed).
