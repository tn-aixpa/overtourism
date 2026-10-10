# Fazzon: plan of activities

Branch: `feat/fazzon-model` (from `refactor/backend-refactor`). Open questions for the
domain experts: `OPEN_QUESTIONS.md`.

## Where we are

- `analyze_new_data.py`: exploratory analysis (2025 panel, road counts, 2025 vs 2026), local
  files only, Italian plots.
- `executor-backend/src/model/fazzon/fazzon_presence_stats.py`: presence statistics are
  **hard-coded** (2025 parking × 2.83; non-car = 0.31 share, ASSUMPTION; no weather; day-type
  buckets from 2025). The model reads no data.
- Data: the context datasets are on the data lake (`most-datalake/overtourism/inputdata/`).
  The Fazzon-specific ones are local only (see Phase 1).

## Phase 0: Housekeeping (now)

- [x] 0.1 Commit the current work on `feat/fazzon-model`: analysis script, `.gitignore`,
      `OPEN_QUESTIONS.md`, `PLAN.md`.

## Phase 1: Data on the platform

- [ ] 1.1 Agree on a layout: `inputdata/fazzon/` for Fazzon-specific raw files.
- [ ] 1.2 Upload the missing raw files:
      - `Dati parcheggio Fazzon.xlsx` (2025 + 2026; living file, one new column block per season)
      - `Riepilogo dati bus estate 2025.xlsx` (Greenbus 2025)
      - `Rilevazione affluenza lago caprioli.*` + `_posti auto.csv` (2021 survey: occupancy, stalls)
      - `TGC_Viaggiato.csv` (only `TGC_Accessi` is on the lake)
      - `meteo_stazione_T0074.csv` (from `data_download/download_meteo_meteotrentino.py`) to
        `fazzon/meteo/`; re-run and re-upload when new days are needed
      - station reference table: road stations (code, name, lat/lon, road, up-valley direction,
        role) + bike counters + Fazzon location (now hard-coded in the script)
- [ ] 1.3 Check what is already there and could replace local copies:
      - `dati_transiti/dati_transito_veicolare.parquet` (15.6 MB): does it keep
        station × date × hour × direction × class? If yes, use it instead of the 2.3 GB CSV.
      - `Gestionestrade_contamezzi.zip`: station metadata?
- [x] 1.4 Weather: fetched from Meteotrentino open data by a `data_download/` script (as the
      ISPAT presences), uploaded to the data lake, read by the pipeline like the other sources.

## Phase 2: Fazzon data-preparation pipeline (`data_preparation/model/fazzon/`)

Reuses `data_preparation/utils` (remote, io, config, `settings.yaml`); own entry point, e.g.
`python -m model.fazzon.prepare [--skip-download] [--upload]`.

| Step | Input | Output |
|------|-------|--------|
| 0 Download | data lake (`inputdata/` + `inputdata/fazzon/`) | `raw_data/fazzon/` (as is) |
| 1 Standardize | raw | `normalized/fazzon/*.parquet`: one clean table per source |
| 2 Reference datasets | normalized | `final_data/fazzon/*.parquet` |
| 3 Upload (optional) | final | platform dataitems (`log_dataframe`) |

Step 1 tables (one per source, cleaned and quality-flagged, no modelling choices):
- `fazzon_parking`: date, year, cars_full, cars_half, moto, bus_up, bus_down, notes, excluded
- `fazzon_bus_2025` (Greenbus, Aug rows dropped)
- `road_counts_hourly` (station, date, hour, direction, cars; valid flag)
- `meteo_male` (tmin, tmax, rain, quality code)
- `vodafone_presences_fazzon` (POI / comuni / APT / province, tourist + visitor)
- `bike_counts_valdisole`, `tgc_accessi_valdisole`, `stations`

Step 2 reference datasets (the contract with the model; to be agreed, see 3.1):
- `fazzon_daily`: one row per day (2023–2026): Fazzon cars/bus/persons + regime flags +
  predictors (road counts, weather, Vodafone, bikes) + data-quality flags
- `fazzon_presence_stats`: season × day-type × regime statistics that today are hard-coded in
  `fazzon_presence_stats.py`

Tasks:
- [x] 2.1 Move the loaders of `analyze_new_data.py` into step-1 modules (no logic change), with
      tests on small fixtures.
- [x] 2.2 Step 2 builders + checks (row counts, date coverage, sums). The presence statistics
      reproduce the hard-coded ones within 0.7% (the model rounded car counts before x 2.83);
      28/08/2025 must be left out to reproduce them (reason not documented).
- [x] 2.3 Upload step and a README section, mirroring `data_preparation/README.md`.
      (`--upload` and the data-lake download are not tested yet: no platform access from the
      development machine.)
- [x] 2.4 Make `analyze_new_data.py` read the step-1/step-2 parquets (single source of truth).
      All analysis outputs unchanged.

## Phase 3: Model reads the reference datasets

- [x] 3.1 What the model consumes: the pre-computed statistics (`fazzon_presence_season_stats`,
      `fazzon_presence_day_type_stats`); the calibration lives in pipeline step 2.
- [x] 3.2 How `executor-backend` gets the data: as the indicator data of the CRUD API. At startup
      (FastAPI lifespan) the dataitems are downloaded from the platform unless
      `DT_OVERTURISM_STANDALONE_MODE=true`; on demand with `POST /models/fazzon/download-data`
      (not authenticated, the wrapping layers add authentication). No built-in values: missing
      data or a failed download raise. Data folder: `FAZZON_DATA_DIR` or
      `executor-backend/database/fazzon`. The statistics are read when the model is built, and the
      cached backend is dropped after a download.
- [x] 3.3 Replace the hard-coded tables in `fazzon_presence_stats.py` with the loader; keep the
      same interface (`season_stats`, `day_type_stats`, `make_presence_factories`). The hard-coded
      tables stay as `DEFAULT_*` (fallback); their comments fixed (per-season SD, 28/08 left out).
- [~] 3.4 Model content updates enabled by the new data (decisions in `ASSUMPTIONS.md`).
      Done: register; 28/08/2025 included (D4); weekday split dropped (D5); lakeside dropped
      (D6); daily parking price in `fazzon_daily` (A7, D3). Next: weather category (D7), modal-
      shift estimation (D8), then the model's visitor numbers on top of it (replacing A3, A4).
      Original list:
      - non-car visitors: 2026 bus counts are measured, so they replace the 0.31 ASSUMPTION
        for the shuttle part
      - regime: separate stats for "no regime" (2025) and "2026 regime" (pricing + cap + shuttle),
        usable as scenarios
      - day-type buckets: the analysis finds ~no weekend effect; re-derive with 2025 + 2026
      - weather: station data now exists, so a weather component is possible
      - capacity: revisit "car saturation" (not confirmed against road counts)
- [x] 3.5 Regression check on the 7 predefined scenarios: without `FAZZON_DATA_DIR` outputs are
      identical; with the pipeline statistics the sustainability fields are identical and the
      visitor samples move by < 0.2 visitors on average.
      Note for 3.4: the overall field is limited by the lakeside constraint, so the shuttle and
      mode-share scenarios (B1, B2, C1) give the same overall result as AS_IS (only the road
      sub-field changes).

## Phase 4: Pending analysis

- [ ] 4.1 Road-count window 12–15h: check for afternoon cars turned away on peak days.
- [ ] 4.2 2026 demand-adjusted estimate of the regime effect, once the 2026 road counts arrive.
- [ ] 4.3 Answers from domain experts (Phase 5) → update code/assumptions (`OPEN_QUESTIONS.md`).
- [ ] 4.4 (optional) Vodafone origin–destination (`vodafone-aixpa/od-mask_*`, 2024): share of
      visitors from the lower valley vs Tonale (what the Pellizzano counter misses).
- [ ] 4.5 Presentation material: select and polish plots for the Italian audience.

## Phase 5: Domain experts and data requests

- [ ] 5.1 Send `OPEN_QUESTIONS.md` to the domain experts (when possible).
- [ ] 5.2 Request 2026 road counts (PAT) and 2026 Vodafone presences.
- [ ] 5.3 Fold the answers back into code and assumptions (see 4.3).
- [ ] 5.4 Inform the backend developers of the Model API changes (3.2): the Model API needs the
      Fazzon data to start, new endpoint `POST /models/{model_key}/download-data` to wrap with
      admin authentication in the CRUD API, `digitalhub` dependency in `executor-backend`.
      Also (3.4): the Fazzon schema no longer has the `day_type` category nor the lakeside
      capacity, and the `lakeside` constraint is gone from the metadata (`mapper`, KPIs);
      the data is one dataitem (`fazzon_presence_season_stats.parquet`).

## Dependencies

- Phase 2 needs 1.1–1.2 (or `update.local_source_dir` to run from local files meanwhile).
- Phase 3.3 needs 3.1, 3.2 and 2.2. 3.4 can start on local data in parallel.
- 4.2 needs the 2026 road counts (5.2).
