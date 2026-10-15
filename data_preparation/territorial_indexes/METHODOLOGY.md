# Methodology: how the phenomena are generated (gen) and updated (update)

```
server -> raw_data -> normalized -> data_processed -> final_data
 (0)         (1)          (2)             (3)
```

Same four steps for the base build (`gen_phenomenon_dataframes.py`) and for the yearly update
(`update_phenomenon_dataframes.py`). Directories, file format and the **time window** are in
`config/settings.yaml`; the decisions behind each step are in [ASSUMPTIONS.md](ASSUMPTIONS.md) (IDs in brackets).

## Time window

| | from | to | content |
|---|---|---|---|
| gen | 2022-01-01 | 2024-12-31 | creates the phenomena |
| update 2025 | 2025-01-01 | 2025-12-31 | appends the 2025 rows to the existing phenomena |

Nothing before 2022-01-01 survives anywhere (D1). The window is applied in step 2 on every dataset (before the heavy
computations, so that e.g. 2025 days do not enter the base build or the estimate of the profile weights), and a last
time in step 3 on the final phenomena. Yearly data are kept by year, daily data by date.

## Step 0 – Download

Raw files are copied exactly as they are from the data lake (or from `update.local_source_dir`) to `raw_data/`;
the json mappings and the geojson of the areas go to `mapping/`. The base build downloads: popolazione 2020–2024,
strutture (Annuario csv + 2024 ods), Vodafone presences (`vodafone_attendences` + `_new`, A1), Vodafone OD co-visits,
the ISPAT arrivals / presences 2022–2024 (A2), the energy category files. The update downloads the files listed in its
yaml (`config/updates/update_2025.yaml`). Energy files keep their folder (`ee_prelievi/`, `gas_prelievi/`) because the
two carriers have identically named files.

## Step 1 – Standardization (`<dataset>_std`)

Uniform names and format, **no filtering and no aggregation**: columns `DATA` / `LOCATION`, comune names cleaned,
Vodafone area ids converted to names (Vigo + Pozza di Fassa → San Giovanni di Fassa), energy categories stacked,
OD co-visits reduced to what the indicators use (area level, TOURIST / VISITOR, total over the countries; read in chunks).
In the update every source first goes through its **adapter**, which checks the layout (a mismatch stops the run)
and renames the columns to the standard ones; then the same standardization functions of the base build are applied.
The ISPAT arrivals / presences are not normalized (only their layout is checked).

## Step 2 – Processing (`<dataset>_pr`, `<dataset>_update_pr`)

The update calls the **same functions** as the base build, so a `_update_pr` has exactly the format of its `_pr`.
Every output has `DATA` and `ID_COMUNE` (6 digits, one comune per row) plus the value columns.

| dataset | granularity out | how |
|---|---|---|
| popolazione | comune × year | resident population at 1 January |
| strutture | comune × year | beds and structures, conventional (alb + extralb) and not; totals recomputed (A5) |
| vodafone | comune × day | TOURIST presences per area × day; areas with several comuni split by beds of the year (A7); sums checked after the split |
| presenze_alb / presenze_extralb | comune × day | ISPAT monthly arrivals and presences per APT / province → daily presences through the average length of stay (A8), then split among comuni by `alberghieri` / `extra alb.` beds; each month adds up exactly to the official figure |
| ee_prelievi / gas_prelievi | comune × day | categories summed per comune-month, spread evenly over the days (A9) |
| vodafone_covisits | comune × year | flow indicators: share of incoming and outgoing users, share of intra-ambito flow, PageRank, for TOURIST, VISITOR and a weighted `_combined`; `giorni_coperti` = days with data (A10) |

Presences and beds: for a day, the beds of its year are used; if the year is missing the nearest one (with a warning).
With the window of D2 the base build never needs 2025 beds; the update processes the 2025 strutture first and uses them
for the 2025 Vodafone presences (U5).

**Profile weights (flussi OD).** The base build estimates, from monthly energy consumption by usage category, how much a
visitor-day weighs compared with a tourist-day, and saves it in `covisits_profile_weights.json` (fallback: fixed
weights). The update loads that file instead of re-estimating, so the years stay comparable.

**Update specifics.** The update only processes the rows of its window: `process_update_std_data` receives the window
and passes it to every processor. The co-visit indicators of a year need all its days, so the base standardized
data of the same year (none for 2025) and the update parts are stacked (the later file wins on a duplicated date) and cut
to the year of the update.

## Step 3 – Phenomena (`final_data/phen_*`)

| phenomenon | built from | rule |
|---|---|---|
| `phen_popolazione` | popolazione | as is |
| `phen_strutture` | strutture | as is |
| `phen_presenze` | vodafone, presenze_alb, presenze_extralb | **inner join** on DATA + ID_COMUNE; the dropped records (e.g. days not covered by Vodafone) are counted in warnings |
| `phen_consumption` | ee_prelievi, gas_prelievi | outer join on DATA + ID_COMUNE |
| `phen_flussi_od` | vodafone_covisits | as is |

* **gen**: the phenomena are created from the `_pr` files and cut to 2022-01-01 .. 2024-12-31.
* **update**: a phenomenon is recomputed only if **all** its input datasets are in the update (hence all the datasets
  of the 2025 yaml); the new rows are appended to the existing file (the update wins on a duplicated key, D5), and the
  result is cut to `period.start` .. end of the update window. `--upload` also logs the phenomena to the platform.

## Running

```
python -m gen_phenomenon_dataframes [--skip-download] [--upload]
python -m update_phenomenon_dataframes --config config/updates/update_2025.yaml [--skip-download] [--upload]
```

Run from `territorial_indexes/`. Run gen first (same `Output/`, same `type_format`), then the update. To move the window edit `period` in
`config/settings.yaml` (or add `period: {start, end}` to the update yaml); to update a new year write a new yaml.
Tests: `python -m pytest tests` (from `territorial_indexes/`).
