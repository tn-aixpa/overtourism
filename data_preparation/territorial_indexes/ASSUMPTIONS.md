# Territorial indexes (gen / update of the phenomena): decisions and assumptions

Register of the decisions (`D*`) and assumptions (`A*`) of the pipeline that builds the phenomenon dataframes:
`gen_phenomenon_dataframes.py` (base build, "gen") and `update_phenomenon_dataframes.py` (yearly update).
The steps are described in [METHODOLOGY.md](METHODOLOGY.md). Where a value is configurable the file is named.

Status: **in use** · **to verify** (written without being able to check it against the real files / data lake: confirm
it before trusting the 2025 numbers) · **retired**.

## Time window

| ID | Decision | Reason / effect | Where | Status |
|----|----------|-----------------|-------|--------|
| D1 | **Nothing before 01/01/2022 is kept**, in the base build and in the update. | Earlier years are discarded (the older strutture were also not comparable, municipal mergers). Applied in step 2 (processing) and again on the final phenomena, so an old `final_data` file cannot bring older rows back. | `config/settings.yaml` `period.start`; `utils/window.py`; `process_std_data.py`; `create_phenomena_df.py` | in use |
| D2 | **gen (base build) uses data up to 31/12/2024** (included). | Whatever the raw files contain after that date (2025 days of the Vodafone files, 2025 energy, ...) is ignored by the base build and is left to the update. This also removes the warning `[beds_weights] no strutture for 2025: using 2024`: the base build no longer processes 2025 days that have no 2025 strutture. | `period.base_end` | in use |
| D3 | **update covers 2025 only**, for ALL the phenomena: popolazione, strutture, presenze (Vodafone + ISPAT alb + ISPAT extralb), consumption (electricity + gas), flussi OD. | The processed update files and the rows appended to the base phenomena only contain 2025. Rows of other periods in the update files (e.g. Vodafone 2023–2024, in the same csv) are ignored. An update that finds no data in its window stops with an error (instead of silently appending nothing). | `period.update_start/update_end`; optional `period: {start, end}` in the update yaml (never before `period.start`) | in use |
| D4 | Yearly data (DATA = `YYYY`: popolazione, strutture, flussi OD) are kept if their **year** is inside the window; daily data by exact date, both ends included. | Window boundaries are whole years in the current settings; with a boundary inside a year a yearly row of that year is kept. | `utils/window.py` | in use |
| D5 | The update **replaces** rows with the same DATA + ID_COMUNE and **appends** the new ones to the existing phenomena; the earlier rows (2022–2024) are untouched. | Reruns of the update are idempotent. | `create_phenomena_df.append_to_existing` | in use |

## Base build (gen)

| ID | Assumption | Evidence / reason | Where | Status |
|----|------------|-------------------|-------|--------|
| A1 | Vodafone presences of 2024-04-01 .. 2024-12-31 come from `vodafone_attendences_new.csv` (2023-01-01 .. 2025-12-31). The base build reads it together with `vodafone_attendences` (2022-01-01 .. 2024-03-31); on the dates present in both files **the newer file wins**. | `vodafone_attendences` stops at 2024-03-31, so without it the base would end in March 2024 (the final presences are an inner join of Vodafone and ISPAT) . The two files are identical on 2023 (Molveno ASSUMPTIONS D8). The file is added to the base download list. | `standardize_raw_data.read_base_vodafone`; `download_data.S3_DATA` | to verify |
| A2 | ISPAT arrivals / presences (alb and extralb) are read for **every year of the window** (2022, 2023, 2024): `Presenze/<arrivi\|presenze>_<alb\|exalb>_<year>.csv`. | `PRESENZE_BASE_YEARS` is derived from the window. The 2024 files were in the former update (`update_2025.yaml`); they are now downloaded by the base build. | `datasets.PRESENZE_BASE_YEARS` | to verify |
| A3 | The 2022 stays that cross the 2021/2022 boundary are approximated (no 2021 files, warning). | Spill-over of the stays uses the previous year's files when present. | `process_presenze_stay` | in use |
| A4 | Popolazione of year Y is the resident population at 1 January of Y. | `popolazione_ispat_1jan` | `adapters/popolazione.py` | in use |
| A5 | Strutture: CONV = alberghieri + extralberghieri, NON-CONV = alloggi privati + a disposizione; the totals are recomputed from the components, not read from the raw totals. | Consistency between years / layouts. | `process_strutture` | in use |
| A6 | Vodafone: only `TOURIST` and `locType = TN_MKT_AL_3` rows; `VIGO DI FASSA` and `POZZA DI FASSA` are unified in `SAN GIOVANNI DI FASSA` (22250). | The comune of Fassa is a merger. | `standardize_vodafone_columns`, `process_vodafone` | in use |
| A7 | An area that covers several comuni is split **proportionally to the beds** (`tot_postiletto`) of the year; ISPAT alb / extralb are split by `alberghieri` / `extra alb.` beds. Without beds of the year, the nearest year is used (earlier on a tie, warning); without beds at all, uniform split. | Beds are the only supply measure available per comune. Rounding with the largest-remainder method keeps the totals exactly. | `disaggregation.beds_weights` | in use |
| A8 | ISPAT presences: the monthly arrivals are spread uniformly over the days of the month; a guest who arrives on day d stays `s = presences / arrivals` nights (also across the end of the month); each month is then rescaled to the official monthly presences. | A uniform spread of the presences would ignore the length of stay. | `stay_distribution.py`, `PRESENZE_SPILL_OVER` | in use |
| A9 | Energy: all usage categories (`codice_utilizzo`) are summed per comune-month, then divided **evenly over the calendar days** of the month. Comuni without an ID are dropped (warning). | Only monthly data are available. | `process_energy` | in use |
| A10 | Flussi OD: only area-level rows (`TN_MKT_AL_3`), profiles TOURIST and VISITOR, total over the countries (`userCountry` empty). Indicators are yearly per comune; `_combined` columns merge the profiles with **fixed weights, TOURIST 1 and VISITOR 0.5**. An area covering several comuni gives its value to each of them. | Simple and transparent; same weights in base build and update, so years are comparable. 0.5 is a chosen coefficient, not estimated. | `datasets.COVISITS_PROFILE_WEIGHTS`, `flows.build_area_comune`, `process_covisits` | in use |

## Update (2025)

| ID | Assumption | Evidence / reason | Where | Status |
|----|------------|-------------------|-------|--------|
| U1 | **Energy 2025**: the files of the update have the same layout as the base ones (semicolon csv, year × comune × usage category, one column per month) and are the objects `Energia/<ee_prelievi\|gas_prelievi>/{49_3,55_1,56_1,93_1,96}.csv`; the 2025 rows are read from them (same names as the base files). | The format of the energy update was undefined (placeholder adapter). If the 2025 data arrive in other files, edit the `sources` of `ee_prelievi` / `gas_prelievi` in `config/updates/update_2025.yaml`. | `adapters/energia.py` (`energia_categorie`), `sources.read_energy_csv` | **to verify** |
| U2 | **Flussi OD 2025**: the 2025 co-visits are in the object `vodafone_covisits.csv` (the same of the base build), read in chunks and filtered to the update period. | The update yaml had no co-visits source. If a separate file is delivered, change `file` in `update_2025.yaml`. | `sources.read_covisits_csv` | **to verify** |
| U3 | The yearly flow indicators of 2025 are computed on all the 2025 days (base std data of earlier years are not touched); the same fixed profile weights (A10) are used. A partial year (< 360 days) is flagged by `giorni_coperti` and a warning. | Comparability across years. | `process_update_std_data._process_covisits` | in use |
| U4 | **Presenze 2025**: only the 2025 ISPAT files are in the update yaml; the 2024 files (downloaded by the base build in the same `raw_data` folder) give the stays that cross the new year. If they are missing the stays crossing 31/12/2024 are approximated (warning). | The 2024 sources moved to the base build (A2). | `_process_presenze` | in use |
| U5 | **Vodafone 2025** is in `vodafone_attendences_new.csv`; areas are split with the 2025 beds (the 2025 strutture are processed first in the same update). | If `strutture` is not part of the update, the 2025 beds are missing and the 2024 beds are used (this is the origin of the old warning). Keep `strutture` enabled. | `process_update_std_data` | in use |
| U6 | Popolazione and strutture of 2025 keep their files and adapters (`popolazione_2026_ISPAT.csv`, `numero_strutture_ISPAT_2025.xlsx`). | unchanged | `update_2025.yaml` | in use |
| U7 | The base build must have been run (in the same `Output/` folder, same `type_format`) before the update. | The update appends to `final_data`, reads the base normalized strutture / co-visits and the 2024 ISPAT files. | `update_phenomenon_dataframes.py` | in use |

## Not changed / out of scope

* The two model pipelines (`../model/molveno`, `../model/fazzon`) have their own registers and are not covered here.
* The raw files and the data lake were not available while making this change: the new window logic is covered by
  unit tests on synthetic data (`tests/test_period_window.py`), the full pipeline has **not** been run on the real data.
