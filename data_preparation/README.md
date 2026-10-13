# Data preparation pipeline

```
server -> Output/data/raw_data -> normalized -> data_processed -> final_data
   (0)                   (1)            (2)            (3)
```

The json **mappings** are in a single place, `Output/mapping/` (`paths.mapping`), used by the base build and by the update.

Every step reads the output of the previous one. **Directories, file format and platform settings are all in
`config/settings.yaml`** (the only file to edit for them; neither script takes directories as arguments).
`type_format` (`parquet` by default) applies to every file, in the base build and in the update.

## Layout

```
data_preparation/
├── README.md
├── gen_phenomenon_dataframes.py     # entry point: base build, ALL steps (download -> final parquet)
├── update_phenomenon_dataframes.py       # entry point: yearly update
├── config/
│   ├── settings.yaml                     # directories, file format, platform (edit here)
│   └── updates/                          # one yaml per yearly update: datasets, files, adapters
└── utils/
    ├── config.py          loads config/settings.yaml, exposes it as constants
    ├── cleaning.py        names / ids / frame helpers
    ├── io.py              read / write dataframes, output-dir guard
    ├── remote.py          S3 + DigitalHub (lazy imports: boto3, geopandas, digitalhub)
    ├── readers.py         low-level file readers
    ├── datasets.py        registry: datasets, value columns, references, phenomena
    ├── phenomena.py       processed -> phenomena computation
    ├── disaggregation.py  space / time disaggregation
    ├── steps/             download_data, standardize_raw_data, process_std_data, compute_phenomena (base build)
    │                      standardize_update_raw_data, process_update_std_data (update; compute_phenomena is shared)
    ├── flows.py           OD co-visits -> yearly flow indicators (areas -> comuni), tourists + visitors combined
    ├── profile_weights.py weights of TOURIST vs VISITOR learnt from the data (used by flows.py)
    ├── adapters/          year-specific layouts -> aligned layout (registry)
    └── update/            sources (readers), spec (config, parts)
```

## Base build: `gen_base_phenomenon_dataframes.py`

Runs all the steps, from the download to the final files:

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 Download | `utils/steps/download_data.py` (`download_raw_base_data`) | server (S3) | `Output/data/raw_data/` |
| 1 Standardization | `utils/steps/standardize_raw_data.py` | raw_data | `Output/data/normalized/` |
| 2 Processing | `utils/steps/process_std_data.py` | normalized (+ mappings in `Output/mapping`) | `Output/data/data_processed/` |
| 3 Phenomena | `utils/steps/compute_phenomena.py` | data_processed | `Output/data/final_data/` (+ platform with `--upload`) |

* **Download** saves data as they arrive: no renaming, no normalization. The mappings go to `Output/mapping/`, not to raw_data. The 2024 strutture ODS is part of the base build; later annual strutture files remain update inputs.
* **Standardization** uniforms column names, `DATA`/`LOCATION` schema, dates and comune names (no filtering/aggregation).
  Output: `popolazione_std`, `strutture_std`, `vodafone_std`, `ee_prelievi_std`, `gas_prelievi_std`,
  `vodafone_covisits_std`.
  The ISPAT arrivals / presences (alb, extralb) are **not normalized**. Electricity and gas category CSVs
  from `Energia/ee_prelievi/` and `Energia/gas_prelievi/` are each stacked into one normalized file;
  category rows remain separate at this step.
* **Processing**: `ID_COMUNE` resolution/padding, filtering, column selection, spatial/temporal disaggregation
  (comune x day). Output: `*_pr`. The presences are processed **in this order**:
  1. `vodafone`: areas mapping to more than one comune are split linearly to the beds (`tot_postiletto` of
     `strutture_pr`, same year or nearest); single-comune areas are unchanged.
  2. `presenze_alb` / 3. `presenze_extralb`: read as downloaded from `raw_data`
     (`Presenze/<arrivi|presenze>_<alb|exalb>_<year>.csv`; base build: 2022, 2023, updates: later years).
     alb: only the `<APT> - Totale` columns are used; extralb: only the first and the last column (provincial
     total); the `Anno` row and the empty row are dropped. Monthly area presences are distributed over the days
     with the **average stay** of the month (presences / arrivals): the arrivals of a month are uniform over its
     days and each arrival adds presences on the following days, also across the end of the month. Daily values
     are rescaled and rounded so that each area-month sums exactly to the official presences. Then area x day
     is split over comuni proportionally to the beds of the normalized strutture (`alberghieri posti_letto` for
     alb, `extra alb. Posti_letto` for extralb; same year or nearest). Arrivals are not in the output.
     The sums are checked (ValueError otherwise). Details: `utils/stay_distribution.py`.
  Every vodafone / ISPAT step checks that summing back gives the original totals.
  `vodafone_covisits` (OD co-visits between Vodafone areas) is aggregated **per year** into flow indicators, see below.
* **Phenomena**: `phen_popolazione`, `phen_strutture`, `phen_presenze` (alb + xalb + vodafone unified).
  `phen_consumption` contains daily municipality usage in `ee_services` (kWh) and `gas_services` (Smc).
  Categories are summed to monthly totals during processing, then each monthly total is divided evenly
  across the calendar days in that month (including leap days). The two measures are combined on `DATA`
  + `ID_COMUNE`. The format of the yearly update of these datasets is not known yet: they are registered
  in the update machinery with a **placeholder** (`energia_placeholder`, see `utils/adapters/energia.py`),
  which the config validation refuses with a clear message until a real adapter is written.
* **Flow indicators** (`phen_flussi_od`): see the next section.

### Flow indicators (`vodafone_covisits` -> `phen_flussi_od`)

The Vodafone OD co-visits (`vodafone_covisits.csv`: day x source area x target area, `covisitType` ND / UD,
profile, country) are turned into **yearly** indicators per comune. The computation is the one of
`notebooks/flow_indicators.ipynb`, in `utils/flows.py`; thresholds and filters are constants in `utils/datasets.py`
(`COVISITS_*`).

| Step | What happens |
|------|--------------|
| 1 Standardization | the file is read in chunks and only the rows used by the indicators are kept: area level (`TN_MKT_AL_3`), `TOURIST` / `VISITOR`, empty `userCountry` (total over the countries). Output `vodafone_covisits_std` (`DATA`, source/target id and ambito, `covisitType`, `userProfile`, `value`). |
| 2 Processing | per year and profile: quota in ingresso `UD(*->j)/T(j)`, quota in uscita `UD(j->*)/T(j)`, quota intra-ambito, PageRank of the UD network; then areas -> comuni (an area passes its value to all its comuni, a comune with several areas gets the weighted mean). Output `vodafone_covisits_pr`: `DATA` (year), `ID_COMUNE`, `<metric>_<tourist, visitor or combined>` for the 4 metrics, `giorni_coperti`. The `_combined` columns merge the two profiles with a weight each, see below. |
| 3 Phenomena | `phen_flussi_od` = `vodafone_covisits_pr` |

* A year with fewer than `COVISITS_FULL_YEAR_DAYS` (360) days is **partial**: it is kept, flagged by `giorni_coperti` and by a warning.
* The mobility shares are `NaN` where `T < COVISITS_MIN_T`; the shares can exceed 1 (users counted in several areas).
* In the **update** the indicators are yearly, so they are recomputed on all the days of the year: the base
  `vodafone_covisits_std` + every part of the update (on common dates the later data win). Only the years touched by the
  update are written in `vodafone_covisits_update_pr`. Config example:
  ```yaml
  vodafone_covisits:
    sources:
      - file: vodafone_covisits_new.csv
        reader: csv
        reader_kwargs: {dtype: str, keep_default_na: false}
        adapter: vodafone_covisits_raw       # columns of KIND_SCHEMAS["vodafone_covisits"]
  ```
* In the backend the four indicators are `quota-mobilita-ingresso`, `quota-mobilita-uscita`, `quota-intra-ambito`
  and `pagerank-discretizzato` (PageRank in quantile classes, 10 by default: `PAGERANK_N_CLASSES` in
  `trentino_indicators.py`). They read the `_combined` columns; the tourist / visitor ones are returned next to them.

#### Tourists and visitors: weighted union (`*_combined`)

A tourist-day (overnight stay) and a visitor-day (excursionist) do not weigh the same on a territory, so the
indicators of the backend use their **weighted union**. The weights are applied on the user-days BEFORE the indicators
are computed (`utils/flows.py: combine_profiles`): with weights `w_T = 1`, `w_V`, the shares are
`sum_p w_p UD_p / sum_p w_p T_p` (a profile counts in proportion to its volume AND its weight) and the PageRank is
computed on the network of the weighted flows. With equal weights this is the plain pooled share; with `w_V = 0` it is
the tourist indicator. The reliability threshold `COVISITS_MIN_T` always refers to the raw user-days.

How the weight is chosen (`COVISITS_WEIGHT_MODE` in `utils/datasets.py`):

* `"estimate"` (default): learnt from the data in `utils/profile_weights.py`. The weight of a profile is its marginal
  pressure on the comune relative to the tourists, measured through the monthly consumption of electricity and gas
  **by usage category** (`codice_utilizzo`, roles in `COVISITS_WEIGHT_CATEGORIES`): non-negative regression of the
  consumption of each comune-month on the user-days of the two profiles, net of the size of the comune and of the month
  (two-way fixed effects, fixed-point on the size), uncertainty by cluster bootstrap over the comuni.
  * `shared` categories (553 restaurants, 602 transport, 920 recreation, 930 other services: used by both profiles) are
    summed per carrier and give the weight;
  * `tourist_only` categories (551 hotels) are a **placebo**: the visitors must not explain them; if they do, the model
    mixes up the profiles and the fixed weights are used instead (hotels are not in the estimate, otherwise the visitors
    would be penalised just because they do not sleep there);
  * each shared category is also fitted alone (`diagnostics.diagnostic_targets`) to see whether they agree.
  Targets that are not reliable (too few comuni, effect of the tourists not clearly positive) are discarded, the others
  averaged with inverse-variance weights; if none is reliable the `COVISITS_FIXED_WEIGHTS` are used and the reason is
  logged. The pipeline warns about configured usage codes that are not in the data. Result, with the coefficients, the 95%
  intervals, the within R2, the placebo and the collinearity of the two presences, in
  `Output/data/data_processed/covisits_profile_weights.json` (**read it after the base build**: `source` must be
  `estimated`, the placebo `passed`, and the interval of the visitor weight narrow).
* `"fixed"`: `COVISITS_FIXED_WEIGHTS` (e.g. `{"TOURIST": 1.0, "VISITOR": 0.3}`), if you prefer to decide the weights.
* The weights are estimated **once, by the base build**, and saved: the yearly updates reuse them, so that the years of the
  base build and of the updates stay comparable. To re-estimate them, rerun the base build (all the years change).
* Why energy consumption, what the weight means and what its limits are: **decision record** at the top of
  `utils/profile_weights.py` (read it before changing the method). In short: it is the only per-comune demand-side
  signal available and it is roughly additive per guest, but it measures accommodation-type pressure, not mobility
  pressure, and it is an indirect proxy. If you cannot defend the estimate, use `"fixed"` weights; other proxies
  (traffic, waste, ...) can be added through the same monthly interface (`build_energy_targets` / `estimate_profile_weights`).

```bash
python -m gen_base_phenomenon_dataframes                   # all steps, parquet
python -m gen_base_phenomenon_dataframes --skip-download   # reuse Output/data/raw_data
python -m gen_base_phenomenon_dataframes --upload          # also log final data to the platform
```
Each step can also be run alone, e.g. `python -m utils.steps.process_std_data`.

## Yearly update: `update_phenomenon_dataframes.py`

Same steps of the base build, with update-specific versions of standardize and process:

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 Download | `utils/steps/download_data.py` (`download_update_raw_data`) | server (S3) or `update.local_source_dir` | `Output/data/raw_data/` + `Output/mapping/` |
| 1 Standardization | `utils/steps/standardize_update_raw_data.py` | raw_data | `Output/data/normalized/<dataset>_update_std` |
| 2 Processing | `utils/steps/process_update_std_data.py` | normalized (+ mappings) | `Output/data/data_processed/<dataset>_update_pr` |
| 3 Phenomena | `utils/steps/compute_phenomena.py` (`datasets=[...]`) | `<dataset>_update_pr` + existing `final_data` | `Output/data/final_data/` (appended) |

* **Download**: `download_raw_base_data` and `download_update_raw_data` live in the same module and share
  `download_mappings`: the mapping json files and the geojson go to `Output/mapping/` in both cases.
  The update downloads the files listed in the update config.
* **Standardization**: for each source it first **checks that the column structure is compatible with the
  expected one** (the adapter of the source, then the columns required by its kind, `KIND_SCHEMAS`), renames the
  columns to the standard names, and applies **the same standardize functions of the base build**.
  The result is saved with the standard name `<dataset>_update_std` (`<dataset>_<label>_update_std` for labeled sources).
* **Processing**: the same `process_*` functions of the base build, so `<dataset>_update_pr` has the same format of
  `<dataset>_pr` (`DATA`, zero-padded `ID_COMUNE`, value columns; comune x day for the presences).
  Sources of the same dataset are stacked, later ones win on a duplicated `DATA` + `ID_COMUNE`.
* **Phenomena**: `compute_phenomena` creates the phenomenon dataframes in the base build; in the update it
  **appends** the new ones to the existing files in `final_data`. On a duplicated `DATA` + `ID_COMUNE` the
  update row is kept. A phenomenon is computed only if **all** its datasets are in the update
  (`phen_presenze` needs `vodafone`, `presenze_alb` and `presenze_extralb`); the others are skipped with a warning.
  The final data of the base build must already exist.

```bash
python -m update_phenomenon_dataframes --config data_preparation/config/updates/update_2026.yaml
python -m update_phenomenon_dataframes --config ... --skip-download   # reuse raw_data and mappings
python -m update_phenomenon_dataframes --config ... --upload          # also log final data to the platform
```
To read source files from a local folder instead of S3 set `update.local_source_dir` in `settings.yaml`.
`--no-strict` turns an adapter schema mismatch into a warning. Directories, reference files and format come from
`settings.yaml`, datasets from the update yaml. From Python: `update_pipeline(config)`.
Each step can also be run alone: `python -m utils.steps.process_update_std_data --config ...`.

### The update config (`config/updates/<round>.yaml`)

Nothing year-specific is in the code: **which datasets**, file names, years, column names and the adapter to use
are here (`update_2025.yaml` reproduces the previous hard-coded behaviour).

```yaml
datasets:                                      # only these are updated
  popolazione:
    enabled: false                             # optional (default true): skip it this round
    sources: [...]
  strutture:
    sources:                                   # several sources are stacked (e.g. one per year)
      - file: numero_strutture_ISPAT_2026.xlsx
        reader: excel                          # csv | excel | ispat_apt_tsv | ispat_grouped_tsv
        reader_kwargs: {header: [0, 1]}        # passed to the reader
        adapter: strutture_annuario            # default handler: 2025 two-row-header layout
        adapter_kwargs: {year: 2026}
        # label: "2026"                        # optional: keeps this source separate until the processed parts are stacked
```
      Sources without a label share the `<dataset>_update` part. A labeled source is kept separate during processing
      (e.g. two strutture years) and parts of the same dataset are stacked in `<dataset>_update_pr` (later sources win on overlap).
To update a subset, list only those datasets or set `enabled: false` on the others; their final data remain unchanged. A phenomenon is appended only if all its datasets are in the update (see Step 3).

### Disclaimer

For now data have been retrived on request and we cannot make assumptions about the structures files will 
have in future versions for yearly updates.

For this reason update_phenonmenon_datafrems requires for year 2025 some temporary files generated by the previous step.

### Outputs

Update files are kept next to the base ones: raw sources in `raw_data`, `<dataset>_update_std` in `normalized`,
`<dataset>_update_pr` in `data_processed` (the base `<dataset>_pr` are **not** modified), and the final
phenomenon files (`phen_popolazione`, `phen_strutture`, `phen_presenze`, `phen_flussi_od`) in `final_data`, with the update rows appended.

### Strutture and popolazione

* **strutture**: the base build combines the historical CSV with the 2024 ODS variant, using
  `strutture_annuario_2024` (`Comuni`). Yearly updates use the default `strutture_annuario` adapter, which
  expects the 2025 XLSX layout (two-row header and `Comune`). If a file does not match its selected
  handler, validation stops and asks for a dedicated adapter rather than guessing its layout. The adapters rename
  columns (`Alloggi turistici` → `all. privati`, `Alloggi a disposizione` → `all. disposizione`,
  `Totale` → `tot convenzionali`) and converts `-` to 0. In processed data (base and update) the raw
  `tot convenzionali strutture / posti_letto` are **discarded**: conv is recomputed as alberghieri + extralberghieri,
  total = conv + non conv (`all. privati`).
* **popolazione**: the value is the column `Popolazione residente al 1.1.<year>` (`year: 2025` → second column of the file).

### A new year arrives

1. Upload the raw files to the data lake.
2. Copy `config/updates/update_2025.yaml` → `update_2026.yaml`; change file names, `year`, column names.
3. If the provider changed a layout, add an adapter (function + decorator) in `utils/adapters/` — nothing else:
   ```python
   @register_adapter("popolazione_ispat_2027", kind="popolazione")
   def popolazione_ispat_2027(raw: RawData, *, year: int) -> pd.DataFrame:
       return raw.df.rename(columns={"Territorio": "comune", "Residenti": "popolazione"}).assign(anno=year)
   ```
   The output must contain the columns of its `kind` (`KIND_SCHEMAS` in `utils/adapters/base.py`); a mismatch
   stops the standardization step with a clear error. Config errors (unknown dataset/adapter/reader, wrong adapter
   parameters) are reported before anything is downloaded.
4. Run the update and inspect `Output/data/data_processed/<dataset>_update_pr` and the appended phenomena in `Output/data/final_data/`.



