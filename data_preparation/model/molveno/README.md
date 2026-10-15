# Molveno data preparation

Reference datasets for the Molveno model. Same structure as the Fazzon pipeline
(`../fazzon/README.md`), reusing `model/common/` (sources, weather readers), the main pipeline's
`model/common/config.py`, `model/common/io.py` and `model/config/settings.yaml` (directories, platform). Run from the `data_preparation` folder:

```bash
python -m model.molveno.prepare                  # steps 0-2
python -m model.molveno.prepare --skip-download  # reuse <normalized>/molveno/ (step 2 only)
python -m model.molveno.prepare --refresh        # download the data-lake files again
python -m model.molveno.prepare --local-sources model/molveno/data  # local copies instead of the data lake
python -m model.molveno.prepare --upload         # also log the reference datasets to the platform
python -m model.molveno.estimator_comparison     # validation of the presence estimators
```

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 Sources | `model/common/sources.py` | data lake or `local_source_dir` (`molveno.yaml`) | raw files |
| 1 Standardize | `standardize.py` | raw files | `Output/data/normalized/molveno/*.parquet` |
| 2 Reference datasets | `reference_datasets.py`, `presence_estimators.py` | normalized tables | `Output/data/final_data/molveno/*.parquet` |
| 3 Upload | `model.common.remote.log_dataframe` | reference datasets | platform dataitems (`--upload`) |

* **Sources** (`molveno.yaml`): Vodafone presences (`vodafone_attendences_new.csv`) and the
  aggregated foreign presences (`vodafone_attendences_STR_new.csv`); the measured weather of
  Meteotrentino station T0414 (San Lorenzo in Banale), fetched by
  `data_download/download_meteo_meteotrentino.py T0414` and uploaded to `molveno/meteo/`.
* **Standardize**: `vodafone_presences` (Molveno tourists and excursionists per day: Italians
  plus the foreign total), `meteo_station`.
* **Reference datasets**:
  * `molveno_daily`: one row per day, presences, calendar, measured weather, weather class.
  * `molveno_presence_stats`: the presence statistics of the model, per summer calendar day
    (`month_day`) x weekday x weather class (good / cool / rainy): mean and SD of tourists and
    excursionists, estimated by gradient boosting on the summers 2023–2025.
  * `molveno_presence_stats_monthly`: the comparison statistics, per month x weekday x weather
    class (least squares).
  * `molveno_weather_freq`: per summer month, the frequency of each weather class.
* **Validation** (`estimator_comparison.py`): leave-one-year-out scores of the two estimators and
  of the alternatives considered; tables in `data/analysis_output/` (not versioned).

Every decision and assumption, with its evidence and status: `ASSUMPTIONS.md`.
`local-docs/` holds local working notes and is never versioned.
Tests: `tests/test_molveno_pipeline.py` (synthetic fixtures, no data needed).
