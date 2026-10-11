# Molveno data preparation

Reference datasets for the Molveno model. Same structure as the Fazzon pipeline
(`../fazzon/README.md`), reusing `model/common/` (sources, weather readers), the main pipeline's
`utils` and `config/settings.yaml`. Run from the `data_preparation` folder:

```bash
python -m model.molveno.prepare                  # steps 0-2
python -m model.molveno.prepare --skip-download  # reuse <normalized>/molveno/ (step 2 only)
python -m model.molveno.prepare --refresh        # download the data-lake files again
python -m model.molveno.prepare --upload         # also log the reference datasets to the platform
```

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 Sources | `model/common/sources.py` | data lake or `local_source_dir` (`molveno.yaml`) | raw files |
| 1 Standardize | `standardize.py` | raw files | `Output/data/normalized/molveno/*.parquet` |
| 2 Reference datasets | `reference_datasets.py` | normalized tables | `Output/data/final_data/molveno/*.parquet` |
| 3 Upload | `utils.remote.log_dataframe` | reference datasets | platform dataitems (`--upload`) |

* **Sources** (`molveno.yaml`): Vodafone presences (`vodafone_attendences_new.csv`) and the
  Meteotrentino bulletin per comune (`meteo_osservato.csv`), both already on the data lake.
* **Standardize**: `vodafone_presences` (Molveno tourists and excursionists per day),
  `meteo_bulletin` (Molveno).
* **Reference datasets**: `molveno_daily` (presences, weekly trend and deviation, season of the
  reference days, weather description), `molveno_season_stats`, `molveno_weekday_stats`.
  The statistics currently used by the Molveno model; see `ASSUMPTIONS.md`.

Every decision and assumption, with its evidence and status: `ASSUMPTIONS.md`.
`local-docs/` holds local working notes and is never versioned.
Tests: `tests/test_molveno_pipeline.py` (synthetic fixtures, no data needed).
