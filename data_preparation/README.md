# data_preparation

Data preparation for the Overtourism project: it turns the raw files of the data lake into the datasets used by the
backend. The folder holds two independent pipelines that write in the same `Output/` folder.

```
data_preparation/
├── README.md                  this file
├── pyproject.toml, uv.lock    dependencies (uv), shared by the two pipelines
├── Output/                    everything the pipelines write (not versioned): mapping/, data/{raw_data,normalized,data_processed,final_data}
├── territorial_indexes/       territorial indexes: base build (gen) + yearly update of the phenomena
│   ├── README.md              how the pipeline works, steps, configuration, how to run it
│   ├── METHODOLOGY.md         the steps of gen / update, in short
│   ├── ASSUMPTIONS.md         decisions and assumptions (time window, files, methods)
│   ├── gen_phenomenon_dataframes.py, update_phenomenon_dataframes.py
│   ├── config/                settings.yaml (directories, format, period, platform), updates/<round>.yaml
│   ├── utils/                 the code of the pipeline (steps, adapters, flows, disaggregation, ...)
│   └── tests/
└── model/                     reference datasets of the local models
    ├── common/                code shared by the models: config, io, remote (data lake / platform), sources, weather
    ├── config/settings.yaml   directories and platform of the models
    ├── molveno/               Molveno model (README.md, ASSUMPTIONS.md, molveno.yaml)
    ├── fazzon/                Fazzon model (Lago dei Caprioli; README.md, ASSUMPTIONS.md, fazzon.yaml)
    └── tests/
```

## The two pipelines

| | Territorial indexes | Models |
|---|---|---|
| What | `phen_popolazione`, `phen_strutture`, `phen_presenze`, `phen_consumption`, `phen_flussi_od` per comune (yearly / daily) | reference datasets (daily presences, calendar, weather, ...) of Molveno and Fazzon |
| Period | gen: 2022-01-01 .. 2024-12-31; update: 2025 | defined by each model (see its ASSUMPTIONS.md) |
| Run from | `territorial_indexes/` | `data_preparation/` |
| Entry points | `python -m gen_phenomenon_dataframes`, `python -m update_phenomenon_dataframes --config config/updates/update_2025.yaml` | `python -m model.molveno.prepare`, `python -m model.fazzon.prepare` |
| Settings | `territorial_indexes/config/settings.yaml` | `model/config/settings.yaml` |
| Documentation | `territorial_indexes/README.md`, `METHODOLOGY.md`, `ASSUMPTIONS.md` | `model/<name>/README.md`, `ASSUMPTIONS.md` |

Both follow the same flow: `server (data lake) -> raw_data -> normalized -> processed / reference -> final_data`
(the models write in a sub-folder of their own, e.g. `final_data/molveno`).

### Shared and duplicated code

The two pipelines do not import each other. What they both need is **duplicated on purpose**, each copy trimmed to its use:

| Territorial indexes | Models | Content |
|---|---|---|
| `territorial_indexes/utils/config.py` | `model/common/config.py` | loads the settings, exposes directories and platform constants. The territorial one also has references, datasets, phenomena and the time window |
| `territorial_indexes/utils/io.py` | `model/common/io.py` | writing of dataframes (the territorial one also reads and has the raw-file path helper) |
| `territorial_indexes/utils/remote.py` | `model/common/remote.py` | S3 / DigitalHub (the model copy has only `init_s3` and `log_dataframe`) |
| `territorial_indexes/config/settings.yaml` | `model/config/settings.yaml` | `output_dir`, `paths.raw / normalized / final` and `platform` must be the same in the two files |

Everything else in `territorial_indexes/utils/` (cleaning, adapters, disaggregation, flows, steps, ...) is used by the
territorial indexes only; `model/common/` (`sources.py`, `weather.py`) by the models only. If you change a shared
constant (a directory, the platform project, the bucket) change it in **both** settings files.

## Setup

```bash
uv sync                      # from data_preparation/ (pyproject.toml, uv.lock); extras: cloud, download
```
S3 and platform access need the credentials of the DHCLI in `~/.dhcore.ini` (section `platform.dhcore_env` of the settings).

## Tests

```bash
cd territorial_indexes && python -m pytest tests     # time window, config files
cd .. && python -m pytest model/tests                # Molveno and Fazzon pipelines
```
No test needs the data lake or the real data.

## Data

Raw and generated data are not versioned: `Output/` is created by the pipelines. The raw files come from the data lake
(bucket and prefix in the settings); external sources are fetched by their own scripts and uploaded to the data lake.
