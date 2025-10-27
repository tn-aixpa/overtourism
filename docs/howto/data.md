# Prepare Overtourism Digital Twin Model Datasets

The model relies on a set of elaborations derived from the raw data collected for the territory and referring to presence statistics from mobile operators, parking repos, etc. The code of the repository includes the corresponding elaborations and in the current version requires that the input raw data is made available in the same datalake as the platform uses. More specifically, 

- all the input data should be placed as is under some S3 path within the datalake. The path is configured by the environment variable `DATA_PREFIX` and is used by the data processing scripts to read data from.
- the intermediate data is being processed by the procedure and is stored locally at the path defined by `BASE_DIR` environment variable. This is necessary to allow for mounting explicit volumes in case there are limitations on the default disk space available to the executed code.

## Running the data preparation process

To run the data preparation procedure, it is possible to use `process_data.py` script available in this repository. The scripts runs either locally or as a Job in the platform and requires that the environment is configured accordingly. If run locally, it is possible to specify to which platform project the elaboration and the resulting data artifacts should be associated (`PROJECT_NAME` environment variable) and the name of the platform environment to use the configuration from if CLI is used (`CLI_ENV` variable).

To run within the platform it is necessary to declare the function, e.g., as follows

```python
func = project.new_function(
    name="process_data",
    kind="python",
    python_version="PYTHON3_12",
    source="git+https://github.com/tn-aixpa/overtourism.git",
    handler="process_data:process_all_data",
    requirements=["civic-digital-twins==0.5.0", "digitalhub==0.14.0", "fastapi[standard]", "geojson>=3.2.0", "geopandas>=1.1.1", "matplotlib>=3.10.0", "orjson>=3.11.3", "plotly>=6.3.0", "pyarrow>19.0", "scikit-learn>=1.7.1", "scipy", "slugify>=0.0.1", "unidecode>=1.4.0", "networkx>=3.5", "contextily>=1.6.2", "matplotlib-scalebar>=0.9.0", "osmnx>=2.0.6", "polars", "haversine", "tqdm", "seaborn", "gtfs_kit"]
)
```

and the create a run of the function, setting also necessary environment and volumes, e.g.,

```python
train_run = func.run(action="job",
                     envs=[
                        {"name": "DATA_PREFIX", "value": "path/to/inputdata/"},
                        {"name": "BASE_DIR", "value":  "/appdata"}
                     ],
                     volumes=[{
                        "volume_type": "persistent_volume_claim",
                        "name": "appdata-volume",
                        "mount_path": "/appdata",
                        "spec": { "size": "50Gi" }}]
                    )
```

Once executed (around 2.5 hours), the procedure creates a set of artifacts and data items:

- map_apt.geojson
- map_vodafone.geojson
- map_comuni.geojson
- map_vodafone_2024.geojson
- weekday_stats_ot	
- weather_stats_ot
- season_stats_ot
- df_turismo_sommerso
- df_overturismo
- df_distribuzione_festivo
- df_distribuzione_prefestivo
- df_distribuzione_feriale
- df_flussi_estate
- df_stagionalita_presenze
- df_tasso_variazione_pecentuale
- df_tasso_turisticita_inverno2023
- df_tasso_turisticita_estate2023
- df_tasso_turisticita_estate2022
- df_tasso_turisticita_estate
- df_tasso_turisticita_2023
- df_tasso_turisticita_2022
- df_tasso_turisticita
- df_tasso_ricettivita
- df_incidenza_postiletto_non_conv
- df_incidenza_strutture_non_conv