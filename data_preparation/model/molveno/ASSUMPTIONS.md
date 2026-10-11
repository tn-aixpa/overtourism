# Molveno: register of decisions and assumptions

Every modelling decision (`D*`) and assumption (`A*`) of the Molveno data-preparation pipeline
and of the Molveno model, with its evidence and the places where it is used. The code refers to
these IDs in comments; a value without an ID is a bug.

Status: **in use** · **to review** (in use, to be reviewed) · **to validate** · **planned** ·
**retired**.

Paths: `pipeline/` = `data_preparation/model/molveno/`, `model/` = `executor-backend/src/model/molveno/`.

## Decisions

| ID | Decision | Evidence / reason | Used in | Status |
|----|----------|-------------------|---------|--------|
| D1 | Reference period: 01/07–30/09/2023 (92 days). | Statistics of the current model. | `pipeline/reference_datasets.py` (`REFERENCE_PERIOD`) | to review |
| D2 | The tourists of a day are the Vodafone tourists dated the following day. | Statistics of the current model. Possibly a date convention of the Vodafone data (overnight presences): to check. | `pipeline/reference_datasets.py` (`TOURIST_DATE_SHIFT`) | to review |
| D3 | Seasons: k-means with 4 clusters on the standardized tourists and excursionists of the reference days, named low / mid / high / very high by increasing mean tourists; season weight = share of days. | Statistics of the current model. Fixed seed: the same seasons on every run. | `pipeline/reference_datasets.py` (`assign_seasons`, `KMEANS_SEED`) | to review |
| D4 | Weekday effect: deviation of each day from the centred 7-day moving average, mean and SD per weekday over the reference days; added to the season mean in the model (variances summed). | Statistics of the current model. | `pipeline/reference_datasets.py` (`build_weekday_stats`); `model/molveno_presence_stats.py` | to review |
| D5 | Vodafone presences of the comune of Molveno (`locId` 27, `TN_MKT_AL_3`): TOURIST = tourists, VISITOR = excursionists, all user countries summed. | Single foreign countries are masked to 0 below 15 (privacy), so foreigners are undercounted; Fazzon uses Italians + the aggregated foreign total (`_STR`, Fazzon D12). | `pipeline/standardize.py` (`standardize_vodafone`) | to review |
| D6 | Weather effect: three classes bad / unsettled / good (weights 0.15 / 0.20 / 0.65) with shifts of the mean and SDs added to the season. | Fixed values in the model, not yet produced by the pipeline. | `model/molveno_presence_stats.py` | to review |
