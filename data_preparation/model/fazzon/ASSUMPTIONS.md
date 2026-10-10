# Fazzon: register of decisions and assumptions

Every modelling decision (`D*`) and assumption (`A*`) of the Fazzon data-preparation pipeline
and of the Fazzon model, with its evidence and the places where it is used. The code refers to
these IDs in comments; a value without an ID is a bug. Questions for the domain experts are in
`OPEN_QUESTIONS.md` (`OQ*` below).

Status: **in use** · **to validate** (in use, waiting for the domain experts) · **planned** ·
**to be replaced** (in use until a planned change lands) · **retired**.

Paths: `pipeline/` = `data_preparation/model/fazzon/`, `model/` = `executor-backend/src/model/fazzon/`.

## Decisions

| ID | Decision | Evidence / reason | Used in | Status |
|----|----------|-------------------|---------|--------|
| D1 | The Greenbus 2025 rows after 20/07/2025 refer to another area: flagged `other_area`, not used. | Confirmed by the operator; the pilot ran 28/06–20/07/2025. | `pipeline/standardize.py` (`GREENBUS_LAST_DAY`), `reference_datasets.py` | in use |
| D2 | 2026 event days (02–03/07/2026, "5days": >2000 people, extra meadows used as parking, cars not counted) are excluded from comparisons. | Operator notes in the parking sheet. | `pipeline/reference_datasets.py` (`EVENT_DAYS`) | in use |
| D3 | 21/08/2026 is a special day: bad weather all day, shuttle not running, parking at 5 euro. | Operator notes in the parking sheet. | `pipeline/reference_datasets.py` (`SHUTTLE_NOT_RUNNING`, `REDUCED_PRICE_DAYS`) | in use |
| D4 | 28/08/2025 (27 cars) is part of the calibration. | Real bad-weather day, not a data error: 34.2 mm rain, tmax 18.6 °C, low road traffic at Pellizzano (1241 cars up 7–12 vs ~1750 the days before). Previously left out without a documented reason. | `pipeline/reference_datasets.py` (`build_presence_stats`) | in use |
| D5 | No day-type (weekday) split in the presence statistics and in the model. | No evidence of a weekday effect: ANOVA on within-month deviations p = 0.42 (2025), 0.53 (2026); the former peak buckets: July Sundays p = 0.22 (2025, n = 4), 0.06 (2026); August Mon+Tue p = 0.10 (2025), 0.09 (2026). | `pipeline/reference_datasets.py`; `model/fazzon_model.py` (no `cv_day_type`) | in use |
| D6 | The lakeside constraint is dropped from the model. | Its capacity (uniform 400–800 people) was never validated with the domain experts, and it dominated the overall sustainability field. | `model/fazzon_model.py`, `fazzon_backend.py` | in use |
| D7 | Weather is a model category with three classes: good / cool / rainy (definitions A19). | 2025: weather explains half of the within-month variation of cars (R² 0.50, tmax + rain as continuous variables). Of the tested class definitions, A19 explains most of it (R² 0.32; rain ≥ 1 mm only: 0.15; rain ≥ 5 mm or cool, two classes: 0.28). | `pipeline/reference_datasets.py` (`_add_weather_class`, `build_presence_stats`); `model/fazzon_model.py` (`cv_weather`) | in use |
| D8 | Modal shift: an explicit behaviour model (car share as a parametric, e.g. exponential, function of the parking cost per person; split of the non-car visitors among shuttle / walk / elsewhere), estimated on the data with bootstrap uncertainty; no ML. It replaces A3 and A4. | Only two price levels (5, 20 euro) and a 2026 change that bundles price, shuttle, booking and traffic light: a flexible model cannot separate them nor extrapolate to new prices. | `pipeline/modal_shift.py` (estimation, A22–A25); model: not yet | estimated; model pending |
| D9 | Presence statistics calibrated on the 2025 parking data; seasons = calendar months June–September. | 2025 is the only season without access regulation. | `pipeline/reference_datasets.py` (`CALIBRATION_YEAR`, `SEASONS`) | in use |
| D10 | Road counts: a station/day/direction is valid with ≥ 23 hours and ≥ 50 cars; zero readings are missing data. | Several sensors report zeros for months (Dimaro from mid-2024, Tonale 2022, Croviana direction 2). | `pipeline/standardize.py` (`road_counts_daily`) | in use |
| D11 | Weather from the Meteotrentino station T0074 (Male', measured), not from the per-comune bulletin. | Bulletin values are rounded and its rain does not match the station (r = 0.67; 188 vs 114 rainy days in 2025); station rain predicts Fazzon cars better (within-month R² 0.26 vs 0.05). | `data_preparation/data_download/download_meteo_meteotrentino.py`, `pipeline/standardize.py` | in use |
| D12 | Vodafone presences = TOURIST + VISITOR profiles, Italians (`ITA`) plus the aggregated foreign total (`_STR`). | Single foreign nationalities are masked to 0 below 15 (privacy threshold). | `pipeline/standardize.py` (`standardize_vodafone`) | in use |
| D13 | In the model the weather category does not depend on the season: its probabilities are the season-weighted average of the climatology (good 0.63, cool 0.20, rainy 0.17). The data keeps P(weather \| season). | The modelling library (civic-digital-twins 0.12) fails when a scenario fixes the season or the weather of a season-dependent category (`ConditionalCategoricalIndex`). By month, the climatological frequencies differ by at most 4 points per class. Revert when the library supports it. | `model/fazzon_presence_stats.py` (`presence_inputs`) | in use |

## Assumptions

| ID | Assumption | Value | Source | Used in | Status |
|----|------------|-------|--------|---------|--------|
| A1 | Persons per car | 2.83 | NetMobility 2021 field study | `pipeline/reference_datasets.py` (`PERSONS_PER_CAR`); `model/fazzon_model.py` (`i_xa_persons_per_car`) | to be replaced by A2 in the model (step 4); consistent with A2 if ~20 % of the 2025 vehicles were motorbikes with ~1.2 persons (A26) |
| A2 | Persons per vehicle as a function of the price paid per vehicle: 2.5 at 5 euro, 3.5 at 20 euro, linear in between, constant outside (afternoon ticket at 10 euro: 2.83) | 2.5 – 3.5 | Operator, parking sheet note ("~2.5 in 2025, motorbikes included; ~3.5 in 2026, people go up only with full cars"); interpolation decided by the user | `pipeline/modal_shift.py` (`occupancy`); `pipeline/analyze_new_data.py` (2025 vs 2026 comparison, 2.5 / 3.5) | to validate (OQ4): it decides how many visitors went elsewhere |
| A3 | Non-car share of visitors | 0.31 (car share 0.69) | 2022 EETRA survey | `pipeline/reference_datasets.py` (`NON_CAR_SHARE`); `model/fazzon_model.py` (`i_car_mode_share`) | to be replaced (D8) |
| A4 | SD of non-car visitors | 50 % of the mean | none (synthetic) | `pipeline/reference_datasets.py` (`OTHER_STD_REL`) | to be replaced (D8) |
| A5 | "N° persone bus" in the parking sheet = people going up | — | Operator sheet | `pipeline/standardize.py` (`bus_up`) | to validate (OQ2) |
| A6 | People coming down by bus but not going up by bus walked up | bus_down − bus_up ≥ walkers | Working hypothesis | `pipeline/modal_shift.py` (`walk_up`) | to validate (OQ1) |
| A7 | 2026 parking price outside 04/07–31/08 | 5 euro/day, as in 2025 | none | `pipeline/reference_datasets.py` (`PRICE_2026_UNREGULATED`, `parking_price_assumed`) | to validate (OQ10) |
| A8 | Parking occupancy ratio (simultaneous / daily cars) | 0.555 (dwell 4.44 h / 8 h) | 2021 calibration | `model/fazzon_model.py` (`i_occupancy_ratio`) | in use |
| A9 | Road passages per car | 2 (one round trip per day) | — | `model/fazzon_model.py` (`i_xo_road_trips`) | in use |
| A10 | Share of visitors using the restaurants | 0.55 | 2022 EETRA survey (>50 %) | `model/fazzon_model.py` (`i_u_food_fraction`) | in use |
| A11 | Restaurant seat turnover | 2 sittings/seat/day | — | `model/fazzon_model.py` (`i_xo_food_sittings`) | in use |
| A12 | Parking capacity (simultaneous cars) | triangular [150, 250], mode 200 | Policy cap 150 + whitelist/enforcement uncertainty | `model/fazzon_model.py` (`i_c_parking`) | in use |
| A13 | Road capacity under semaphore control | uniform [800, 1400] passages/day | 2025 peak 422 cars × 2 = 844 passages without saturation | `model/fazzon_model.py` (`i_c_road`) | in use |
| A14 | Season weights = share of the 2025 days with parking data per season | 3, 31, 31, 17 of 82 days | 2025 parking data | `pipeline/reference_datasets.py` (`season_freq_rel`); `model/fazzon_model.py` (`cv_season`) | in use; note: the June weight reflects the season start (28/06), not June demand |
| A15 | Restaurant seats | 520 (4 venues) | `ristoranti_posti_a_sedere.csv` | `model/fazzon_model.py` (`i_c_seats`) | in use (data) |
| A16 | Pellizzano road counter (station 110): direction 2 = up valley, below the Fazzon turn-off | — | Confirmed by the user (SS42 km increase downhill) | `pipeline/reference_datasets.py` (`ROAD_UP_VALLEY_DIR`) | in use (confirmed) |
| A17 | Demand proxy: cars up valley at Pellizzano 07:00–11:59 | — | Best within-month correlation with Fazzon cars (R² 0.71) | `pipeline/reference_datasets.py` (`ROAD_MORNING_HOURS`) | in use |
| A18 | Rainy day (exploratory analysis) | rain ≥ 1 mm | Standard climatological threshold | `pipeline/analyze_new_data.py` (`RAINY_DAY_MM`) | in use |
| A19 | Weather classes: rainy = rain ≥ 5 mm; cool = not rainy and tmax more than 2 °C below the month's climatological mean tmax; good = the rest | 5 mm, 2 °C | Chosen among the tested definitions as the one explaining most of the within-month variation of 2025 cars (D7); chosen on the same data, so the fit may be optimistic | `pipeline/reference_datasets.py` (`RAINY_MM`, `COOL_DEGREES`) | in use |
| A20 | Weather climatology: P(weather class \| season) from the Male' station, June–September 2023–2026 | e.g. July: good 0.61, cool 0.21, rainy 0.18 | Meteotrentino T0074 | `pipeline/reference_datasets.py` (`CLIMATOLOGY_YEARS`, `weather_climatology`) | in use |
| A21 | Weather effect: multiplicative and the same in every season; cars(season, weather) = level(season) × factor(weather), factor(good) = 1; within-cell SD = cell mean × the season's coefficient of variation of the weather-adjusted daily cars | factors: cool 0.73, rainy 0.60 | Fitted on the 2025 parking data; with 82 days several season × weather cells have 0–2 days, so separate effects per season cannot be estimated | `pipeline/reference_datasets.py` (`weather_factors`, `build_presence_stats`) | in use |
| A22 | 2026 year effect: 2026 demand in 2025 conditions = 2025 level × one factor, estimated on the 2026 days at 5 euro without the access rules (June, 1 July day, September; events excluded) | 1.01 (90 %: 0.84–1.22) | `fazzon_modal_shift` | `pipeline/modal_shift.py` | in use; weak (18 days, mostly September; by season 0.36–1.34); to be replaced by 2026 road counts / Vodafone (OQ8, OQ9) |
| A23 | Response of the visitors by private vehicle (cars + motorbikes) to the price: visitors(p) = visitors(5 euro) × exp(−β (p − 5)), p = euro per vehicle | β = 0.090 per euro (90 %: 0.070–0.112); visitor response at 15 euro = 0.41 (0.33–0.50); vehicles: 0.31 (0.25–0.38) | Two price levels only: the exponential is a two-point fit | `pipeline/modal_shift.py` | in use; extrapolation to other prices rests on the exponential form |
| A24 | Price per car = average paid: full-day (20 euro) and afternoon (10 euro) tickets weighted by their counts | 15.0 euro under the 2026 rules | Parking sheet | `pipeline/modal_shift.py` (`price_paid`) | in use; afternoon tickets recorded only from 18/07/2026 (OQ6) |
| A25 | The whole 2026 package (price, booking, traffic light, shuttle) is attributed to the price | — | Not separable with the 2026 data | `pipeline/modal_shift.py` (β) | in use; a scenario that changes the price without the rest of the package is an extrapolation |
| A26 | Vehicle counts: in 2025 (and in 2026 at 5 euro without the access rules) the parking count includes the motorbikes; under the 2026 rules motorbikes are counted apart, as persons (`moto_persons`) | — | Operator, parking sheet note; the 2025 motorbike column is always 0 | `pipeline/modal_shift.py` (`private_visitors`) | to validate (OQ4) |
