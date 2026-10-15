# SPDX-License-Identifier: Apache-2.0
"""Single source of truth about the datasets of the pipeline.

Everything that used to be repeated (dataset keys, processed file names, value columns used
for merging, which reference mappings each dataset needs, which datasets feed which phenomenon)
is defined here once.
"""

from dataclasses import dataclass

from utils.config import BASE_WINDOW, DATASET_FILES, PHENOMENA_FILES

# value columns of the processed dataframes
POPOLAZIONE_VALUE_COLS = ["popolazione"]
STRUTTURE_VALUE_COLS = [
    "tot_postiletto_non_conv",
    "tot_postiletto",
    "tot_strutture_non_conv",
    "tot_strutture",
]
VODAFONE_VALUE_COLS = ["presenze"]
PRESENZE_ALB_VALUE_COLS = ["presenze_alb"]
PRESENZE_XALB_VALUE_COLS = ["presenze_xalb"]
EE_PRELIEVI_VALUE_COLS = ["ee_services"]
GAS_PRELIEVI_VALUE_COLS = ["gas_services"]

# Vodafone OD co-visits (flows between areas) -> yearly flow indicators per comune, see utils/flows.py.
# One column per metric and profile (`<metric>_<profile in lower case>`), plus the days covered by the year
COVISITS_RAW_FILE = "vodafone_covisits.csv"  # data-lake object / name in raw_data of the base build
COVISITS_LOC_TYPE = "TN_MKT_AL_3"  # area level kept (same as the vodafone presences)
COVISITS_PROFILES = ("TOURIST", "VISITOR")  # kept separate, the other profiles are discarded
COVISITS_TOTAL_ROWS_ONLY = True  # only the rows with an empty userCountry (total over all the countries)
COVISITS_METRICS = ("quota_ingresso", "quota_uscita", "quota_intra_ambito", "pagerank")
COVISITS_MIN_T = 100  # minimum yearly user-days in an area to compute the mobility shares
COVISITS_PAGERANK_DAMPING = 0.85
COVISITS_FULL_YEAR_DAYS = 360  # fewer days with data: the year is partial (warning, `giorni_coperti`)
# Weighted union of the two profiles (a tourist-day and a visitor-day do not weigh the same): columns
# `<metric>_combined`, computed on user-days multiplied by the weight of their profile (see utils/flows.py).
COVISITS_COMBINED = "combined"
# "estimate": learn the weights (utils/profile_weights.py) in the base build, save them in WEIGHTS_FILE and
#             reuse them in the yearly updates; "fixed": use COVISITS_FIXED_WEIGHTS
COVISITS_WEIGHT_MODE = "estimate"
COVISITS_FIXED_WEIGHTS = {"TOURIST": 1.0, "VISITOR": 1.0}  # used in "fixed" mode, and as fallback if the estimate fails
COVISITS_WEIGHTS_FILE = "covisits_profile_weights.json"  # in the processed directory
# Proxy used to learn the weights: monthly consumption of electricity and gas BY USAGE CATEGORY (`codice_utilizzo` of the
# raw energy files). WHY consumption, what it does (not) measure and the alternatives: see the DECISION RECORD at the top
# of utils/profile_weights.py. The categories are NOT all alike, so they get a role:
#   "shared"       used by tourists AND visitors  -> their sum is the target that gives the weight (like-for-like
#                  comparison of a visitor-day and a tourist-day on the same services)
#   "tourist_only" used only by overnight guests  -> PLACEBO: the visitors must not explain it, otherwise the model
#                  mixes up the two profiles and the weights are not used
#   "ignore"       left out
# Codes (ATECO): 551 hotels (55.10), 553 restaurants / bars (56.10), 602 other passenger land transport (49.3),
# 920 sport / recreation / culture (93.1), 930 other personal services (96.0). They are matched with the values of
# `codice_utilizzo` as strings (551 = 551.0): the pipeline warns about the configured codes it does not find.
COVISITS_WEIGHT_CATEGORIES = {
    "551": "tourist_only",
    "553": "shared",
    "602": "shared",
    "920": "shared",
    "930": "shared",
}
# the placebo fails if the LOWER end of the 95% interval of (visitor effect / tourist effect) in a tourist_only
# category is above this value (it should be ~0)
COVISITS_WEIGHT_PLACEBO_MAX = 0.2
COVISITS_WEIGHT_BOOTSTRAP = 300  # bootstrap draws (over the comuni) for the uncertainty of the weights
COVISITS_WEIGHT_MIN = 0.05  # a profile never weighs less than this (it is never dropped)
COVISITS_VALUE_COLS = [
    f"{metric}_{profile.lower()}"
    for metric in COVISITS_METRICS
    for profile in (*COVISITS_PROFILES, COVISITS_COMBINED)
] + ["giorni_coperti"]

# ISPAT presences (alb / extralb): arrivals + presences per month, read from the raw files (no normalization).
# Files: Presenze/<arrivi|presenze>_<alb|exalb>_<year>.csv ; the base build downloads the years of its window (period.start .. period.base_end), the updates the others.
PRESENZE_FOLDER = "Presenze"
PRESENZE_BASE_YEARS = tuple(range(BASE_WINDOW.start.year, BASE_WINDOW.end.year + 1))  # 2022..2024
PRESENZE_MEASURES = ("arrivi", "presenze")
# True: a stay that starts at the end of a month goes on in the next one (uses the previous month's arrivals and stay);
# False: every month is distributed using only its own arrivals and stay
PRESENZE_SPILL_OVER = True
PRESENZE_TAGS = {
    "presenze_alb": "alb",
    "presenze_extralb": "exalb",
}  # dataset -> tag in the file names

# reference files that a dataset may need to be standardized / processed
REF_MAPPING_COMUNI = "mapping_comuni"
REF_MAPPING_VODAFONE = "mapping_vodafone"
REF_MAPPING_APT = "mapping_apt"
REF_GEOJSON = "geojson"


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # key used for selection, e.g. "presenze_alb"
    value_cols: tuple  # columns merged on (DATA, ID_COMUNE) when updating
    references: tuple  # reference files needed to process it
    std_name: str = (
        ""  # standardized dataframe name (step 1); "" = not normalized, processed from raw_data
    )
    processed_name: str = ""  # processed dataframe name (step 2)
    update_name: str = ""  # processed dataframe name in the yearly update

    @property
    def normalized(self) -> bool:
        return bool(self.std_name)


def _spec(name, value_cols, references, normalized=True):
    dataset_cfg = DATASET_FILES.get(name, {})
    return DatasetSpec(
        name=name,
        value_cols=tuple(value_cols),
        references=tuple(references),
        std_name=dataset_cfg.get("std", f"{name}_std" if normalized else ""),
        processed_name=dataset_cfg.get("processed", f"{name}_pr"),
        update_name=dataset_cfg.get("update", f"{name}_update_pr"),
    )


DATASETS = {
    spec.name: spec
    for spec in (
        _spec("popolazione", POPOLAZIONE_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("strutture", STRUTTURE_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("vodafone", VODAFONE_VALUE_COLS, [REF_MAPPING_VODAFONE, REF_GEOJSON]),
        _spec("ee_prelievi", EE_PRELIEVI_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec("gas_prelievi", GAS_PRELIEVI_VALUE_COLS, [REF_MAPPING_COMUNI]),
        _spec(
            "vodafone_covisits",
            COVISITS_VALUE_COLS,
            [REF_MAPPING_VODAFONE, REF_GEOJSON],
        ),
        _spec(
            "presenze_alb",
            PRESENZE_ALB_VALUE_COLS,
            [REF_MAPPING_APT, REF_MAPPING_COMUNI],
            normalized=False,
        ),
        _spec(
            "presenze_extralb",
            PRESENZE_XALB_VALUE_COLS,
            [REF_MAPPING_COMUNI],
            normalized=False,
        ),
    )
}
ALL_DATASETS = tuple(DATASETS)

# final phenomenon -> datasets that must ALL be updated for it to be recomputed
PHENOMENA = {name: frozenset(deps) for name, deps in PHENOMENA_FILES.items()}
PRESENZE_DATASETS = PHENOMENA["phen_presenze"]


def phenomena_to_recompute(selected) -> list:
    """Phenomena whose inputs are all among the selected datasets."""
    return [p for p, required in PHENOMENA.items() if required <= set(selected)]
