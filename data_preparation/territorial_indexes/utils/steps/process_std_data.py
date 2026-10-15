# SPDX-License-Identifier: Apache-2.0
"""
STEP 2 - Processing.

Input : Output/data/normalized/   (+ mapping json files, read from Output/mapping; the ISPAT presences
        arrivals / presences are read from Output/data/raw_data/, they are not normalized)
Output: Output/data/data_processed/   (popolazione_pr, strutture_pr, vodafone_pr,
                                       presenze_alb_pr, presenze_extralb_pr,
                                       ee_prelievi_pr, gas_prelievi_pr, vodafone_covisits_pr)

Transformations: ID_COMUNE resolution, filtering, selection of the columns, computation of
aggregated columns, disaggregation of the presences to comune x day (vodafone: areas -> comuni by beds;
ISPAT: APT / provincia x month -> comune x day, with the average stay of the month, see stay_distribution.py). Every row has a single ID_COMUNE.
Energy categories are summed per comune-month and divided evenly over the calendar days of each month.
The Vodafone OD co-visits are aggregated per YEAR and turned into flow indicators per comune (quota in
ingresso / in uscita, quota intra-ambito, PageRank, per profile): one row per comune and year, see flows.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.cleaning import (
    ids_to_int,
    normalize_id_comune,
    pad_id_comune,
    remove_provincia,
    resolve_id_comune,
    standard_ordering_cols,
)
from utils.config import (
    BASE_WINDOW,
    DATA_START,
    MAPPING_DIR,
    NORMALIZED_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REFERENCES,
    TYPE_FORMAT,
    MAPPING_FILES,
    setup_logging,
)
from utils.datasets import (  # noqa: F401  (re-exported for convenience)
    COVISITS_FIXED_WEIGHTS,
    COVISITS_FULL_YEAR_DAYS,
    COVISITS_WEIGHT_BOOTSTRAP,
    COVISITS_WEIGHT_MIN,
    COVISITS_WEIGHT_MODE,
    COVISITS_WEIGHT_CATEGORIES,
    COVISITS_WEIGHT_PLACEBO_MAX,
    COVISITS_WEIGHTS_FILE,
    COVISITS_METRICS,
    COVISITS_MIN_T,
    COVISITS_PAGERANK_DAMPING,
    COVISITS_PROFILES,
    COVISITS_VALUE_COLS,
    DATASETS,
    PRESENZE_BASE_YEARS,
    PRESENZE_MEASURES,
    PRESENZE_SPILL_OVER,
    PRESENZE_TAGS,
    POPOLAZIONE_VALUE_COLS,
    PRESENZE_ALB_VALUE_COLS,
    PRESENZE_XALB_VALUE_COLS,
    EE_PRELIEVI_VALUE_COLS,
    GAS_PRELIEVI_VALUE_COLS,
    STRUTTURE_VALUE_COLS,
    VODAFONE_VALUE_COLS,
)
from utils.steps.standardize_raw_data import ENERGY_MONTHS
from utils.window import Window, clip
from utils.adapters.presenze import MONTHS_MAPPING
from utils.disaggregation import (
    assert_sums_match,
    beds_weights,
    disaggregate,
    is_whole,
)
from utils.flows import build_area_comune, comune_flow_indicators
from utils.stay_distribution import distribute_by_stay
from utils.io import read_df, read_json, save_computed_dfs
from utils.profile_weights import (
    build_energy_targets,
    estimate_profile_weights,
    fixed_profile_weights,
    load_profile_weights,
    normalize_code,
    save_profile_weights,
)

logger = logging.getLogger(__name__)

BEDS_ALB_COL = "posti_letto_alb"
BEDS_XALB_COL = "posti_letto_xalb"


def _filtering_strutture(df, window, year_col="DATA"):
    """Keeps the years of the time window (nothing before `period.start`: geography changes of the
    municipalities make the older years not comparable)."""
    return clip(df, window, col=year_col, name="strutture")


def _filtering_vodafone_attendences(df):
    """Filtering presences on tourists and municipalities"""
    return df[
        (df["userProfile"] == "TOURIST") & (df["locType"] == "TN_MKT_AL_3")
    ].copy()


def process_popolazione(df, mapping_comuni, window=BASE_WINDOW):
    df = clip(df, window, name="popolazione")
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda x: resolve_id_comune(x, mapping_comuni)
    )
    df = remove_provincia(df, comune_col="LOCATION")
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])
    return standard_ordering_cols(df[["DATA", "ID_COMUNE"] + POPOLAZIONE_VALUE_COLS])


def process_strutture(df, mapping_comuni, window=BASE_WINDOW):
    df = _filtering_strutture(df, window)
    df = remove_provincia(df, comune_col="LOCATION")

    # CONV = alberghieri + extralberghieri (from the raw components, not from the raw CONV totals)
    df["tot_strutture_conv"] = df["alberghieri strutture"] + df["extra alb. Strutture"]
    df["tot_postiletto_conv"] = (
        df["alberghieri posti_letto"] + df["extra alb. Posti_letto"]
    )

    df["tot_strutture_non_conv"] = (
        df["all. privati numero"] + df["all.disposizione numero"]
    )
    df["tot_postiletto_non_conv"] = (
        df["all. privati posti_letto"] + df["all. disposizione posti_letto"]
    )

    # Compute total as the sum of CONV and NON CONV
    df["tot_strutture"] = df["tot_strutture_conv"] + df["tot_strutture_non_conv"]
    df["tot_postiletto"] = df["tot_postiletto_conv"] + df["tot_postiletto_non_conv"]

    # Set ID_COMUNE (resolving the bilingual overrides)
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda x: resolve_id_comune(x, mapping_comuni)
    )
    missing = df.loc[df["ID_COMUNE"].isna(), "LOCATION"].unique()
    if len(missing) > 0:
        logger.warning(
            "[process_strutture] No ID_COMUNE found (even with overrides) for: %s",
            sorted(missing),
        )

    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])
    return standard_ordering_cols(df[["DATA", "ID_COMUNE"] + STRUTTURE_VALUE_COLS])


def process_vodafone(
    df,
    mapping_vodafone,
    strutture=None,
    bed_col="tot_postiletto",
    check=True,
    window=BASE_WINDOW,
):
    """vodafone presences: vodafone areas x day -> comune x day.

    Areas that map to a single comune are unchanged. Areas that map to several comuni are split
    linearly to the beds (`bed_col` of `strutture`, year of the day or nearest year) of their comuni.
    Without `strutture` the split is uniform. With check=True the totals are summed back
    (per area and day) and a mismatch raises ValueError.
    """
    df = _filtering_vodafone_attendences(df)  # only COMUNI & TURISTI
    df["ID_COMUNE"] = df["LOCATION"].map(mapping_vodafone)
    mask = df["LOCATION"] == "SAN GIOVANNI DI FASSA"
    df.loc[mask, "ID_COMUNE"] = pd.Series(
        [[22250]] * mask.sum(), index=df.index[mask], dtype=object
    )
    if df["ID_COMUNE"].isna().any():
        locations = sorted(
            df.loc[df["ID_COMUNE"].isna(), "LOCATION"].dropna().unique().tolist()
        )
        logger.warning(
            "[process_vodafone] %d rows Vodafone (%d aree) with no mapping ID_COMUNE; "
            "rows will be excluded by groupby: %s",
            int(df["ID_COMUNE"].isna().sum()),
            len(locations),
            locations,
        )
    df["DATA"] = pd.to_datetime(df["DATA"].astype(str), errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    df = clip(df, window, name="vodafone")  # before the (expensive) split over the comuni
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"]).apply(
        normalize_id_comune
    )  # hashable

    # Sum per (day, vodafone area), then split each area over its comuni
    df = df.groupby(["DATA", "ID_COMUNE"], as_index=False)[VODAFONE_VALUE_COLS].sum()
    integer = is_whole(df, VODAFONE_VALUE_COLS)
    df["_TID"] = np.arange(len(df))

    kwargs = {}
    if strutture is not None:
        years = pd.to_datetime(df["DATA"]).dt.year.unique()
        kwargs = dict(
            space_weights=beds_weights(strutture, years, bed_col),
            space_weight_col=bed_col,
            space_time_freq="Y",  # beds are yearly
        )
    else:
        logger.warning("[process_vodafone] no strutture given: uniform split")

    out = disaggregate(
        df, cols=VODAFONE_VALUE_COLS, axis="space", integer=integer, **kwargs
    )
    if check:
        assert_sums_match(df, out, VODAFONE_VALUE_COLS, ["_TID"], name="vodafone")
        assert_sums_match(df, out, VODAFONE_VALUE_COLS, ["DATA"], name="vodafone/day")
    return standard_ordering_cols(out.drop(columns="_TID").reset_index(drop=True))


def stack_covisits(frames):
    """Standardized OD frames in order: on the dates present in a later frame the later one wins
    (a later file corrects the earlier ones, like vodafone_attendences / _new)."""
    frames = list(frames)
    for k in range(len(frames) - 1):
        later_dates = set().union(*[set(f["DATA"]) for f in frames[k + 1 :]])
        n0 = len(frames[k])
        frames[k] = frames[k][~frames[k]["DATA"].isin(later_dates)]
        if len(frames[k]) < n0:
            logger.info(
                "[covisits] %d rows replaced by the later data", n0 - len(frames[k])
            )
    od = pd.concat(frames, ignore_index=True)
    key = [c for c in od.columns if c != "value"]
    return od.groupby(key, as_index=False, observed=True)[
        "value"
    ].sum()  # duplicated rows


def covisits_area_comune(mapping_dir, mapping_vodafone, mapping_comuni):
    """Area -> ID_COMUNE table of the OD co-visits (geojson of the areas + mapping_vodafone)."""
    geojson = Path(mapping_dir) / Path(REFERENCES["geojson"]).name
    return build_area_comune(geojson, mapping_vodafone, mapping_comuni)


def covisits_profile_weights(
    od, area_comune, energy, strutture, out_dir, *, reuse=False, mode=COVISITS_WEIGHT_MODE
):
    """Weights {userProfile: w} used to merge TOURIST and VISITOR in the `_combined` indicators.

    mode "fixed": COVISITS_FIXED_WEIGHTS. mode "estimate": learnt from the consumption by usage category,
    `energy` = {carrier: energy_monthly_by_category(...)} e.g. {"ee": ..., "gas": ...} (see profile_weights.py and
    the categories in COVISITS_WEIGHT_CATEGORIES), and saved in out_dir/COVISITS_WEIGHTS_FILE.
    reuse=True (yearly updates): the saved weights are loaded instead, so that the years of the base build
    and of the updates stay comparable; without a saved file the fixed weights are used."""
    file = Path(out_dir) / COVISITS_WEIGHTS_FILE
    if mode == "fixed":
        res = fixed_profile_weights(COVISITS_FIXED_WEIGHTS)
    elif reuse:
        res = load_profile_weights(file)
        if res is None:
            logger.warning(
                "[covisits] %s not found (run the base build): fixed weights %s",
                file, COVISITS_FIXED_WEIGHTS,
            )
            res = fixed_profile_weights(COVISITS_FIXED_WEIGHTS, reason="no saved weights")
        else:
            logger.info("[covisits] weights loaded from %s: %s", file, res["weights"])
        return res["weights"]
    else:
        t = build_energy_targets(energy, COVISITS_WEIGHT_CATEGORIES)
        for carrier, found in t["codes_found"].items():
            absent = sorted(set(normalize_code(c) for c in COVISITS_WEIGHT_CATEGORIES) - set(found))
            if absent:
                logger.warning(
                    "[covisits] %s: usage codes %s configured in COVISITS_WEIGHT_CATEGORIES but not in the data "
                    "(codes in the data: %s)", carrier, absent, found,
                )
        res = estimate_profile_weights(
            od,
            area_comune,
            t["targets"],
            strutture,
            profiles=list(COVISITS_PROFILES),
            diagnostic_targets=t["diagnostic"],
            placebo_targets=t["placebo"],
            placebo_max=COVISITS_WEIGHT_PLACEBO_MAX,
            n_boot=COVISITS_WEIGHT_BOOTSTRAP,
            min_weight=COVISITS_WEIGHT_MIN,
            fallback=COVISITS_FIXED_WEIGHTS,
        )
    save_profile_weights(res, file)
    logger.info("[covisits] weights (%s): %s -> %s", res["source"], res["weights"], file)
    return res["weights"]


def process_covisits(df, area_comune, weights=None, window=BASE_WINDOW):
    """Vodafone OD co-visits: area x area x day -> comune x YEAR flow indicators.

    Wide output, one row per comune and year: DATA (year), ID_COMUNE (zero-padded) and
    `<metric>_<profile>` (quota_ingresso, quota_uscita, quota_intra_ambito, pagerank; tourist, visitor and
    combined) + `giorni_coperti`. `weights` ({userProfile: w}, default COVISITS_FIXED_WEIGHTS) merge the two
    profiles in the `_combined` columns. A year with fewer than COVISITS_FULL_YEAR_DAYS days is partial: it
    is kept, flagged by `giorni_coperti` and by a warning."""
    df = clip(df, window, name="covisits")  # only the days of the window enter the yearly indicators
    out = comune_flow_indicators(
        df,
        area_comune,
        min_t=COVISITS_MIN_T,
        damping=COVISITS_PAGERANK_DAMPING,
        profiles=COVISITS_PROFILES,
        metrics=COVISITS_METRICS,
        weights=weights or COVISITS_FIXED_WEIGHTS,
    )
    partial = out.groupby("DATA")["giorni_coperti"].first()
    for year, days in partial[partial < COVISITS_FULL_YEAR_DAYS].items():
        logger.warning(
            "[process_covisits] %s is a partial year (%d days): compare it with care",
            year,
            days,
        )
    return standard_ordering_cols(out[["DATA", "ID_COMUNE"] + COVISITS_VALUE_COLS])


def process_beds(strutture_std, mapping_comuni, window=BASE_WINDOW):
    """Beds per comune and year from the NORMALIZED strutture: DATA (year), ID_COMUNE, posti_letto_alb
    (`alberghieri posti_letto`), posti_letto_xalb (`extra alb. Posti_letto`). The years up to the end of
    `window` are kept (the update needs the previous years too, hence only the end matters beyond start)."""
    df = _filtering_strutture(strutture_std, window)
    df = remove_provincia(df, comune_col="LOCATION")
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda x: resolve_id_comune(x, mapping_comuni)
    )
    missing = df.loc[df["ID_COMUNE"].isna(), "LOCATION"].unique()
    if len(missing) > 0:
        logger.warning("[process_beds] No ID_COMUNE found for: %s", sorted(missing))
    df = df[df["ID_COMUNE"].notna()].copy()
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])
    df = df.rename(
        columns={
            "alberghieri posti_letto": BEDS_ALB_COL,
            "extra alb. Posti_letto": BEDS_XALB_COL,
        }
    )
    return df[["DATA", "ID_COMUNE", BEDS_ALB_COL, BEDS_XALB_COL]].reset_index(drop=True)


def _energy_monthly(df, mapping_comuni, value_col, source, by_category=False):
    """Standardized energy rows -> monthly consumption per comune (DATA = first day of the month, ID_COMUNE,
    value_col), summing the usage categories (`codice_utilizzo`) or, with by_category=True, keeping them
    (one row per comune, month and category, with the column `codice_utilizzo`)."""
    df = df.copy()
    df = remove_provincia(df, comune_col="LOCATION", upper=True)
    df["ID_COMUNE"] = df["LOCATION"].apply(
        lambda name: resolve_id_comune(name, mapping_comuni)
    )
    missing = sorted(df.loc[df["ID_COMUNE"].isna(), "LOCATION"].dropna().unique())
    if missing:
        logger.warning(
            "[%s] Excluding locations without municipality IDs: %s", source, missing
        )
    df = df[df["ID_COMUNE"].notna()].copy()
    df["ID_COMUNE"] = pad_id_comune(df["ID_COMUNE"])

    keys = ["DATA", "ID_COMUNE"] + (["codice_utilizzo"] if by_category else [])
    long = df.melt(
        id_vars=keys,
        value_vars=list(ENERGY_MONTHS),
        var_name="_MONTH",
        value_name=value_col,
    )
    raw_values = long[value_col]
    long[value_col] = pd.to_numeric(raw_values, errors="coerce")
    invalid = raw_values.notna() & long[value_col].isna()
    if invalid.any():
        raise ValueError(
            f"[{source}] non-numeric consumption values: "
            f"{raw_values.loc[invalid].head().tolist()}"
        )
    month_numbers = {month: number for number, month in enumerate(ENERGY_MONTHS, 1)}
    long["DATA"] = pd.to_datetime(
        {
            "year": pd.to_numeric(long["DATA"], errors="raise").astype(int),
            "month": long["_MONTH"].map(month_numbers),
            "day": 1,
        }
    )
    return (
        long.groupby(keys, as_index=False)[value_col]
        .sum(min_count=1)
        .sort_values(keys)
        .reset_index(drop=True)
    )


def energy_monthly_by_category(df, mapping_comuni, source):
    """Monthly consumption per comune AND usage category, for the estimation of the profile weights:
    ID_COMUNE, month ('YYYY-MM'), code (`codice_utilizzo` as a string), y. Rows without a value are dropped."""
    m = _energy_monthly(df, mapping_comuni, "y", source, by_category=True)
    m["month"] = m["DATA"].dt.strftime("%Y-%m")
    m["code"] = m["codice_utilizzo"].map(normalize_code)
    return m.dropna(subset=["y"])[["ID_COMUNE", "month", "code", "y"]].reset_index(drop=True)


def process_energy(df, mapping_comuni, value_col, source, window=BASE_WINDOW):
    """Sum energy categories, then spread each municipality-month total evenly over its days.
    Only the days of the time window are kept."""
    df = clip(df, window, name=source)  # years (DATA = anno) first: cheap
    monthly = _energy_monthly(df, mapping_comuni, value_col, source)
    daily = monthly.loc[monthly.index.repeat(monthly["DATA"].dt.days_in_month)].copy()
    if not daily.empty:
        days_in_month = daily["DATA"].dt.days_in_month
        day_offsets = daily.groupby(level=0).cumcount()
        daily["DATA"] = daily["DATA"] + pd.to_timedelta(day_offsets, unit="D")
        daily[value_col] = daily[value_col] / days_in_month
    out = clip(daily.reset_index(drop=True), window, name=source)  # exact days
    # Keep dates as ISO strings in parquet as well: timestamp columns are exposed by
    # some downstream readers as epoch-millisecond integers.
    out["DATA"] = out["DATA"].dt.strftime("%Y-%m-%d")
    return standard_ordering_cols(out)


def read_strutture_std(normalized_dir, type_format=TYPE_FORMAT):
    """Every normalized strutture file (base `strutture_std` + the ones of the updates, e.g.
    `strutture_2024_update_std`), stacked: on a duplicated DATA + LOCATION the later file wins.
    """
    normalized_dir = Path(normalized_dir)
    base = DATASETS["strutture"].std_name
    names = sorted(
        f.stem
        for f in normalized_dir.glob(f"strutture*_std.{type_format}")
        if f.stem != base
    )
    frames = [read_df(normalized_dir, n, type_format) for n in [base, *names]]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["DATA", "LOCATION"], keep="last").reset_index(drop=True)


# ---- ISPAT arrivals / presences: read as downloaded, no normalization
def _monthly_long(df, year, source):
    """Rows of the 12 months (the `Anno` total row and the empty row are dropped) -> long frame with
    DATA = first day of the month, `variable` (column name) and `value`."""
    first = df.columns[0]
    month = df[first].astype(str).str.strip().map(MONTHS_MAPPING)
    if month.notna().sum() != 12 or month.dropna().nunique() != 12:
        raise ValueError(
            f"{source}: expected the 12 months in the first column, found {df[first].tolist()}"
        )
    dropped = df.loc[month.isna(), first].tolist()
    logger.info("[%s] rows dropped: %s", source, dropped)
    df = df.loc[month.notna()].copy()
    df["DATA"] = [
        pd.Timestamp(year=int(year), month=int(m), day=1) for m in month[month.notna()]
    ]
    long = df.drop(columns=first).melt(
        id_vars="DATA", var_name="variable", value_name="value"
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    if long["value"].isna().any():
        raise ValueError(
            f"{source}: missing / non numeric values in {sorted(long.loc[long['value'].isna(), 'variable'].unique())}"
        )
    return long


def clean_presenze_alb(df, year, source="presenze_alb"):
    """ISPAT alb table (months x APT): only the `<APT> - Totale` columns are kept (`- Italiani`,
    `- Stranieri` and the `Provincia - ...` columns are ignored). -> (AREA = APT, DATA, value).
    """
    cols = [
        c
        for c in df.columns[1:]
        if str(c).endswith(" - Totale") and not str(c).startswith("Provincia - ")
    ]
    long = _monthly_long(df[[df.columns[0], *cols]], year, source)
    long["AREA"] = long.pop("variable").str.removesuffix(" - Totale").str.strip()
    return long[["AREA", "DATA", "value"]]


def clean_presenze_exalb(df, year, source="presenze_extralb"):
    """ISPAT extra alb table (months x type of structure): only the first column and the last one
    (the provincial total, `Totale - Totale`) are kept. -> (AREA = "PROVINCIA", DATA, value).
    """
    long = _monthly_long(df.iloc[:, [0, -1]], year, source)
    long["AREA"] = "PROVINCIA"
    return long[["AREA", "DATA", "value"]]


def presenze_files(raw_dir, tag, years):
    """Convention of the file names: {(measure, year): raw_dir/<measure>_<tag>_<year>.csv} (existing files)."""
    raw_dir = Path(raw_dir)
    files = {}
    for y in years:
        for m in PRESENZE_MEASURES:
            f = raw_dir / f"{m}_{tag}_{y}.csv"
            if f.exists():
                files[(m, int(y))] = f
    return files


def process_presenze_stay(
    files,
    tag,
    years,
    area_ids,
    beds,
    *,
    bed_col,
    out_col,
    spill_over=PRESENZE_SPILL_OVER,
    window=BASE_WINDOW,
):
    """ISPAT arrivals + presences (month x area) of `years` -> comune x day presences (see stay_distribution.py).

    files    {(measure, year): path} with measure in ("arrivi", "presenze"); every year needs both.
             The files of the year before `years` (if present) give the stays that cross the new year.
    tag      "alb" (areas = APT, with the `- Totale` columns) or "exalb" (province).
    area_ids {area: [ids]}; beds: (DATA, ID_COMUNE, bed_col) weights of the split among comuni.
    """
    years = sorted(int(y) for y in years)
    context = years[0] - 1
    has_context = spill_over and all((m, context) in files for m in PRESENZE_MEASURES)
    use = [context, *years] if has_context else years
    if spill_over and not has_context:
        logger.warning(
            "[%s] no files of %s: the stays that cross the beginning of %s are approximated",
            out_col,
            context,
            years[0],
        )
    clean = clean_presenze_alb if tag == "alb" else clean_presenze_exalb
    tables = {m: [] for m in PRESENZE_MEASURES}
    for y in use:
        for m in PRESENZE_MEASURES:
            if (m, y) not in files:
                raise FileNotFoundError(
                    f"missing file for {m} {tag} {y}: run the download step"
                )
            raw = pd.read_csv(files[(m, y)])
            tables[m].append(clean(raw, y, source=f"{m}_{tag}_{y}"))
    arrivals, presences = (
        pd.concat(tables[m], ignore_index=True) for m in PRESENZE_MEASURES
    )

    names = set(arrivals["AREA"])
    if names != set(area_ids):
        raise ValueError(
            f"[{out_col}] areas of the files and of the mapping differ: "
            f"only in files {sorted(names - set(area_ids))}, only in mapping {sorted(set(area_ids) - names)}"
        )
    out = distribute_by_stay(
        arrivals,
        presences,
        area_ids,
        beds,
        bed_col=bed_col,
        out_col=out_col,
        keep_from=pd.Timestamp(years[0], 1, 1),
        spill_over=spill_over,
    )
    return standard_ordering_cols(clip(out, window, name=out_col))


def presenze_alb_area_ids(mapping_apt):
    return {k: [int(i) for i in v] for k, v in mapping_apt.items()}


def presenze_exalb_area_ids(mapping_comuni):
    return {"PROVINCIA": sorted({int(i) for i in mapping_comuni.values()})}


def load_mappings(mapping_dir):
    mapping_dir = Path(mapping_dir)
    return {key: read_json(mapping_dir / fname) for key, fname in MAPPING_FILES.items()}


def process_data(
    normalized_dir=NORMALIZED_DIR,
    mapping_dir=MAPPING_DIR,
    out_dir=PROCESSED_DIR,
    type_format=TYPE_FORMAT,
    raw_dir=RAW_DIR,
    window=BASE_WINDOW,
):
    """Base build: every dataset is restricted to `window` (default: period.start .. period.base_end of
    settings.yaml), so nothing before 2022-01-01 and nothing after 2024-12-31 reaches the processed data
    (nor the estimation of the covisits profile weights)."""
    normalized_dir = Path(normalized_dir)
    logger.info("Time window of the base build: %s", window)

    logger.info("Reading standardized data from %s", normalized_dir)
    std = {
        name: read_df(normalized_dir, spec.std_name, type_format)
        for name, spec in DATASETS.items()
        if spec.normalized
    }
    for name in ("ee_prelievi", "gas_prelievi", "vodafone_covisits"):  # big / used by the weights
        std[name] = clip(std[name], window, name=name)
    maps = load_mappings(mapping_dir)
    mapping_comuni, mapping_vodafone, mapping_apt = (
        maps["mapping_comuni"],
        maps["mapping_vodafone"],
        maps["mapping_apt"],
    )

    # Order: strutture -> vodafone (areas with several comuni split by beds) -> alb -> extralb.
    # alb / extralb (ISPAT arrivals + presences, read from raw_data) are distributed taking the average
    # stay into account, and split among comuni by `alberghieri` / `extra alb.` beds.
    strutture_pr = process_strutture(std["strutture"], mapping_comuni, window)
    vodafone_pr = process_vodafone(
        std["vodafone"], mapping_vodafone, strutture_pr, window=window
    )
    beds = process_beds(
        read_strutture_std(normalized_dir, type_format), mapping_comuni, window
    )
    years = range(window.start.year, window.end.year + 1)
    alb_pr = process_presenze_stay(
        presenze_files(raw_dir, PRESENZE_TAGS["presenze_alb"], years),
        "alb",
        years,
        presenze_alb_area_ids(mapping_apt),
        beds,
        bed_col=BEDS_ALB_COL,
        out_col=PRESENZE_ALB_VALUE_COLS[0],
        window=window,
    )
    extralb_pr = process_presenze_stay(
        presenze_files(raw_dir, PRESENZE_TAGS["presenze_extralb"], years),
        "exalb",
        years,
        presenze_exalb_area_ids(mapping_comuni),
        beds,
        bed_col=BEDS_XALB_COL,
        out_col=PRESENZE_XALB_VALUE_COLS[0],
        window=window,
    )
    ee_pr = process_energy(
        std["ee_prelievi"], mapping_comuni, EE_PRELIEVI_VALUE_COLS[0], "ee_prelievi", window
    )
    gas_pr = process_energy(
        std["gas_prelievi"], mapping_comuni, GAS_PRELIEVI_VALUE_COLS[0], "gas_prelievi", window
    )
    area_comune = covisits_area_comune(mapping_dir, mapping_vodafone, mapping_comuni)
    covisits_weights = covisits_profile_weights(
        std["vodafone_covisits"],
        area_comune,
        {  # consumption BY usage category (not the aggregate): see COVISITS_WEIGHT_CATEGORIES
            "ee": energy_monthly_by_category(std["ee_prelievi"], mapping_comuni, "ee_prelievi"),
            "gas": energy_monthly_by_category(std["gas_prelievi"], mapping_comuni, "gas_prelievi"),
        },
        strutture_pr,
        out_dir,
    )
    covisits_pr = process_covisits(
        std["vodafone_covisits"], area_comune, covisits_weights, window
    )
    dict_processed = {
        "popolazione_pr": process_popolazione(std["popolazione"], mapping_comuni, window),
        "strutture_pr": strutture_pr,
        "vodafone_pr": vodafone_pr,
        "presenze_alb_pr": alb_pr,
        "presenze_extralb_pr": extralb_pr,
        DATASETS["ee_prelievi"].processed_name: ee_pr,
        DATASETS["gas_prelievi"].processed_name: gas_pr,
        DATASETS["vodafone_covisits"].processed_name: covisits_pr,
    }

    save_computed_dfs(
        dict_processed, local=True, type_format=type_format, path_saving=out_dir
    )
    logger.info("Processed data saved in %s", out_dir)
    return dict_processed


if __name__ == "__main__":
    setup_logging()
    process_data()
