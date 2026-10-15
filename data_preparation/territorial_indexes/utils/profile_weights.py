"""Data-driven weights of the user profiles (TOURIST vs VISITOR) used to merge them in the flow indicators.

DECISION RECORD - why the weights are learnt from energy consumption (read this before changing it)
==================================================================================================
Question. The flow indicators (quota in ingresso / uscita, intra-ambito, PageRank) are computed for tourists and
for visitors (excursionists). To publish ONE value per comune the two profiles are summed, and a tourist-day and a
visitor-day are not expected to count the same. The weight w_V (w_T = 1) says "how many tourist-days is a
visitor-day worth". There is no ground truth for it: it has to be inferred from some outcome that both profiles
produce and that we can measure per comune.

Outcomes available in the pipeline when this was written, and why they were (not) used:
  * ISPAT presences (alb / extralb): overnight stays only, so they cannot tell the two profiles apart -> no.
  * beds (strutture): supply, not pressure; it does not vary with the profile mix -> no.
  * consumption of electricity / gas by usage category (`codice_utilizzo`: ee_prelievi, gas_prelievi), monthly per
    comune -> USED. It is demand-side, available for every comune, and approximately additive: every guest present
    adds heating, lighting, kitchens, vehicles, services. That makes the coefficient of a regression physically
    readable (energy per user-day of each profile).
  * traffic, waste, water, parking, perceived crowding: probably closer to the pressure of the day visitors, but
    not available (yet). They plug in through the same interface (monthly frames) with no change to the method.

The categories are NOT used as one aggregate (a first version did, and it was wrong to). They differ in WHO consumes:
  * 551 hotels: only overnight guests. A visitor does not sleep there, so the visitors must NOT explain it. It is used
    as a PLACEBO: if the model attributes to the visitors a sizeable share of the hotel consumption (lower end of the
    95% interval of visitor/tourist above COVISITS_WEIGHT_PLACEBO_MAX), the model is mixing up the two profiles
    (confounded seasonality, bad presence data) and NO weight is trusted (fallback). Hotels are kept OUT of the
    estimate of the weight on purpose: including them would push the weight of the visitors down only because they
    cannot use a bed, not because their presence weighs less.
  * 553 restaurants, 602 passenger land transport, 920 sport / recreation, 930 other personal services: consumed by
    both profiles. Their sum (per carrier) is the target that gives the weight: a like-for-like comparison of a
    visitor-day and a tourist-day on the same services, and the ratio is energy-weighted (the categories that use
    more energy count more). Each category is also fitted alone and reported in `diagnostics["diagnostic_targets"]`,
    so that disagreement between categories (e.g. visitors weigh a lot in transport, little in restaurants) is
    visible. The estimate is one number, so heterogeneity is averaged out: look at it.
  * Roles and codes live in COVISITS_WEIGHT_CATEGORIES (utils/datasets.py); change them there, not here.

What the weight therefore means. It is the relative ENERGY pressure of a user-day on the non-accommodation services
(restaurants, transport, recreation, other services), i.e. how much more or less a visitor-day uses of what both
profiles use. It is NOT the relative weight of the two profiles on mobility or congestion, which is what the flow
indicators are about, and there the day visitors could weigh MORE than the tourists. It also ignores the (large)
accommodation energy of the tourists on purpose. If the indicators are read as "mobility pressure", the
energy weights are a proxy of the intensity of use of local services, not a neutral choice. The alternatives are legitimate:
  * COVISITS_WEIGHT_MODE = "fixed" with weights decided by the domain experts (e.g. 1 / 0.3), or 1 / 1 (plain pooling);
  * a different target (traffic counts, waste): pass its monthly frame as a target of estimate_profile_weights;
  * different weights per indicator (not implemented: the combination is done once, in flows.combine_profiles).
The per-profile columns (`*_tourist`, `*_visitor`) are always kept, so the choice can be revisited without recomputing
the OD data.

Known weaknesses of the estimate (what to check in `diagnostics` of the saved json):
  1. Indirect proxy: the consumption of the shared categories also comes from residents (restaurants, transport and
     personal services serve them too), second homes, businesses. Fixed
     effects remove what is constant per comune or common to all comuni in a month, not what varies in between.
  2. Seasonality is only partly controlled: comune-specific seasonality (altitude, climate) that is not captured by
     size x common seasonality, and happens to correlate with the profile mix, leaks into the coefficients.
  3. Collinearity: tourists and visitors peak in the same months. `collinearity` close to 1, a wide `ratio_ci95`, or a
     low `r2_within` mean the data do not separate the two effects; the code then warns, or falls back.
  4. Measurement error in the presences (Vodafone counts, split of the areas into comuni by beds) attenuates the
     coefficients, possibly differently for the two profiles, so the RATIO is biased in an unknown direction.
  5. Time span: with few years the identification comes mostly from the cross-section of the comuni.
  6. Validation is only synthetic (tests/): on simulated data with a known weight the method recovers it; on real data
     there is no ground truth, so treat the number as an informed estimate, not as a measurement.
If the diagnostics are poor, prefer "fixed" weights chosen explicitly over a number nobody can defend.

Method
======
Idea. A user-day of a tourist (overnight stay) and one of a visitor (excursionist) do not weigh the same on
a territory. We measure the weight as the *marginal pressure* that a user-day of each profile puts on the
comune, observed through a physical proxy that both profiles drive: the monthly consumption of energy
(electricity / gas, shared service categories) of the comune. Model, per comune c and month m, for each target y:

        y[c,m]  =  s[c] * g[m]  +  sum_p  beta_p * P_p[c,m]  +  e[c,m]                  beta_p >= 0

  * P_p[c,m]  user-days of the profile p present in the comune (ND(*->j) of the OD co-visits, areas -> comuni
    split by beds, like the Vodafone presences; months with missing days are rescaled to the full month)
  * s[c] * g[m]  the consumption that does not depend on the guests: a size of the comune times a common
    seasonality (weather, tariffs). Dividing by s[c] gives  y/s = g[m] + sum_p beta_p P_p/s + e/s, i.e. a
    regression with month effects (plus comune effects, for robustness) on size-free variables, so the comuni are
    comparable and the weights are learnt from how the *mix* of the profiles lifts the consumption of a comune
    above its own level, not from the size of the comuni or from the season
  * s[c] is not observed: it is the mean consumption of the comune net of the guests, s[c] = ybar[c] -
    sum_p beta_p Pbar_p[c], found by a fixed-point iteration together with beta
  * beta_p >= 0 (non-negative least squares, solved exactly): a presence cannot lower the consumption

The weight of a profile is its pressure relative to the reference profile, w_p = beta_p / beta_ref (the
reference, TOURIST, has weight 1). Uncertainty: cluster bootstrap over the comuni. An estimate is used only if
it is reliable (enough comuni, the reference profile has a clearly positive effect in almost every bootstrap
draw); the estimates of the different targets are averaged with inverse-variance weights. Otherwise the
pipeline falls back to the configured neutral weights and says why (`source` / `reason` in the output).

Caveats (also in the README): the proxy is indirect, tourists and visitors move together over the year (their
collinearity is reported), and the result is only as good as the consumption data: read `diagnostics` and, if
needed, set fixed weights in utils/datasets.py (COVISITS_WEIGHT_MODE = "fixed").
"""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils.disaggregation import beds_weights

logger = logging.getLogger(__name__)

STAR = "*"


# ------------------------------------------------------------------ presences per comune and month
def area_month_presence(od: pd.DataFrame, profiles) -> pd.DataFrame:
    """User-days present per (userProfile, areaId, month): T(j) = ND(*->j), summed over the days of the month.

    Same convention of flows.py (rows with source '*', else rows with target '*'). The month total is
    rescaled to the full month when some days of the month are missing in the data.
    Returns: userProfile, areaId, month ('YYYY-MM'), T."""
    nd = od[(od["covisitType"] == "ND") & od["userProfile"].isin(profiles)].copy()
    nd["month"] = nd["DATA"].astype(str).str[:7]
    keys = ["userProfile", "areaId", "month"]
    src = (
        nd[nd["sourceId"] == STAR]
        .rename(columns={"targetId": "areaId"})
        .groupby(keys)["value"]
        .sum()
    )
    tgt = (
        nd[nd["targetId"] == STAR]
        .rename(columns={"sourceId": "areaId"})
        .groupby(keys)["value"]
        .sum()
    )
    t = src.combine_first(tgt).rename("T").reset_index()

    covered = nd.groupby("month")["DATA"].nunique()
    in_month = pd.PeriodIndex(covered.index, freq="M").days_in_month
    scale = pd.Series(in_month / covered.values, index=covered.index)
    t["T"] = t["T"] * t["month"].map(scale)
    return t


def comune_month_presence(area_pres, area_comune, strutture=None, bed_col="tot_postiletto"):
    """Areas -> comuni: an area that covers several comuni is split by their beds (uniform without beds
    or where an area has no beds), as in process_vodafone.

    Returns the wide frame ID_COMUNE, month, P_<profile> (user-days)."""
    years = sorted({int(m[:4]) for m in area_pres["month"].unique()})
    ac = area_comune.drop_duplicates(["areaId", "ID_COMUNE"]).copy()
    frames = []
    for y in years:
        f = ac.copy()
        f["year"] = str(y)
        frames.append(f)
    ac = pd.concat(frames, ignore_index=True)
    if strutture is not None:
        bw = beds_weights(strutture, years, bed_col)
        bw["year"] = bw["DATA"].str[:4]
        ac = ac.merge(
            bw[["year", "ID_COMUNE", bed_col]], on=["year", "ID_COMUNE"], how="left"
        )
        ac[bed_col] = ac[bed_col].fillna(0.0)
    else:
        ac[bed_col] = 0.0
    tot = ac.groupby(["year", "areaId"])[bed_col].transform("sum")
    n = ac.groupby(["year", "areaId"])[bed_col].transform("size")
    ac["share"] = np.where(tot > 0, ac[bed_col] / tot.where(tot > 0, 1.0), 1.0 / n)

    p = area_pres.copy()
    p["year"] = p["month"].str[:4]
    p = p.merge(ac[["year", "areaId", "ID_COMUNE", "share"]], on=["year", "areaId"])
    p["P"] = p["T"] * p["share"]
    wide = p.pivot_table(
        index=["ID_COMUNE", "month"], columns="userProfile", values="P", aggfunc="sum"
    )
    wide.columns = [f"P_{c}" for c in wide.columns]
    return wide.fillna(0.0).reset_index()


def monthly_target(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Daily (DATA, ID_COMUNE, col) -> monthly total per comune: ID_COMUNE, month, y."""
    d = df[["DATA", "ID_COMUNE", col]].copy()
    d["month"] = d["DATA"].astype(str).str[:7]
    return (
        d.groupby(["ID_COMUNE", "month"], as_index=False)[col]
        .sum(min_count=1)
        .rename(columns={col: "y"})
        .dropna(subset=["y"])
    )


def normalize_code(x) -> str:
    """Usage code as a plain string: 551, 551.0 and ' 551 ' are the same category."""
    t = str(x).strip()
    return t[:-2] if t.endswith(".0") and t[:-2].isdigit() else t


def build_energy_targets(monthly_by_carrier: dict, categories: dict) -> dict:
    """Consumption targets of the weight estimation, built from the consumption BY CATEGORY.

    monthly_by_carrier  {carrier: DataFrame(ID_COMUNE, month, code, y)}  e.g. {"ee": ..., "gas": ...}
    categories          {usage code: role}; role "shared" = used by BOTH profiles (restaurants, transport, ...),
                        "tourist_only" = used only by overnight guests (hotels), anything else = ignored.

    Returns {"targets":    {"<carrier>_shared": monthly sum of the shared categories}   -> estimate the weight,
             "diagnostic": {"<carrier>_<code>": one shared category}                      -> reported, not used,
             "placebo":    {"<carrier>_<code>": one tourist_only category}                -> validity check,
             "codes_found": {carrier: sorted codes in the data}}
    A category missing in a comune-month counts as 0 in the sum of the shared categories (no report = no consumption)."""
    bad = {c: r for c, r in categories.items() if r not in ("shared", "tourist_only", "ignore")}
    if bad:
        raise ValueError(f"unknown category roles {bad}: use 'shared', 'tourist_only' or 'ignore'")
    roles = {normalize_code(c): r for c, r in categories.items()}
    out = {"targets": {}, "diagnostic": {}, "placebo": {}, "codes_found": {}}
    for carrier, df in monthly_by_carrier.items():
        df = df.assign(code=df["code"].map(normalize_code))
        out["codes_found"][carrier] = sorted(df["code"].unique())
        shared = [c for c, r in roles.items() if r == "shared"]
        sel = df[df["code"].isin(shared)]
        if not sel.empty:
            out["targets"][f"{carrier}_shared"] = sel.groupby(["ID_COMUNE", "month"], as_index=False)["y"].sum()
        for code, role in roles.items():
            one = df[df["code"] == code]
            if one.empty or role not in ("shared", "tourist_only"):
                continue
            key = "diagnostic" if role == "shared" else "placebo"
            out[key][f"{carrier}_{code}"] = one[["ID_COMUNE", "month", "y"]].reset_index(drop=True)
    return out


def _as_monthly(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """A target is either already monthly (ID_COMUNE, month, y) or daily (DATA, ID_COMUNE, <name>)."""
    if {"month", "y"} <= set(df.columns):
        return df[["ID_COMUNE", "month", "y"]]
    return monthly_target(df, name)


# ------------------------------------------------------------------ estimation
def _demean_two_way(M, g1, g2, n_iter=200, tol=1e-10):
    """Residuals of M (n x k) after removing the means of the groups g1 and g2 (alternating projections:
    the two-way fixed effects of an unbalanced panel)."""
    X = M.astype(float).copy()
    k = X.shape[1]
    for _ in range(n_iter):
        before = X.copy()
        for g in (g1, g2):
            cnt = np.bincount(g)
            for j in range(k):
                X[:, j] -= (np.bincount(g, weights=X[:, j]) / cnt)[g]
        if np.max(np.abs(X - before)) < tol:
            break
    return X


def _nnls_small(XtX, Xty):
    """min ||y - X b||^2 with b >= 0, exact for a handful of regressors: it tries every subset of
    regressors left free (the others at 0) and keeps the feasible solution with the smallest error."""
    k = len(Xty)
    best_b, best_sse = np.zeros(k), 0.0  # sse up to the constant y'y
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            S = list(S)
            A, c = XtX[np.ix_(S, S)], Xty[S]
            if abs(np.linalg.det(A)) < 1e-12 * max(1.0, np.trace(A) ** r):
                continue
            b_s = np.linalg.solve(A, c)
            if np.any(b_s < 0):
                continue
            b = np.zeros(k)
            b[S] = b_s
            sse = -2 * b @ Xty + b @ XtX @ b
            if sse < best_sse:
                best_b, best_sse = b, sse
    return best_b


def _fit(y, X, g1, g2):
    """NNLS of y on X after the removal of the two-way fixed effects (Frisch-Waugh).
    Returns beta (in the units of X) and the within R^2."""
    scale = X.std(axis=0)
    scale[scale <= 0] = 1.0
    Z = _demean_two_way(np.column_stack([y, X / scale]), g1, g2)
    yd, Xd = Z[:, 0], Z[:, 1:]
    beta = _nnls_small(Xd.T @ Xd, Xd.T @ yd)
    ss_tot = float(yd @ yd)
    ss_res = float(((yd - Xd @ beta) ** 2).sum())
    return beta / scale, (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _fit_scaled(y, P, g_com, g_time, *, max_iter=30, tol=1e-6, min_size=0.05):
    """beta of  y = s[c] g[m] + P beta  (see the module docstring) by fixed-point iteration on the size s[c].
    Returns beta (user-day units), the within R^2 of the last step and the scaled regressors."""
    cnt = np.bincount(g_com)
    ybar = np.bincount(g_com, weights=y) / cnt
    Pbar = np.column_stack(
        [np.bincount(g_com, weights=P[:, j]) / cnt for j in range(P.shape[1])]
    )
    s = ybar.copy()
    beta = np.zeros(P.shape[1])
    for _ in range(max_iter):
        ys, Ps = y / s[g_com], P / s[g_com][:, None]
        new, r2 = _fit(ys, Ps, g_com, g_time)
        s = np.maximum(ybar - Pbar @ new, min_size * ybar)
        done = np.max(np.abs(new - beta)) <= tol * max(np.max(np.abs(new)), 1e-12)
        beta = new
        if done:
            break
    return beta, r2, P / s[g_com][:, None]


def _codes(values):
    return pd.factorize(values)[0]


def estimate_target(
    panel, profiles, reference, *, n_boot, seed, min_comuni, max_zero_share
) -> dict:
    """Fits one target. panel: ID_COMUNE, month, y, P_<profile>... Returns coefficients, ratios and the checks."""
    cols = [f"P_{p}" for p in profiles]
    panel = panel.dropna(subset=["y"]).copy()
    panel = panel[panel["y"] > 0]
    y = panel["y"].to_numpy(float)
    P = panel[cols].to_numpy(float)
    g_com = _codes(panel["ID_COMUNE"])
    g_time = _codes(panel["month"])
    n_com = int(g_com.max()) + 1 if len(panel) else 0
    out = {
        "n_obs": int(len(panel)),
        "n_comuni": n_com,
        "ok": False,
        "reason": "",
    }
    if n_com < min_comuni:
        out["reason"] = f"only {n_com} comuni (< {min_comuni})"
        return out
    if np.any(P.std(axis=0) <= 0):
        out["reason"] = "a profile has no variation"
        return out

    beta, r2, Xs = _fit_scaled(y, P, g_com, g_time)
    ref = profiles.index(reference)
    corr = float(np.corrcoef(Xs[:, 0], Xs[:, 1])[0, 1]) if len(profiles) == 2 else np.nan

    rng = np.random.default_rng(seed)
    rows_by_com = [np.flatnonzero(g_com == c) for c in range(n_com)]
    boots = np.zeros((n_boot, len(profiles)))
    for b in range(n_boot):
        pick = rng.integers(0, n_com, n_com)
        rows = np.concatenate([rows_by_com[c] for c in pick])
        g1 = np.repeat(np.arange(n_com), [len(rows_by_com[c]) for c in pick])
        boots[b], _, _ = _fit_scaled(y[rows], P[rows], g1, g_time[rows])

    zero_share = float((boots[:, ref] <= 1e-12).mean())
    ratio_point = beta / beta[ref] if beta[ref] > 1e-12 else np.full(len(profiles), np.nan)
    valid = boots[:, ref] > 1e-12
    ratio_boot = boots[valid] / boots[valid][:, [ref]]
    out.update(
        beta=dict(zip(profiles, beta.tolist())),
        r2_within=float(r2),
        collinearity=corr,
        ref_zero_share=zero_share,
        ratio=dict(zip(profiles, ratio_point.tolist())),
        ratio_ci95={
            p: [float(np.percentile(ratio_boot[:, j], 2.5)), float(np.percentile(ratio_boot[:, j], 97.5))]
            for j, p in enumerate(profiles)
        }
        if len(ratio_boot) >= 20
        else None,
        ratio_se={p: float(ratio_boot[:, j].std()) for j, p in enumerate(profiles)}
        if len(ratio_boot) >= 20
        else None,
    )
    if zero_share > max_zero_share:
        out["reason"] = (
            f"the effect of {reference} is not clearly positive (zero in {zero_share:.0%} of the bootstrap draws)"
        )
    elif not np.all(np.isfinite(ratio_point)) or out["ratio_se"] is None:
        out["reason"] = "ratio not finite"
    else:
        out["ok"] = True
    return out


def estimate_profile_weights(
    od,
    area_comune,
    targets,
    strutture=None,
    *,
    profiles,
    reference=None,
    diagnostic_targets=None,
    placebo_targets=None,
    placebo_max=0.2,
    n_boot=300,
    seed=0,
    min_comuni=30,
    max_zero_share=0.05,
    min_weight=0.05,
    fallback=None,
) -> dict:
    """Weights {profile: w} (reference profile = 1) learnt from the consumption proxies.

    od                  standardized OD rows (DATA, covisitType, sourceId, targetId, userProfile, value)
    area_comune         areaId -> ID_COMUNE
    targets             {name: monthly dataframe (ID_COMUNE, month, y)} used to estimate the weight, see
                        build_energy_targets (a daily (DATA, ID_COMUNE, <name>) frame is accepted too)
    diagnostic_targets  same format, fitted and reported in `diagnostics` but NOT used (e.g. one category at a time,
                        to see whether the categories agree)
    placebo_targets     same format: consumption that only the reference profile should drive (hotels: visitors do not
                        sleep there). The weight of the other profiles must come out ~0: if even the LOWER end of its
                        95% interval is above `placebo_max` the model is mixing up the profiles (confounding) and the
                        weights are NOT used (fallback). No placebo / inconclusive placebo: warning only.
    strutture           processed strutture (beds, to split the areas among their comuni)
    fallback            {profile: w} used if the estimate is not reliable (default: all 1 = plain pooling)
    min_weight          lower bound of a weight, so that a profile is never dropped completely

    Returns {"weights", "source" ('estimated' | 'fallback'), "reason", "reference", "diagnostics"}."""
    profiles = list(profiles)
    reference = reference or profiles[0]
    others = [p for p in profiles if p != reference]
    fallback = dict(fallback or {p: 1.0 for p in profiles})
    diagnostics = {"targets": {}, "diagnostic_targets": {}, "placebo": {}}

    def _fallback(reason):
        logger.warning("[profile_weights] fallback weights %s: %s", fallback, reason)
        return {
            "weights": fallback,
            "source": "fallback",
            "reason": reason,
            "reference": reference,
            "diagnostics": diagnostics,
        }

    area_pres = area_month_presence(od, profiles)
    if area_pres.empty or set(area_pres["userProfile"]) != set(profiles):
        return _fallback("no presences (ND rows) for every profile")
    pres = comune_month_presence(area_pres, area_comune, strutture)
    for p in profiles:
        if f"P_{p}" not in pres:
            pres[f"P_{p}"] = 0.0

    def _fit_target(name, df):
        panel = _as_monthly(df, name).merge(pres, on=["ID_COMUNE", "month"], how="inner")
        return estimate_target(
            panel, profiles, reference, n_boot=n_boot, seed=seed,
            min_comuni=min_comuni, max_zero_share=max_zero_share,
        )

    # ---- placebo: the visitors must NOT explain the consumption of the tourist-only categories
    for name, df in (placebo_targets or {}).items():
        res = _fit_target(name, df)
        status = "inconclusive"
        if res["ok"]:
            worst = max(res["ratio_ci95"][p][0] for p in others)
            status = "failed" if worst > placebo_max else "passed"
        res["status"] = status
        diagnostics["placebo"][name] = res
        logger.info(
            "[profile_weights] placebo %s: %s (ratio %s, %s)", name, status, res.get("ratio"),
            res.get("reason") or "95%% interval of the other profiles' ratio: %s" % res.get("ratio_ci95"),
        )
        if status == "failed":
            return _fallback(
                f"placebo {name} failed: the profiles other than {reference} explain the consumption of a "
                f"{reference}-only category (ratio {res['ratio']}, 95% interval {res['ratio_ci95']}): "
                "the model cannot separate the profiles"
            )
    if not diagnostics["placebo"]:
        logger.warning("[profile_weights] no placebo category available: the validity of the model is not checked")
    elif all(r["status"] == "inconclusive" for r in diagnostics["placebo"].values()):
        logger.warning(
            "[profile_weights] the placebo is inconclusive (%s): the presence of %s does not clearly explain "
            "its own consumption, check the presence data",
            {k: v["reason"] for k, v in diagnostics["placebo"].items()}, reference,
        )

    for name, df in (diagnostic_targets or {}).items():
        diagnostics["diagnostic_targets"][name] = _fit_target(name, df)

    ok = {}
    for name, df in targets.items():
        res = _fit_target(name, df)
        diagnostics["targets"][name] = res
        logger.info(
            "[profile_weights] %s: %s comuni, %s obs, ratio %s, %s",
            name, res["n_comuni"], res["n_obs"], res.get("ratio"),
            "used" if res["ok"] else f"discarded ({res['reason']})",
        )
        if res["ok"]:
            ok[name] = res
            for prof, (lo, hi) in res["ratio_ci95"].items():
                if prof != reference and hi - lo > res["ratio"][prof]:
                    logger.warning(
                        "[profile_weights] %s: the weight of %s is imprecise (95%% interval %.2f-%.2f)",
                        name, prof, lo, hi,
                    )
    if not ok:
        return _fallback(
            "no reliable estimate: "
            + "; ".join(f"{k}: {v['reason']}" for k, v in diagnostics["targets"].items())
            if diagnostics["targets"]
            else "no consumption target available (no configured usage code found in the data)"
        )

    weights = {}
    for p in profiles:
        r = np.array([o["ratio"][p] for o in ok.values()])
        var = np.array([max(o["ratio_se"][p] ** 2, 1e-8) for o in ok.values()])
        weights[p] = 1.0 if p == reference else float(max(np.sum(r / var) / np.sum(1 / var), min_weight))
    for p, w in weights.items():
        if w > 1.0:
            logger.warning(
                "[profile_weights] %s weighs MORE than the reference %s (%.2f): check the diagnostics", p, reference, w
            )
    logger.info("[profile_weights] estimated weights: %s (targets used: %s)", weights, sorted(ok))
    return {
        "weights": weights,
        "source": "estimated",
        "reason": f"targets used: {sorted(ok)}",
        "reference": reference,
        "diagnostics": diagnostics,
    }


def fixed_profile_weights(weights, reference=None, reason="set in the configuration") -> dict:
    weights = dict(weights)
    return {
        "weights": weights,
        "source": "fixed",
        "reason": reason,
        "reference": reference or next(iter(weights)),
        "diagnostics": {},
    }


# ------------------------------------------------------------------ persistence
# The weights are estimated ONCE, by the base build, and saved: the yearly updates reuse them, so that the
# years of the base build and of the updates are comparable (re-estimating would change the old years too).
def save_profile_weights(result: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return path


def load_profile_weights(path):
    """The saved result, or None if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
