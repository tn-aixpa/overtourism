from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from overtourism.overtourism.backend_extension.api.models.indicator import Indicator
from overtourism.overtourism.backend_extension.api.models.trenitno_phenomena import (
    BedsPhenomenon,
    ExtraBedsPhenomenon,
    ExtraFacilitiesPhenomenon,
    FacilitiesTotalPhenomenon,
    FlowIndicatorPhenomenon,
    PopulationPhenomenon,
    PresencesPhenomenon,
)
from overtourism.overtourism.setup import index_data_path

# ---------------------------------------------------------------------------
# Source paths
# ---------------------------------------------------------------------------

MAP_SHAPEFILE = index_data_path / "Com01012026_g" / "Com01012026_g_WGS84.shp"
MAP_IDS = index_data_path / "cities_gdf_base_columns.geojson"
POPULATION_SOURCE = index_data_path / "phen_popolazione.parquet"
STRUCTURES_SOURCE = index_data_path / "phen_strutture.parquet"
PRESENCES_SOURCE = index_data_path / "phen_presenze.parquet"
FLOWS_OD_SOURCE = (
    index_data_path / "phen_flussi_od.parquet"
)  # yearly OD flow indicators per comune
MACRO_AREAS_FILE = index_data_path / "map_comuni_into_apt.json"
CODICI_COMUNI_FILE = index_data_path / "mapping_comuni_ISTAT.json"

# ---------------------------------------------------------------------------
# Types definition
# ---------------------------------------------------------------------------
INDICATORS_TYPES = [
    "Gestione della Sostenibilità",
    "Impatti socio-economici",
    "Impatti culturali",
    "Impatti ambientali",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Number of classes of the discretized PageRank map (quantiles, 1 = lowest ... N = highest)
PAGERANK_N_CLASSES = 10

_VARIATION_RATE_KEY = "tasso-variazione"
_PERIOD_PERCENTAGE_IMPACT_KEY = "incidenza-periodo"

_REGISTRY: dict[str, Callable[[], Indicator]] = {
    "ricettivita": lambda: AccommodationCapacityIndicator(
        STRUCTURES_SOURCE, POPULATION_SOURCE
    ),
    "turisticita": lambda: TourismIndexIndicator(
        POPULATION_SOURCE, PRESENCES_SOURCE, "presenze_vodafone"
    ),
    "ospitalita-strutture": lambda: HospitalityIndexFacilitiesIndicator(
        STRUCTURES_SOURCE
    ),
    "ospitalita-letti": lambda: HospitalityIndexBedsIndicator(STRUCTURES_SOURCE),
    "turismo-sommerso": lambda: HiddenTourismIndicator(PRESENCES_SOURCE),
    "quota-mobilita-ingresso": lambda: InboundMobilityShareIndicator(FLOWS_OD_SOURCE),
    "quota-mobilita-uscita": lambda: OutboundMobilityShareIndicator(FLOWS_OD_SOURCE),
    "quota-intra-ambito": lambda: IntraAmbitoShareIndicator(FLOWS_OD_SOURCE),
    "pagerank-discretizzato": lambda: DiscretizedPageRankIndicator(
        FLOWS_OD_SOURCE, PAGERANK_N_CLASSES
    ),
    _VARIATION_RATE_KEY: lambda: VariationRateIndicator(),
    _PERIOD_PERCENTAGE_IMPACT_KEY: lambda: PeriodPercentageImpactIndicator(),
}

_INDICATOR_CACHE: dict[str, Indicator] = {}


def get_indicator(key: str) -> Indicator:
    if key not in _INDICATOR_CACHE:
        if key not in _REGISTRY:
            raise KeyError(f"Unknown indicator '{key}'. Available: {sorted(_REGISTRY)}")
        _INDICATOR_CACHE[key] = _REGISTRY[key]()
    return _INDICATOR_CACHE[key]


# ---------------------------------------------------------------------------
# Concrete indicators
# ---------------------------------------------------------------------------


class AccommodationCapacityIndicator(Indicator):
    """Beds per resident: NUMLETTI_TOT / POPOLAZIONE."""

    name = "Indice di ricettività"
    type = INDICATORS_TYPES[1]
    description = "L'indice di ricettività definisce il <strong>rapporto</strong> fra i <strong>letti presenti negli esercizi ricettivi</strong> e gli <strong>abitanti</strong> di una stessa area. L'indice è una misura della capacità turistica rispetto alla dimensione, in termini di popolazione, di un'area. L'indice è calcolato partendo dai dati ISPAT relativi alla popolazione residente e alla consistenza degli esercizi alberghieri e extra-alberghieri."
    index_value_unit_description = "Rapporto posti letto / popolazione"

    def __init__(self, source_file1, source_file2):
        super().__init__(
            phenomena=[
                BedsPhenomenon(source_file1),
                PopulationPhenomenon(source_file2),
            ],
            combinator=self.divide("beds", "population"),
        )


class TourismIndexIndicator(Indicator):
    """Total presences per resident."""

    name = "Indice di turisticità"
    type = INDICATORS_TYPES[1]
    description = "L'indice di turisticità definisce il <strong>rapporto</strong> fra il <strong>numero medio giornaliero di turisti negli esercizi ricettivi</strong> di una specifica area e il<strong>numero di abitanti</strong> della stessa area. L'indice fornisce una misura dell'effettiva incidenza del turismo rispetto alla dimensione, in termini di popolazione, di un'area. L'indice è calcolato partendo dai dati Vodafone per quanto riguarda le presenze turistiche e dai dati ISPAT relativi alla popolazione residente."
    index_value_unit_description = "Rapporto fra numero turisti e popolazione"

    def __init__(self, source_file1, source_file2, presences_col_name):
        super().__init__(
            phenomena=[
                PresencesPhenomenon(source_file2, presences_col_name),
                PopulationPhenomenon(source_file1),
            ],
            combinator=self.divide("presences", "population"),
        )


class HospitalityIndexFacilitiesIndicator(Indicator):
    """Share of non-hotel facilities over total facilities."""

    name = "Indice di incidenza ospitalità non convenzionale (strutture)"
    type = INDICATORS_TYPES[1]
    description = "Questo indice di incidenza dell'ospitalità non convenzionale misura il <strong>rapporto</strong> fra il  <strong>numero di strutture ricettive non convenzionali</strong> e il <strong>numero totale delle strutture</strong> presenti in un'area. L'indice è calcolato partendo dai dati ISPAT relativi al numero degli esercizi alberghieri e extra-alberghieri."
    index_value_unit_description = (
        "Percentuale strutture ricettive non conv. rispetto al totale"
    )

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                ExtraFacilitiesPhenomenon(source_file),
                FacilitiesTotalPhenomenon(source_file),
            ],
            combinator=self.divide("extra_Facilities", "Facilities_total"),
        )


class HospitalityIndexBedsIndicator(Indicator):
    """Share of non-hotel beds over total beds."""

    name = "Indice di incidenza ospitalità non convenzionale (posti letto)"
    type = INDICATORS_TYPES[1]
    description = "Questo indice di incidenza dell'ospitalità non convenzionale misura il <strong>rapporto</strong> fra il <strong>numero di posti letto in strutture ricettive non convenzionali</strong> e il <strong>numero totale di posti letto</strong> in tutte le strutture di un'area. L'indice è calcolato partendo dai dati ISPAT al numero degli esercizi alberghieri e extra-alberghieri."
    index_value_unit_description = (
        "Percentuale letti in strututre ricettive non conv. rispetto al totale"
    )

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                ExtraBedsPhenomenon(source_file),
                BedsPhenomenon(source_file),
            ],
            combinator=self.divide("extra_beds", "beds"),
        )


class HiddenTourismIndicator(Indicator):
    """Calculates the hidden tourism indicator as the ratio between Vodafone registered attendences and accomodancy ones"""

    name = "Indice di turismo sommerso"
    type = INDICATORS_TYPES[0]
    description = "L'indice misura il <strong>rapporto</strong> fra le <strong>presenze di turisti misurate</strong> attraverso l'analisi di dati da rete di telefonia mobile e le <strong>presenze ufficiali</strong> di turisti in strutture alberghiere e extra-alberghiere. L'indice è stato calcolato partendo dai dati ISPAT sul movimento turistico e dai dati Vodafone relativi alle presenze misurate."
    index_value_unit_description = "Rapporto fra presenze misurate e ufficiali"

    def __init__(self, source_file):
        super().__init__(
            phenomena=[
                PresencesPhenomenon(
                    source_file, col="presenze_vodafone", name="presenze_vodafone"
                ),
                PresencesPhenomenon(
                    source_file,
                    col="presenze_alb",
                    name="presenze_alb",
                ),
                PresencesPhenomenon(
                    source_file,
                    col="presenze_xalb",
                    name="presenze_xalb",
                ),
            ],
            combinator=self.compute_hidden_factor,
        )

    def compute_hidden_factor(
        self,
        df: pd.DataFrame,
        **extra,
    ) -> pd.Series:
        denominator = df["presenze_alb"] + df["presenze_xalb"]
        return (
            df["presenze_vodafone"]
            .div(denominator.where(denominator != 0))
            .fillna(df["presenze_vodafone"])
        )


# ---------------------------------------------------------------------------
# OD flow indicators (Vodafone co-visits between areas)
# ---------------------------------------------------------------------------
# They read `phen_flussi_od`: one row per comune and YEAR with, for every metric, a column for the TOURIST
# profile, one for the VISITOR (excursionist) profile and one for their weighted union (`<metric>_combined`).
# The union is computed in the data preparation on the user-days (a visitor-day counts `w` tourist-days,
# w = 0.5 fixed, see COVISITS_PROFILE_WEIGHTS in data_preparation), so the indicators here only read it.
# The values are yearly: a requested window returns the value of its year(s) (mean over the years of the window),
# and the seasonality filter does not change them.

_FLOW_PROFILE_SUFFIXES = ("combined", "tourist", "visitor")
_FLOW_COMMON_DESCRIPTION = (
    " Turisti ed escursionisti sono <strong>aggregati con un peso per ciascun profilo</strong>, "
    "stimato dai dati (una giornata-utente di escursionista non pesa come una di turista); "
    "i valori per singolo profilo sono riportati a fianco. "
    "L'indice è annuale e si basa sui dati Vodafone di co-visitazione fra aree."
)


class _FlowIndicator(Indicator):
    """Base of the indicators that read one metric of phen_flussi_od (tourists + excursionists, weighted)."""

    metric: str = ""
    type = INDICATORS_TYPES[0]

    def __init__(self, source_file, combinator=None):
        super().__init__(
            phenomena=[
                FlowIndicatorPhenomenon(
                    source_file, f"{self.metric}_{suffix}", f"{self.metric}_{suffix}"
                )
                for suffix in _FLOW_PROFILE_SUFFIXES
            ],
            combinator=combinator or self._weighted_union,
        )

    def _weighted_union(self, df: pd.DataFrame, **_extra) -> pd.Series:
        return df[f"{self.metric}_combined"]


class InboundMobilityShareIndicator(_FlowIndicator):
    """UD(*->j) / T(j): share of the users present in j who arrive from another area."""

    metric = "quota_ingresso"
    name = "Flussi - Quota di mobilità in ingresso"
    description = (
        "La quota di mobilità in ingresso misura la <strong>quota degli utenti presenti in un'area che "
        "arrivano da un'altra area</strong> (rapporto fra gli utenti in arrivo da altre aree e gli utenti "
        "presenti). Valori alti indicano località di tappa, raggiunte durante uno spostamento; valori bassi "
        "indicano località di destinazione o di pernottamento stabile."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = (
        "Rapporto fra utenti in arrivo da altre aree e utenti presenti"
    )


class OutboundMobilityShareIndicator(_FlowIndicator):
    """UD(j->*) / T(j): share of the users present in j who go on to another area."""

    metric = "quota_uscita"
    name = "Flussi - Quota di mobilità in uscita"
    description = (
        "La quota di mobilità in uscita misura la <strong>quota degli utenti presenti in un'area che "
        "proseguono verso un'altra area</strong> (rapporto fra gli utenti diretti verso altre aree e gli "
        "utenti presenti). Valori alti indicano località da cui i visitatori ripartono per visitare altre aree."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = (
        "Rapporto fra utenti diretti verso altre aree e utenti presenti"
    )


class IntraAmbitoShareIndicator(_FlowIndicator):
    """Share of the UD flow of an area that stays inside its own ambito."""

    metric = "quota_intra_ambito"
    name = "Flussi - Quota intra-ambito"
    description = (
        "La quota intra-ambito misura la <strong>quota dei flussi di un'area (in ingresso e in uscita) "
        "scambiati con aree dello stesso ambito turistico</strong>; il resto è mobilità inter-ambito. "
        "Valori alti indicano un'area i cui flussi restano all'interno del proprio ambito."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = "Quota dei flussi con aree dello stesso ambito"


class DiscretizedPageRankIndicator(_FlowIndicator):
    """
    PageRank of the network of the flows between areas, discretized in ``n_classes`` quantile classes.

    INDICE is the class: 1 = lowest PageRank ... n_classes = highest, 0 = no data. Classes are quantiles of
    the comuni of the request (values that are tied stay in the same class, so the classes can have different
    sizes), so they show the *relative* position of a comune. The continuous PageRank values stay in the
    columns next to INDICE. Classes of different years are not comparable as a time series, hence
    ``availableForVariation = False``.
    """

    metric = "pagerank"
    name = "Flussi - PageRank discretizzato"
    description = (
        "Il PageRank misura la <strong>centralità di un'area nella rete dei flussi</strong> fra aree: è alto "
        "per le tappe che ricevono flusso da aree a loro volta rilevanti. Per la lettura su mappa il valore è "
        "<strong>discretizzato in classi di quantili</strong> (1 = centralità più bassa, "
        "{n} = più alta; 0 = dato non disponibile): le classi indicano la posizione relativa fra i comuni."
        + _FLOW_COMMON_DESCRIPTION
    )
    index_value_unit_description = "Classe di PageRank (quantili: 1 = più basso)"
    availableForVariation = False

    def __init__(self, source_file, n_classes: int = PAGERANK_N_CLASSES):
        if (
            not isinstance(n_classes, int)
            or isinstance(n_classes, bool)
            or n_classes < 2
        ):
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}")
        self.n_classes = n_classes
        self.description = self.description.format(n=n_classes)
        super().__init__(source_file, combinator=self._discretize)

    def _discretize(self, df: pd.DataFrame, **_extra) -> pd.Series:
        """Quantile class (1..n_classes) of the combined PageRank among the comuni of ``df``; NaN where missing."""
        value = df["pagerank_combined"]
        classes = pd.Series(np.nan, index=df.index, dtype=float)
        valid = value.notna()
        if valid.sum() < 2:
            return classes
        share_below = value[valid].rank(method="average", pct=True)
        classes[valid] = np.ceil(share_below * self.n_classes).clip(1, self.n_classes)
        return classes


class VariationRateIndicator(Indicator):
    """
    Generic "Tasso di variazione" (rate of variation) index.

    Computes the percentage change of *any other registered indicator*
    between two periods: the request's regular baseline period
    (``start_date`` / ``end_date``) and a comparison period
    (``start_date_comparison`` / ``end_date_comparison``). The same
    indicator is used as the phenomenon in both periods.

    Rather than hard-coding one variation subclass per wrapped indicator,
    the wrapped indicator is itself a parameter: it is selected at request
    time via the ``indicator`` extra field (its registry key, e.g.
    ``"indice-turisticita"``) — exactly like ``seasonality`` is an extra
    field on ``SeasonalityIndicator`` above. This keeps a single
    "tasso-variazione" entry in the registry that works for every
    combinable indicator.

    Because the wrapped indicator is only known once a request comes in,
    this class doesn't use the phenomena/combinator pipeline from the base
    class at all — ``_compute_indicator`` is fully overridden below, and
    ``phenomena``/``combinator`` are left empty/no-op.
    """

    name = "Tasso di variazione"
    description = "Il tasso di variazione misura la <strong>variazione percentuale</strong> di un indicatore fra il <strong>periodo selezionato</strong> ed un <strong>periodo storicizzato di confronto</strong>. È utile per confrontare due intervalli temporali (es. due stagioni, due anni) sullo stesso indice."
    availableForVariation = False
    extraFields = ["indicator", "start_date_comparison", "end_date_comparison"]
    internal_only = True
    index_value_unit_description = "Percentuale di variazione tra due periodi"

    def __init__(self):
        # No fixed phenomena: which indicator (and therefore which
        # phenomena) to use is only known at request time, via the
        # "indicator" extra field handled in _compute_indicator().
        super().__init__(phenomena=[], combinator=self._unused_combinator)

    @staticmethod
    def _unused_combinator(df: pd.DataFrame, **_extra) -> pd.Series:  # pragma: no cover
        # Never invoked: _compute_indicator() is fully overridden below and
        # bypasses the base class's phenomena-driven pipeline entirely.
        raise RuntimeError(
            "VariationRateIndicator.combinator should never be called directly"
        )

    @property
    def years_range(self) -> dict[str, int]:
        """
        Broadest year range across every combinable indicator in the
        registry (excluding this one), since the actual wrapped indicator
        isn't known until request time. Validity of the *chosen* indicator
        for the requested dates is enforced in ``_compute_indicator``.
        """
        if self._years_range is None:
            min_years, max_years = [], []
            for key in _REGISTRY:
                if key == _VARIATION_RATE_KEY:
                    continue
                try:
                    yr = get_indicator(key).years_range
                except ValueError:
                    continue
                min_years.append(yr["min_year"])
                max_years.append(yr["max_year"])

            if not min_years:
                raise ValueError(
                    f"{self!r}: no combinable indicators available to compute "
                    "a years_range for"
                )

            self._years_range = {
                "min_year": min(min_years),
                "max_year": max(max_years),
            }
        return self._years_range

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        indicator_key = extra.pop("indicator", None)
        start_date_comparison = extra.pop("start_date_comparison", None)
        end_date_comparison = extra.pop("end_date_comparison", None)

        if not indicator_key:
            raise ValueError(
                "'indicator' extra field is required for 'tasso-variazione'"
            )
        if indicator_key == _VARIATION_RATE_KEY:
            raise ValueError("'tasso-variazione' cannot wrap itself")
        if start_date_comparison is None or end_date_comparison is None:
            raise ValueError(
                "'start_date_comparison' and 'end_date_comparison' extra "
                "fields are required for 'tasso-variazione'"
            )

        wrapped = get_indicator(indicator_key)

        # Current period (e.g. 2025)
        current_df = wrapped.get_indicator(
            start_date=start_date,
            end_date=end_date,
            **extra,
        )

        # Reference period (e.g. 2024)
        previous_df = wrapped.get_indicator(
            start_date=start_date_comparison,
            end_date=end_date_comparison,
            **extra,
        )

        if current_df is None or previous_df is None:
            return None

        merged = (
            current_df[["ID_COMUNE", "INDICE"]]
            .rename(columns={"INDICE": "Periodo base"})
            .merge(
                previous_df[["ID_COMUNE", "INDICE"]].rename(
                    columns={"INDICE": "Periodo comparazione"}
                ),
                on="ID_COMUNE",
                how="outer",
            )
        )

        # Percentage variation: (baseline - comparison) / comparison * 100
        merged["INDICE"] = (
            (merged["Periodo base"] - merged["Periodo comparazione"])
            / merged["Periodo comparazione"].replace(0, np.nan)
        ) * 100

        return merged[["ID_COMUNE", "INDICE", "Periodo base", "Periodo comparazione"]]


class PeriodPercentageImpactIndicator(Indicator):
    """
    Percentage weight that a seasonal sub-period has on the total value of
    another indicator over the full requested period.

    Hybrid of VariationRateIndicator (wrapped indicator selected at request
    time via the "indicator" extra field) and SeasonalityIndicator (result
    expresses a sub-period's share of a total period, via "seasonality").

    result = 100 * INDICE(wrapped, full_period, seasonality=X)
                 / INDICE(wrapped, full_period, no seasonality filter)
    """

    name = "Incidenza percentuale del periodo"
    description = (
        "L'incidenza percentuale del periodo misura il <strong>peso "
        "percentuale</strong> che un <strong>sotto-periodo stagionale</strong> "
        "(es. alta stagione, periodo di spalla, weekend) ha sul "
        "<strong>totale del periodo selezionato</strong>, per un indicatore "
        "a scelta."
    )
    availableForVariation = False
    extraFields = ["indicator", "seasonality"]
    internal_only = True
    index_value_unit_description = (
        "Percentuale di impatto del sottoperiodo rispetto al periodo totale"
    )

    def __init__(self):
        # Same rationale as VariationRateIndicator: the wrapped indicator is
        # only known at request time via the "indicator" extra field, so the
        # base class's phenomena/combinator pipeline is bypassed entirely.
        super().__init__(phenomena=[], combinator=self._unused_combinator)

    @staticmethod
    def _unused_combinator(df: pd.DataFrame, **_extra) -> pd.Series:  # pragma: no cover
        raise RuntimeError(
            "PeriodPercentageImpactIndicator.combinator should never be called directly"
        )

    @property
    def years_range(self) -> dict[str, int]:
        """Broadest year range across every combinable indicator, same
        rationale as VariationRateIndicator.years_range."""
        if self._years_range is None:
            min_years, max_years = [], []
            for key in _REGISTRY:
                if key == _PERIOD_PERCENTAGE_IMPACT_KEY:
                    continue
                try:
                    yr = get_indicator(key).years_range
                except ValueError:
                    continue
                min_years.append(yr["min_year"])
                max_years.append(yr["max_year"])

            if not min_years:
                raise ValueError(
                    f"{self!r}: no combinable indicators available to compute "
                    "a years_range for"
                )

            self._years_range = {
                "min_year": min(min_years),
                "max_year": max(max_years),
            }
        return self._years_range

    def _compute_indicator(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        **extra,
    ) -> pd.DataFrame | None:
        indicator_key = extra.pop("indicator", None)
        seasonality = extra.pop("seasonality", None)

        if not indicator_key:
            raise ValueError(
                "'indicator' extra field is required for 'incidenza-periodo'"
            )
        if indicator_key == _PERIOD_PERCENTAGE_IMPACT_KEY:
            raise ValueError("'incidenza-periodo' cannot wrap itself")
        if indicator_key == _VARIATION_RATE_KEY:
            # Not strictly wrong, but tasso-variazione already needs its own
            # extra fields (start/end_date_comparison); reject early with a
            # clear message rather than a confusing missing-field error.
            raise ValueError("'incidenza-periodo' cannot wrap 'tasso-variazione'")
        if not seasonality:
            raise ValueError(
                "'seasonality' extra field is required for 'incidenza-periodo'"
            )

        wrapped = get_indicator(indicator_key)

        total_df = wrapped.get_indicator(
            start_date=start_date,
            end_date=end_date,
            **extra,
        )
        seasonal_df = wrapped.get_indicator(
            start_date=start_date,
            end_date=end_date,
            seasonality=seasonality,
            **extra,
        )

        if total_df is None or seasonal_df is None:
            return None

        merged = (
            total_df[["ID_COMUNE", "INDICE"]]
            .rename(columns={"INDICE": "INDICE_periodo"})
            .merge(
                seasonal_df[["ID_COMUNE", "INDICE"]].rename(
                    columns={"INDICE": "INDICE_sottoperiodo"}
                ),
                on="ID_COMUNE",
                how="outer",
            )
        )
        merged["INDICE_sottoperiodo"] = merged["INDICE_sottoperiodo"].fillna(0)

        merged["INDICE"] = (
            merged["INDICE_sottoperiodo"] / merged["INDICE_periodo"].replace(0, np.nan)
        ) * 100

        return merged[["ID_COMUNE", "INDICE", "INDICE_periodo", "INDICE_sottoperiodo"]]
