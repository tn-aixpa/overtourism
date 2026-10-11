# SPDX-License-Identifier: Apache-2.0
"""Layer 3 computation backend for the Fazzon model — see `overtourism/BACKEND_DESIGN.md` §5.

The single evaluation path for the Fazzon model: builds its own seeded
`CrossProductEnsemble`/`Evaluation` and targets
`overtourism.model.common.sustainability_field.SustainabilityFieldOutput`
directly via the shared field-math functions.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import numpy as np
from civic_digital_twins.dt_model import (
    CategoricalIndex,
    CrossProductEnsemble,
    Evaluation,
    Scenario,
    build_scenario,
    sample_across,
)

from src.model.common.sustainability_field import (
    OvertourismEvaluationConfig,
    OvertourismParameterMeta,
    SustainabilityFieldOutput,
    arrange_frontend_data,
    compute_sustainability_field,
)
from src.model.fazzon.fazzon_model import FazzonModel
from src.model.fazzon.fazzon_presence_stats import (
    PRICE_INDEX_NAME,
    PRICE_LEVELS,
    shuttle_capacity_level,
    shuttle_usable_capacity,
)


def price_level(price: float) -> str:
    """The price level (5-euro class, D15) of a price in euro: nearest class, clipped to the
    levels' range."""
    step = PRICE_LEVELS[1] - PRICE_LEVELS[0]
    level = int(round(float(price) / step) * step)
    return str(min(max(level, PRICE_LEVELS[0]), PRICE_LEVELS[-1]))


class FazzonBackend:
    """Frontend-agnostic computation backend for the Fazzon model.

    Constructor builds the model, the index map, and holds an
    `OvertourismEvaluationConfig` with fixed seeds and sample counts — seeds
    are a compute-quality concern and live here, not in the frontend.
    """

    # Subsystem/constraint display names and axis labels — the backend's own
    # minimal, structural presentation metadata, hardcoded here the same way
    # `_schema` hardcodes parameter metadata. `schema()` returns these
    # verbatim
    MAPPER: ClassVar[dict[str, str]] = {
        "parking": "Parcheggi",
        "road": "Viabilità",
        "food": "Ristorazione",
    }
    # The axes are the arrivals under the access policy (D15): the policy moves the samples.
    X_AXIS_NAME = "Visitatori arrivati in auto / giorno"
    Y_AXIS_NAME = "Visitatori arrivati con altri mezzi / giorno"
    # TODO: check: are thesepointers really needed?
    X_FIELD = "visitors_car"
    Y_FIELD = "visitors_other"

    def __init__(self) -> None:
        model = FazzonModel(inputs=FazzonModel.default_inputs())
        self._model = model
        self._index_map: dict[str, Any] = {idx.name: idx for idx in model.indexes}
        self._parameter_axes = [
            model.inputs.pv_visitors_car,
            model.inputs.pv_visitors_other,
        ]
        # Grid-resolution: structural axis choices, not per-evaluation config.
        # Non-car arrivals reach ~1600/day under a strong price policy (modal shift, D15).
        self._c_max, self._o_max = 2000, 2000
        self._c_sample, self._o_sample = 100, 100
        self._config = OvertourismEvaluationConfig(
            ensemble_size=12,  # 4 seasons x 3 weather classes (D5, D7)
            ensemble_seed=0,
            n_samples_per_combo=10,
            sample_seed=1,
            target_presence_samples=2000,
            confidence=0.8,
        )

    def parameter_schema(self) -> list[OvertourismParameterMeta]:
        """Return the ordered, self-describing parameter schema."""
        return list(self._schema.values())

    def schema(self) -> dict[str, Any]:
        """Return Fazzon indexes and placeholder frontend metadata."""
        return {
            "metadata": {
                "mapper": self.MAPPER,
                "x_axis_name": self.X_AXIS_NAME,
                "y_axis_name": self.Y_AXIS_NAME,
                "x_field": self.X_FIELD,
                "y_field": self.Y_FIELD,
            },
            "indexes": self.parameter_schema(),
        }

    @property
    def model(self) -> FazzonModel:
        """The live `FazzonModel` instance.

        Needed by frontends that resolve predefined what-if scenarios
        (`Index`-keyed ``overrides_fn`` callables) against a live model, e.g.
        `fazzon_scenarios.ALL_SCENARIOS`.
        """
        return self._model

    def evaluate(self, param_overrides: dict[str, Any]) -> SustainabilityFieldOutput:
        """Evaluate the model under the given string-keyed parameter overrides.

        The parking price is a slider in the schema (euro) but a category in the model
        (D15): it is converted to its price level here, outside ``build_scenario``.
        """
        params = dict(param_overrides)
        price = params.pop(PRICE_INDEX_NAME, None)
        scenario = build_scenario(
            self._model,
            params,
            self._index_map,
            self._schema,
            self._parameter_axes,
        )
        if price is not None:
            overrides = {**scenario.overrides, self._model.cv_parking_price: price_level(price)}
            scenario = Scenario(self._model, overrides=overrides, parameter_axes=self._parameter_axes)
        return self._evaluate_scenario(scenario)

    def _with_policy(self, scenario: Scenario) -> Scenario:
        """Fix the access-policy categories (D15, D16): the price level (default: the 2025
        price, 5 euro) and the usable shuttle capacity class, computed from the shuttle
        timetable parameters (overridden or default).  Left free, the ensemble would mix
        the levels."""
        model = self._model
        inp = model.inputs
        overrides = dict(scenario.overrides)
        overrides.setdefault(model.cv_parking_price, str(PRICE_LEVELS[0]))

        def value(idx):
            return float(overrides.get(idx, idx.concrete_default))

        capacity = shuttle_usable_capacity(
            buses=value(inp.i_shuttle_buses),
            seats_per_bus=value(inp.i_shuttle_seats_per_bus),
            interval_min=value(inp.i_shuttle_interval_min),
            start_hour=value(inp.i_shuttle_start_hour),
            end_hour=value(inp.i_shuttle_end_hour),
            peak_end_hour=value(inp.i_shuttle_peak_end_hour),
            peak_share=value(inp.i_shuttle_peak_share),
        )
        overrides[model.cv_shuttle] = shuttle_capacity_level(capacity)
        return Scenario(model, overrides=overrides, parameter_axes=self._parameter_axes)

    def arrange_data(self, output: SustainabilityFieldOutput) -> dict[str, Any]:
        """Arrange Fazzon output for the frontend presentation format."""
        return arrange_frontend_data(output)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @functools.cached_property
    def _schema(self) -> dict[str, OvertourismParameterMeta]:
        """Hand-authored parameter metadata; the single source of truth (§3.1)."""
        inp = self._model.inputs
        return {
            inp.cv_parking_price.name: OvertourismParameterMeta(
                name=inp.cv_parking_price.name,
                kind="scalar",  # a category in the model, converted by evaluate() (D15)
                label="Tariffa parcheggio (€ per veicolo)",
                description=(
                    "Prezzo medio pagato per veicolo, a passi di 5 €. 5 € = regole 2025; 15 € = "
                    "media pagata con le regole 2026 (20 € giornata, 10 € dopo le 12).  Una tariffa "
                    "più alta riduce gli arrivi in auto (risposta stimata sui dati 2025-2026) e "
                    "aumenta le persone per veicolo; chi rinuncia all'auto prende la navetta (se "
                    "attiva), sale a piedi o va altrove."
                ),
                unit="€",
                category="Mobilità",
                default=float(PRICE_LEVELS[0]),
                min_value=float(PRICE_LEVELS[0]),
                max_value=float(PRICE_LEVELS[-1]),
                step=float(PRICE_LEVELS[1] - PRICE_LEVELS[0]),
            ),
            inp.i_shuttle_buses.name: OvertourismParameterMeta(
                name=inp.i_shuttle_buses.name,
                kind="scalar",
                label='Navetta: numero di bus',
                description='Bus che partono insieme a ogni corsa. 0 = nessuna navetta. Lakebus 2026: 2.',
                unit='bus',
                category="Navetta",
                default=0.0,
                min_value=0.0,
                max_value=6.0,
                step=1.0,
            ),
            inp.i_shuttle_seats_per_bus.name: OvertourismParameterMeta(
                name=inp.i_shuttle_seats_per_bus.name,
                kind="scalar",
                label='Navetta: posti per bus',
                description='Posti a sedere di ogni bus. Lakebus 2026: 30.',
                unit='posti',
                category="Navetta",
                default=30.0,
                min_value=10.0,
                max_value=60.0,
                step=5.0,
            ),
            inp.i_shuttle_interval_min.name: OvertourismParameterMeta(
                name=inp.i_shuttle_interval_min.name,
                kind="scalar",
                label='Navetta: frequenza (minuti)',
                description='Minuti fra due corse. Lakebus 2026: 30 (salita nei primi 15, discesa nei secondi 15).',
                unit='min',
                category="Navetta",
                default=30.0,
                min_value=10.0,
                max_value=60.0,
                step=5.0,
            ),
            inp.i_shuttle_start_hour.name: OvertourismParameterMeta(
                name=inp.i_shuttle_start_hour.name,
                kind="scalar",
                label='Navetta: prima corsa (ora)',
                description='Ora della prima partenza da Pellizzano. Lakebus 2026: 9.',
                unit='h',
                category="Navetta",
                default=9.0,
                min_value=6.0,
                max_value=12.0,
                step=1.0,
            ),
            inp.i_shuttle_end_hour.name: OvertourismParameterMeta(
                name=inp.i_shuttle_end_hour.name,
                kind="scalar",
                label='Navetta: fine servizio (ora)',
                description="Ora di fine servizio, cioè dell'ultimo rientro a Pellizzano. Lakebus 2026: 19 (ultima partenza 18:30).",
                unit='h',
                category="Navetta",
                default=19.0,
                min_value=14.0,
                max_value=22.0,
                step=1.0,
            ),
            inp.i_shuttle_peak_end_hour.name: OvertourismParameterMeta(
                name=inp.i_shuttle_peak_end_hour.name,
                kind="scalar",
                label='Navetta: fine picco del mattino (ora)',
                description="I passeggeri si concentrano fra la prima corsa e quest'ora. ASSUNZIONE: 12.",
                unit='h',
                category="Navetta",
                default=12.0,
                min_value=10.0,
                max_value=15.0,
                step=1.0,
            ),
            inp.i_shuttle_peak_share.name: OvertourismParameterMeta(
                name=inp.i_shuttle_peak_share.name,
                kind="scalar",
                label='Navetta: quota passeggeri nel picco',
                description='Quota dei passeggeri che vuole salire nel picco del mattino: limita la capacità utilizzabile. ASSUNZIONE: 0.6 (da validare; dati 2026: al massimo 0.69).',
                unit='',
                category="Navetta",
                default=0.6,
                min_value=0.1,
                max_value=1.0,
                step=0.05,
            ),
            inp.cv_season.name: OvertourismParameterMeta(
                name=inp.cv_season.name,
                kind="categorical",
                label="Stagione",
                description=(
                    "Vincola la valutazione a una singola stagione. "
                    "Lasciare su '(tutte)' per la media ponderata stagionale."
                ),
                category="Contesto",
                support=list(inp.cv_season.support),
            ),
            inp.cv_weather.name: OvertourismParameterMeta(
                name=inp.cv_weather.name,
                kind="categorical",
                label="Meteo",
                description=(
                    "Vincola la valutazione a un tipo di giornata: good = bel tempo, "
                    "cool = fresca (tmax > 2 °C sotto la media del mese), rainy = pioggia >= 5 mm. "
                    "Lasciare su '(tutte)' per la media ponderata sulla climatologia."
                ),
                category="Contesto",
                support=list(inp.cv_weather.support),
            ),
            inp.i_c_parking.name: OvertourismParameterMeta(
                name=inp.i_c_parking.name,
                kind="distribution",
                label="Cap parcheggio (auto simultanee)",
                description=(
                    "Range [min, max] del numero massimo di auto in parcheggio simultaneo. "
                    "Riferimento 2025: triangolare moda 200, range [150, 250]."
                ),
                unit="auto",
                category="Capacità",
                default_range=(150.0, 250.0),
                min_value=50.0,
                max_value=500.0,
                step=10.0,
                distribution_family="triang",
                distribution_fixed_params={"c": 0.5},
            ),
        }

    def _evaluate_scenario(self, scenario: Scenario) -> SustainabilityFieldOutput:
        """Run the seeded field + presence-sample evaluation for `scenario`."""
        scenario = self._with_policy(scenario)
        model = self._model
        config = self._config
        pv_car, pv_other = model.inputs.pv_visitors_car, model.inputs.pv_visitors_other
        cc = np.linspace(0, self._c_max, self._c_sample + 1)
        oo = np.linspace(0, self._o_max, self._o_sample + 1)

        # Presence samples for the scatter overlay — seeded independently of the field.
        sampling_overrides = {
            idx: (
                [val]
                if isinstance(idx, CategoricalIndex) and isinstance(val, str)
                else val
            )
            for idx, val in scenario.overrides.items()
        }
        sampling_scenario = Scenario(
            model, overrides=sampling_overrides, parameter_axes=[pv_car, pv_other]
        )
        sampling_ensemble = CrossProductEnsemble(
            sampling_scenario,
            max_categorical_size=config.ensemble_size,
            rng=np.random.default_rng(config.sample_seed),
        )
        pv_samples = sample_across(
            sampling_ensemble,
            [pv_car, pv_other],
            total=config.target_presence_samples,
            rng=np.random.default_rng(config.sample_seed),
        )

        # Field ensemble — seeded, n_samples_per_combo capacity-distribution replicates
        # per categorical combo. `scenario` already carries parameter_axes (set by
        # build_scenario), so it's used as-is — no internal rebuild.
        field_ensemble = CrossProductEnsemble(
            scenario,
            max_categorical_size=config.ensemble_size,
            n_samples_per_combo=config.n_samples_per_combo,
            rng=np.random.default_rng(config.ensemble_seed),
        )
        result = Evaluation(scenario).evaluate(
            ensemble=field_ensemble,
            parameters={pv_car: cc, pv_other: oo},
        )
        field, field_elements, usage_fields, capacity_distributions = (
            compute_sustainability_field(
                model.constraints, result, pv_car, pv_other, scenario=scenario
            )
        )

        return SustainabilityFieldOutput(
            field=field,
            field_elements=field_elements,
            x_values=cc,
            y_values=oo,
            samples_x=list(pv_samples[pv_car]),
            samples_y=list(pv_samples[pv_other]),
            usage_fields=usage_fields,
            capacity_distributions=capacity_distributions,
            confidence=config.confidence,
        )
