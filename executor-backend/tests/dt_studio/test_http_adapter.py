# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from src.dt_studio.dashboard.http_adapter import (
    HttpOvertourismAdapter,
)


def test_parameter_specs_reads_indexes_from_schema_response() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "metadata": {
                    "mapper": {},
                    "color_map": [],
                    "kpi_mapper": {},
                    "plot_mapper": {},
                },
                "indexes": [
                    {
                        "name": "parking capacity",
                        "kind": "distribution",
                        "default_range": [350.0, 450.0],
                    }
                ],
            }

    class FakeClient:
        def get(self, path: str) -> FakeResponse:
            assert path == "/models/molveno/schema"
            return FakeResponse()

    adapter = HttpOvertourismAdapter(
        model_key="molveno",
        title="Molveno",
        base_url="http://example.test",
    )
    adapter._client = FakeClient()

    specs = adapter.parameter_specs()

    assert len(specs) == 1
    assert specs[0].name == "parking capacity"
    assert specs[0].default_range == (350.0, 450.0)


def _fake_api(evaluate_body: dict | None = None):
    """A fake API client: the schema has a labelled categorical and a model-specific KPI label."""

    class FakeResponse:
        def __init__(self, body: dict) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._body

    schema = {
        "metadata": {
            "mapper": {"parking": "Parcheggi"},
            "color_map": [],
            "kpi_mapper": {"dropouts mean": "Rinunce: media"},
            "plot_mapper": {"bidimensional": {"x": {"label": "X"}, "y": {"label": "Y"}}},
        },
        "indexes": [
            {
                "name": "weather",
                "kind": "categorical",
                "support": ["good", "rainy"],
                "support_labels": {"good": "bel tempo", "rainy": "pioggia"},
            }
        ],
    }

    class FakeClient:
        def get(self, path: str) -> FakeResponse:
            return FakeResponse(schema)

        def post(self, path: str, json: dict) -> FakeResponse:
            return FakeResponse(evaluate_body or {})

    adapter = HttpOvertourismAdapter(model_key="fazzon", title="Fazzon", base_url="http://example.test")
    adapter._client = FakeClient()
    return adapter


def test_parameter_specs_read_support_labels() -> None:
    (spec,) = _fake_api().parameter_specs()
    assert spec.support_labels == {"good": "bel tempo", "rainy": "pioggia"}


def test_run_shows_model_specific_kpis() -> None:
    body = {
        "field": [[1.0]],
        "field_elements": {"parking": [[1.0]]},
        "x_values": [0.0],
        "y_values": [0.0],
        "samples_x": [],
        "samples_y": [],
        "sustainability_index": {"value": 1.0, "ci": 0.0},
        "sustainability_by_constraint": {"parking": {"value": 1.0, "ci": 0.0}},
        "modal_lines": {},
        "kpis": {
            "sustainability_level": {"level": 100.0, "confidence": 0.0},
            "critical constraint": {"name": "parking", "level": 100.0, "confidence": 0.0},
            "constraint level parking": {"level": 100.0, "confidence": 0.0},
            "dropouts mean": {"value": 632.6, "share": 12.47},
        },
    }
    plot = _fake_api(body).run({})
    assert plot.extra_kpis == {"Rinunce: media": "633 (12.5%)"}
