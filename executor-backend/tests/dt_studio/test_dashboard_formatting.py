# SPDX-License-Identifier: Apache-2.0
"""Formatting shared by the dashboards: model-specific KPIs and categorical labels."""

from __future__ import annotations

from src.dt_studio.dashboard.adapter import format_extra_kpis
from src.model.common.sustainability_field import get_index_diffs


def test_format_extra_kpis_skips_the_standard_kpis() -> None:
    kpis = {
        "sustainability_level": {"level": 90.0, "confidence": 1.0},
        "critical constraint": {"name": "parking", "level": 80.0, "confidence": 1.0},
        "constraint level parking": {"level": 80.0, "confidence": 1.0},
        "dropouts mean": {"value": 23.4, "share": 3.14},
        "dropouts peak": {"value": 44.0},
    }
    assert format_extra_kpis(kpis, {"dropouts mean": "Rinunce: media"}) == {
        "Rinunce: media": "23 (3.1%)",
        "dropouts peak": "44",
    }


def test_index_diffs_use_the_support_labels() -> None:
    schema = {
        "indexes": [
            {"name": "weekday", "kind": "categorical", "support_labels": {"saturday": "sabato"}},
            {"name": "season", "kind": "categorical"},
        ]
    }
    assert get_index_diffs(schema, {"weekday": "saturday", "season": "june"}) == {
        "weekday": "(tutte) -> sabato",
        "season": "(tutte) -> june",
    }
