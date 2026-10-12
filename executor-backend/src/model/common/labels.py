# SPDX-License-Identifier: Apache-2.0
"""Italian display labels of the categorical values shared by the models.

Used as ``OvertourismParameterMeta.support_labels``: the model and the API keep the values
(``"monday"``, ``"rainy"``, ...), the dashboards show the labels.
"""

WEEKDAY_LABELS: dict[str, str] = {
    "monday": "lunedì",
    "tuesday": "martedì",
    "wednesday": "mercoledì",
    "thursday": "giovedì",
    "friday": "venerdì",
    "saturday": "sabato",
    "sunday": "domenica",
}

WEATHER_LABELS: dict[str, str] = {
    "good": "bel tempo",
    "cool": "fresco",
    "rainy": "pioggia",
}

MONTH_LABELS: dict[str, str] = {
    "june": "giugno",
    "july": "luglio",
    "august": "agosto",
    "september": "settembre",
}
