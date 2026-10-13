# SPDX-License-Identifier: Apache-2.0
"""Placeholder adapter for the energy sources (electricity and gas withdrawals).

The base build reads the energy category files, but nothing is known yet about the format
of the data that will arrive with the yearly updates. Until it is, this placeholder keeps
`ee_prelievi` and `gas_prelievi` known to the update machinery (config, parts, standardization
and processing hooks) without guessing a layout.

When the format is known: write a real adapter with `@register_adapter(..., kind="energia")`,
fill `KIND_SCHEMAS["energia"]` in base.py and replace the placeholders in
standardize_update_raw_data.py (STANDARDIZERS) and process_update_std_data.py (PROCESSORS).
"""

import pandas as pd

from utils.adapters.base import RawData, register_adapter


@register_adapter("energia_placeholder", kind="energia", placeholder=True)
def energia_placeholder(raw: RawData, **params) -> pd.DataFrame:
    """PLACEHOLDER: the format of the yearly update of the energy data is not defined yet."""
    raise NotImplementedError(
        "The format of the yearly update of the energy data (ee_prelievi, gas_prelievi) "
        "is not defined yet: write an adapter for it."
    )
