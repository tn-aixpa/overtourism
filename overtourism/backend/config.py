# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

# Default paths
ROOT = Path(__file__).parent.parent
default_module_path = str(ROOT / "model" / "setup.py")
default_metadata_path = str(ROOT / "data" / "metadata")
default_index_data_path = str(ROOT / "data" / "index_data")

# Classes names
default_viewer_name = "viewer"
default_manager_name = "problem_manager"
default_model_module_name = "setup"

# Get environment variables
module_name = os.getenv("DT_MODEL_MODULE_NAME", default_model_module_name)
module_path = os.getenv("DT_MODEL_MODULE_PATH", default_module_path)
manager_name = os.getenv("DT_MODEL_PROBLEM_MANAGER_NAME", default_manager_name)
viewer_name = os.getenv("DT_STUDIO_VIEWER_NAME", default_viewer_name)
metadata_path = os.getenv("DT_MODEL_METADATA_PATH", default_metadata_path)
index_data_path = os.getenv("DT_OVERTURISM_INDEX_DATA_PATH", default_index_data_path)

# whether execute standalone, with data already prepared
standalone_mode = os.getenv("DT_OVERTURISM_STANDALONE_MODE", "false").lower() == "true"

# Platform specific settings
project_name = os.getenv("PROJECT_NAME", "overtourism")
dataitems = [
    "df_distribuzione_feriale",
    "df_distribuzione_festivo",
    "df_distribuzione_prefestivo",
    "df_flussi_estate",
    "df_incidenza_postiletto_non_conv",
    "df_incidenza_strutture_non_conv",
    "df_overturismo",
    "df_stagionalita_presenze",
    "df_tasso_ricettivita",
    "df_tasso_turisticita_estate",
    "df_tasso_turisticita",
    "df_tasso_variazione_pecentuale",
    "df_turismo_sommerso",
]
artifacts = [
    "map_apt.geojson",
    "map_comuni.geojson",
    "map_vodafone_2024.geojson",
    "map_vodafone.geojson",
]
