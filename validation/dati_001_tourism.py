# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import digitalhub as dh

PROJECT = "overtourism"

list = dh.list_dataitems(PROJECT)

print(f"Datasets in project {PROJECT}:")
for di in list:
    name = di.metadata.name
    updated = datetime.strptime(di.metadata.updated, '%Y-%m-%dT%H:%M:%S.%fZ')
    print(f"  {name} (last updated: {updated:%d/%m/%Y %H:%M:%S})")
print(f"{len(list)} datasets.")