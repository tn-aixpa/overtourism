# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import digitalhub as dh

PROJECT = "overtourism"

list = dh.list_dataitems(PROJECT)

print(list)