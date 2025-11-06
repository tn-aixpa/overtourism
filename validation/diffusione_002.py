# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sklearn.metrics

import digitalhub as dh

PROJECT = "overtourism"

def main():
    correlazione = dh.get_dataitem("df_overturismo", project=PROJECT).as_df()

    correlazione['flusso'] = correlazione['flusso'] == '**'
    correlazione.loc[correlazione['flusso'], 'level'] -= 2
    correlazione['level'] = correlazione['level'] >= 8
    correlazione.rename(columns={'comune':'count', 'level':'overtourism', 'flusso':'high flow'}, inplace=True)

    ((vn, fp),(fn, vp)) = sklearn.metrics.confusion_matrix(correlazione['overtourism'], correlazione['high flow'])
    tot = vn+fp+fn+vp
    print(f"Recupero      : {vp/(vp+fn):.2%}")
    print(f"Precisione    : {vp/(vp+fp):.2%}")
    print(f"Real. positivi: {(vp+fn)/tot:.2%}")
    print(f"Real. negativi: {(vn+fp)/tot:.2%}")
    print(f"Veri negativi : {vn/tot:.2%}")
    print(f"Veri positivi : {vp/tot:.2%}")
    print(f"Falsi negativi: {fn/tot:.2%}")
    print(f"Falsi positivi: {fp/tot:.2%}")


if __name__ == "__main__":
    main()
