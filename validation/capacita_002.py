# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sklearn import metrics

from data_preparation.compute_system_capacity_indexes import (
    get_spiaggia, get_parcheggi, get_presenze_vodafone, model_parcheggi, model_spiaggia
)

def main():
    parcheggi = get_parcheggi()
    presenze = get_presenze_vodafone()
    reg_parcheggi = model_parcheggi(parcheggi, presenze)
    coef_t, coef_e =  reg_parcheggi.coef_[0,0], reg_parcheggi.coef_[0,1]
    r2_score = metrics.r2_score(parcheggi, reg_parcheggi.predict(presenze))
    nmae = metrics.mean_absolute_error(parcheggi, reg_parcheggi.predict(presenze)) / parcheggi["PARCHEGGI"].mean()
    print("MOELLO PARCHEGGI")
    print(f"- Coefficiente turisti      : {coef_t:.4f}")
    print(f"- Coefficiente escursionisti: {coef_e:.4f}")
    print(f"- Coefficiente R^2          : {r2_score:.4f}")
    print(f"- MAE normalizzata          : {nmae:.4f}")

    spiaggia = get_spiaggia()
    presenze_spiaggia = presenze[presenze.index.isin(spiaggia.index)]
    reg_spiaggia = model_spiaggia(spiaggia, presenze_spiaggia)
    coef_t, coef_e =  reg_spiaggia.coef_[0,0], reg_spiaggia.coef_[0,1]
    r2_score = metrics.r2_score(spiaggia, reg_spiaggia.predict(presenze_spiaggia))
    nmae = (metrics.mean_absolute_error(spiaggia, reg_parcheggi.predict(presenze_spiaggia)) /
            spiaggia["SPIAGGIA"].mean())
    print("MOELLO PARCHEGGI")
    print(f"- Coefficiente turisti      : {coef_t:.4f}")
    print(f"- Coefficiente escursionisti: {coef_e:.4f}")
    print(f"- Coefficiente R^2          : {r2_score:.4f}")
    print(f"- MAE normalizzata          : {nmae:.4f}")


if __name__ == "__main__":
    main()