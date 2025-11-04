# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import slugify

from scipy import stats

import pandas as pd

from overtourism.model.setup import problem_manager

from pandas.core.interchange.dataframe_protocol import DataFrame
import digitalhub as dh

PROJECT = "overtourism"

def get_measured_presences():
    START_DATE = "2023-06-16"
    END_DATE = "2023-09-15"
    LOC_ID_COMUNE_MOLVENO = "27"
    LOC_TYPE_COMUNE_MOLVENO = "TN_MKT_AL_3"

    presences = dh.get_dataitem("vodafone_attendences", project=PROJECT).as_df()

    presences = presences[
        (presences["date"] >= START_DATE)
        & (presences["date"] <= END_DATE)
        & (presences["value"] != 0)
        & (presences["userProfile"] != "INHABITANT")
        & (presences["userProfile"] != "COMMUTER")
        & (presences["locId"] == LOC_ID_COMUNE_MOLVENO)
        & (presences["locType"] == LOC_TYPE_COMUNE_MOLVENO)
        ]
    presences = presences[["date", "userProfile", "value", "locId"]]

    # tourists
    tourist_presences = (
        presences[presences["userProfile"] == "TOURIST"][["date", "value", "locId"]]
        .groupby(["locId", "date"])
        .sum()
        .reset_index()
    )
    tourist_presences.rename(columns={"value": "presences_tourists"}, inplace=True)

    # excursionists
    excursionist_presences = (
        presences[presences["userProfile"] == "VISITOR"][["date", "value", "locId"]]
        .groupby(["locId", "date"])
        .sum()
        .reset_index()
    )
    excursionist_presences.rename(
        columns={"value": "presences_excursionists"}, inplace=True
    )

    # all
    presences = pd.merge(
        tourist_presences, excursionist_presences, how="inner", on=["date", "locId"]
    )[["presences_tourists", "presences_excursionists"]]

    return presences


PROBLEM_NAME = "validation-monitoraggio-001-problem"
PROBLEM_ID = slugify.slugify(PROBLEM_NAME)
SCENARIO_NAME = "validation-monitoraggio-001-scenario"
SCENARIO_ID = slugify.slugify(SCENARIO_NAME)


def main():
    problem_manager.import_problem(PROBLEM_ID + ".yaml")  # TODO: remove suffix
    scenario_manager = problem_manager.get_problem(PROBLEM_ID)

    scenario_manager.evaluate_scenario(SCENARIO_ID)
    data = scenario_manager.get_scenario_data(SCENARIO_ID)
    predicted_tourists = data.sample_x
    predicted_excursionists = data.sample_y

    measured_presences = get_measured_presences()
    measured_tourists = measured_presences["presences_tourists"]
    measured_excursionists = measured_presences["presences_excursionists"]

    ks_test_tourists = stats.ks_2samp(measured_tourists, predicted_tourists)
    ks_test_excursionists = stats.ks_2samp(measured_excursionists, predicted_excursionists)

    pvalue_tourists = ks_test_tourists.pvalue
    pvalue_excursionists = ks_test_excursionists.pvalue

    print(f"Info: KS-test on tourists (p-value): {pvalue_tourists:.5f}")
    print(f"Info: KS-test on excursionists (p-value): {pvalue_excursionists:.5f}")

    if pvalue_tourists < 0.05:
        print(f"** WARNING: the measured presences of tourists DO NOT match " 
              "the forecasts of scenario {SCENARIO_NAME} (p-value {pvalue_tourists:.5f})")
    elif pvalue_tourists < 0.75:
        print(f"** WARNING: the measured presences of tourists MAY NOT match " 
              "the forecasts of scenario {SCENARIO_NAME} (p-value {pvalue_tourists:.5f})")

    if pvalue_excursionists < 0.05:
        print(f"** WARNING: the measured presences of excursionists DO NOT match " 
              f"the forecasts of scenario {SCENARIO_NAME} (p-value {pvalue_excursionists:.5f})")
    elif pvalue_excursionists < 0.75:
        print(f"** WARNING: the measured presences of excursionists MAY NOT match " 
              f"the forecasts of scenario {SCENARIO_NAME} (p-value {pvalue_excursionists:.5f})")


if __name__ == "__main__":
    main()