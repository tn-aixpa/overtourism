# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing

import numpy as np
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.symbols.index import Distribution
from scipy import stats

if typing.TYPE_CHECKING:
    from civic_digital_twins.dt_model import Constraint, Ensemble, PresenceVariable
    from civic_digital_twins.dt_model.model.instantiated_model import InstantiatedModel


def sample_with_transformation(
    p_var: PresenceVariable,
    model: InstantiatedModel,
    evaluation: Evaluation,
    ensemble: Ensemble,
    target_presence_samples: int,
    reduction_index_name: str,
    saturation_index_name: str,
) -> list:
    """
    Sample the presence variable and apply the transformation
    to each sample.

    Parameters
    ----------
    p_var : PresenceVariable
    model : InstantiatedModel
    evaluation : Evaluation
    ensemble : Ensemble
    target_presence_samples : int
    reduction_index_name : str
    saturation_index_name : str

    Returns
    -------
    list
        List of transformed presence samples
    """
    sample_p_var = []
    for condition in ensemble:
        # Extract the weights and the context variables
        weight, context = condition
        # Sample the presence variable
        samples = p_var.sample(
            cvs=context, nr=max(1, round(weight * target_presence_samples))
        )
        # Transform each sample
        transformed_samples = [
            presence_transformation(
                presence,
                get_index_mean_value(model, evaluation, reduction_index_name),
                get_index_mean_value(model, evaluation, saturation_index_name),
            )
            for presence in samples
        ]
        sample_p_var.extend(transformed_samples)
    return sample_p_var


def presence_transformation(
    presence,
    reduction_factor,
    saturation_level,
    sharpness=3,
):
    tmp = presence * reduction_factor / 100.0
    return (
        tmp
        * saturation_level
        / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))
    )


def get_index_mean_value(
    model: InstantiatedModel, evaluation: Evaluation, index_name: str
):
    for i in model.abs.indexes:
        if i.name == index_name:
            return evaluation.get_index_mean_value(i)
    raise ValueError


def compute_kpis(
    evaluation: Evaluation,
    constraints: list[Constraint],
    zip_sample: list[tuple[float, float]],
) -> dict:
    kpis = {}

    # Overtourism level
    index, confidence = evaluation.compute_sustainability_index_with_ci(
        zip_sample, confidence=0.9
    )
    kpis["overtourism_level"] = {
        "level": round(((1 - index) * 100), 4),
        "confidence": round(((confidence) * 100), 4),
    }

    # Critical constraint
    indexes = evaluation.compute_sustainability_index_with_ci_per_constraint(
        zip_sample, confidence=0.9
    )
    critical = min(indexes, key=lambda i: indexes.get(i)[0])
    critical_param_level = round(((1 - indexes[critical][0]) * 100), 4)
    critical_param_confidence = round(((indexes[critical][1]) * 100), 4)
    kpis["critical constraint"] = {
        "name": critical.name,
        "level": critical_param_level,
        "confidence": critical_param_confidence,
    }

    # Constraint level
    for constraint in constraints:
        constraint_level = round(((1 - indexes[constraint][0]) * 100), 4)
        constraint_confidence = round(((indexes[constraint][1]) * 100), 4)
        kpis["constraint level " + constraint.name] = {
            "level": constraint_level,
            "confidence": constraint_confidence,
        }
    return kpis


def compute_constraint_curves(
    evaluation: Evaluation,
) -> dict:
    constraint_curves = evaluation.compute_modal_line_per_constraint()
    return {k.name: v for k, v in constraint_curves.items()}


def compute_uncertainty(
    evaluation: Evaluation,
    constraints: list[Constraint],
    zip_sample: list[tuple[float, float]],
) -> tuple[list[float], dict[str, list[float]]]:
    uncertainty = []
    for i in zip_sample:
        sust = evaluation.compute_sustainability_index([i])
        uncertainty.append(float("{:.4f}".format(sust)))

    uncertainty_by_constraint = {}
    for c in constraints:
        uncertainty_by_constraint[c.name] = []
    for i in zip_sample:
        sust = evaluation.compute_sustainability_index_per_constraint([i])
        for k, v in sust.items():
            uncertainty_by_constraint[k.name].append(float("{:.4f}".format(v)))

    return uncertainty, uncertainty_by_constraint


def compute_usage_capacity(
    evaluation: Evaluation,
    samples: list[tuple[float, float]],
    len_sample: int,
    constraints: list[Constraint],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    float,
    dict[str, float],
]:
    usage_by_constraint = evaluation.evaluate_usage(samples)
    usage_by_constraint = {
        k.name: [int(u) for u in v.tolist()] for k, v in usage_by_constraint.items()
    }

    capacity_by_constraint = {}
    usage = np.ones(len_sample)
    variance = 0.0
    for constraint in constraints:
        capacity_by_constraint[constraint.name] = constraint.capacity.value
        if isinstance(constraint.capacity.value, Distribution):
            cap = constraint.capacity.value.mean()
            var = constraint.capacity.value.std() ** 2
        else:
            cap = constraint.capacity.value
            var = 0.0
        usage += usage_by_constraint[constraint.name] / cap
        variance += var / (cap**2)
    usage *= 100.0 / len(constraints)
    std = (variance**0.5) * 100.0 / len(constraints)
    capacity = stats.norm(loc=100.0, scale=std)

    capacity_mean = capacity.mean()
    capacity_mean_by_constraint = {}
    for constraint in constraints:
        c = constraint.name
        capacity_mean_by_constraint[c] = capacity_by_constraint[c].mean()

    return (
        [int(u) for u in usage.tolist()],
        usage_by_constraint,
        capacity,
        capacity_by_constraint,
        capacity_mean,
        capacity_mean_by_constraint,
    )


def compute_usage_uncertainty(capacity: np.ndarray, usage: list[float]) -> list[float]:
    capacity_mean = float(capacity.mean())
    y_max = int(max(max(usage), capacity_mean) * 1.2)
    rangey = range(int(max(max(usage), capacity_mean) * 1.2))
    capacity = [[float((capacity.cdf(y)))][0] for y in rangey]
    heatmap_y = np.linspace(0, y_max, len(capacity))

    usage_uncertainty = []
    for u in usage:
        idx = np.abs(heatmap_y - u).argmin()
        usage_uncertainty.append(float("{:.4f}".format(capacity[idx])))

    return usage_uncertainty


def build_output(
    model: InstantiatedModel,
    evaluation: Evaluation,
    ensemble: Ensemble,
    sampling_dicts: list[dict],
) -> dict:
    # Compute get sample
    sample_dict = {
        "model": model,
        "evaluation": evaluation,
        "ensemble": ensemble,
    }
    sample_x = sample_with_transformation(
        **sampling_dicts[1],
        **sample_dict,
    )
    sample_y = sample_with_transformation(
        **sampling_dicts[0],
        **sample_dict,
    )

    constraints = model.abs.constraints
    zip_sample = list(zip(sample_y, sample_x))

    # Compute constraint curves
    constraint_curves = compute_constraint_curves(evaluation)

    # Compute kpis
    kpis = compute_kpis(
        evaluation,
        constraints,
        zip_sample,
    )

    # Compute uncertainty
    uncertainty, uncertainty_by_constraint = compute_uncertainty(
        evaluation,
        constraints,
        zip_sample,
    )

    # Compute usage and capacity
    (
        usage,
        usage_by_constraint,
        capacity,
        capacity_by_constraint,
        capacity_mean,
        capacity_mean_by_constraint,
    ) = compute_usage_capacity(
        evaluation,
        [sample_x, sample_y],
        len(sample_x),
        constraints,
    )

    # Compute usage uncertainty
    usage_uncertainty = compute_usage_uncertainty(capacity, usage)
    usage_uncertainty_by_constraint = {}
    for c in constraints:
        usage_uncertainty_by_constraint[c.name] = compute_usage_uncertainty(
            capacity_by_constraint[c.name],
            usage_by_constraint[c.name],
        )

    return {
        "sample_x": [int(x) for x in sample_x],
        "sample_y": [int(y) for y in sample_y],
        "kpis": kpis,
        "uncertainty": uncertainty,
        "uncertainty_by_constraint": uncertainty_by_constraint,
        "constraint_curves": constraint_curves,
        "usage": usage,
        "usage_by_constraint": usage_by_constraint,
        "usage_uncertainty": usage_uncertainty,
        "usage_uncertainty_by_constraint": usage_uncertainty_by_constraint,
        "capacity_mean": capacity_mean,
        "capacity_mean_by_constraint": capacity_mean_by_constraint,
    }
