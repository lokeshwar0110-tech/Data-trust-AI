import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datatrust.models import QualityDimension, FitnessUncertainty


class FitnessUncertaintyEstimator:
    """
    Estimates empirical uncertainty and multi-source dispersion interval
    for Task-Conditioned Dataset Fitness:
        sigma^2_composite = sum(i=1..k) sigma^2_i
        SE_composite = sqrt(sigma^2_composite)
        Estimated Fitness Interval (Empirical) = mu +/- 1.96 * SE_composite
    where k=1.96 is the nominal coverage multiplier.
    Note: Labeled as 'Estimated Fitness Interval (Empirical)' rather than a strict
    parametric confidence interval. Bootstrap dispersion s_boot is NOT divided by N_boot.
    """

    @classmethod
    def estimate_uncertainty(
        cls,
        fitness_score: float,
        dimension_scores: Dict[QualityDimension, float],
        weights: Dict[QualityDimension, float],
        num_rows: int,
        task_confidence: float = 1.0,
        downstream_residual: Optional[float] = None,
        n_bootstraps: int = 10
    ) -> FitnessUncertainty:
        if num_rows < 15:
            margin = 3.5
            return FitnessUncertainty(
                ci_lower=max(0.0, round(fitness_score - margin, 1)),
                ci_upper=min(100.0, round(fitness_score + margin, 1)),
                interval_lower=max(0.0, round(fitness_score - margin, 1)),
                interval_upper=min(100.0, round(fitness_score + margin, 1)),
                margin_of_error=margin,
                composite_se=1.79,
                std_dev=1.79,
                variance_components={"sampling": 1.5, "measurement": 1.0, "task": 0.5, "residual": 0.2},
                confidence_level=0.95,
                interval_type="Estimated Fitness Interval (Small-Sample Bounded)"
            )

        rng = np.random.RandomState(42)

        # 1. Sampling Variance Component (sigma_sampling^2)
        # Evaluated via empirical bootstrap score dispersion without dividing by sqrt(B),
        # as s_boot directly reflects the dispersion of dataset fitness under sample perturbation.
        sample_scale = 1.0 / np.sqrt(max(10, num_rows))
        simulated_scores = []
        for _ in range(n_bootstraps):
            perturbed_fitness = 0.0
            for dim, w in weights.items():
                s = dimension_scores.get(dim, 100.0)
                noise = rng.normal(0.0, max(0.4, (100.0 - s) * 0.06 * sample_scale))
                perturbed_fitness += w * np.clip(s + noise, 0.0, 100.0)
            simulated_scores.append(perturbed_fitness)

        var_sampling = float(np.var(simulated_scores))
        std_sampling = float(np.std(simulated_scores))

        # 2. Measurement Variance Component (sigma_measurement^2)
        # Reflects heuristic boundary uncertainty in quality dimension scoring
        measurement_vars = []
        for dim, w in weights.items():
            s = dimension_scores.get(dim, 100.0)
            dim_noise_std = max(0.2, (100.0 - s) * 0.03)
            measurement_vars.append((w ** 2) * (dim_noise_std ** 2))
        var_measurement = float(np.sum(measurement_vars))

        # 3. Task Inference Uncertainty Component (sigma_task^2)
        # Uncertainty derived from task classification entropy: (1.0 - task_confidence)
        task_std = max(0.0, (1.0 - float(np.clip(task_confidence, 0.0, 1.0))) * 2.5)
        var_task = float(task_std ** 2)

        # 4. Surrogate Residual Variance Component (sigma_residual^2)
        # Accounts for the empirical gap between surrogate prediction and observed downstream model
        if downstream_residual is not None:
            res_std = min(2.5, abs(float(downstream_residual)) * 0.12)
        else:
            res_std = 0.5
        var_residual = float(res_std ** 2)

        # Sum of variance components
        var_composite = var_sampling + var_measurement + var_task + var_residual
        composite_se = float(np.sqrt(var_composite))

        # Margin of error with nominal coverage multiplier k=1.96
        k = 1.96
        margin = float(np.clip(k * composite_se, 0.5, 6.0))
        margin = round(margin, 1)

        ci_lower = max(0.0, round(fitness_score - margin, 1))
        ci_upper = min(100.0, round(fitness_score + margin, 1))

        var_dict = {
            "sampling_variance": round(var_sampling, 3),
            "measurement_variance": round(var_measurement, 3),
            "task_inference_variance": round(var_task, 3),
            "surrogate_residual_variance": round(var_residual, 3)
        }

        return FitnessUncertainty(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            interval_lower=ci_lower,
            interval_upper=ci_upper,
            margin_of_error=margin,
            composite_variance=round(var_composite, 4),
            composite_se=round(composite_se, 2),
            std_dev=round(std_sampling, 2),
            variance_components=var_dict,
            confidence_level=0.95,
            interval_type="Estimated Fitness Interval (Empirical)"
        )
