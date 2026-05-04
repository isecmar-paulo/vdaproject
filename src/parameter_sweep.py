import pandas as pd

from src.privacy_methods import apply_privacy_technique

from src.statistical_analysis import (
    compare_datasets,
    compute_utility_score,
    compute_privacy_score,
    compute_tradeoff_score,
)


def evaluate_configuration(
    df_prepared,
    technique,
    bins,
    numeric_columns,
    sigma=1.0,
    epsilon=1.0,
    sampling_rate=0.8,
    bin_size=None,
    data_range=None,
):
    """
    Apply one configuration and compute utility, privacy and trade-off scores.
    """

    if technique in ["Gaussian Noise", "Laplace Noise"]:
        columns = numeric_columns
    else:
        columns = []

    df_transformed = apply_privacy_technique(
        df=df_prepared,
        technique=technique,
        columns=columns,
        sigma=sigma,
        epsilon=epsilon,
        sampling_rate=sampling_rate,
        bin_size=bin_size,
    )

    comparison = compare_datasets(
        df_original=df_prepared,
        df_transformed=df_transformed,
        bins=bins,
    )

    utility_score = compute_utility_score(
        comparison["distribution_similarity"]
    )

    privacy_score = compute_privacy_score(
        technique=technique,
        sigma=sigma,
        epsilon=epsilon,
        sampling_rate=sampling_rate,
    )

    tradeoff_score = compute_tradeoff_score(
        utility_score,
        privacy_score,
    )

    return {
        "Technique": technique,
        "Utility Score": utility_score,
        "Privacy Score": privacy_score,
        "Trade-off Score": tradeoff_score,
    }


def run_parameter_sweep(
    df_prepared,
    numeric_columns,
    bins,
    selected_techniques,
):
    """
    Run an automatic parameter sweep for multiple privacy-preserving techniques.
    """

    results = []

    # Gaussian Noise
    if "Gaussian Noise" in selected_techniques:
        for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
            result = evaluate_configuration(
                df_prepared=df_prepared,
                technique="Gaussian Noise",
                bins=bins,
                numeric_columns=numeric_columns,
                sigma=sigma,
            )

            result["Parameter"] = "sigma"
            result["Parameter Value"] = sigma

            results.append(result)

    # Laplace Noise
    if "Laplace Noise" in selected_techniques:
    
        for epsilon in [0.2, 0.5, 1.0, 2.0, 5.0]:
            result = evaluate_configuration(
                df_prepared=df_prepared,
                technique="Laplace Noise",
                bins=bins,
                numeric_columns=numeric_columns,
                epsilon=epsilon,
            )

            result["Parameter"] = "epsilon"
            result["Parameter Value"] = epsilon

            results.append(result)

    # Sampling
    if "Sampling" in selected_techniques:

        for sampling_rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            result = evaluate_configuration(
                df_prepared=df_prepared,
                technique="Sampling",
                bins=bins,
                numeric_columns=numeric_columns,
                sampling_rate=sampling_rate,
            )

            result["Parameter"] = "sampling_rate"
            result["Parameter Value"] = sampling_rate

            results.append(result)

    # Generalization - Age
    if "Generalization - Age" in selected_techniques:
        if "age" in df_prepared.columns:
            age_range = df_prepared["age"].max() - df_prepared["age"].min()

            for bin_size in [5, 10, 15, 20]:
                if bin_size <= age_range:
                    result = evaluate_configuration(
                        df_prepared=df_prepared,
                        technique="Generalization - Age",
                        bins=bins,
                        numeric_columns=numeric_columns,
                        bin_size=bin_size,
                        data_range=age_range,
                    )

                    result["Parameter"] = "bin_size"
                    result["Parameter Value"] = bin_size

                    results.append(result)

    # Generalization - BMI
    if "Generalization - BMI" in selected_techniques:
        if "bmi" in df_prepared.columns:
            bmi_range = df_prepared["bmi"].max() - df_prepared["bmi"].min()

            for bin_size in [1.0, 2.0, 5.0, 10.0]:
                if bin_size <= bmi_range:
                    result = evaluate_configuration(
                        df_prepared=df_prepared,
                        technique="Generalization - BMI",
                        bins=bins,
                        numeric_columns=numeric_columns,
                        bin_size=bin_size,
                        data_range=bmi_range,
                    )

                    result["Parameter"] = "bin_size"
                    result["Parameter Value"] = bin_size

                    results.append(result)

    df_results = pd.DataFrame(results)

    return df_results.sort_values(
        by="Trade-off Score",
        ascending=False,
    ).reset_index(drop=True)