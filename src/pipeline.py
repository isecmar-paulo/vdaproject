import streamlit as st
import pandas as pd

from src.data_preparation import preprocess_dataset
from src.privacy_methods import (
    apply_privacy_technique,
    generalize_numeric_column,
    generalize_region,
)


from src.statistical_analysis import (
    compare_datasets,
    compute_utility_score,
    compute_privacy_score,
    compute_tradeoff_score,
)

from src.data_preparation import (
    load_dataset,
    exploratory_inspection,
    preprocess_dataset,
    validate_prepared_dataset,
)



def load_and_prepare_data(data_path: str):
    """
    Load dataset, perform exploratory inspection, preprocess data,
    and validate the prepared dataset.

    Returns a dictionary with all relevant objects.
    """

    # ------------------------------------------------------------
    # Load original dataset
    # ------------------------------------------------------------
    df_original = load_dataset(data_path)

    # ------------------------------------------------------------
    # Exploratory inspection
    # ------------------------------------------------------------
    inspection = exploratory_inspection(df_original)

    # ------------------------------------------------------------
    # Preprocess dataset
    # ------------------------------------------------------------
    df_prepared = preprocess_dataset(df_original)

    # ------------------------------------------------------------
    # Validate prepared dataset
    # ------------------------------------------------------------
    validation = validate_prepared_dataset(df_prepared)

    # ------------------------------------------------------------
    # Extract numeric columns
    # ------------------------------------------------------------
    numeric_columns = df_prepared.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if len(numeric_columns) == 0:
        st.error("No numeric columns available after preprocessing.")
        st.stop()

    # ------------------------------------------------------------
    # Return structured data
    # ------------------------------------------------------------
    return {
        "df_original": df_original,
        "df_prepared": df_prepared,
        "inspection": inspection,
        "validation": validation,
        "numeric_columns": numeric_columns,
    }



def apply_selected_transformation(df_original, df_prepared, sidebar):
    technique = sidebar["technique"]

    if technique == "None":
        return df_prepared.copy()

    if technique == "Generalization":
        return apply_generalization(
            df_original=df_original,
            sidebar=sidebar,
        )

    return apply_privacy_technique(
        df=df_prepared,
        technique=technique,
        columns=sidebar["selected_columns"],
        sigma=sidebar["sigma"],
        sampling_rate=sidebar["sampling_rate"],
        epsilon=sidebar["epsilon"],
    )


def compute_evaluation_results(df_prepared, df_transformed, sidebar):
    """
    Compute statistical comparison, utility, privacy, and trade-off scores.
    """

    # ------------------------------------------------------------
    # Dataset comparison
    # ------------------------------------------------------------
    comparison = compare_datasets(
        df_original=df_prepared,
        df_transformed=df_transformed,
        bins=sidebar["bins"],
    )

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    utility_score = compute_utility_score(
        comparison["distribution_similarity"]
    )

    # ------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------
    privacy_score = compute_privacy_score(
        technique=sidebar["technique"],
        sigma=sidebar["sigma"],
        epsilon=sidebar["epsilon"],
        sampling_rate=sidebar["sampling_rate"],
        generalized_attributes=sidebar["generalized_attributes"],
        age_bin_size=sidebar["age_bin_size"],
        bmi_bin_size=sidebar["bmi_bin_size"],
        age_range=sidebar["age_range"],
        bmi_range=sidebar["bmi_range"],
    )

    # ------------------------------------------------------------
    # Trade-off
    # ------------------------------------------------------------
    tradeoff_score = compute_tradeoff_score(
        utility_score,
        privacy_score,
    )

    # ------------------------------------------------------------
    # Return structured results
    # ------------------------------------------------------------
    return {
        "comparison": comparison,
        "utility_score": utility_score,
        "privacy_score": privacy_score,
        "tradeoff_score": tradeoff_score,
    }


def build_tradeoff_comparison(
    df_prepared,
    numeric_columns,
    sidebar,
    evaluation,
):
    """
    Build a trade-off comparison table.

    The current user-selected configuration is included first.
    Other techniques are evaluated using default configurations.
    """

    tradeoff_results = []

    # ------------------------------------------------------------
    # 1. Current user-selected configuration
    # ------------------------------------------------------------
    tradeoff_results.append(
        {
            "Technique": f'{sidebar["technique"]} (Current)',
            "Utility Score": evaluation["utility_score"],
            "Privacy Score": evaluation["privacy_score"],
            "Trade-off Score": evaluation["tradeoff_score"],
        }
    )

    # ------------------------------------------------------------
    # 2. Default configurations for comparison
    # ------------------------------------------------------------
    techniques_to_compare = [
        {
            "name": "Gaussian Noise",
            "sigma": 1.0,
            "epsilon": 1.0,
            "sampling_rate": 0.8,
            "columns": numeric_columns,
            "bin_size": None,
            "data_range": None,
        },
        {
            "name": "Laplace Noise",
            "sigma": 1.0,
            "epsilon": 1.0,
            "sampling_rate": 0.8,
            "columns": numeric_columns,
            "bin_size": None,
            "data_range": None,
        },
        {
            "name": "Sampling",
            "sigma": 1.0,
            "epsilon": 1.0,
            "sampling_rate": 0.8,
            "columns": numeric_columns,
            "bin_size": None,
            "data_range": None,
        },
    ]

    # ------------------------------------------------------------
    # 3. Evaluate default configurations, excluding current technique
    # ------------------------------------------------------------
    for item in techniques_to_compare:
        if item["name"] == sidebar["technique"]:
            continue

        df_temp = apply_privacy_technique(
            df=df_prepared,
            technique=item["name"],
            columns=item["columns"],
            sigma=item["sigma"],
            epsilon=item["epsilon"],
            sampling_rate=item["sampling_rate"],
            bin_size=item["bin_size"],
        )

        temp_comparison = compare_datasets(
            df_original=df_prepared,
            df_transformed=df_temp,
            bins=sidebar["bins"],
        )

        temp_utility = compute_utility_score(
            temp_comparison["distribution_similarity"]
        )

        temp_privacy = compute_privacy_score(
            technique=item["name"],
            sigma=item["sigma"],
            epsilon=item["epsilon"],
            sampling_rate=item["sampling_rate"],
            
        )

        temp_tradeoff = compute_tradeoff_score(
            temp_utility,
            temp_privacy,
        )

        tradeoff_results.append(
            {
                "Technique": f'{item["name"]} (Default)',
                "Utility Score": temp_utility,
                "Privacy Score": temp_privacy,
                "Trade-off Score": temp_tradeoff,
            }
        )

    df_tradeoff_comparison = pd.DataFrame(tradeoff_results)

    return df_tradeoff_comparison.sort_values(
        by="Trade-off Score",
        ascending=False,
    ).reset_index(drop=True)


def apply_generalization(df_original, sidebar):
    df_generalized = df_original.copy()

    if "age" in sidebar["generalized_attributes"]:
        df_generalized = generalize_numeric_column(
            df=df_generalized,
            column="age",
            bin_size=sidebar["age_bin_size"],
        )

    if "bmi" in sidebar["generalized_attributes"]:
        df_generalized = generalize_numeric_column(
            df=df_generalized,
            column="bmi",
            bin_size=sidebar["bmi_bin_size"],
        )

    if "region" in sidebar["generalized_attributes"]:
        df_generalized = generalize_region(df_generalized)

    return preprocess_dataset(df_generalized)