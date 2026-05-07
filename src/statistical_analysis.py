import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return numeric columns from a DataFrame.
    """
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic descriptive statistics for numeric variables.
    """
    numeric_columns = get_numeric_columns(df)

    if not numeric_columns:
        return pd.DataFrame()

    return df[numeric_columns].agg(["mean", "var", "std", "min", "max"]).T


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for numeric variables.
    """
    numeric_columns = get_numeric_columns(df)

    if not numeric_columns:
        return pd.DataFrame()

    return df[numeric_columns].corr(method="pearson")


def compare_descriptive_statistics(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare descriptive statistics between original and transformed datasets.
    """
    original_stats = compute_descriptive_statistics(df_original)
    transformed_stats = compute_descriptive_statistics(df_transformed)

    common_columns = original_stats.index.intersection(transformed_stats.index)

    comparison = pd.DataFrame()

    for col in common_columns:
        for metric in original_stats.columns:
            comparison.loc[col, f"original_{metric}"] = original_stats.loc[col, metric]
            comparison.loc[col, f"transformed_{metric}"] = transformed_stats.loc[col, metric]
            comparison.loc[col, f"difference_{metric}"] = (
                transformed_stats.loc[col, metric] - original_stats.loc[col, metric]
            )

    return comparison


def compare_correlation_matrices(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare Pearson correlation matrices between original and transformed datasets.
    """
    original_corr = compute_correlation_matrix(df_original)
    transformed_corr = compute_correlation_matrix(df_transformed)

    common_columns = original_corr.columns.intersection(transformed_corr.columns)

    if len(common_columns) == 0:
        return pd.DataFrame()

    original_corr = original_corr.loc[common_columns, common_columns]
    transformed_corr = transformed_corr.loc[common_columns, common_columns]

    return transformed_corr - original_corr


def _prepare_histograms(
    original_series: pd.Series,
    transformed_series: pd.Series,
    bins: int = 20
) -> tuple:
    """
    Convert two numeric series into aligned probability distributions.
    """
    original_series = original_series.dropna()
    transformed_series = transformed_series.dropna()

    min_value = min(original_series.min(), transformed_series.min())
    max_value = max(original_series.max(), transformed_series.max())

    original_hist, bin_edges = np.histogram(
        original_series,
        bins=bins,
        range=(min_value, max_value),
        density=False
    )

    transformed_hist, _ = np.histogram(
        transformed_series,
        bins=bin_edges,
        density=False
    )

    original_prob = original_hist / original_hist.sum()
    transformed_prob = transformed_hist / transformed_hist.sum()

    epsilon = 1e-10

    original_prob = original_prob + epsilon
    transformed_prob = transformed_prob + epsilon

    original_prob = original_prob / original_prob.sum()
    transformed_prob = transformed_prob / transformed_prob.sum()

    return original_prob, transformed_prob


def compute_kl_divergence(
    original_series: pd.Series,
    transformed_series: pd.Series,
    bins: int = 20
) -> float:
    """
    Compute Kullback-Leibler divergence between two numeric distributions.
    """
    original_prob, transformed_prob = _prepare_histograms(
        original_series,
        transformed_series,
        bins
    )

    return float(entropy(original_prob, transformed_prob))


def compute_js_divergence(
    original_series: pd.Series,
    transformed_series: pd.Series,
    bins: int = 20
) -> float:
    """
    Compute Jensen-Shannon divergence between two numeric distributions.
    """
    original_prob, transformed_prob = _prepare_histograms(
        original_series,
        transformed_series,
        bins
    )

    return float(jensenshannon(original_prob, transformed_prob) ** 2)


def compare_distribution_similarity(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    bins: int = 20
) -> pd.DataFrame:
    """
    Compute KL and JS divergence for common numeric columns.
    """
    original_numeric = get_numeric_columns(df_original)
    transformed_numeric = get_numeric_columns(df_transformed)

    common_columns = sorted(set(original_numeric).intersection(transformed_numeric))

    results = []

    for col in common_columns:
        results.append({
            "column": col,
            "kl_divergence": compute_kl_divergence(
                df_original[col],
                df_transformed[col],
                bins
            ),
            "js_divergence": compute_js_divergence(
                df_original[col],
                df_transformed[col],
                bins
            )
        })

    return pd.DataFrame(results)


# def compare_datasets(
#     df_original: pd.DataFrame,
#     df_transformed: pd.DataFrame,
#     bins: int = 20
# ) -> dict:
#     """
#     Compare original and transformed datasets using statistical metrics.
#     """
#     return {
#         "descriptive_statistics": compare_descriptive_statistics(
#             df_original,
#             df_transformed
#         ),
#         "correlation_difference": compare_correlation_matrices(
#             df_original,
#             df_transformed
#         ),
#         "distribution_similarity": compare_distribution_similarity(
#             df_original,
#             df_transformed,
#             bins
#         )
#     }

def compare_datasets(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    bins: int = 20
) -> dict:
    return {
        "descriptive_statistics": compare_descriptive_statistics(
            df_original,
            df_transformed
        ),
        "original_correlation": compute_correlation_matrix(df_original),
        "transformed_correlation": compute_correlation_matrix(df_transformed),
        "correlation_difference": compare_correlation_matrices(
            df_original,
            df_transformed
        ),
        "distribution_similarity": compare_distribution_similarity(
            df_original,
            df_transformed,
            bins
        )
    }


def compute_utility_score(distribution_df: pd.DataFrame) -> float:
    """
    Utility based on JS divergence (lower divergence → higher utility)
    """
    if distribution_df.empty:
        return 0.0

    mean_js = distribution_df["js_divergence"].mean()

    return float(1 / (1 + mean_js))

def compute_privacy_score(
    technique,
    sigma=1.0,
    epsilon=1.0,
    sampling_rate=0.8,
    generalized_attributes=None,
    age_bin_size=None,
    bmi_bin_size=None,
    charges_bin_size=None,
    age_range=None,
    bmi_range=None,
    charges_range=None,
):
    if generalized_attributes is None:
        generalized_attributes = []

    if technique == "Gaussian Noise":
        return float(sigma / (sigma + 1))

    if technique == "Laplace Noise":
        return float(1 / (1 + epsilon))

    if technique == "Sampling":
        return float(1 - sampling_rate)

    if technique == "Generalization":
        return compute_generalization_privacy_score(
            generalized_attributes=generalized_attributes,
            age_bin_size=age_bin_size,
            bmi_bin_size=bmi_bin_size,
            age_range=age_range,
            bmi_range=bmi_range,
            charges_bin_size=charges_bin_size, 
            charges_range=charges_range
        )

    return 0.0


# def compute_privacy_score(
#     technique: str,
#     sigma: float = None,
#     epsilon: float = None,
#     sampling_rate: float = None
# ) -> float:
#     """
#     Approximate privacy score depending on the technique.
#     """
#     if technique == "Gaussian Noise" and sigma is not None:
#         return float(sigma)

#     if technique == "Laplace Noise" and epsilon is not None:
#         return float(1 / epsilon)

#     if technique == "Sampling" and sampling_rate is not None:
#         return float(1 - sampling_rate)

#     if technique.startswith("Generalization"):
#         return 0.5  # heuristic (can be refined)

#     return 0.0


# def compute_tradeoff_score(utility: float, privacy: float) -> float:
#     """
#     Combine utility and privacy into a trade-off score.
#     """
#     return float(utility / (1 + privacy))

def compute_tradeoff_score(utility: float, privacy: float) -> float:
    """
    Compute a balanced privacy-utility trade-off score.

    The geometric mean penalizes configurations where either
    privacy or utility is low.
    """
    return float((utility * privacy) ** 0.5)

def compute_generalization_privacy_score(
    generalized_attributes,
    age_bin_size=None,
    bmi_bin_size=None,
    age_range=None,
    bmi_range=None,
    charges_bin_size=None,
    charges_range=None,
):
    scores = []

    if "age" in generalized_attributes and age_bin_size is not None and age_range:
        scores.append(min(1.0, age_bin_size / age_range))

    if "bmi" in generalized_attributes and bmi_bin_size is not None and bmi_range:
        scores.append(min(1.0, bmi_bin_size / bmi_range))

    if "charges" in generalized_attributes and charges_bin_size is not None and charges_range > 0:
        scores.append(min(1.0, charges_bin_size / charges_range))
            

    if "region" in generalized_attributes:
        scores.append(0.5)

    if not scores:
        return 0.0

    return float(sum(scores) / len(scores))