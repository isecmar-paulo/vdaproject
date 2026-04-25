import numpy as np
import pandas as pd


def generalize_age(
    df: pd.DataFrame,
    age_column: str = "age",
    bins: list = None,
    labels: list = None
) -> pd.DataFrame:
    """
    Generalize the age attribute into age intervals.
    """
    df_generalized = df.copy()

    if age_column not in df_generalized.columns:
        raise ValueError(f"Column '{age_column}' not found in dataset.")

    if bins is None:
        bins = [0, 18, 30, 45, 60, 100]

    if labels is None:
        labels = ["0-18", "19-30", "31-45", "46-60", "60+"]

    df_generalized[age_column] = pd.cut(
        df_generalized[age_column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df_generalized


def generalize_bmi(
    df: pd.DataFrame,
    bmi_column: str = "bmi"
) -> pd.DataFrame:
    """
    Generalize BMI into standard BMI categories.
    """
    df_generalized = df.copy()

    if bmi_column not in df_generalized.columns:
        raise ValueError(f"Column '{bmi_column}' not found in dataset.")

    bins = [0, 18.5, 25, 30, 100]
    labels = ["underweight", "normal", "overweight", "obese"]

    df_generalized[bmi_column] = pd.cut(
        df_generalized[bmi_column],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df_generalized


def add_gaussian_noise(
    df: pd.DataFrame,
    columns: list,
    sigma: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add Gaussian noise to selected numerical columns.
    """
    df_noisy = df.copy()

    rng = np.random.default_rng(random_state)

    for col in columns:
        if col not in df_noisy.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(df_noisy[col]):
            raise TypeError(f"Column '{col}' must be numeric to add Gaussian noise.")

        noise = rng.normal(
            loc=0,
            scale=sigma,
            size=len(df_noisy)
        )

        df_noisy[col] = df_noisy[col] + noise

    return df_noisy


def sample_dataset(
    df: pd.DataFrame,
    sampling_rate: float = 0.8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly sample a percentage of the dataset.
    """
    if sampling_rate <= 0 or sampling_rate > 1:
        raise ValueError("sampling_rate must be between 0 and 1.")

    df_sampled = df.sample(
        frac=sampling_rate,
        random_state=random_state
    ).reset_index(drop=True)

    return df_sampled


def add_laplace_noise(
    df: pd.DataFrame,
    columns: list,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Add Laplace noise to selected numerical columns as a simplified
    differential privacy mechanism.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0.")

    df_private = df.copy()

    rng = np.random.default_rng(random_state)

    scale = sensitivity / epsilon

    for col in columns:
        if col not in df_private.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(df_private[col]):
            raise TypeError(f"Column '{col}' must be numeric to add Laplace noise.")

        noise = rng.laplace(
            loc=0,
            scale=scale,
            size=len(df_private)
        )

        df_private[col] = df_private[col] + noise

    return df_private


def apply_privacy_technique(
    df: pd.DataFrame,
    technique: str,
    columns: list = None,
    sigma: float = 1.0,
    sampling_rate: float = 0.8,
    epsilon: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Apply a selected privacy-preserving technique to the dataset.
    """
    if technique == "Generalization - Age":
        return generalize_age(df)

    if technique == "Generalization - BMI":
        return generalize_bmi(df)

    if technique == "Gaussian Noise":
        if columns is None:
            raise ValueError("columns must be provided for Gaussian Noise.")
        return add_gaussian_noise(
            df=df,
            columns=columns,
            sigma=sigma,
            random_state=random_state
        )

    if technique == "Sampling":
        return sample_dataset(
            df=df,
            sampling_rate=sampling_rate,
            random_state=random_state
        )

    if technique == "Laplace Noise":
        if columns is None:
            raise ValueError("columns must be provided for Laplace Noise.")
        return add_laplace_noise(
            df=df,
            columns=columns,
            epsilon=epsilon,
            random_state=random_state
        )

    raise ValueError(f"Unknown privacy technique: {technique}")