import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the Medical Cost Personal Dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def exploratory_inspection(df: pd.DataFrame) -> dict:
    """
    Perform an initial exploratory inspection of the dataset.

    Includes:
    - identifying data types
    - checking missing values
    - computing basic descriptive statistics
    """
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    inspection = {
        "data_types": df.dtypes.astype(str).reset_index().rename(
            columns={"index": "column", 0: "data_type"}
        ),
        "missing_values": df.isnull().sum().reset_index().rename(
            columns={"index": "column", 0: "missing_values"}
        ),
        "numeric_statistics": df[numeric_columns].describe(),
        "categorical_statistics": df[categorical_columns].describe()
        if len(categorical_columns) > 0 else pd.DataFrame(),
    }

    return inspection


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean invalid or inconsistent values.
    """
    df_clean = df.copy()

    df_clean.columns = df_clean.columns.str.strip()

    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    df_clean = df_clean.drop_duplicates()

    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, if present.
    """
    df_processed = df.copy()

    numeric_columns = df_processed.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = df_processed.select_dtypes(include=["object", "category"]).columns

    for col in numeric_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    return df_processed


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using one-hot encoding.
    """
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    return df_encoded


def validate_prepared_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the prepared dataset after preprocessing.
    """
    validation = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "all_columns_numeric": bool(
            all(dtype != "object" for dtype in df.dtypes.astype(str))
        ),
    }

    return validation


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete preprocessing pipeline.
    """
    df_clean = clean_dataset(df)
    df_no_missing = handle_missing_values(df_clean)
    df_encoded = encode_categorical_variables(df_no_missing)

    return df_encoded