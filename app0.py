import streamlit as st

from src.data_preparation import (
    load_dataset,
    exploratory_inspection,
    preprocess_dataset,
    validate_prepared_dataset,
)

from src.privacy_methods import apply_privacy_technique

from src.privacy_methods import apply_privacy_technique
from src.statistical_analysis import compare_datasets


DATA_PATH = "data/medicalData.csv"


st.set_page_config(
    page_title="Privacy-Preserving Visual Analytics Dashboard",
    layout="wide",
)


st.title("Visual Analytics Dashboard for Privacy-Preserving Techniques")

st.markdown(
    """
    This dashboard supports the exploratory inspection and preparation of the
    Medical Cost Personal Dataset.
    """
)


try:
    # Load dataset
    df = load_dataset(DATA_PATH)

    # Initial exploratory inspection
    inspection = exploratory_inspection(df)

    # Preprocess dataset
    df_prepared = preprocess_dataset(df)

    df_transformed = apply_privacy_technique(
        df=df_prepared,
        technique="Gaussian Noise",
        columns=["age", "bmi", "charges"],
        sigma=1.0,
    )

    comparison = compare_datasets(
        df_original=df_prepared,
        df_transformed=df_transformed,
        bins=20,
    )

    st.subheader("Test - Descriptive Statistics Comparison")
    st.dataframe(comparison["descriptive_statistics"])

    st.subheader("Test - Correlation Matrix Difference")
    st.dataframe(comparison["correlation_difference"])

    st.subheader("Test - Distribution Similarity")
    st.dataframe(comparison["distribution_similarity"])


    df_noisy = apply_privacy_technique(
        df=df_prepared, 
        technique="Gaussian Noise", 
        columns=["age", "bmi", "charges"], 
        sigma=1.0
    )

    st.subheader("Test - Gaussian Noise Applied")
    st.dataframe(df_noisy.head())

    # Validate prepared dataset
    validation = validate_prepared_dataset(df_prepared)

    # -----------------------------
    # Original dataset preview
    # -----------------------------
    st.subheader("Original Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Exploratory inspection
    # -----------------------------
    st.subheader("Initial Exploratory Inspection")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Rows", df.shape[0])

    with col2:
        st.metric("Original Columns", df.shape[1])

    st.subheader("Data Types")
    st.dataframe(inspection["data_types"])

    st.subheader("Missing Values")
    st.dataframe(inspection["missing_values"])

    st.subheader("Basic Descriptive Statistics - Numerical Variables")
    st.dataframe(inspection["numeric_statistics"])

    if not inspection["categorical_statistics"].empty:
        st.subheader("Basic Descriptive Statistics - Categorical Variables")
        st.dataframe(inspection["categorical_statistics"])

    # -----------------------------
    # Prepared dataset preview
    # -----------------------------
    st.subheader("Prepared Dataset Preview")
    st.dataframe(df_prepared.head())

    # -----------------------------
    # Prepared dataset validation
    # -----------------------------
    st.subheader("Prepared Dataset Validation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Prepared Rows", validation["rows"])

    with col2:
        st.metric("Prepared Columns", validation["columns"])

    with col3:
        st.metric("Missing Values", validation["missing_values_total"])

    with col4:
        st.metric("Duplicate Rows", validation["duplicate_rows"])

    st.write("All columns numeric after encoding:", validation["all_columns_numeric"])

except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")

except Exception as e:
    st.error(f"An error occurred: {e}")