
import streamlit as st

from src.data_preparation import (
    load_dataset,
    exploratory_inspection,
    preprocess_dataset,
    validate_prepared_dataset,
)

from src.privacy_methods import apply_privacy_technique

from src.statistical_analysis import compare_datasets

from src.visualizations import (
    plot_histogram_comparison,
    plot_boxplot_comparison,
    plot_scatter_comparison,
    plot_correlation_heatmap,
    plot_metric_bar_chart,
)


DATA_PATH = "data/medicalData.csv"

st.set_page_config(
    page_title="Privacy-Preserving Visual Analytics Dashboard",
    layout="wide",
)


st.title("Visual Analytics Dashboard for Privacy-Preserving Techniques")

st.markdown(
    """
    This dashboard supports the exploratory inspection, preprocessing, and comparative
    evaluation of privacy-preserving techniques applied to the Medical Cost Personal Dataset.
    """
)


try:
    # ============================================================
    # 1. DATA PIPELINE — LOAD, INSPECT, PREPROCESS
    # ============================================================

    df_original = load_dataset(DATA_PATH)

    inspection = exploratory_inspection(df_original)

    df_prepared = preprocess_dataset(df_original)

    validation = validate_prepared_dataset(df_prepared)

    numeric_columns = df_prepared.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if len(numeric_columns) == 0:
        st.error("No numeric columns available after preprocessing.")
        st.stop()

    # ============================================================
    # 2. SIDEBAR — CONTROL PANEL
    # ============================================================

    st.sidebar.header("Privacy Configuration")

    technique = st.sidebar.selectbox(
        "Select Privacy-Preserving Technique",
        [
            "None",
            "Generalization - Age",
            "Generalization - BMI",
            "Gaussian Noise",
            "Sampling",
            "Laplace Noise",
        ],
    )

    sigma = 1.0
    epsilon = 1.0
    sampling_rate = 0.8
    selected_columns = []

    if technique == "Gaussian Noise":
        sigma = st.sidebar.slider(
            "Sigma (Noise Level)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        selected_columns = st.sidebar.multiselect(
            "Select Numeric Columns",
            numeric_columns,
            default=numeric_columns,
        )

    elif technique == "Laplace Noise":
        epsilon = st.sidebar.slider(
            "Epsilon (Privacy Parameter)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )

        selected_columns = st.sidebar.multiselect(
            "Select Numeric Columns",
            numeric_columns,
            default=numeric_columns,
        )

    elif technique == "Sampling":
        sampling_rate = st.sidebar.slider(
            "Sampling Rate",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05,
        )

    st.sidebar.header("Visualization Controls")

    selected_column = st.sidebar.selectbox(
        "Variable for Histogram and Boxplot",
        numeric_columns,
    )

    x_column = st.sidebar.selectbox(
        "X-axis for Scatter Plot",
        numeric_columns,
        index=0,
    )

    y_column_index = 1 if len(numeric_columns) > 1 else 0

    y_column = st.sidebar.selectbox(
        "Y-axis for Scatter Plot",
        numeric_columns,
        index=y_column_index,
    )

    bins = st.sidebar.slider(
        "Number of Bins",
        min_value=10,
        max_value=60,
        value=20,
        step=5,
    )

    # ============================================================
    # 3. DATA PIPELINE — APPLY SELECTED PRIVACY TECHNIQUE
    # ============================================================

    if technique == "None":
        df_transformed = df_prepared.copy()

    else:
        df_transformed = apply_privacy_technique(
            df=df_prepared,
            technique=technique,
            columns=selected_columns,
            sigma=sigma,
            sampling_rate=sampling_rate,
            epsilon=epsilon,
        )

    comparison = compare_datasets(
        df_original=df_prepared,
        df_transformed=df_transformed,
        bins=bins,
    )

    # ============================================================
    # 4. OUTPUT — DATASET INSPECTION
    # ============================================================

    st.header("1. Dataset Preparation")

    with st.expander("Original Dataset Preview", expanded=True):
        st.dataframe(df_original.head())

    with st.expander("Initial Exploratory Inspection", expanded=False):
        st.subheader("Data Types")
        st.dataframe(inspection["data_types"])

        st.subheader("Missing Values")
        st.dataframe(inspection["missing_values"])

        st.subheader("Basic Descriptive Statistics - Numerical Variables")
        st.dataframe(inspection["numeric_statistics"])

        if not inspection["categorical_statistics"].empty:
            st.subheader("Basic Descriptive Statistics - Categorical Variables")
            st.dataframe(inspection["categorical_statistics"])

    with st.expander("Prepared Dataset Preview", expanded=False):
        st.dataframe(df_prepared.head())

    with st.expander("Prepared Dataset Validation", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", validation["rows"])

        with col2:
            st.metric("Columns", validation["columns"])

        with col3:
            st.metric("Missing Values", validation["missing_values_total"])

        with col4:
            st.metric("Duplicate Rows", validation["duplicate_rows"])

        st.write(
            "All columns numeric after encoding:",
            validation["all_columns_numeric"],
        )

    # ============================================================
    # 5. OUTPUT — PRIVACY CONFIGURATION SUMMARY
    # ============================================================

    st.header("2. Privacy-Preserving Transformation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Technique", technique)

    with col2:
        st.metric("Original Rows", df_prepared.shape[0])

    with col3:
        st.metric("Transformed Rows", df_transformed.shape[0])

    with st.expander("Transformed Dataset Preview", expanded=True):
        st.dataframe(df_transformed.head())

    # ============================================================
    # 6. OUTPUT — STATISTICAL METRICS
    # ============================================================

    st.header("3. Statistical Comparison")

    st.subheader("Descriptive Statistics Comparison")
    st.dataframe(comparison["descriptive_statistics"])

    st.subheader("Distribution Similarity Metrics")
    st.dataframe(comparison["distribution_similarity"])

    st.subheader("Correlation Matrix Difference")
    st.dataframe(comparison["correlation_difference"])

    # ============================================================
    # 7. OUTPUT — VISUALIZATIONS
    # ============================================================

    st.header("4. Visual Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Histogram Comparison")
        fig_hist = plot_histogram_comparison(
            df_original=df_prepared,
            df_transformed=df_transformed,
            column=selected_column,
            bins=bins,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Boxplot Comparison")
        fig_box = plot_boxplot_comparison(
            df_original=df_prepared,
            df_transformed=df_transformed,
            column=selected_column,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Scatter Plot Comparison")
    fig_scatter = plot_scatter_comparison(
        df_original=df_prepared,
        df_transformed=df_transformed,
        x_column=x_column,
        y_column=y_column,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation Difference Heatmap")
    fig_corr = plot_correlation_heatmap(
        comparison["correlation_difference"],
        title="Correlation Difference: Transformed - Original",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Distribution Similarity - JS Divergence")
    fig_js = plot_metric_bar_chart(
        comparison["distribution_similarity"],
        metric_column="js_divergence",
    )
    st.plotly_chart(fig_js, use_container_width=True)

    st.subheader("Distribution Similarity - KL Divergence")
    fig_kl = plot_metric_bar_chart(
        comparison["distribution_similarity"],
        metric_column="kl_divergence",
    )
    st.plotly_chart(fig_kl, use_container_width=True)


except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")

except Exception as e:
    st.error(f"An error occurred: {e}")