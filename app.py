import streamlit as st
import pandas as pd

from src.data_preparation import (
    load_dataset,
    exploratory_inspection,
    preprocess_dataset,
    validate_prepared_dataset,
)

from src.privacy_methods import apply_privacy_technique

from src.statistical_analysis import (
    compare_datasets,
    compute_utility_score,
    compute_privacy_score,
    compute_tradeoff_score,
)

from src.visualizations import (
    plot_histogram_comparison,
    plot_boxplot_comparison,
    plot_scatter_comparison,
    plot_correlation_heatmap,
    plot_metric_bar_chart,
    plot_tradeoff, 
    plot_tradeoff_comparison,
)



from src.visualizations import plot_tradeoff


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
    # DATA PIPELINE — LOAD, INSPECT, PREPROCESS
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
    # SIDEBAR — CONTROL PANEL
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

    if technique in ["Gaussian Noise", "Laplace Noise"] and not selected_columns:
        st.warning("Please select at least one numeric column.")
        st.stop()

    # ============================================================
    # DATA PIPELINE — APPLY SELECTED PRIVACY TECHNIQUE
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

    utility_score = compute_utility_score(
        comparison["distribution_similarity"]
    )

    privacy_score = compute_privacy_score(
        technique=technique,
        sigma=sigma,
        epsilon=epsilon,
        sampling_rate=sampling_rate
    )

    tradeoff_score = compute_tradeoff_score(
        utility_score,
        privacy_score
    )

    tradeoff_results = []

    techniques_to_compare = [
        {
            "name": "Gaussian Noise",
            "sigma": 1.0,
            "epsilon": 1.0,
            "sampling_rate": 0.8,
            "columns": numeric_columns,
        },
        {
            "name": "Laplace Noise",
            "sigma": 1.0,
            "epsilon": 1.0,
            "sampling_rate": 0.8,
            "columns": numeric_columns,
        },
        {
            "name": "Sampling",
            "sigma": 1.0,
            "epsilon": 1.0,
            "sampling_rate": 0.8,
            "columns": numeric_columns,
        },
    ]

    for item in techniques_to_compare:
        df_temp = apply_privacy_technique(
            df=df_prepared,
            technique=item["name"],
            columns=item["columns"],
            sigma=item["sigma"],
            epsilon=item["epsilon"],
            sampling_rate=item["sampling_rate"],
        )

        temp_comparison = compare_datasets(
            df_original=df_prepared,
            df_transformed=df_temp,
            bins=bins,
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
                "Technique": item["name"],
                "Utility Score": temp_utility,
                "Privacy Score": temp_privacy,
                "Trade-off Score": temp_tradeoff,
            }
        )

    df_tradeoff_comparison = pd.DataFrame(tradeoff_results)



    # ============================================================
    # OUTPUT — TABS
    # ============================================================

    tab_data, tab_transformation, tab_statistics, tab_visual, tab_tradeoff = st.tabs(
        [
            "Data Preparation",
            "Transformation",
            "Statistical Analysis",
            "Visual Analysis",
            "Trade-off Evaluation",
        ]
    )

    # ============================================================
    # TAB 1 — DATA PREPARATION
    # ============================================================

    with tab_data:
        st.header("Dataset Preparation")

        st.subheader("Original Dataset Preview")
        st.dataframe(df_original.head())

        st.subheader("Initial Exploratory Inspection")

        st.write("Data Types")
        st.dataframe(inspection["data_types"])

        st.write("Missing Values")
        st.dataframe(inspection["missing_values"])

        st.write("Basic Descriptive Statistics - Numerical Variables")
        st.dataframe(inspection["numeric_statistics"])

        if not inspection["categorical_statistics"].empty:
            st.write("Basic Descriptive Statistics - Categorical Variables")
            st.dataframe(inspection["categorical_statistics"])

        st.subheader("Prepared Dataset Preview")
        st.dataframe(df_prepared.head())

        st.subheader("Prepared Dataset Validation")

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
    # TAB 2 — TRANSFORMATION
    # ============================================================

    with tab_transformation:
        st.header("Privacy-Preserving Transformation")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Selected Technique", technique)

        with col2:
            st.metric("Original Rows", df_prepared.shape[0])

        with col3:
            st.metric("Transformed Rows", df_transformed.shape[0])

        if technique == "Gaussian Noise":
            st.write("Noise level σ:", sigma)
            st.write("Affected columns:", selected_columns)

        elif technique == "Laplace Noise":
            st.write("Privacy parameter ε:", epsilon)
            st.write("Affected columns:", selected_columns)

        elif technique == "Sampling":
            st.write("Sampling rate:", sampling_rate)

        st.subheader("Transformed Dataset Preview")
        st.dataframe(df_transformed.head())

        st.download_button(
            label="Download Transformed Dataset",
            data=df_transformed.to_csv(index=False),
            file_name="transformed_dataset.csv",
            mime="text/csv",
        )

    # ============================================================
    # TAB 3 — STATISTICAL ANALYSIS
    # ============================================================

    with tab_statistics:
        st.header("Statistical Comparison")

        st.info(
            """
            JS Divergence measures similarity between distributions.
            Lower values indicate higher similarity and therefore better utility.

            KL Divergence measures distributional difference between the original
            and transformed data. Higher values indicate greater distortion.
            """
        )

        st.subheader("Utility Score")

        st.metric(
            "Utility Score",
            round(utility_score, 4),
        )

        st.caption(
            "The utility score is computed from JS divergence. Higher values indicate better preservation of data utility."
        )

        st.subheader("Most Affected Variables")

        most_affected = comparison["distribution_similarity"].sort_values(
            by="js_divergence",
            ascending=False,
        ).head(3)

        st.dataframe(most_affected)

        st.subheader("Descriptive Statistics Comparison")
        st.dataframe(comparison["descriptive_statistics"])

        st.subheader("Distribution Similarity Metrics")
        st.dataframe(comparison["distribution_similarity"])

        st.subheader("Correlation Matrix Difference")
        st.dataframe(comparison["correlation_difference"])

    # ============================================================
    # TAB 4 — VISUAL ANALYSIS
    # ============================================================

    with tab_visual:
        st.header("Visual Comparison")

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

        if x_column == y_column:
            st.warning("Select different variables for the X-axis and Y-axis.")
        else:
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

    # ============================================================
    # TAB 5 — Trade-off Evaluation
    # ============================================================

    with tab_tradeoff:
        st.header("Privacy-Utility Trade-off Evaluation")

        st.info(
            """
            This section evaluates the balance between privacy protection and data utility.

            Utility is estimated from distribution similarity using JS divergence.
            Privacy is estimated from the intensity of the selected transformation.
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Utility Score", round(utility_score, 4))

        with col2:
            st.metric("Privacy Score", round(privacy_score, 4))

        with col3:
            st.metric("Trade-off Score", round(tradeoff_score, 4))

        df_tradeoff = pd.DataFrame([
            {
                "Technique": technique,
                "Utility Score": utility_score,
                "Privacy Score": privacy_score,
                "Trade-off Score": tradeoff_score,
            }
        ])

        st.subheader("Current Configuration")
        st.dataframe(df_tradeoff)

        st.subheader("Privacy-Utility Plot")
        fig_tradeoff = plot_tradeoff_comparison(df_tradeoff)
        st.plotly_chart(fig_tradeoff, use_container_width=True)

        st.subheader("Comparison Across Techniques")

        st.dataframe(df_tradeoff_comparison)

        st.subheader("Privacy-Utility Comparison Plot")

        fig_tradeoff_comparison = plot_tradeoff_comparison(
            df_tradeoff_comparison
        )

        st.plotly_chart(
            fig_tradeoff_comparison,
            use_container_width=True,
        )



except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")

except Exception as e:
    st.error(f"An error occurred: {e}")