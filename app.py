import streamlit as st
import pandas as pd
import plotly.express as px

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
    plot_privacy_utility_dual_bar,
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
    # DATA PIPELINE — carregar, inspecionar, preprocessar
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
    # SIDEBAR — painél de controlo
    # ============================================================

    # st.sidebar.header("Privacy Configuration")

    # technique = st.sidebar.selectbox(
    #     "Select Privacy-Preserving Technique",
    #     [
    #         "None",
    #         "Generalization - Age",
    #         "Generalization - BMI",
    #         "Gaussian Noise",
    #         "Sampling",
    #         "Laplace Noise",
    #     ],
    # )

    # sigma = 1.0
    # epsilon = 1.0
    # sampling_rate = 0.8
    # selected_columns = []

    # if technique == "Gaussian Noise":
    #     sigma = st.sidebar.slider(
    #         "Sigma (Noise Level)",
    #         min_value=0.1,
    #         max_value=10.0,
    #         value=1.0,
    #         step=0.1,
    #     )

    #     selected_columns = st.sidebar.multiselect(
    #         "Select Numeric Columns",
    #         numeric_columns,
    #         default=numeric_columns,
    #     )

    # elif technique == "Laplace Noise":
    #     epsilon = st.sidebar.slider(
    #         "Epsilon (Privacy Parameter)",
    #         min_value=0.1,
    #         max_value=5.0,
    #         value=1.0,
    #         step=0.1,
    #     )

    #     selected_columns = st.sidebar.multiselect(
    #         "Select Numeric Columns",
    #         numeric_columns,
    #         default=numeric_columns,
    #     )

    # elif technique == "Sampling":
    #     sampling_rate = st.sidebar.slider(
    #         "Sampling Rate",
    #         min_value=0.1,
    #         max_value=1.0,
    #         value=0.8,
    #         step=0.05,
    #     )

    # st.sidebar.header("Visualization Controls")

    # selected_column = st.sidebar.selectbox(
    #     "Variable for Histogram and Boxplot",
    #     numeric_columns,
    # )

    # x_column = st.sidebar.selectbox(
    #     "X-axis for Scatter Plot",
    #     numeric_columns,
    #     index=0,
    # )

    # y_column_index = 1 if len(numeric_columns) > 1 else 0

    # y_column = st.sidebar.selectbox(
    #     "Y-axis for Scatter Plot",
    #     numeric_columns,
    #     index=y_column_index,
    # )

    # bins = st.sidebar.slider(
    #     "Number of Bins",
    #     min_value=10,
    #     max_value=60,
    #     value=20,
    #     step=5,
    # )

    #st.sidebar.header("Data Analysis")

    #st.sidebar.caption(
    #    "Adjust parameters to control the level of privacy applied to the dataset."
    #)

# ============================================================
# SIDEBAR — PRIVACY TRANSFORMATION SETTINGS
# ============================================================

    st.sidebar.header("Privacy Transformation Settings")

    technique = st.sidebar.selectbox(
        "Privacy-preserving technique",
        [
            "None",
            "Generalization - Age",
            "Generalization - BMI",
            "Gaussian Noise",
            "Sampling",
            "Laplace Noise",
        ],
        help="Select the privacy-preserving transformation to apply to the prepared dataset.",
    )

    sigma = 1.0
    epsilon = 1.0
    sampling_rate = 0.8
    selected_columns = []
    bin_size = None
    data_range = None

    if technique == "Generalization - Age":
        col = "age"
        data_range = df_prepared[col].max() - df_prepared[col].min()

        bin_size = st.sidebar.slider(
            "Age generalization interval",
            min_value=1,
            max_value=int(data_range),
            value=10,
            step=1,
            help=(
                "Defines the size of the age intervals used for generalization. "
                "Larger intervals increase privacy but reduce data precision."
            ),
        )

    elif technique == "Generalization - BMI":
        col = "bmi"
        data_range = df_prepared[col].max() - df_prepared[col].min()

        bin_size = st.sidebar.slider(
            "BMI generalization interval",
            min_value=0.5,
            max_value=float(data_range),
            value=2.0,
            step=0.5,
            help=(
                "Defines the size of the BMI intervals used for generalization. "
                "Larger intervals increase privacy but reduce detail in the data."
            ),
        )

    elif technique == "Gaussian Noise":
        sigma = st.sidebar.slider(
            "Gaussian noise level (σ)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help=(
                "Controls the standard deviation of the Gaussian noise added to selected numeric variables. "
                "Higher values introduce stronger perturbation."
            ),
        )

        selected_columns = st.sidebar.multiselect(
            "Variables to perturb",
            numeric_columns,
            default=numeric_columns,
            help="Select the numeric variables to which Gaussian noise will be applied.",
        )

    elif technique == "Laplace Noise":
        epsilon = st.sidebar.slider(
            "Privacy budget (ε)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help=(
                "Controls the privacy budget used for Laplace noise. "
                "Lower ε values provide stronger privacy but usually reduce utility."
            ),
        )

        selected_columns = st.sidebar.multiselect(
            "Variables to perturb",
            numeric_columns,
            default=numeric_columns,
            help="Select the numeric variables to which Laplace noise will be applied.",
        )

    elif technique == "Sampling":
        sampling_rate = st.sidebar.slider(
            "Retained data proportion",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help=(
                "Defines the proportion of records retained after sampling. "
                "Lower values expose less data and therefore increase privacy."
            ),
        )


    # ============================================================
    # SIDEBAR — VISUAL EVALUATION SETTINGS
    # ============================================================

    st.sidebar.header("Visual Evaluation Settings")

    selected_column = st.sidebar.selectbox(
        "Variable for distribution plots",
        numeric_columns,
        help="Select the numeric variable used in the histogram and boxplot comparisons.",
    )

    x_column = st.sidebar.selectbox(
        "Scatter plot X variable",
        numeric_columns,
        index=0,
        help="Select the variable shown on the X-axis of the scatter plot.",
    )

    y_column_index = 1 if len(numeric_columns) > 1 else 0

    y_column = st.sidebar.selectbox(
        "Scatter plot Y variable",
        numeric_columns,
        index=y_column_index,
        help="Select the variable shown on the Y-axis of the scatter plot.",
    )

    bins = st.sidebar.slider(
        "Histogram bin count",
        min_value=10,
        max_value=60,
        value=20,
        step=5,
        help=(
            "Defines the number of bins used in histograms and in the histogram-based "
            "distribution comparison for KL and JS divergence."
        ),
    )

    if technique in ["Gaussian Noise", "Laplace Noise"] and not selected_columns:
        st.warning("Please select at least one numeric variable to perturb.")
        st.stop()
    
    # footer com texto informativo do trabalho
    with st.sidebar:
        st.markdown("---") # Linha divisória após os controlos
        st.markdown(
            """
            <div style="text-align: center; color: #808495; font-size: 0.85em;">
                <p><strong>VDA Project Dashboard</strong><br>
                Comparative Evaluation of Privacy Techniques.</p>
                <p style="text-align: justify;  padding-left: 1rem; padding-right: 1rem; " >
                Project developed by <span  style="color: #1f4e79; font-weight: 600;">Juliana Jesus and Paulo Silva</span> for the Visualization and 
                Data Analysis module, integrated into the Transversal and Transferable Competences II 
                course unit of the University of Aveiro's third-cycle programme.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    

    # ===========================================================================
    # DATA PIPELINE — Aplicar a técnica de preservação de privacidade selecionada
    # ===========================================================================

    bin_size = None
    data_range = None

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
        sampling_rate=sampling_rate, 
        bin_size=bin_size,
        data_range=data_range,
    )

    tradeoff_score = compute_tradeoff_score(
        utility_score,
        privacy_score
    )

    tradeoff_results = []

    # 1. Add current user-selected configuration
    tradeoff_results.append(
        {
            "Technique": f"{technique} (Current)",
            "Utility Score": utility_score,
            "Privacy Score": privacy_score,
            "Trade-off Score": tradeoff_score,
        }
    )

    # 2. Define default configurations for comparison
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

    # 3. Add default configurations, excluding the current technique
    for item in techniques_to_compare:

        if item["name"] == technique:
            continue

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
                "Technique": f'{item["name"]} (Default)',
                "Utility Score": temp_utility,
                "Privacy Score": temp_privacy,
                "Trade-off Score": temp_tradeoff,
            }
        )

    df_tradeoff_comparison = pd.DataFrame(tradeoff_results)

    df_tradeoff_comparison = df_tradeoff_comparison.sort_values(
        by="Trade-off Score",
        ascending=False,
    ).reset_index(drop=True)



    # ============================================================
    # OUTPUT — Tabs 
    # ============================================================

    tab_data, tab_transformation, tab_statistics, tab_visual, tab_tradeoff = st.tabs(
        [
            "Exploratory Inspection",
            "Privacy-Preserving Transformation",
            "Statistical Evaluation",
            "Visual Comparision",
            "Trade-off Analysis",
        ]
    )

    # ============================================================
    # TAB 1 — Preparação de Dados
    # ============================================================

    with tab_data:
      #  st.header("Dataset Preparation")

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

        st.subheader("Exploratory Visualization")

        numeric_cols_original = df_original.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_cols_original = df_original.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        selected_num_col = st.selectbox(
            "Select numeric variable",
            numeric_cols_original,
            key="explore_numeric"
        )

        # fig_hist = px.histogram(
        #         df_original,
        #         x=selected_num_col,
        #         nbins=30,
        #         title=f"Histogram - {selected_num_col}"
        # )

        # st.plotly_chart(fig_hist, use_container_width=True)


        mean_value = df_original[selected_num_col].mean()

        fig_hist = px.histogram(
            df_original,
            x=selected_num_col,
            nbins=30,
            title=f"Distribution of {selected_num_col}",
        )

        fig_hist.add_vline(
            x=mean_value,
            line_dash="dash",
            annotation_text="Mean",
            annotation_position="top right",
        )

        fig_hist.update_layout(
            xaxis_title=selected_num_col,
            yaxis_title="Count",
            bargap=0.08,
            height=450,
            showlegend=False,
        )

        fig_hist.update_traces(
            marker_line_width=1,
            opacity=0.85,
        )

        st.plotly_chart(fig_hist, use_container_width=True)


        fig_box = px.box(
            df_original,
            y=selected_num_col,
            title=f"Boxplot - {selected_num_col}"
        )

        st.plotly_chart(fig_box, use_container_width=True)

        if len(categorical_cols_original) > 0:
            selected_cat_col = st.selectbox(
                "Select categorical variable",
                categorical_cols_original,
                key="explore_categorical"
            )

            category_counts = (
                df_original[selected_cat_col]
                .value_counts()
                .reset_index()
            )

            category_counts.columns = [selected_cat_col, "count"]

            category_counts["percentage"] = (
                category_counts["count"] / category_counts["count"].sum() * 100
            ).round(2)

            category_counts = category_counts.sort_values(
                by="count",
                ascending=True
            )

            fig_bar = px.bar(
                category_counts,
                x="count",
                y=selected_cat_col,
                orientation="h",
                text="percentage",
                title=f"Category Distribution - {selected_cat_col}"
            )

            fig_bar.update_traces(
                texttemplate="%{text}%",
                textposition="outside",
                marker_line_width=1,
                opacity=0.85,
            )

            fig_bar.update_layout(
                xaxis_title="Count",
                yaxis_title=selected_cat_col,
                height=450,
                showlegend=False,
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        
    # ============================================================
    # TAB 2 — Transformação ao dataset 
    # ============================================================

    with tab_transformation:
    #    st.header("Privacy-Preserving Transformation")

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
    # TAB 3 — Análise Estatistica
    # ============================================================

    with tab_statistics:
        #st.header("Statistical Evaluation")

        # ------------------------------------------------------------
        # 1. Statistical Properties
        # ------------------------------------------------------------

        st.subheader("1. Statistical Properties")

        st.caption(
            """
            This section evaluates how basic numerical properties changed after
            applying the selected privacy-preserving technique.
            It compares measures such as mean, variance, standard deviation,
            minimum, and maximum.
            """
        )

        st.dataframe(comparison["descriptive_statistics"])

        st.divider()

        # ------------------------------------------------------------
        # 2. Relationship Preservation
        # ------------------------------------------------------------

        st.subheader("2. Relationship Preservation")

        st.caption(
            """
            This section evaluates whether relationships between variables were
            preserved after transformation. The table shows the difference between
            the transformed and original correlation matrices.
            """
        )

        st.dataframe(comparison["correlation_difference"])

        st.divider()

        # ------------------------------------------------------------
        # 3. Distribution Similarity
        # ------------------------------------------------------------

        st.subheader("3. Distribution Similarity")

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

        st.subheader("Distribution Similarity Metrics")

        st.dataframe(comparison["distribution_similarity"])

    # ============================================================
    # TAB 4 — Análise Visual
    # ============================================================

    with tab_visual:

        st.subheader("1. Statistical Properties")

        st.caption(
            "Visual comparison of distributions to observe changes in central tendency and dispersion."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("Histogram Comparison")
            st.plotly_chart(
                plot_histogram_comparison(df_prepared, df_transformed, selected_column),
                use_container_width=True
            )

        with col2:
            st.write("Boxplot Comparison")
            st.plotly_chart(
                plot_boxplot_comparison(df_prepared, df_transformed, selected_column),
                use_container_width=True
            )
        
        st.subheader("2. Relationship Preservation")

        st.caption(
            "Visual inspection of relationships between variables after transformation."
        )

        if x_column == y_column:
            st.warning("Select different variables for X and Y.")
        else:
            st.plotly_chart(
                plot_scatter_comparison(df_prepared, df_transformed, x_column, y_column),
                use_container_width=True
            )

        st.subheader("Correlation Heatmap")

        st.plotly_chart(
            plot_correlation_heatmap(
                comparison["correlation_difference"],
                "Correlation Difference (Transformed - Original)"
            ),
            use_container_width=True
        )

        st.subheader("3. Distribution Similarity")

        st.caption(
            "Visual comparison of distribution changes using overlay histograms and divergence metrics."
        )

        st.plotly_chart(
            plot_histogram_comparison(
                df_prepared,
                df_transformed,
                selected_column
            ),
            use_container_width=True,
            key="distribution_similarity_histogram"
        )

        st.subheader("JS Divergence")

        st.plotly_chart(
            plot_metric_bar_chart(
                comparison["distribution_similarity"],
                "js_divergence"
            ),
            use_container_width=True,
            key="distribution_similarity_js"
        )

        st.subheader("KL Divergence")

        st.plotly_chart(
            plot_metric_bar_chart(
                comparison["distribution_similarity"],
                "kl_divergence"
            ),
            use_container_width=True,
            key="distribution_similarity_kl"
        )


    

    # ============================================================
    # TAB 5 — Avaliação do Trade-off
    # ============================================================

    with tab_tradeoff:
        st.info(
            """
            This section evaluates the balance between privacy protection and data utility.

            Utility measures how well the original data characteristics are preserved and is estimated from distribution similarity using JS divergence.

            Privacy is estimated from the intensity of the selected transformation.

            The trade-off score combines utility and privacy to indicate the overall effectiveness of the selected configuration.
            """
        )

        st.subheader("1. Current Configuration")

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

        st.dataframe(df_tradeoff)

        st.subheader("2. Current Privacy-Utility Plot")

        fig_tradeoff = plot_tradeoff_comparison(df_tradeoff)

        st.plotly_chart(
            fig_tradeoff,
            use_container_width=True,
            key="current_tradeoff_plot",
        )

        st.divider()

        st.subheader("3. Comparison Across Techniques")

        st.caption(
            """
            This comparison evaluates multiple privacy-preserving techniques using a fixed default configuration.
            It supports the identification of techniques that provide a better balance between privacy and utility.
            """
        )

        st.dataframe(df_tradeoff_comparison)

        st.subheader("4. Privacy-Utility Comparison Plot")

        fig_tradeoff_comparison = plot_tradeoff_comparison(
            df_tradeoff_comparison
        )

        st.plotly_chart(
            fig_tradeoff_comparison,
            use_container_width=True,
            key="technique_tradeoff_comparison_plot",
        )

        st.subheader("Trade-off Ranking")

        fig_bar = px.bar(
            df_tradeoff_comparison.sort_values(
                by="Trade-off Score", ascending=True
            ),
            x="Trade-off Score",
            y="Technique",
            orientation="h",
            title="Technique Ranking by Trade-off Score"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Privacy vs Utility (Side-by-Side)")

        fig_dual = plot_privacy_utility_dual_bar(df_tradeoff_comparison)

        st.plotly_chart(
            fig_dual,
            use_container_width=True,
            key="dual_bar_chart"
        )

    


except FileNotFoundError:
    st.error(f"Dataset not found: {DATA_PATH}")

except Exception as e:
    st.error(f"An error occurred: {e}")