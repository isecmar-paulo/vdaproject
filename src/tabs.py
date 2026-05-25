import pandas as pd
import plotly.express as px
import streamlit as st

from src.visualizations import (
    plot_density_comparison,
    plot_histogram_comparison,
    plot_boxplot_comparison,
    plot_privacy_utility_dual_bar,
    plot_scatter_comparison,
    plot_correlation_heatmap,
    plot_metric_bar_chart,
    plot_top_correlation_changes,
    plot_tradeoff_comparison,
)

from src.parameter_sweep import run_parameter_sweep

from src.visualizations import (
    plot_parameter_sweep_tradeoff,
    plot_parameter_sweep_ranking,
    plot_parameter_sweep_lines,
    plot_density_comparison,
    plot_correlation_matrix_comparison,
    plot_correlation_scatter,
)

def render_exploratory_tab(data):
    df_original = data["df_original"]
    df_prepared = data["df_prepared"]
    inspection = data["inspection"]
    validation = data["validation"]

    st.subheader("Original Dataset Preview")
    st.dataframe(df_original.head())

    st.subheader("Initial Exploratory Inspection")

    st.write("Data Types")
    st.dataframe(inspection["data_types"])

    st.write("Missing Values")
    st.dataframe(inspection["missing_values"])

    st.write("Basic Descriptive Statistics - Numerical Variables")
    st.dataframe(inspection["numeric_statistics"])

  #  if not inspection["categorical_statistics"].empty:
  #      st.write("Basic Descriptive Statistics - Categorical Variables")
  #      st.dataframe(inspection["categorical_statistics"])

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
        key="explore_numeric",
    )

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

    st.plotly_chart(
        fig_hist,
        use_container_width=True,
        key="exploratory_histogram",
    )

    fig_box = px.box(
        df_original,
        y=selected_num_col,
        title=f"Boxplot - {selected_num_col}",
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True,
        key="exploratory_boxplot",
    )

    if len(categorical_cols_original) > 0:
        selected_cat_col = st.selectbox(
            "Select categorical variable",
            categorical_cols_original,
            key="explore_categorical",
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
            ascending=True,
        )

        fig_bar = px.bar(
            category_counts,
            x="count",
            y=selected_cat_col,
            orientation="h",
            text="percentage",
            title=f"Category Distribution - {selected_cat_col}",
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

        st.plotly_chart(
            fig_bar,
            use_container_width=True,
            key="exploratory_categorical_bar",
        )


def render_transformation_tab(df_prepared, df_transformed, sidebar):
    technique = sidebar["technique"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Technique", technique)

    with col2:
        st.metric("Original Rows", df_prepared.shape[0])

    with col3:
        st.metric("Transformed Rows", df_transformed.shape[0])

    if technique == "Gaussian Noise":
        st.write("Noise level σ:", sidebar["sigma"])
        st.write("Affected columns:", sidebar["selected_columns"])

    elif technique == "Laplace Noise":
        st.write("Privacy parameter ε:", sidebar["epsilon"])
        st.write("Affected columns:", sidebar["selected_columns"])

    elif technique == "Sampling":
        st.write("Sampling rate:", sidebar["sampling_rate"])
    
    elif technique == "Generalization":
      st.write("Generalized attributes:", sidebar["generalized_attributes"])

      if "age" in sidebar["generalized_attributes"]:
          st.write("Age interval:", sidebar["age_bin_size"])

      if "bmi" in sidebar["generalized_attributes"]:
          st.write("BMI interval:", sidebar["bmi_bin_size"])

      if "region" in sidebar["generalized_attributes"]:
          st.write("Region hierarchy: northeast/northwest → north; southeast/southwest → south")

      if "children" in sidebar["generalized_attributes"]:
        st.write("Children interval:", sidebar["children_bin_size"])

    # elif technique in ["Generalization - Age", "Generalization - BMI"]:
    #     st.write("Generalization interval:", sidebar["bin_size"])
    #     st.write("Data range:", sidebar["data_range"])

    st.subheader("Transformed Dataset Preview")
    st.dataframe(df_transformed.head())

    st.download_button(
        label="Download Transformed Dataset",
        data=df_transformed.to_csv(index=False),
        file_name="transformed_dataset.csv",
        mime="text/csv",
    )

def render_statistical_tab(evaluation, sidebar):
    comparison = evaluation["comparison"]
    utility_score = evaluation["utility_score"]

    #technique = sidebar["technique"]

    st.subheader("1. Statistical Properties")

    st.caption(
        """
        This section evaluates how statistical properties changed after applying
        the selected privacy-preserving technique.
        Numerical and categorical properties are shown separately when applicable.
        """
    )

    st.subheader("1.1 Numerical Properties")

    st.caption(
        """
        Comparison of numerical statistics such as mean, variance, standard deviation,
        minimum, and maximum.
        """
    )

    st.dataframe(comparison["descriptive_statistics"])

    if sidebar["technique"] in ["Generalization", "Sampling"]:

        if "categorical_statistics" in comparison:
            st.subheader("1.2 Categorical Properties")

            st.caption(
                """
                Comparison of categorical distributions before and after transformation.
                This is especially relevant for techniques that affect categorical
                attributes or the composition of the dataset.
                """
            )

            st.dataframe(comparison["categorical_statistics"])
    st.divider()

    st.subheader("2. Relationship Preservation")

    st.caption(
        """
        This section evaluates whether relationships between numerical variables were preserved after transformation.
        The table shows the difference between the transformed and original correlation matrices.
        """
    )

    st.dataframe(comparison["correlation_difference"])

    st.divider()

    st.subheader("3. Distribution Similarity")

    st.info(
        """
        JS Divergence measures similarity between distributions.
        Lower values indicate higher similarity and therefore better utility.
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
    ).head(4)

    st.dataframe(
        most_affected,
        column_config={
            "kl_divergence": None
        }
    )

    #st.subheader("Distribution Similarity Metrics")
    #st.dataframe(comparison["distribution_similarity"])


def render_visual_tab(df_prepared, df_transformed, sidebar, evaluation):
    comparison = evaluation["comparison"]
    technique = sidebar["technique"]
    
    selected_column = sidebar["selected_column"]
    x_column = sidebar["x_column"]
    y_column = sidebar["y_column"]

    
    st.metric("Selected Technique", technique)
    

    #st.subheader("1. Statistical Properties")
    st.info("**1. Statistical Properties**")

    st.caption(
        "Visual comparison of distributions to observe changes in central tendency and dispersion."
    )

    #col1, col2 = st.columns(2)

    #with col1:
    #html_title = """
    #<div style="background-color: #f0f2f6; color: #31333F; padding: 4px 12px; border-radius: 6px; margin-bottom: 15px; display: inline-block; font-weight: bold;">
    #    Histogram Comparison
    #</div>
    #"""

    #st.markdown(html_title, unsafe_allow_html=True)


    st.plotly_chart(
            plot_histogram_comparison(
                df_prepared,
                df_transformed,
                selected_column,
            ),
        use_container_width=True,
        key="visual_statistical_histogram",
    )
    with st.expander("What does 'Density' mean?"):
            st.markdown(
                """
                **Probability Density Interpretation**

                In this histogram, the Y-axis does not represent the number of observations.

                Instead, it represents *probability density*, which indicates how concentrated the data is in each region.

                - Higher bars → higher concentration of values
                - Lower bars → fewer values
                - Total area under the histogram = 1

                The probability of observing values within a given interval is approximated by:

                **probability ≈ density × bin width**                
               """
    )

    #with col2:
    #st.info("Boxplot Comparison")
    st.plotly_chart(
            plot_boxplot_comparison(
                df_prepared,
                df_transformed,
                selected_column,
            ),
            use_container_width=True,
            key="visual_statistical_boxplot",
    )

    # st.subheader("2. Relationship Preservation")

    # st.caption(
    #     "Visual inspection of relationships between variables after transformation."
    # )

    # if x_column == y_column:
    #     st.warning("Select different variables for X and Y.")
    # else:
    #     fig_original_density, fig_transformed_density = plot_density_comparison(
    #         df_prepared,
    #         df_transformed,
    #         x_column,
    #         y_column,
    #     )

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.write("Original Data Density")
    #         st.plotly_chart(
    #             fig_original_density,
    #             use_container_width=True,
    #             key="visual_original_density",
    #         )

    #     with col2:
    #         st.write("Transformed Data Density")
    #         st.plotly_chart(
    #             fig_transformed_density,
    #             use_container_width=True,
    #             key="visual_transformed_density",
    #         )
    
    st.info("**2. Relationship Preservation**")

    st.caption(
        "Visual inspection of relationships between variables after transformation."
    )

    

    st.subheader("Correlation Matrix Comparison")

    fig_original_corr, fig_transformed_corr = plot_correlation_matrix_comparison(
        comparison["original_correlation"],
        comparison["transformed_correlation"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            fig_original_corr,
            use_container_width=True,
            key="visual_original_correlation2",
        )

    with col2:
        st.plotly_chart(
            fig_transformed_corr,
            use_container_width=True,
            key="visual_transformed_correlation2",
        )

    

    st.subheader("Correlation Difference Heatmap")

    st.plotly_chart(
        plot_correlation_heatmap(
            comparison["correlation_difference"],
            "Correlation Difference (Transformed - Original)",
        ),
        use_container_width=True,
        key="visual_correlation_heatmap2",
    )

    st.info("**3. Distribution Similarity**")

    st.caption(
        "Visual comparison of distribution changes using divergence metrics."
    )

    #st.plotly_chart(
    #    plot_histogram_comparison(
    #        df_prepared,
    #        df_transformed,
    #        selected_column,
    #    ),
    #    use_container_width=True,
    #    key="distribution_similarity_histogram",
    #)

    st.subheader(
        "JS Divergence",
        help=(
            "This chart identifies which variables were most affected by the transformation. "
            "Variables with higher JS Divergence experienced greater distributional change. "
            "Values close to zero indicate strong preservation of the original distribution."
        )
    )
    st.plotly_chart(
        plot_metric_bar_chart(
            comparison["distribution_similarity"],
            "js_divergence",
        ),
        use_container_width=True,
        key="distribution_similarity_js",
    )

    

def render_tradeoff_tab(sidebar, evaluation, df_tradeoff_comparison, df_prepared,
    numeric_columns,):
    technique = sidebar["technique"]

    utility_score = evaluation["utility_score"]
    privacy_score = evaluation["privacy_score"]
    tradeoff_score = evaluation["tradeoff_score"]

    st.info(
        """
        This section evaluates the balance between privacy protection and data utility.

        Utility measures how well the original data characteristics are preserved and is estimated from distribution similarity using JS divergence.

        Privacy is estimated from the intensity of the selected transformation.

        The trade-off score combines utility and privacy to indicate the overall effectiveness of the selected configuration.
        """
    )

    st.success("**1. Current Configuration**")
    #html_title1 = """
    #<div style="background-color: #e0f2fe; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
    #  <style="color: #0369a1; margin: 0;">1. Current configuration</style=>
    #</div>
    #"""

    #st.markdown(html_title1, unsafe_allow_html=True)

    df_tradeoff = pd.DataFrame([
        {
            "Technique": technique,
            "Utility Score": utility_score,
            "Privacy Score": privacy_score,
            "Trade-off Score": tradeoff_score,
        }
    ])

    nome_da_tecnica = df_tradeoff["Technique"].iloc[0]
    st.metric(label="Selected Technique", value=nome_da_tecnica)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Utility Score", round(utility_score, 4))

    with col2:
        st.metric("Privacy Score", round(privacy_score, 4))

    with col3:
        st.metric("Trade-off Score", round(tradeoff_score, 4))

    

    st.success("**2. Current Privacy-Utility Plot**")

    #html_title2 = """
    
    #<div style="background-color: #e0f2fe; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
    #  <style="color: #0369a1; margin: 0;">2. Privacy-Utility Plot</style=>
    #</div>
    #"""
    #st.markdown(html_title2, unsafe_allow_html=True)

    fig_tradeoff = plot_tradeoff_comparison(df_tradeoff)

    st.plotly_chart(
        fig_tradeoff,
        use_container_width=True,
        key="current_tradeoff_plot",
    )

    