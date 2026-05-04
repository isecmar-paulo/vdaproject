import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_histogram_comparison(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    column: str,
    bins: int = 30
):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df_original[column],
            name="Original",
            opacity=0.5,
            nbinsx=bins,
            histnorm="probability density",  #
            marker=dict(line=dict(width=1)),
        )
    )

    fig.add_trace(
        go.Histogram(
            x=df_transformed[column],
            name="Transformed",
            opacity=0.5,
            nbinsx=bins,
            histnorm="probability density",
            marker=dict(line=dict(width=1)),
        )
    )

    # Média (referência visual)
    mean_original = df_original[column].mean()
    mean_transformed = df_transformed[column].mean()

    fig.add_vline(
        x=mean_original,
        line_dash="dash",
        annotation_text="Mean (Original)",
        annotation_position="top left",
    )

    fig.add_vline(
        x=mean_transformed,
        line_dash="dot",
        annotation_text="Mean (Transformed)",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Distribution Comparison — {column}",
        xaxis_title=column,
        yaxis_title="Density",
        barmode="overlay",
        height=450,
        bargap=0.05,
        legend=dict(
            orientation="h",
            y=1.1,
            x=0.5,
            xanchor="center"
        ),
    )

    return fig


def plot_boxplot_comparison(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    column: str
):
    """
    Create a boxplot comparing original and transformed data.
    """
    combined_df = pd.DataFrame({
        "Original": df_original[column],
        "Transformed": df_transformed[column]
    })

    long_df = combined_df.melt(
        var_name="Dataset",
        value_name=column
    )

    fig = px.box(
        long_df,
        x="Dataset",
        y=column,
        title=f"Boxplot Comparison - {column}"
    )

    return fig


def plot_scatter_comparison(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    x_column: str,
    y_column: str
):
    """
    Create side-by-side scatter plots comparing original and transformed data.
    """
    original = df_original[[x_column, y_column]].copy()
    original["Dataset"] = "Original"

    transformed = df_transformed[[x_column, y_column]].copy()
    transformed["Dataset"] = "Transformed"

    combined_df = pd.concat([original, transformed], ignore_index=True)

    fig = px.scatter(
        combined_df,
        x=x_column,
        y=y_column,
        color="Dataset",
        title=f"Scatter Plot Comparison - {x_column} vs {y_column}"
    )

    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap"
):
    """
    Create a heatmap from a correlation matrix.
    """
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title=title
    )

    return fig


def plot_metric_bar_chart(
    df_metrics: pd.DataFrame,
    metric_column: str,
    x_column: str = "column"
):
    """
    Create a bar chart for statistical comparison metrics such as KL or JS divergence.
    """
    fig = px.bar(
        df_metrics,
        x=x_column,
        y=metric_column,
        title=f"{metric_column} by Variable"
    )

    return fig

def plot_tradeoff(utility: float, privacy: float):
    """
    Plot a single point representing the trade-off.
    """
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame({
        "Privacy": [privacy],
        "Utility": [utility],
        "Label": ["Current Configuration"]
    })

    fig = px.scatter(
        df,
        x="Privacy",
        y="Utility",
        text="Label",
        title="Privacy vs Utility Trade-off"
    )

    fig.update_traces(textposition="top center")

    return fig




def plot_tradeoff_comparison(df_tradeoff):
    """
    Plot privacy vs utility for different techniques.
    """
    fig = px.scatter(
        df_tradeoff,
        x="Privacy Score",
        y="Utility Score",
        size="Trade-off Score",
        color="Technique",
        hover_data=["Trade-off Score"],
        title="Privacy-Utility Trade-off Comparison"
    )

    # Ideal region: high privacy and high utility
    fig.add_shape(
        type="rect",
        x0=0.7,
        x1=1.0,
        y0=0.7,
        y1=1.0,
        fillcolor="LightGreen",
        opacity=0.2,
        line_width=0,
    )

    fig.add_annotation(
        x=0.85,
        y=0.95,
        text="Ideal region",
        showarrow=False,
    )

    fig.update_layout(
        xaxis=dict(range=[-0.10, 1.10], title="Privacy Score"),
        yaxis=dict(range=[-0.10, 1.10], title="Utility Score"),
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig

def plot_privacy_utility_dual_bar(df_tradeoff):
    df_melt = df_tradeoff.melt(
        id_vars="Technique",
        value_vars=["Privacy Score", "Utility Score"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        df_melt,
        x="Technique",
        y="Score",
        color="Metric",
        barmode="group",
        title="Privacy vs Utility by Technique",
    )

    fig.update_layout(
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Technique",
    )

    return fig


def plot_parameter_sweep_tradeoff(df_sweep):
    """
    Plot Privacy vs Utility for all parameter configurations.
    """

    fig = px.scatter(
        df_sweep,
        x="Privacy Score",
        y="Utility Score",
        color="Technique",
        size="Trade-off Score",
        hover_data=["Parameter", "Parameter Value", "Trade-off Score"],
        title="Parameter Sweep: Privacy vs Utility",
    )

    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.05], title="Privacy Score"),
        yaxis=dict(range=[-0.05, 1.05], title="Utility Score"),
        height=500,
    )

    return fig


def plot_parameter_sweep_ranking(df_sweep):
    """
    Plot the best configurations ranked by trade-off score.
    """

    top_results = df_sweep.head(10).copy()

    top_results["Configuration"] = (
        top_results["Technique"]
        + " | "
        + top_results["Parameter"]
        + "="
        + top_results["Parameter Value"].astype(str)
    )

    fig = px.bar(
        top_results.sort_values("Trade-off Score", ascending=True),
        x="Trade-off Score",
        y="Configuration",
        orientation="h",
        title="Top Configurations by Trade-off Score",
    )

    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Trade-off Score"),
        yaxis_title="Configuration",
        height=500,
    )

    return fig


def plot_parameter_sweep_lines(df_sweep):
    """
    Plot trade-off score as a function of parameter value for each technique.
    """

    fig = px.line(
        df_sweep,
        x="Parameter Value",
        y="Trade-off Score",
        color="Technique",
        markers=True,
        title="Trade-off Score vs Parameter Value",
    )

    fig.update_layout(
        xaxis_title="Parameter Value",
        yaxis_title="Trade-off Score",
        height=500,
    )

    return fig


def plot_density_comparison(
    df_original,
    df_transformed,
    x_column,
    y_column,
):
    import plotly.express as px

    # Original
    fig_original = px.density_heatmap(
        df_original,
        x=x_column,
        y=y_column,
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale="Blues",
        title="Original Data Density",
    )

    # Transformed
    fig_transformed = px.density_heatmap(
        df_transformed,
        x=x_column,
        y=y_column,
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale="Reds",
        title="Transformed Data Density",
    )

    return fig_original, fig_transformed

def plot_correlation_matrix_comparison(
    original_corr,
    transformed_corr,
):
    """
    Create side-by-side correlation heatmaps for original and transformed data.
    """
    fig_original = px.imshow(
        original_corr,
        text_auto=True,
        aspect="auto",
        title="Original Correlation Matrix",
        zmin=-1,
        zmax=1,
    )

    fig_transformed = px.imshow(
        transformed_corr,
        text_auto=True,
        aspect="auto",
        title="Transformed Correlation Matrix",
        zmin=-1,
        zmax=1,
    )

    return fig_original, fig_transformed


def plot_correlation_scatter(original_corr, transformed_corr):
    """
    Scatter plot comparing pairwise correlations before and after transformation.
    Each point represents one pair of variables.
    """

    common_columns = original_corr.columns.intersection(transformed_corr.columns)

    original_corr = original_corr.loc[common_columns, common_columns]
    transformed_corr = transformed_corr.loc[common_columns, common_columns]

    rows = []

    for i, col_i in enumerate(common_columns):
        for j, col_j in enumerate(common_columns):
            if i < j:
                original_value = original_corr.loc[col_i, col_j]
                transformed_value = transformed_corr.loc[col_i, col_j]

                rows.append(
                    {
                        "Variable Pair": f"{col_i} - {col_j}",
                        "Original Correlation": original_value,
                        "Transformed Correlation": transformed_value,
                        "Absolute Difference": abs(transformed_value - original_value),
                    }
                )

    df_corr = pd.DataFrame(rows)

    min_value = min(
        df_corr["Original Correlation"].min(),
        df_corr["Transformed Correlation"].min(),
    )

    max_value = max(
        df_corr["Original Correlation"].max(),
        df_corr["Transformed Correlation"].max(),
    )

    padding = 0.1
    axis_min = max(-1, min_value - padding)
    axis_max = min(1, max_value + padding)

    fig = px.scatter(
        df_corr,
        x="Original Correlation",
        y="Transformed Correlation",
        size="Absolute Difference",
        color="Absolute Difference",
        hover_data=[
            "Variable Pair",
            "Original Correlation",
            "Transformed Correlation",
            "Absolute Difference",
        ],
        title="Pairwise Correlation Preservation",
        size_max=22,
    )

    fig.add_shape(
        type="line",
        x0=axis_min,
        y0=axis_min,
        x1=axis_max,
        y1=axis_max,
        line=dict(dash="dash", color="black"),
    )

    fig.update_layout(
        xaxis=dict(
            range=[axis_min, axis_max],
            title="Original Correlation",
        ),
        yaxis=dict(
            range=[axis_min, axis_max],
            title="Transformed Correlation",
        ),
        coloraxis_colorbar=dict(
            title="Abs. Difference"
        ),
        height=520,
    )

    return fig

def plot_top_correlation_changes(original_corr, transformed_corr, top_n=10):
    common_columns = original_corr.columns.intersection(transformed_corr.columns)

    original_corr = original_corr.loc[common_columns, common_columns]
    transformed_corr = transformed_corr.loc[common_columns, common_columns]

    rows = []

    for i, col_i in enumerate(common_columns):
        for j, col_j in enumerate(common_columns):
            if i < j:
                diff = abs(transformed_corr.loc[col_i, col_j] - original_corr.loc[col_i, col_j])

                rows.append({
                    "Variable Pair": f"{col_i} - {col_j}",
                    "Absolute Difference": diff,
                })

    df_changes = pd.DataFrame(rows).sort_values(
        by="Absolute Difference",
        ascending=False,
    ).head(top_n)

    fig = px.bar(
        df_changes.sort_values("Absolute Difference", ascending=True),
        x="Absolute Difference",
        y="Variable Pair",
        orientation="h",
        title="Most Affected Correlation Pairs",
    )

    fig.update_layout(
        xaxis_title="Absolute Correlation Change",
        yaxis_title="Variable Pair",
        height=500,
    )

    return fig