import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_histogram_comparison(
    df_original: pd.DataFrame,
    df_transformed: pd.DataFrame,
    column: str,
    bins: int = 30
):
    """
    Create an overlaid histogram comparing original and transformed data.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df_original[column],
            name="Original",
            opacity=0.6,
            nbinsx=bins
        )
    )

    fig.add_trace(
        go.Histogram(
            x=df_transformed[column],
            name="Transformed",
            opacity=0.6,
            nbinsx=bins
        )
    )

    fig.update_layout(
        title=f"Distribution Comparison - {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
        barmode="overlay"
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

import plotly.express as px


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

    return fig
