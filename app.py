import streamlit as st
import pandas as pd
import plotly.express as px

from src.sidebar import render_sidebar

from src.tabs import (
    render_exploratory_tab,
    render_transformation_tab,
    render_statistical_tab,
    render_visual_tab,
    render_tradeoff_tab,
)

from src.pipeline import (
    build_tradeoff_comparison, 
    compute_evaluation_results,
    apply_selected_transformation,
    load_and_prepare_data
) 

DATA_PATH = "data/medicalData.csv"

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
def configure_page():
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

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    configure_page()

    try:
        # ============================================================
        # DATA PIPELINE — carregar, inspecionar, preprocessar
        # ============================================================   

        data = load_and_prepare_data(DATA_PATH)

        df_original = data["df_original"]
        df_prepared = data["df_prepared"]
        inspection = data["inspection"]
        validation = data["validation"]
        numeric_columns = data["numeric_columns"]    
        
        sidebar = render_sidebar(df_prepared, numeric_columns)


        # ===========================================================================
        # DATA PIPELINE — Aplicar a técnica de preservação de privacidade selecionada
        # ===========================================================================


      
        df_transformed_raw, df_transformed = apply_selected_transformation(
            df_original=df_original,
            df_prepared=df_prepared,
            sidebar=sidebar,
        )

        evaluation = compute_evaluation_results(
            df_original=df_original,
            df_prepared=df_prepared,
            df_transformed_raw=df_transformed_raw,
            df_transformed=df_transformed,
            sidebar=sidebar,
        )


        df_tradeoff_comparison = build_tradeoff_comparison(
            df_prepared=df_prepared,
            numeric_columns=numeric_columns,
            sidebar=sidebar,
            evaluation=evaluation,
        )


        # ============================================================
        # OUTPUT — Tabs 
        # ============================================================

        tab_data, tab_transformation, tab_statistics, tab_visual, tab_tradeoff = st.tabs(
            [
                "Exploratory Inspection",
                "Privacy-Preserving Transformation",
                "Statistical Evaluation",
                "Visual Comparison",
                "Trade-off Analysis",
            ]
        )

        # ============================================================
        # TAB 1 — Preparação de Dados
        # ============================================================

        with tab_data:
            render_exploratory_tab(data)

        # ============================================================
        # TAB 2 — Transformação ao dataset 
        # ============================================================

        with tab_transformation:
            render_transformation_tab(
                df_prepared=df_prepared,
                df_transformed=df_transformed,
                sidebar=sidebar,
            )


        # ============================================================
        # TAB 3 — Análise Estatistica
        # ============================================================

        
        with tab_statistics:
            render_statistical_tab(
                evaluation=evaluation,
                sidebar=sidebar,
            )

        # ============================================================
        # TAB 4 — Análise Visual
        # ============================================================

        with tab_visual:
            render_visual_tab(
                df_prepared=df_prepared,
                df_transformed=df_transformed,
                sidebar=sidebar,
                evaluation=evaluation,
            )    

        # ============================================================
        # TAB 5 — Avaliação do Trade-off
        # ============================================================

        with tab_tradeoff:
            render_tradeoff_tab(
                sidebar=sidebar,
                evaluation=evaluation,
                df_tradeoff_comparison=df_tradeoff_comparison,
                 df_prepared=df_prepared,
                numeric_columns=numeric_columns,
            )


    except FileNotFoundError:
        st.error(f"Dataset not found: {DATA_PATH}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main()