import streamlit as st

def render_sidebar(df_prepared, numeric_columns):
    
# ============================================================
# SIDEBAR — PRIVACY TRANSFORMATION SETTINGS
# ============================================================

    st.sidebar.header("Privacy Mechanism Settings")

    technique = st.sidebar.selectbox(
        "Privacy-preserving technique",
        [
            "None",
            "Generalization",
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
    generalized_attributes = []
    age_bin_size = None
    bmi_bin_size = None
    age_range = None
    bmi_range = None
    charges_bin_size = None
    charges_range = None
    children_bin_size = None
    children_range = None


    # if technique == "Generalization - Age":
    #     col = "age"
    #     data_range = df_prepared[col].max() - df_prepared[col].min()

    #     bin_size = st.sidebar.slider(
    #         "Age generalization interval",
    #         min_value=1,
    #         max_value=int(data_range),
    #         value=10,
    #         step=1,
    #         help="Larger age intervals increase privacy but reduce data precision.",
    #     )

    # elif technique == "Generalization - BMI":
    #     col = "bmi"
    #     data_range = df_prepared[col].max() - df_prepared[col].min()

    #     bin_size = st.sidebar.slider(
    #         "BMI generalization interval",
    #         min_value=0.5,
    #         max_value=float(data_range),
    #         value=2.0,
    #         step=0.5,
    #         help="Larger BMI intervals increase privacy but reduce detail.",
    #     )

    if technique == "Generalization":
        generalized_attributes = st.sidebar.multiselect(
            "Attributes to generalize",
            ["age", "bmi", "charges", "children", "region"],
            default=["age"],
            help="Select one or more attributes to generalize. " \
            "Low-cardinality discrete variables such as children "  
            "generally provide limited privacy gains through generalization."
        )

        if "age" in generalized_attributes:
            age_range = df_prepared["age"].max() - df_prepared["age"].min()

            age_bin_size = st.sidebar.slider(
                "Age generalization interval",
                min_value=1,
                max_value=int(age_range),
                value=10,
                step=1,
            )

        if "bmi" in generalized_attributes:
            bmi_range = df_prepared["bmi"].max() - df_prepared["bmi"].min()

            bmi_bin_size = st.sidebar.slider(
                "BMI generalization interval",
                min_value=0.5,
                max_value=float(bmi_range),
                value=2.0,
                step=0.5,
            )
        
        if "charges" in generalized_attributes:
            charges_range = df_prepared["charges"].max() - df_prepared["charges"].min()

            charges_bin_size = st.sidebar.slider(
                "Charges generalization interval",
                min_value=100.0,
                max_value=float(charges_range),
                value=5000.0,
                step=100.0,
                help=(
                    "Defines the interval size used to generalize medical charges. "
                    "Larger intervals increase privacy."
                ),
            )


        if "region" in generalized_attributes:
            st.sidebar.caption(
                "Region will be generalized from four categories into two broader groups: north and south."
            )
        
        if "children" in generalized_attributes:
            children_range = df_prepared["children"].max() - df_prepared["children"].min()

            children_bin_size = st.sidebar.slider(
                "Children generalization interval",
                min_value=1,
                max_value=int(children_range),
                value=2,
                step=1,
                help=(
                    "Defines the interval size used to generalize the number of children. "
                ),
            )

        

    elif technique == "Gaussian Noise":
        sigma = st.sidebar.slider(
            "Gaussian noise level (σ)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Higher σ introduces stronger perturbation.",
        )

        selected_columns = st.sidebar.multiselect(
            "Variables to perturb",
            numeric_columns,
            default=numeric_columns,
        )

    elif technique == "Laplace Noise":
        epsilon = st.sidebar.slider(
            "Privacy budget (ε)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Lower ε provides stronger privacy.",
        )

        selected_columns = st.sidebar.multiselect(
            "Variables to perturb",
            numeric_columns,
            default=numeric_columns,
        )

    elif technique == "Sampling":
        sampling_rate = st.sidebar.slider(
            "Retained data proportion",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Lower values expose fewer records and increase privacy.",
        )
# ============================================================
# SIDEBAR — VISUAL EVALUATION SETTINGS
# ============================================================

    st.sidebar.header("Visual Evaluation Settings")

    selected_column = st.sidebar.selectbox(
        "Variable for distribution plots",
        numeric_columns,
    )

    x_column = st.sidebar.selectbox(
        "Scatter plot X variable",
        numeric_columns,
        index=0,
    )

    y_column_index = 1 if len(numeric_columns) > 1 else 0

    y_column = st.sidebar.selectbox(
        "Scatter plot Y variable",
        numeric_columns,
        index=y_column_index,
    )

    bins = st.sidebar.slider(
        "Histogram bin count",
        min_value=10,
        max_value=60,
        value=20,
        step=5,
        help="Used in histograms and in JS divergence calculation.",
    )

    if technique in ["Gaussian Noise", "Laplace Noise"] and not selected_columns:
        st.warning("Please select at least one numeric variable to perturb.")
        st.stop()

    render_sidebar_footer()

    return {
        "technique": technique,
        "sigma": sigma,
        "epsilon": epsilon,
        "sampling_rate": sampling_rate,
        "selected_columns": selected_columns,
        "generalized_attributes": generalized_attributes,
        "age_bin_size": age_bin_size,
        "bmi_bin_size": bmi_bin_size,
        "charges_bin_size": charges_bin_size,
        "charges_range": charges_range,
        "age_range": age_range,
        "bmi_range": bmi_range,
        "children_bin_size": children_bin_size,
        "children_range": children_range,
        "bins": bins,
        "selected_column": selected_column,
        "x_column": x_column,
        "y_column": y_column,
    }


# ============================================================
# render_sidebar_footer: render footer section
# ============================================================

def render_sidebar_footer():
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #808495; font-size: 0.85em;">
                <p><strong>VDA Dashboard</strong><br>
                Comparative Evaluation of Privacy Techniques.
                Done by <br>
                <span style="color: #1f4e79; font-weight: 600;">
                Juliana Jesus and Paulo Silva
                </span></p>
                <p style="text-align: justify; padding-left: 1rem; padding-right: 1rem;">
                Project developed for the Visualization and Data Analysis module, 
                as part of the Transversal and Transferable Competences II course  
                in the University of Aveiro’s third-cycle program.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
