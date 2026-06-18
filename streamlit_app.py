import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import linprog
from pathlib import Path
from chatbot_helpers import (
    load_chatbot_tables,
    retrieve_relevant_context,
    generate_llm_answer,
)


st.set_page_config(
    page_title="Production Optimization",
    layout="wide",
)


st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        color: #000000 !important;
        font-size: 20px !important;
    }

    .stApp {
        color: #000000 !important;
        font-size: 1.20rem !important;
    }

    p, div, span, label {
        color: #000000 !important;
    }

    /* Main text */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        font-size: 1.20rem !important;
        line-height: 1.60 !important;
        color: #000000 !important;
    }

    /* Main page title */
    h1,
    h1 span,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h1 span,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h1 span,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h1 span {
        font-size: 2.90rem !important;
        line-height: 1.12 !important;
        font-weight: 800 !important;
        color: #000000 !important;
        margin-bottom: 1.10rem !important;
    }

    /* Section headings */
    h2,
    h2 span,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h2 span {
        font-size: 2.00rem !important;
        line-height: 1.25 !important;
        color: #000000 !important;
    }

    h3,
    h3 span,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h3 span {
        font-size: 1.65rem !important;
        line-height: 1.30 !important;
        color: #000000 !important;
    }

    /* Sidebar / Scenario settings */
    [data-testid="stSidebar"] * {
        color: #000000 !important;
        font-size: 1.25rem !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        font-size: 1.25rem !important;
        line-height: 1.50 !important;
        color: #000000 !important;
    }

    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        font-size: 1.25rem !important;
        color: #000000 !important;
    }

    /* Sidebar sliders: spacing and larger slider numbers */
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        margin-top: 0.70rem !important;
        margin-bottom: 2.10rem !important;
    }

    /* More space between slider title and slider value/track */
    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        display: block !important;
        margin-bottom: 1.60rem !important;
    }

    /* Slider title */
    [data-testid="stSidebar"] [data-testid="stSlider"] label p,
    [data-testid="stSidebar"] [data-testid="stSlider"] label span {
        font-size: 1.25rem !important;
        line-height: 1.45 !important;
        color: #000000 !important;
    }

    /* Slider value and min/max numbers */
    [data-testid="stSidebar"] [data-testid="stSlider"] div,
    [data-testid="stSidebar"] [data-testid="stSlider"] span,
    [data-testid="stSidebar"] [data-testid="stSlider"] p {
        font-size: 1.60rem !important;
        color: #000000 !important;
    }

    /* Space above the actual slider track */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top: 1.00rem !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] * {
        color: #000000 !important;
        font-size: 1.15rem !important;
        line-height: 1.55 !important;
    }

    /* Tabs */
    [data-baseweb="tab"] p {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }

    [data-baseweb="tab"] {
        padding-left: 0.80rem !important;
        padding-right: 0.80rem !important;
    }

    /* Metrics */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        font-size: 1.25rem !important;
        color: #000000 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.65rem !important;
        color: #000000 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1.20rem !important;
        color: #000000 !important;
    }

    [data-testid="stMetricDelta"] svg {
        height: 1.05rem !important;
        width: 1.05rem !important;
    }

    /* Buttons, including example questions */
    .stButton button,
    .stDownloadButton button {
        font-size: 1.20rem !important;
        font-weight: 600 !important;
        color: #000000 !important;
        min-height: 3.30rem !important;
        padding: 0.70rem 1.15rem !important;
    }

    .stButton button p,
    .stDownloadButton button p,
    .stButton button span,
    .stDownloadButton button span {
        font-size: 1.20rem !important;
        font-weight: 600 !important;
        line-height: 1.40 !important;
        color: #000000 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        font-size: 1.20rem !important;
        color: #000000 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder {
        font-size: 1.20rem !important;
        color: #666666 !important;
        opacity: 1 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p {
        font-size: 1.20rem !important;
        line-height: 1.60 !important;
        color: #000000 !important;
    }

    /* Info boxes, warnings, success messages */
    [data-testid="stAlert"] *,
    [data-testid="stInfo"] *,
    [data-testid="stWarning"] *,
    [data-testid="stSuccess"] * {
        font-size: 1.20rem !important;
        line-height: 1.60 !important;
        color: #000000 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] *,
    details *,
    summary * {
        font-size: 1.20rem !important;
        color: #000000 !important;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] *,
    .stDataFrame *,
    .stTable * {
        font-size: 1.15rem !important;
        color: #000000 !important;
    }

    [data-testid="stDataFrame"] th,
    [data-testid="stTable"] th,
    [data-testid="stDataFrame"] td,
    [data-testid="stTable"] td {
        font-size: 1.15rem !important;
        color: #000000 !important;
    }

    /* Input widgets */
    input,
    textarea,
    [data-baseweb="select"] * {
        font-size: 1.20rem !important;
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "03_kpi_by_factory_line_product.csv"
TOLERANCE = 1e-4

ROBOT_IMAGE_CANDIDATES = [
    APP_DIR / "assets" / "iposa_robot.png",
    APP_DIR / "assets" / "iposa_robot.png.png",
]

ROBOT_IMAGE = next(
    (image_path for image_path in ROBOT_IMAGE_CANDIDATES if image_path.exists()),
    ROBOT_IMAGE_CANDIDATES[0],
)


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error(f"Required file not found: {DATA_FILE}")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "factory_id",
        "line_id",
        "product_id",
        "total_units",
        "oee",
        "downtime_rate",
        "scrap_rate",
        "setup_rate",
        "avg_cycle_time_min",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["factory_id"] = df["factory_id"].astype(str).str.strip()
    df["line_id"] = df["line_id"].astype(str).str.strip()

    numeric_columns = [
        "total_units",
        "oee",
        "downtime_rate",
        "scrap_rate",
        "setup_rate",
        "avg_cycle_time_min",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    return df


@st.cache_data
def cached_load_chatbot_tables():
    return load_chatbot_tables()


def normalize(series):
    series = series.astype(float)
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(1.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def numeric_suffix(value):
    digits = "".join(character for character in str(value) if character.isdigit())

    if digits:
        return int(digits)

    return 0


def sort_routes_for_display(df):
    sorted_df = df.copy()

    sorted_df["line_sort"] = sorted_df["line_id"].map(numeric_suffix)
    sorted_df["product_sort"] = sorted_df["product_id"].map(numeric_suffix)

    sorted_df = sorted_df.sort_values(
        by=[
            "factory_id",
            "line_sort",
            "product_sort",
        ]
    )

    sorted_df = sorted_df.drop(
        columns=[
            "line_sort",
            "product_sort",
        ]
    )

    return sorted_df


def apply_readable_plot_style(fig):
    fig.update_layout(
        font=dict(size=18, color="#000000"),
        title=dict(
            font=dict(size=30, color="#000000"),
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        legend=dict(
            font=dict(size=20, color="#000000"),
            title_font=dict(size=21, color="#000000"),
        ),
        xaxis=dict(
            title_font=dict(size=21, color="#000000"),
            tickfont=dict(size=16, color="#000000"),
        ),
        yaxis=dict(
            title_font=dict(size=21, color="#000000"),
            tickfont=dict(size=16, color="#000000"),
        ),
        margin=dict(l=30, r=30, t=90, b=90),
    )

    return fig


def optimize_allocation(input_df, max_route_change, max_line_increase, efficiency_weight):
    df = input_df.copy().reset_index(drop=True)

    df["current_units"] = df["total_units"].astype(float)
    df["avg_oee"] = df["oee"].clip(0, 1).astype(float)

    df["line_key"] = (
        df["factory_id"].astype(str)
        + "_"
        + df["line_id"].astype(str)
    )

    df["route"] = (
        df["factory_id"].astype(str)
        + " / "
        + df["line_id"].astype(str)
        + " / "
        + df["product_id"].astype(str)
    )

    efficiency_score = normalize(df["avg_oee"])
    speed_score = normalize(1 / df["avg_cycle_time_min"].astype(float))

    output_weight = 1 - efficiency_weight

    df["objective_score"] = (
        efficiency_weight * efficiency_score
        + output_weight * speed_score
    )

    number_of_routes = len(df)
    objective = -df["objective_score"].to_numpy()

    products = sorted(df["product_id"].unique())

    equality_matrix = np.zeros((len(products), number_of_routes))
    product_demand = np.zeros(len(products))

    for row_number, product in enumerate(products):
        product_mask = df["product_id"] == product
        equality_matrix[row_number, np.where(product_mask)[0]] = 1
        product_demand[row_number] = df.loc[product_mask, "current_units"].sum()

    lines = sorted(df["line_key"].unique())

    capacity_matrix = np.zeros((len(lines), number_of_routes))
    line_capacity = np.zeros(len(lines))

    for row_number, line in enumerate(lines):
        line_mask = df["line_key"] == line
        capacity_matrix[row_number, np.where(line_mask)[0]] = 1

        current_line_units = df.loc[line_mask, "current_units"].sum()
        line_capacity[row_number] = current_line_units * (1 + max_line_increase)

    allocation_bounds = []

    for current_units in df["current_units"]:
        minimum_units = current_units * (1 - max_route_change)
        maximum_units = current_units * (1 + max_route_change)
        allocation_bounds.append((minimum_units, maximum_units))

    result = linprog(
        c=objective,
        A_ub=capacity_matrix,
        b_ub=line_capacity,
        A_eq=equality_matrix,
        b_eq=product_demand,
        bounds=allocation_bounds,
        method="highs",
    )

    if not result.success:
        return None, None, result.message

    df["scenario_units"] = result.x
    df["change_units"] = df["scenario_units"] - df["current_units"]

    df["change_percent"] = np.where(
        df["current_units"] > 0,
        df["change_units"] / df["current_units"],
        0,
    )

    current_total_units = df["current_units"].sum()
    scenario_total_units = df["scenario_units"].sum()

    current_weighted_oee = (
        df["current_units"] * df["avg_oee"]
    ).sum() / current_total_units

    scenario_weighted_oee = (
        df["scenario_units"] * df["avg_oee"]
    ).sum() / scenario_total_units

    current_avg_cycle_time = (
        df["current_units"] * df["avg_cycle_time_min"]
    ).sum() / current_total_units

    scenario_avg_cycle_time = (
        df["scenario_units"] * df["avg_cycle_time_min"]
    ).sum() / scenario_total_units

    theoretical_output_potential = (
        current_avg_cycle_time / scenario_avg_cycle_time
    ) - 1

    potential_additional_units = (
        current_total_units * theoretical_output_potential
    )

    estimated_process_time_saving_min = (
        current_total_units
        * (current_avg_cycle_time - scenario_avg_cycle_time)
    )

    redistributed_units = df.loc[
        df["change_units"] > 0,
        "change_units",
    ].sum()

    result_info = {
        "optimization_success": True,
        "current_weighted_oee": current_weighted_oee,
        "scenario_weighted_oee": scenario_weighted_oee,
        "current_units": current_total_units,
        "scenario_units": scenario_total_units,
        "redistributed_units": redistributed_units,
        "current_avg_cycle_time": current_avg_cycle_time,
        "scenario_avg_cycle_time": scenario_avg_cycle_time,
        "theoretical_output_potential": theoretical_output_potential,
        "potential_additional_units": potential_additional_units,
        "estimated_process_time_saving_min": estimated_process_time_saving_min,
    }

    return df, result_info, "Optimization successful"


def validate_model(result_df, max_route_change, max_line_increase):
    validation_rows = []

    total_current = result_df["current_units"].sum()
    total_scenario = result_df["scenario_units"].sum()
    total_difference = abs(total_current - total_scenario)

    validation_rows.append({
        "Check": "Total product demand remains unchanged",
        "Result": "Passed" if total_difference <= TOLERANCE else "Failed",
        "Value": f"{total_difference:,.6f}",
    })

    product_check = (
        result_df
        .groupby("product_id", as_index=False)
        .agg(
            current_units=("current_units", "sum"),
            scenario_units=("scenario_units", "sum"),
        )
    )

    product_check["difference"] = (
        product_check["current_units"] - product_check["scenario_units"]
    ).abs()

    max_product_difference = product_check["difference"].max()

    validation_rows.append({
        "Check": "Demand per product remains unchanged",
        "Result": "Passed" if max_product_difference <= TOLERANCE else "Failed",
        "Value": f"{max_product_difference:,.6f}",
    })

    max_route_change_actual = result_df["change_percent"].abs().max()

    validation_rows.append({
        "Check": "Route reallocation limit is respected",
        "Result": "Passed" if max_route_change_actual <= max_route_change + TOLERANCE else "Failed",
        "Value": f"{max_route_change_actual:.2%}",
    })

    line_check = (
        result_df
        .groupby("line_key", as_index=False)
        .agg(
            current_units=("current_units", "sum"),
            scenario_units=("scenario_units", "sum"),
        )
    )

    line_check["capacity_limit"] = (
        line_check["current_units"] * (1 + max_line_increase)
    )

    line_check["excess"] = (
        line_check["scenario_units"] - line_check["capacity_limit"]
    )

    max_line_excess = max(0, line_check["excess"].max())

    validation_rows.append({
        "Check": "Line capacity limit is respected",
        "Result": "Passed" if max_line_excess <= TOLERANCE else "Failed",
        "Value": f"{max_line_excess:,.6f}",
    })

    min_scenario_units = result_df["scenario_units"].min()

    validation_rows.append({
        "Check": "No negative production quantities",
        "Result": "Passed" if min_scenario_units >= -TOLERANCE else "Failed",
        "Value": f"{min_scenario_units:,.6f}",
    })

    validation_df = pd.DataFrame(validation_rows)

    return validation_df, product_check, line_check


def format_display_table(df):
    display = df.copy()

    percent_columns = [
        "avg_oee",
        "change_percent",
        "downtime_rate",
        "scrap_rate",
        "setup_rate",
    ]

    unit_columns = [
        "current_units",
        "scenario_units",
        "change_units",
    ]

    decimal_columns = [
        "avg_cycle_time_min",
    ]

    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda x: f"{x:.2%}")

    for column in unit_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda x: f"{x:,.0f}")

    for column in decimal_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda x: f"{x:.4f}")

    return display


kpi = load_data()

st.markdown(
    """
    <h1 style="
        font-size: 2.90rem;
        line-height: 1.12;
        font-weight: 800;
        color: #000000;
        margin-bottom: 1.10rem;
    ">
        Interactive Production Optimization Scenario Analysis
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This prototype simulates production reallocation between factory-line-product routes.
    Demand remains unchanged. The goal is to improve weighted OEE and, depending on the selected focus,
    reduce average cycle time.
    """
)

st.sidebar.header("Scenario settings")

with st.sidebar.form("scenario_form"):
    products = ["All products"] + sorted(kpi["product_id"].unique())

    selected_product = st.selectbox(
        "Select product",
        products,
    )

    max_route_change_percent = st.slider(
        "Maximum reallocation per product-line route (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
    )

    max_line_increase_percent = st.slider(
        "Maximum line capacity increase (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
    )

    optimization_goal = st.slider(
        "Optimization focus: output potential ↔ efficiency",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
    )

    st.caption(
        "0 = shorter cycle time / theoretical output potential, "
        "100 = higher OEE"
    )

    st.form_submit_button("Calculate scenario")


max_route_change = max_route_change_percent / 100
max_line_increase = max_line_increase_percent / 100
efficiency_weight = optimization_goal / 100

if selected_product == "All products":
    data = kpi.copy()
else:
    data = kpi.loc[kpi["product_id"] == selected_product].copy()


result_df, result_info, optimization_message = optimize_allocation(
    data,
    max_route_change,
    max_line_increase,
    efficiency_weight,
)

if result_df is None:
    st.error(f"Optimization failed: {optimization_message}")
    st.stop()


validation_df, product_check, line_check = validate_model(
    result_df,
    max_route_change,
    max_line_increase,
)

oee_change_pp = (
    result_info["scenario_weighted_oee"]
    - result_info["current_weighted_oee"]
) * 100

cycle_time_change_percent = (
    result_info["scenario_avg_cycle_time"]
    / result_info["current_avg_cycle_time"]
    - 1
) * 100

display_df = result_df[
    [
        "factory_id",
        "line_id",
        "product_id",
        "route",
        "avg_oee",
        "current_units",
        "scenario_units",
        "change_units",
        "change_percent",
        "downtime_rate",
        "scrap_rate",
        "setup_rate",
        "avg_cycle_time_min",
        "line_key",
    ]
].copy()

display_df = sort_routes_for_display(display_df)

route_order = display_df["route"].tolist()

tab_chatbot, tab_overview, tab_allocation, tab_charts, tab_validation = st.tabs(
    ["Chatbot", "Overview", "Allocation", "Charts", "Validation"]
)


with tab_chatbot:
    col_robot, col_text = st.columns([1, 4])

    with col_robot:
        if ROBOT_IMAGE.exists():
            st.image(
                str(ROBOT_IMAGE),
                width=130,
            )
        else:
            st.warning(f"Robot image not found: {ROBOT_IMAGE}")

    with col_text:
        st.header("IPOSA Chatbot")

        st.write(
            "IPOSA is an AI assistant for production analysis. "
            "It answers natural-language questions about KPIs, losses, anomalies and optimization results."
        )

    tables = cached_load_chatbot_tables()

    example_questions = [
        "Which factory has the better OEE?",
        "Which line has the highest downtime?",
        "Which product-line route has the highest OEE?",
        "What does the optimization recommend?",
        "How much production is redistributed?",
        "Which loss factor has the biggest improvement potential?",
        "What do the clusters mean?",
        "How many anomalies were detected?",
    ]

    st.subheader("Example questions")

    cols = st.columns(2)

    for i, question_text in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question_text, key=f"chatbot_example_{i}"):
                st.session_state["chatbot_selected_question"] = question_text

    question = st.chat_input("Ask IPOSA about the production data")

    if "chatbot_selected_question" in st.session_state:
        question = st.session_state.pop("chatbot_selected_question")

    if question:
        with st.chat_message("user"):
            st.write(question)

        context = retrieve_relevant_context(question, tables)

        with st.chat_message("assistant"):
            with st.spinner("IPOSA is analyzing the retrieved data..."):
                try:
                    answer = generate_llm_answer(question, context)
                    st.write(answer)
                except Exception as error:
                    st.error(
                        "The LLM answer could not be generated. "
                        "Please check the API key, billing and internet connection."
                    )
                    st.exception(error)

            with st.expander("Show retrieved context"):
                st.text(context)
    else:
        st.info("Select an example question or ask IPOSA directly.")


with tab_overview:
    st.subheader("Scenario overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current weighted OEE",
        f"{result_info['current_weighted_oee']:.2%}",
    )

    col2.metric(
        "Scenario weighted OEE",
        f"{result_info['scenario_weighted_oee']:.2%}",
        f"{oee_change_pp:.2f} pp",
    )

    col3.metric(
        "Redistributed units",
        f"{result_info['redistributed_units']:,.0f}",
        f"{result_info['redistributed_units'] / result_info['current_units']:.2%}",
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Total units",
        f"{result_info['current_units']:,.0f}",
        "demand unchanged",
    )

    col5.metric(
        "Scenario avg. cycle time",
        f"{result_info['scenario_avg_cycle_time']:.4f} min",
        f"{cycle_time_change_percent:.2f}%",
        delta_color="inverse",
    )

    col6.metric(
        "Theoretical output potential",
        f"{result_info['theoretical_output_potential']:.2%}",
    )

    st.subheader("Business potential")

    col7, col8 = st.columns(2)

    col7.metric(
        "Potential additional units",
        f"{result_info['potential_additional_units']:,.0f}",
    )

    col8.metric(
        "Estimated process time saving",
        f"{result_info['estimated_process_time_saving_min']:,.0f} min",
    )

    st.info(
        f"Current scenario settings: product = {selected_product}, "
        f"maximum reallocation = {max_route_change_percent}%, "
        f"maximum line capacity increase = {max_line_increase_percent}%, "
        f"optimization focus = {optimization_goal}%."
    )

    with st.expander("Model assumptions"):
        st.write(
            """
            - Product demand remains unchanged.
            - Only existing factory-line-product routes are used.
            - No new demand or new production lines are created.
            - Reallocation and line capacity are limited by the sidebar settings.
            - Output potential is based on shorter average cycle time.
            - No exact cost calculation is performed.
            """
        )

    st.subheader("Scenario interpretation")

    st.write(
        f"""
        Weighted OEE changes from **{result_info['current_weighted_oee']:.2%}**
        to **{result_info['scenario_weighted_oee']:.2%}**
        (**{oee_change_pp:.2f} percentage points**).

        **{result_info['redistributed_units']:,.0f} units** are reallocated.
        Average cycle time changes from **{result_info['current_avg_cycle_time']:.4f} min**
        to **{result_info['scenario_avg_cycle_time']:.4f} min**.

        The theoretical output potential is **{result_info['theoretical_output_potential']:.2%}**,
        equal to about **{result_info['potential_additional_units']:,.0f} additional units**
        or **{result_info['estimated_process_time_saving_min']:,.0f} saved process minutes**.

        This is an operational potential, not a cost calculation. The business decision depends on
        whether additional volume can be sold or whether the potential should be used for time savings.
        """
    )

    st.subheader("Top recommended changes")

    increases = display_df[display_df["change_units"] > 0].nlargest(
        5,
        "change_units",
    )

    decreases = display_df[display_df["change_units"] < 0].nsmallest(
        5,
        "change_units",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Increase production on:**")
        st.dataframe(
            format_display_table(
                increases[
                    [
                        "factory_id",
                        "line_id",
                        "product_id",
                        "avg_oee",
                        "avg_cycle_time_min",
                        "change_units",
                        "change_percent",
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
            row_height=38,
        )

    with col_b:
        st.markdown("**Decrease production on:**")
        st.dataframe(
            format_display_table(
                decreases[
                    [
                        "factory_id",
                        "line_id",
                        "product_id",
                        "avg_oee",
                        "avg_cycle_time_min",
                        "change_units",
                        "change_percent",
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
            row_height=38,
        )


with tab_allocation:
    st.subheader("Scenario allocation bubble view")

    bubble_df = display_df.copy()

    if selected_product == "All products":
        product_for_bubble = st.selectbox(
            "Display product in bubble chart",
            sorted(bubble_df["product_id"].unique()),
        )

        bubble_df = bubble_df.loc[
            bubble_df["product_id"] == product_for_bubble
        ].copy()
    else:
        product_for_bubble = selected_product

    bubble_df["absolute_change_units"] = bubble_df["change_units"].abs()

    maximum_bubble_size = bubble_df["absolute_change_units"].max()

    if maximum_bubble_size == 0:
        bubble_df["bubble_size"] = 1
    else:
        bubble_df["bubble_size"] = np.where(
            bubble_df["absolute_change_units"] > 0,
            bubble_df["absolute_change_units"],
            maximum_bubble_size * 0.04,
        )

    bubble_df["change_direction"] = np.select(
        [
            bubble_df["change_units"] > TOLERANCE,
            bubble_df["change_units"] < -TOLERANCE,
        ],
        [
            "Increase",
            "Decrease",
        ],
        default="No relevant change",
    )

    fig_bubble = px.scatter(
        bubble_df,
        x="current_units",
        y="scenario_units",
        size="bubble_size",
        color="change_direction",
        color_discrete_map={
            "Increase": "#1f77b4",
            "Decrease": "#d62728",
            "No relevant change": "#8c8c8c",
        },
        hover_name="route",
        hover_data={
            "factory_id": True,
            "line_id": True,
            "product_id": True,
            "current_units": ":,.0f",
            "scenario_units": ":,.0f",
            "change_units": ":,.0f",
            "change_percent": ":.2%",
            "bubble_size": False,
        },
        labels={
            "current_units": "Current units",
            "scenario_units": "Scenario units",
            "change_direction": "Change direction",
        },
        title=f"Scenario allocation changes for {product_for_bubble}",
    )

    fig_bubble.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_bubble = apply_readable_plot_style(fig_bubble)

    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown(
        "<div style='height: 60px;'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Current vs scenario allocation")

    st.dataframe(
        format_display_table(
            display_df.drop(columns=["line_key"])
        ),
        use_container_width=True,
        hide_index=True,
        row_height=38,
    )

    st.caption(
        "Positive values increase production on a route; negative values reduce it."
    )

    export_df = display_df.drop(columns=["line_key"]).copy()

    export_df.insert(0, "scenario_product", selected_product)
    export_df.insert(1, "max_reallocation_percent", max_route_change_percent)
    export_df.insert(2, "max_line_capacity_increase_percent", max_line_increase_percent)
    export_df.insert(3, "optimization_focus_percent", optimization_goal)
    export_df.insert(4, "current_weighted_oee", result_info["current_weighted_oee"])
    export_df.insert(5, "scenario_weighted_oee", result_info["scenario_weighted_oee"])
    export_df.insert(6, "theoretical_output_potential", result_info["theoretical_output_potential"])
    export_df.insert(7, "potential_additional_units", result_info["potential_additional_units"])
    export_df.insert(8, "estimated_process_time_saving_min", result_info["estimated_process_time_saving_min"])

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download scenario result as CSV",
        data=csv_data,
        file_name="scenario_allocation_result.csv",
        mime="text/csv",
    )


with tab_charts:
    fig_change = px.bar(
        display_df,
        x="route",
        y="change_units",
        color="factory_id",
    )

    fig_change.update_layout(
        title_text="Production change by route",
        xaxis_tickangle=-45,
        xaxis_title="Route",
        yaxis_title="Change in units",
        legend_title_text="Factory",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_change.update_xaxes(
        categoryorder="array",
        categoryarray=route_order,
    )

    fig_change = apply_readable_plot_style(fig_change)

    st.plotly_chart(fig_change, use_container_width=True)

    st.markdown(
        "<div style='height: 80px;'></div>",
        unsafe_allow_html=True,
    )

    fig_units = px.bar(
        display_df,
        x="route",
        y=["current_units", "scenario_units"],
        barmode="group",
    )

    fig_units.update_layout(
        title_text="Current vs scenario units by route",
        xaxis_tickangle=-45,
        xaxis_title="Route",
        yaxis_title="Units",
        legend_title_text="Allocation",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_units.update_xaxes(
        categoryorder="array",
        categoryarray=route_order,
    )

    fig_units = apply_readable_plot_style(fig_units)

    st.plotly_chart(fig_units, use_container_width=True)

    st.markdown(
        "<div style='height: 80px;'></div>",
        unsafe_allow_html=True,
    )

    fig_oee = px.bar(
        display_df,
        x="route",
        y="avg_oee",
        color="factory_id",
        text="avg_oee",
    )

    fig_oee.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside",
    )

    fig_oee.update_yaxes(
        tickformat=".0%",
        title="Average OEE",
    )

    fig_oee.update_layout(
        title_text="Average OEE by route",
        xaxis_tickangle=-45,
        xaxis_title="Route",
        yaxis_title="Average OEE",
        legend_title_text="Factory",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig_oee.update_xaxes(
        categoryorder="array",
        categoryarray=route_order,
    )

    fig_oee = apply_readable_plot_style(fig_oee)

    st.plotly_chart(fig_oee, use_container_width=True)


with tab_validation:
    st.subheader("Model validation")

    st.write(
        """
        The scenario is checked against demand, route, capacity and non-negativity constraints.
        """
    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
        row_height=38,
    )

    if (validation_df["Result"] == "Passed").all():
        st.success("All validation checks passed.")
    else:
        st.warning("At least one validation check failed. Review the constraints.")

    st.subheader("Product demand check")

    product_check_display = product_check.copy()
    product_check_display["current_units"] = product_check_display["current_units"].map(lambda x: f"{x:,.0f}")
    product_check_display["scenario_units"] = product_check_display["scenario_units"].map(lambda x: f"{x:,.0f}")
    product_check_display["difference"] = product_check_display["difference"].map(lambda x: f"{x:,.6f}")

    st.dataframe(
        product_check_display,
        use_container_width=True,
        hide_index=True,
        row_height=38,
    )

    st.subheader("Line capacity check")

    line_check_display = line_check.copy()

    for column in ["current_units", "scenario_units", "capacity_limit", "excess"]:
        line_check_display[column] = line_check_display[column].map(lambda x: f"{x:,.6f}")

    st.dataframe(
        line_check_display,
        use_container_width=True,
        hide_index=True,
        row_height=38,
    )
