from pathlib import Path
import pandas as pd
from openai import OpenAI


CHATBOT_DATA_DIR = Path("chatbot_data")

CHATBOT_FILES = {
    "factory_kpis": "02_kpi_by_factory.csv",
    "route_kpis": "03_kpi_by_factory_line_product.csv",
    "regression_coefficients": "05_ml_regression_coefficients.csv",
    "regression_metrics": "05_ml_regression_metrics.csv",
    "cluster_summary": "07_ml_cluster_summary.csv",
    "sensitivity": "09_sensitivity_simulation.csv",
    "anomaly_summary": "13_anomaly_summary.csv",
    "optimized_allocation": "14_optimized_allocation.csv",
    "optimization_summary": "15_optimization_summary.csv",
    "optimization_by_line": "16_optimization_by_line.csv",
}


def load_chatbot_tables():
    tables = {}

    for table_name, file_name in CHATBOT_FILES.items():
        path = CHATBOT_DATA_DIR / file_name

        if path.exists():
            tables[table_name] = pd.read_csv(path)

    return tables


def dataframe_to_context(name, df, max_rows=50):
    if len(df) <= max_rows:
        table_text = df.to_string(index=False)
        row_label = "Complete table"
    else:
        table_text = df.head(max_rows).to_string(index=False)
        row_label = f"First {max_rows} rows"

    return f"""
Data source: {name}
Columns: {", ".join(df.columns)}
{row_label}:
{table_text}
"""


def retrieve_relevant_context(question, tables):
    question_lower = question.lower()
    selected_contexts = []

    def add_context(table_name):
        if table_name in tables:
            selected_contexts.append(
                dataframe_to_context(table_name, tables[table_name])
            )

    # Anomalies / outliers
    if any(keyword in question_lower for keyword in [
        "anomaly", "anomalies", "outlier", "outliers", "unusual", "abnormal", "detected"
    ]):
        add_context("anomaly_summary")
        return "\n\n".join(selected_contexts)

    # Clusters
    if any(keyword in question_lower for keyword in [
        "cluster", "clusters", "group", "groups", "pattern", "patterns", "loss pattern"
    ]):
        add_context("cluster_summary")
        return "\n\n".join(selected_contexts)

    # Optimization / allocation / redistribution
    if any(keyword in question_lower for keyword in [
        "optimization", "optimisation", "recommend", "recommendation",
        "allocation", "redistribution", "redistributed", "reallocated",
        "shift", "shifting", "increase production", "decrease production"
    ]):
        add_context("optimization_summary")
        add_context("optimized_allocation")
        add_context("optimization_by_line")
        return "\n\n".join(selected_contexts)

    # Regression / drivers / model quality
    if any(keyword in question_lower for keyword in [
        "regression", "coefficient", "impact", "influence", "driver", "factor",
        "r2", "mae", "rmse", "model quality", "accuracy"
    ]):
        add_context("regression_coefficients")
        add_context("regression_metrics")
        return "\n\n".join(selected_contexts)

    # Sensitivity / what-if
    if any(keyword in question_lower for keyword in [
        "sensitivity", "simulation", "what if", "improvement potential", "potential"
    ]):
        add_context("sensitivity")
        return "\n\n".join(selected_contexts)

    # Product-line route / line / downtime / scrap / setup / cycle time
    if any(keyword in question_lower for keyword in [
        "route", "product-line", "product line", "line", "linie",
        "downtime", "scrap", "setup", "cycle", "highest oee", "lowest oee"
    ]):
        add_context("route_kpis")
        return "\n\n".join(selected_contexts)

    # Factory comparison
    if any(keyword in question_lower for keyword in [
        "factory", "werk", "plant", "better oee", "factory oee"
    ]):
        add_context("factory_kpis")
        return "\n\n".join(selected_contexts)

    # Fallback context
    add_context("factory_kpis")
    add_context("route_kpis")
    add_context("optimization_summary")

    return "\n\n".join(selected_contexts)

def generate_llm_answer(question, context):
    client = OpenAI()

    prompt = f"""
You are a production KPI assistant for a diploma thesis prototype.

Answer the user's question only based on the provided retrieved data context.
If the context does not contain enough information, say that the available data is not sufficient.
Use a clear business-oriented explanation.
Do not invent values that are not present in the context.
Do not add follow-up offers at the end of the answer.
If the context contains a complete table, do not say "previewed records".

User question:
{question}

Retrieved data context:
{context}
"""

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    return response.output_text
def generate_direct_answer(question, tables):
    q = question.lower()

    if ("highest oee" in q) and (
        "product-line" in q or "product line" in q or "prod-line" in q or "route" in q
    ):
        df = tables["route_kpis"]
        row = df.loc[df["oee"].idxmax()]

        return (
            f"The product-line route with the highest OEE is "
            f"factory {row['factory_id']}, line {row['line_id']}, product {row['product_id']}.\n\n"
            f"- OEE: {row['oee']:.4f} ({row['oee'] * 100:.2f}%)\n"
            f"- Availability: {row['availability']:.4f}\n"
            f"- Performance: {row['performance']:.4f}\n"
            f"- Quality: {row['quality']:.4f}"
        )

    if "line" in q and "highest downtime" in q:
        df = tables["route_kpis"]

        line_summary = (
            df.groupby(["factory_id", "line_id"], as_index=False)["downtime_total_min"]
            .sum()
            .sort_values("downtime_total_min", ascending=False)
        )

        row = line_summary.iloc[0]

        return (
            f"The line with the highest total downtime is "
            f"factory {row['factory_id']}, line {row['line_id']}.\n\n"
            f"- Total downtime: {row['downtime_total_min']:,.2f} minutes"
        )

    if "factory" in q and "better oee" in q:
        df = tables["factory_kpis"].copy()
        df = df.sort_values("oee", ascending=False)

        best_factory = df.iloc[0]

        answer = (
            f"Factory {best_factory['factory_id']} has the better OEE "
            f"compared to the other factory in the dataset.\n\n"
            f"Overall Equipment Effectiveness (OEE) is calculated as "
            f"Availability x Performance x Quality. "
            f"The component values are shown to make the result transparent.\n\n"
        )

        for _, row in df.iterrows():
            answer += (
                f"- Factory {row['factory_id']}: "
                f"OEE {row['oee'] * 100:.2f}%, "
                f"Availability {row['availability'] * 100:.2f}%, "
                f"Performance {row['performance'] * 100:.2f}%, "
                f"Quality {row['quality'] * 100:.2f}%\n"
            )

        return answer

    if "how many anomalies" in q or "anomalies were detected" in q:
        df = tables["anomaly_summary"]
        row = df[df["anomaly_label"] == "anomaly"].iloc[0]

        return (
            f"The anomaly detection identified {int(row['event_count']):,} anomalous production events.\n\n"
            f"- Share of all events: {row['event_share'] * 100:.2f}%\n"
            f"- Average OEE of anomalies: {row['avg_oee'] * 100:.2f}%"
        )

    if "production is redistributed" in q or "redistributed" in q:
        df = tables["optimization_summary"]
        row = df.iloc[0]

        return (
            f"The optimization redistributes {row['redistributed_units']:,.0f} production units.\n\n"
            f"- Redistributed share: {row['redistributed_share'] * 100:.2f}%\n"
            f"- Current weighted OEE: {row['current_weighted_oee'] * 100:.2f}%\n"
            f"- Optimized weighted OEE: {row['optimized_weighted_oee'] * 100:.2f}%"
        )

    return None
