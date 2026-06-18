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