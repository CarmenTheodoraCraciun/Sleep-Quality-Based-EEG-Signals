import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "dashboard"))

from data import load_data, load_architectures
from export import collect_export_charts
from ui import (
    render_dataset_overview,
    render_export_section,
    render_indepth_section,
    render_leaderboard,
    render_model_inspection,
    render_performance_section,
    render_stability_section,
    render_title,
)


st.set_page_config(page_title="Models Performance - Dashboard", layout="wide")


def create_dashboard():
    df = load_data()
    architectures = load_architectures()

    render_title()
    render_leaderboard(df)
    render_dataset_overview()
    st.divider()

    charts = render_performance_section(df)
    st.divider()
    render_stability_section(df)
    st.divider()
    optimization_figure = render_indepth_section(df)
    st.divider()
    render_model_inspection(df, architectures)
    st.divider()

    charts.update(collect_export_charts(df, optimization_figure))
    render_export_section(charts)


if __name__ == "__main__":
    create_dashboard()
