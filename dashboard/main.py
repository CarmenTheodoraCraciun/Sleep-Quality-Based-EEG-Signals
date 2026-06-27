import sys
from pathlib import Path
import pandas as pd

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "dashboard"))

from data import load_data, load_architectures
from export import collect_export_charts
from ui import (
    render_dataset_overview,
    render_export_section,
    render_indepth_section,
    render_performance_section,
    render_stability_section,
    render_title,
    render_speed_and_benchmark_sections,
    render_hypnogram_discrepancy_section,
    render_model_details_section,
)


st.set_page_config(page_title="Models Performance - Dashboard", layout="wide")

st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #0077BB;
    }
    .stPlotlyChart {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
    hr {
        margin: 1.5rem 0;
    }
    /* Stil card gri ca la Demographics */
    .metric-card {
        background-color: #f0f2f6;
        padding: 12px 8px;
        border-radius: 10px;
        text-align: center;
        height: 100%;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card .label {
        font-size: 11px;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 24px;
        font-weight: 700;
        margin: 2px 0;
        line-height: 1.2;
    }
    .metric-card .model {
        font-size: 14px;
        font-weight: 600;
        color: #333;
        margin: 2px 0;
    }
    .metric-card .sub {
        font-size: 11px;
        color: #888;
        margin-top: 2px;
    }
    .metric-card .highlight-blue {
        color: #0077BB;
    }
    .metric-card .highlight-orange {
        color: #EE7733;
    }
    .metric-card .highlight-green {
        color: #009988;
    }
    .metric-card .highlight-pink {
        color: #EE3377;
    }
    .metric-card .highlight-teal {
        color: #00b09b;
    }
    .metric-card .highlight-red {
        color: #f5576c;
    }
    .metric-card .highlight-sky {
        color: #4facfe;
    }
</style>
""", unsafe_allow_html=True)

def render_performance_overview(df):
    """Render all 7 metric cards on the same row"""
    
    fastest = df.loc[df["gen_time"].idxmin()]
    best_kappa = df.loc[df["test_kappa"].idxmax()]
    best_acc = df.loc[df["test_acc"].idxmax()]
    
    df_copy = df.copy()
    df_copy["acc_per_sec"] = df_copy["test_acc"] / df_copy["gen_time"]
    best_ratio = df_copy.loc[df_copy["acc_per_sec"].idxmax()]
    
    best_test = df.loc[df["test_acc"].idxmax()]
    best_hypno = df.loc[df["hypno_acc"].idxmax()]
    fastest_inf = df.loc[df["gen_time"].idxmin()]
    most_stable = df.loc[df["acc_std"].idxmin()]
    
    cols = st.columns(6, gap="small")
    
    # Card 1: Fastest Model
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Fastest Model</div>
            <div class="value highlight-teal">{fastest['gen_time']:.3f}s</div>
            <div class="model">{fastest['name']}</div>
            <div class="sub">{fastest['type']} · Test Acc: {fastest['test_acc']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 2: Best kappa
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Cohen's Kappa Score</div>
            <div class="value highlight-red">{best_kappa['test_kappa']:.3f}</div>
            <div class="model">{best_kappa['name']}</div>
            <div class="sub">{best_kappa['type']} · Time: {best_kappa['gen_time']:.3f}s</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 3: Best Efficiency
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Efficiency</div>
            <div class="value highlight-sky">{best_ratio['acc_per_sec']:.2f}</div>
            <div class="model">{best_ratio['name']}</div>
            <div class="sub">{best_ratio['type']} · Acc/s</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 4: Best Test Accuracy (Leaderboard)
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Test Accuracy</div>
            <div class="value highlight-blue">{best_test['test_acc']:.3f}</div>
            <div class="model">{best_test['name']}</div>
            <div class="sub">{best_test['type']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 5: Best Hypnogram
    with cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Hypnogram</div>
            <div class="value highlight-orange">{best_hypno['hypno_acc']:.3f}</div>
            <div class="model">{best_hypno['name']}</div>
            <div class="sub">{best_hypno['type']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 7: Most Stable (CV)
    with cols[5]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Most Stable (CV)</div>
            <div class="value highlight-pink">{most_stable['acc_std']:.3f}</div>
            <div class="model">{most_stable['name']}</div>
            <div class="sub">{most_stable['type']}</div>
        </div>
        """, unsafe_allow_html=True)

def create_dashboard():
    df = load_data()
    architectures = load_architectures()
    
    render_title()
    
    render_performance_overview(df)
    st.divider()
    
    render_dataset_overview()
    st.divider()
    
    charts = render_performance_section(df)
    st.divider()
    
    render_stability_section(df)
    st.divider()
    
    optimization_figure = render_indepth_section(df)
    st.divider()
    
    render_speed_and_benchmark_sections(df)
    st.divider()
    
    render_hypnogram_discrepancy_section(df)
    st.divider()
    
    render_model_details_section(df, architectures)
    st.divider()
    
    charts.update(collect_export_charts(df, optimization_figure))
    render_export_section(charts)

if __name__ == "__main__":
    create_dashboard()