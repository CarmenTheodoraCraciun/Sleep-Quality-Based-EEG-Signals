from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from constants import DEFAULT_BG, DEFAULT_FONT_COLOR, PERFORMANCE_COLORS, COLOR_PALETTE

MODEL_FULL_NAMES = {
    "LGBM": "LightGBM",
    "XGB": "XGBoost",
    "RF": "Random Forest",
    "LDA": "Linear Discriminant Analysis",
    "LR": "Logistic Regression",
    "MLP": "Multi-Layer Perceptron",
    "LSTM": "Long-Short Term-Memory",
    "ResNet": "Residual Neural Network",
    "CNN": "Convolution Neural Network",
    "SSN": "SeqSleepNet",
    "LSTM_ResNet": "LSTM + ResNet"
}

def get_full_name(short_name):
    """Get full name from short name"""
    return MODEL_FULL_NAMES.get(short_name, short_name)

def style_figure(fig, y_range=None, x_reverse=False):
    if y_range is not None:
        fig.update_layout(yaxis_range=y_range)
    if x_reverse:
        fig.update_layout(xaxis_autorange="reversed")
    fig.update_layout(
        font_color=DEFAULT_FONT_COLOR,
        plot_bgcolor=DEFAULT_BG,
        paper_bgcolor=DEFAULT_BG,
        font=dict(size=12)
    )
    return fig

def build_bar_chart(df, x, y, color, title, color_sequence, y_range=None):
    filtered_df = df.loc[df[y].notna() & (df[y] != 0)].sort_values(y, ascending=False)
    fig = px.bar(filtered_df, x=x, y=y, color=color, title=title, 
                 text_auto=".3f", color_discrete_sequence=color_sequence)
    return style_figure(fig, y_range=y_range)

def build_scatter_chart(df, x, y, color, text, title, color_sequence, x_reverse=False):
    fig = px.scatter(df, x=x, y=y, color=color, text=text, title=title, 
                     color_discrete_sequence=color_sequence)
    fig.update_traces(textposition="top center", marker=dict(size=12))
    return style_figure(fig, x_reverse=x_reverse)

def build_speed_accuracy_scatter(df):
    """Scatter plot: Speed vs Accuracy trade-off"""
    fig = px.scatter(df, 
                     x="gen_time", 
                     y="test_acc",
                     color="type",
                     text="name",
                     hover_data=["hypno_acc", "test_kappa"],
                     title="Speed vs Accuracy: Inference Time vs Test Accuracy",
                     color_discrete_map=PERFORMANCE_COLORS,
                     labels={"gen_time": "Generation Time (s)", 
                            "test_acc": "Test Accuracy"})
    
    fig.update_traces(textposition="top center", marker=dict(size=15))
    fig.update_layout(
        xaxis_type="log" if df["gen_time"].max() / df["gen_time"].min() > 10 else None,
        hovermode="closest"
    )
    return style_figure(fig)

def build_hypnogram_comparison(df):
    """Comparison chart: Test Accuracy vs Hypnogram Accuracy"""
    df_sorted = df.sort_values("hypno_acc", ascending=True)
    
    fig = go.Figure()
    
    # Add Test Accuracy bars
    fig.add_trace(go.Bar(
        name="Test Accuracy",
        x=df_sorted["name"],
        y=df_sorted["test_acc"],
        marker_color=COLOR_PALETTE[0],
        text=df_sorted["test_acc"].round(3),
        textposition="outside"
    ))
    
    # Add Hypnogram Accuracy bars
    fig.add_trace(go.Bar(
        name="Hypnogram Accuracy",
        x=df_sorted["name"],
        y=df_sorted["hypno_acc"],
        marker_color=COLOR_PALETTE[2],
        text=df_sorted["hypno_acc"].round(3),
        textposition="outside"
    ))
    
    fig.update_layout(
        title="General Performance vs Sequential Performance (Hypnogram)",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        barmode="group",
        yaxis_range=[0.4, 0.85],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return style_figure(fig)

def build_model_weight_chart(df=None):
    """Pie chart showing the ensemble weights assigned to each model."""
    weights_path = Path(__file__).resolve().parent / "results" / "ensemble_weights.csv"

    if not weights_path.exists():
        fig = go.Figure()
        fig.update_layout(title="Ensemble Model Weights")
        return style_figure(fig)

    weights_df = pd.read_csv(weights_path)
    if "weight" not in weights_df.columns or "model_name" not in weights_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Ensemble Model Weights")
        return style_figure(fig)

    weights_df = weights_df.dropna(subset=["weight", "model_name"]).copy()
    weights_df["weight"] = pd.to_numeric(weights_df["weight"], errors="coerce")
    weights_df = weights_df.dropna(subset=["weight"]).sort_values("weight", ascending=False)

    if df is not None and "name" in df.columns:
        available_names = {str(name) for name in df["name"].dropna().astype(str)}
        weights_df = weights_df[weights_df["model_name"].astype(str).isin(available_names)].copy()

    if weights_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ensemble Model Weights")
        return style_figure(fig)

    fig = px.pie(
        weights_df,
        names=weights_df["model_name"].astype(str),
        values=weights_df["weight"],
        title="Ensemble Model Weights",
        hole=0.35,
        color_discrete_sequence=COLOR_PALETTE,
    )
    fig.update_traces(textinfo="percent+label", textposition="inside", textfont=dict(size=11))
    fig.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20), showlegend=True)
    return style_figure(fig)

def build_inference_speed_chart(df):
    """Horizontal bar chart for inference speed - compact version"""
    df_sorted = df.sort_values("gen_time", ascending=True)
    
    colors = []
    for _, row in df_sorted.iterrows():
        if row["gen_time"] < 1:
            colors.append(COLOR_PALETTE[2])
        elif row["gen_time"] < 5:
            colors.append(COLOR_PALETTE[0])
        elif row["gen_time"] < 10:
            colors.append(COLOR_PALETTE[1])
        else:
            colors.append(COLOR_PALETTE[5])
    
    fig = go.Figure(go.Bar(
        y=df_sorted["name"],
        x=df_sorted["gen_time"],
        orientation="h",
        marker_color=colors,
        text=df_sorted["gen_time"].round(2),
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Time: %{x:.2f}s<br>Test Acc: %{customdata[0]:.3f}<br>Hypno Acc: %{customdata[1]:.3f}<extra></extra>",
        customdata=np.column_stack([df_sorted["test_acc"], df_sorted["hypno_acc"]])
    ))
    
    fig.add_vline(
        x=10, 
        line_dash="dash", 
        line_color=COLOR_PALETTE[5],
        line_width=2,
        annotation_text="⏱️ 10s Budget",
        annotation_position="top right",
        annotation_font_size=10
    )
    
    fig.update_layout(
        title=dict(
            text="Inference Speed Benchmark (Fastest to Slowest)",
            font=dict(size=16)
        ),
        xaxis_title="Generation Time (seconds)",
        yaxis_title="",
        height=400, 
        showlegend=False,
        xaxis=dict(
            type="log",
            gridcolor="lightgray",
            tickformat=".0f",
            range=[-0.5, 2.5],
            dtick=1,
            tickvals=[0.1, 1, 10, 100],
            ticktext=["0.1", "1", "10", "100"]
        ),
        yaxis=dict(
            gridcolor="lightgray",
            categoryorder="total ascending",
            tickfont=dict(size=11)
        ),
        margin=dict(l=100, r=80, t=50, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black")
    )
    
    return fig

def build_performance_card_metrics(df):
    """Extract key metrics for the leaderboard cards"""
    metrics = {}
    
    # Best test accuracy
    best_test = df.loc[df["test_acc"].idxmax()]
    metrics["best_test_acc"] = {
        "model": best_test["name"],
        "value": best_test["test_acc"],
        "type": best_test["type"]
    }
    
    # Best hypnogram accuracy
    best_hypno = df.loc[df["hypno_acc"].idxmax()]
    metrics["best_hypno_acc"] = {
        "model": best_hypno["name"],
        "value": best_hypno["hypno_acc"],
        "type": best_hypno["type"]
    }
    
    # Fastest model
    fastest = df.loc[df["gen_time"].idxmin()]
    metrics["fastest_model"] = {
        "model": fastest["name"],
        "value": fastest["gen_time"],
        "type": fastest["type"]
    }
    
    # Most stable model (lowest CV std)
    most_stable = df.loc[df["acc_std"].idxmin()]
    metrics["most_stable"] = {
        "model": most_stable["name"],
        "value": most_stable["acc_std"],
        "type": most_stable["type"]
    }
    
    return metrics

def style_figure(fig, y_range=None, x_reverse=False):
    if y_range is not None:
        fig.update_layout(yaxis_range=y_range)
    if x_reverse:
        fig.update_layout(xaxis_autorange="reversed")
    fig.update_layout(font_color=DEFAULT_FONT_COLOR, plot_bgcolor=DEFAULT_BG, paper_bgcolor=DEFAULT_BG)
    return fig


def build_bar_chart(df, x, y, color, title, color_sequence, y_range=None):
    filtered_df = df.loc[df[y].notna() & (df[y] != 0)].sort_values(y, ascending=False)
    fig = px.bar(filtered_df, x=x, y=y, color=color, title=title, text_auto=".3f", color_discrete_sequence=color_sequence)
    return style_figure(fig, y_range=y_range)


def build_scatter_chart(df, x, y, color, text, title, color_sequence, x_reverse=False):
    fig = px.scatter(df, x=x, y=y, color=color, text=text, title=title, color_discrete_sequence=color_sequence)
    fig.update_traces(textposition="top center", marker=dict(size=12))
    return style_figure(fig, x_reverse=x_reverse)
