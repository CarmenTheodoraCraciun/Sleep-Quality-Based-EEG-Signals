# Chart export and figure collection helpers for use by the dashboard.

import os

import plotly.express as px

from charts import build_bar_chart, build_scatter_chart, style_figure
from constants import COLOR_PALETTE, EXPORT_FOLDER, HIGH_CONTRAST_PALETTE

# fits in ./dashboard/export.py
def collect_export_charts(df, optimization_figure):
    charts = {
        "cv_acc": None,
        "cv_kappa": None,
        "test_acc": None,
        "test_kappa": None,
        "generalization": None,
        "pareto": None,
        "fold_variance": None,
        "optimism": optimization_figure,
    }

    charts["cv_acc"] = build_bar_chart(
        df,
        "name",
        "acc_mean",
        "type",
        "CV Mean Accuracy by Model",
        HIGH_CONTRAST_PALETTE,
        y_range=[0.5, 1.0],
    )
    charts["cv_kappa"] = build_bar_chart(
        df,
        "name",
        "kappa_mean",
        "type",
        "CV Mean Kappa by Model",
        HIGH_CONTRAST_PALETTE,
        y_range=[0.3, 1.0],
    )
    charts["test_acc"] = build_bar_chart(
        df,
        "name",
        "test_acc",
        "type",
        "Final Test Accuracy by Model",
        HIGH_CONTRAST_PALETTE,
        y_range=[0.5, 1.0],
    )
    charts["test_kappa"] = build_bar_chart(
        df,
        "name",
        "test_kappa",
        "type",
        "Final Test Kappa by Model",
        HIGH_CONTRAST_PALETTE,
        y_range=[0.3, 1.0],
    )
    charts["generalization"] = px.bar(
        df[["name", "type", "acc_mean", "test_acc"]]
        .melt(id_vars=["name", "type"], var_name="Metric", value_name="Score"),
        x="name",
        y="Score",
        color="Metric",
        barmode="group",
        title="Generalization: CV Mean vs. Holdout Test Accuracy",
        color_discrete_sequence=[COLOR_PALETTE[0], COLOR_PALETTE[4]],
    )
    style_figure(charts["generalization"], y_range=[0.5, 1.0])
    charts["pareto"] = build_scatter_chart(
        df,
        "acc_std",
        "acc_mean",
        "type",
        "name",
        "Risk vs Reward: CV Std Dev vs. CV Mean Accuracy",
        HIGH_CONTRAST_PALETTE,
        x_reverse=True,
    )
    charts["fold_variance"] = px.box(
        df.explode("fold_accs").rename(columns={"fold_accs": "Fold_Accuracy"}),
        x="name",
        y="Fold_Accuracy",
        color="type",
        points="all",
        title="CV Variance: Fold Accuracy Distribution per Model",
        color_discrete_sequence=HIGH_CONTRAST_PALETTE,
    )
    style_figure(charts["fold_variance"])
    return charts


# fits in ./dashboard/export.py
def export_chart_images(charts, export_dir=EXPORT_FOLDER):
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    file_mapping = {
        "cv_acc": "0a_cv_mean_accuracy.png",
        "cv_kappa": "0b_cv_mean_kappa.png",
        "test_acc": "1_test_accuracy.png",
        "test_kappa": "2_test_kappa.png",
        "generalization": "3_generalization.png",
        "pareto": "4_pareto_front.png",
        "fold_variance": "5_fold_variance.png",
        "optimism": "6_optimism_bias.png",
    }

    for key, fig in charts.items():
        filename = file_mapping.get(key)
        if filename and fig is not None:
            fig.write_image(os.path.join(export_dir, filename), scale=4)

    return export_dir
