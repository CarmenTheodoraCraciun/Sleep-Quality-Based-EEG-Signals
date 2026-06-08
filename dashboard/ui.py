# Suggested file: ./dashboard/ui.py
# Streamlit rendering functions for dashboard sections and page layout.

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from charts import build_bar_chart, build_scatter_chart, style_figure
from constants import COLOR_PALETTE, EXPORT_FOLDER, HIGH_CONTRAST_PALETTE, DEFAULT_BG, DEFAULT_FONT_COLOR
from export import export_chart_images

# fits in ./dashboard/ui.py
def render_title():
    st.title("Machine Learning Systems for Sleep Quality Assessment Based on EEG Signals")
    st.markdown("## Model Performance Dashboard\n\n**Author**: Carmen-Theodora Craciun")


# fits in ./dashboard/ui.py
def render_leaderboard(df):
    best_acc_row = df.loc[df["test_acc"].idxmax()]
    best_kappa_row = df.loc[df["test_kappa"].idxmax()]

    st.subheader("Model Leaderboard")
    cols = st.columns(4)
    cols[0].metric("Top Test Accuracy", f"{best_acc_row['test_acc']:.4f}", best_acc_row["name"])
    cols[1].metric("Top Test Kappa", f"{best_kappa_row['test_kappa']:.4f}", best_kappa_row["name"])
    cols[2].metric("Total Models Evaluated", len(df))
    cols[3].metric("Model Families", len(df["type"].unique()))


# fits in ./dashboard/ui.py
def render_dataset_overview():
    st.subheader("Dataset Demographics")
    total_subjects = 372
    disordered_subjects = 175
    disordered_pct = (disordered_subjects / total_subjects) * 100
    train_count = int(total_subjects * 0.70)
    val_count = int(total_subjects * 0.15)
    test_count = total_subjects - train_count - val_count

    cols = st.columns(5)
    cols[0].metric("Total Subjects", total_subjects, "Across 3 Databases", delta_color="off")
    cols[1].metric("Clinical / Sleep Disorders", disordered_subjects, f"{disordered_pct:.1f}% of dataset", delta_color="off")
    cols[2].metric("Training Set", train_count, "70% Split", delta_color="off")
    cols[3].metric("Validation Set", val_count, "15% Split", delta_color="off")
    cols[4].metric("Holdout Test Set", test_count, "15% Split", delta_color="off")


# fits in ./dashboard/ui.py
def render_performance_section(df):
    st.header("1. Performance")
    st.subheader("A. Cross-Validation Performance (Mean Across Folds)")
    colA, colB = st.columns(2)

    with colA:
        fig = build_bar_chart(df, "name", "acc_mean", "type", "CV Mean Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0])
        st.plotly_chart(fig, width="stretch")

    with colB:
        fig = build_bar_chart(df, "name", "kappa_mean", "type", "CV Mean Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0])
        st.plotly_chart(fig, width="stretch")

    st.subheader("B. Final Holdout Test Performance")
    colA2, colB2 = st.columns(2)

    with colA2:
        fig = build_bar_chart(df, "name", "test_acc", "type", "Final Test Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0])
        st.plotly_chart(fig, width="stretch")

    with colB2:
        fig = build_bar_chart(df, "name", "test_kappa", "type", "Final Test Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0])
        st.plotly_chart(fig, width="stretch")

    return {
        "cv_acc": build_bar_chart(df, "name", "acc_mean", "type", "CV Mean Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0]),
        "cv_kappa": build_bar_chart(df, "name", "kappa_mean", "type", "CV Mean Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0]),
        "test_acc": build_bar_chart(df, "name", "test_acc", "type", "Final Test Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0]),
        "test_kappa": build_bar_chart(df, "name", "test_kappa", "type", "Final Test Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0]),
    }


# fits in ./dashboard/ui.py
def render_stability_section(df):
    st.header("2. Model Stability and Generalization")
    colC, colD = st.columns(2)

    with colC:
        melted = df[["name", "type", "acc_mean", "test_acc"]].melt(id_vars=["name", "type"], var_name="Metric", value_name="Score")
        fig = px.bar(
            melted,
            x="name",
            y="Score",
            color="Metric",
            barmode="group",
            title="Generalization: CV Mean vs. Holdout Test Accuracy",
            color_discrete_sequence=[COLOR_PALETTE[0], COLOR_PALETTE[4]],
        )
        style_figure(fig, y_range=[0.5, 1.0])
        st.plotly_chart(fig, width="stretch")

    with colD:
        fig = build_scatter_chart(
            df,
            "acc_std",
            "acc_mean",
            "type",
            "name",
            "Risk vs Reward: CV Std Dev vs. CV Mean Accuracy",
            HIGH_CONTRAST_PALETTE,
            x_reverse=True,
        )
        fig.update_layout(xaxis_title="Accuracy Std Dev (Lower is Better)")
        st.plotly_chart(fig, width="stretch")


# fits in ./dashboard/ui.py
def render_indepth_section(df):
    st.header("3. In-Depth Analysis")
    colE, colF = st.columns(2)

    with colE:
        exploded = df.explode("fold_accs").rename(columns={"fold_accs": "Fold_Accuracy"})
        exploded["Fold_Accuracy"] = exploded["Fold_Accuracy"].astype(float)
        fig = px.box(
            exploded,
            x="name",
            y="Fold_Accuracy",
            color="type",
            points="all",
            title="CV Variance: Fold Accuracy Distribution per Model",
            color_discrete_sequence=HIGH_CONTRAST_PALETTE,
        )
        style_figure(fig)
        st.plotly_chart(fig, width="stretch")

    with colF:
        df_copy = df.copy()
        df_copy["drop_off"] = df_copy["acc_best"] - df_copy["test_acc"]
        df_sorted = df_copy.sort_values("drop_off")
        fig = go.Figure(go.Bar(x=df_sorted["name"], y=df_sorted["drop_off"], marker_color=COLOR_PALETTE[3]))
        fig.update_layout(
            title="Optimism Bias: Gap Between 'Best' CV Fold and Final Test Acc",
            yaxis_title="Accuracy Drop (Best CV - Test)",
            xaxis_title="Model",
            font_color=DEFAULT_FONT_COLOR,
            plot_bgcolor=DEFAULT_BG,
            paper_bgcolor=DEFAULT_BG,
        )
        st.plotly_chart(fig, width="stretch")

    return fig


# fits in ./dashboard/ui.py
def render_model_inspection(df, architectures):
    st.header("4. Model Inspection & Architecture")
    st.markdown("Select a model to view its multi-class ROC curve and the exact Python code used to compile it.")

    available_models = df["name"].tolist()
    selected_model = st.selectbox("Choose Model to Inspect:", available_models)
    selected_row = df[df["name"] == selected_model].iloc[0]

    st.info(f"**{selected_model}**\n - Family: {selected_row['type']}\n - Final Test Acc: {selected_row['test_acc']:.3f}")
    st.subheader("Source Code")

    model_code = architectures.get(selected_model)
    if model_code:
        st.code(model_code, language="python")
    else:
        st.warning(f"Source code not found for {selected_model}.")


# fits in ./dashboard/ui.py
def render_export_section(charts):
    st.header("5. Export Images")
    st.markdown("Download high-resolution, print-ready PNGs of all dashboard charts.")

    if st.button("Generate High-Res Images"):
        with st.spinner("Saving high-resolution images... This might take a few seconds."):
            try:
                export_chart_images(charts)
                st.success(f"Success! All charts have been saved to the `{EXPORT_FOLDER}` folder in your project directory.")
            except Exception as error:
                st.error(f"Error saving images. Make sure you ran 'pip install -U kaleido' in your terminal. Error details: {error}")
