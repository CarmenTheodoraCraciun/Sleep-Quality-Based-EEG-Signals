from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from charts import build_bar_chart, build_scatter_chart, style_figure
from constants import COLOR_PALETTE, EXPORT_FOLDER, HIGH_CONTRAST_PALETTE, DEFAULT_BG, DEFAULT_FONT_COLOR
from export import export_chart_images

from charts import (
    build_speed_accuracy_scatter, 
    build_hypnogram_comparison,
    build_inference_speed_chart,
)
        
def render_hypnogram_discrepancy_section(df):
    """Section 7: General vs Sequential Performance"""
    st.header("7. General Performance vs Sequential Performance (Hypnogram)")
    st.markdown("""
    Some models excel at classifying individual sleep epochs but lose coherence when 
    evaluating the continuous sequential structure of a full hypnogram.
    """)
    
    fig = build_hypnogram_comparison(df)
    st.plotly_chart(fig, width="stretch")
    
    st.markdown("""
    **Observations:**
    - **ResNet** shows a dramatic drop: 78.2% test accuracy → 50.0% hypnogram accuracy
    - **Tree-based models** (XGB, LGBM) excel on hypnogram, reaching 80-82%
    - **CNN** shows consistent performance across both metrics
    """)

def render_speed_and_benchmark_sections(df):
    """Combine Section 4 and Section 6 in two columns"""
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.header("4. Speed vs Accuracy Trade-off")
        st.markdown("""
        In real-time clinical applications, inference speed is as important as accuracy. 
        This plot helps identify the "sweet spot" models that balance both metrics.
        """)
        
        fig1 = build_speed_accuracy_scatter(df)
        st.plotly_chart(fig1, width="stretch", use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - **Sweet Spot**: Top-left quadrant (fast & accurate)
        - **LSTM_ResNet**: 0.798 accuracy in 0.386s
        - **SSN**: Good accuracy but slow (98s)
        """)
    
    with col2:
        st.header("5. Inference Speed Benchmark")
        st.markdown("""
        For deployment decisions, inference speed is critical. The red dashed line shows 
        a 10-second acceptable time budget for clinical applications.
        """)
        
        fig2 = build_inference_speed_chart(df)
        st.plotly_chart(fig2, width="stretch", use_container_width=True)
        
        st.markdown("""
        **Deployment Recommendations:**
        - **Wearable (< 1s)**: LSTM_ResNet, CNN, XGB
        - **Server (< 10s)**: Most models
        - **Batch**: SSN (98s) for max accuracy
        """)

def render_model_details_section(df, architectures):
    """Enhanced model inspection with visualizations and metrics"""
    st.header("7. Model Deep Dive")
    st.markdown("""
    Select a model to view its complete performance profile, confusion matrix, 
    hypnogram visualization.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        available_models = df["name"].tolist()
        selected_model = st.selectbox("Choose Model to Inspect:", available_models)
        selected_row = df[df["name"] == selected_model].iloc[0].to_dict()
 
        st.subheader("Performance Metrics")
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric("Test Accuracy", f"{selected_row['test_acc']:.3f}")
            st.metric("Inference Speed", f"{selected_row['gen_time']:.2f}s")
        with col_metrics2:
            st.metric("Hypnogram Accuracy", f"{selected_row['hypno_acc']:.3f}")
            st.metric("CV Stability (Std)", f"{selected_row['acc_std']:.3f}")
        
        st.subheader("Model Details")
        full_name = selected_row.get('full_name', selected_model)
        st.info(f"""
        - **Full Name**: {full_name}
        - **Family**: {selected_row['type']}
        - **Test Kappa**: {selected_row['test_kappa']:.3f}
        - **CV Mean Accuracy**: {selected_row['acc_mean']:.3f} ± {selected_row['acc_std']:.3f}
        - **CV Mean Kappa**: {selected_row['kappa_mean']:.3f} ± {selected_row['kappa_std']:.3f}
        """)
    
    with col2:
        st.subheader(f"Confusion Matrix - {selected_model}")
        cf_path = selected_row.get("cf_matrix_path")
        if cf_path and pd.notna(cf_path):
            cf_path = Path(cf_path)
            if cf_path.exists():
                st.image(str(cf_path), use_container_width=True)
            else:
                st.warning(f"Confusion matrix image not found at: {cf_path}")
        else:
            st.warning("Confusion matrix path not available")
    
    st.divider()
    st.subheader(f"Hypnogram Visualization - {selected_model}")
    hypno_path = selected_row.get("hypno_path")
    if hypno_path and pd.notna(hypno_path):
        hypno_path = Path(hypno_path)
        if hypno_path.exists():
            st.image(str(hypno_path), use_container_width=True)
        else:
            st.warning(f"Hypnogram image not found at: {hypno_path}")
    else:
        st.warning("Hypnogram path not available")

def render_performance_overview(df):
    """Render all 7 metric cards on the same row"""
    fastest = df.loc[df["gen_time"].idxmin()].to_dict()
    best_acc = df.loc[df["test_acc"].idxmax()].to_dict()
    
    df_copy = df.copy()
    df_copy["acc_per_sec"] = df_copy["test_acc"] / df_copy["gen_time"]
    best_ratio = df_copy.loc[df_copy["acc_per_sec"].idxmax()].to_dict()
    
    best_test = df.loc[df["test_acc"].idxmax()].to_dict()
    best_hypno = df.loc[df["hypno_acc"].idxmax()].to_dict()
    fastest_inf = df.loc[df["gen_time"].idxmin()].to_dict()
    most_stable = df.loc[df["acc_std"].idxmin()].to_dict()

    cols = st.columns(7, gap="small")
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Fastest Model</div>
            <div class="value highlight-teal">{fastest['gen_time']:.3f}s</div>
            <div class="model">{fastest['name']}</div>
            <div class="sub">{fastest['type']} · Test Acc: {fastest['test_acc']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Card 2: Best Accuracy
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Accuracy</div>
            <div class="value highlight-red">{best_acc['test_acc']:.3f}</div>
            <div class="model">{best_acc['name']}</div>
            <div class="sub">{best_acc['type']} · Time: {best_acc['gen_time']:.3f}s</div>
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
    
    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Test Accuracy</div>
            <div class="value highlight-blue">{best_test['test_acc']:.3f}</div>
            <div class="model">{best_test['name']}</div>
            <div class="sub">{best_test['type']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Hypnogram</div>
            <div class="value highlight-orange">{best_hypno['hypno_acc']:.3f}</div>
            <div class="model">{best_hypno['name']}</div>
            <div class="sub">{best_hypno['type']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[5]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Fastest Inference</div>
            <div class="value highlight-green">{fastest_inf['gen_time']:.3f}s</div>
            <div class="model">{fastest_inf['name']}</div>
            <div class="sub">{fastest_inf['type']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[6]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Most Stable (CV)</div>
            <div class="value highlight-pink">{most_stable['acc_std']:.3f}</div>
            <div class="model">{most_stable['name']}</div>
            <div class="sub">{most_stable['type']}</div>
        </div>
        """, unsafe_allow_html=True)    

def render_title():
    st.title("Machine Learning Systems for Sleep Quality Assessment Based on EEG Signals")
    st.markdown("## Model Performance Dashboard\n\n**Author**: Carmen-Theodora Craciun")


def render_leaderboard(df):
    best_acc_row = df.loc[df["test_acc"].idxmax()]
    best_kappa_row = df.loc[df["test_kappa"].idxmax()]

    st.subheader("Model Leaderboard")
    cols = st.columns(4)
    cols[0].metric("Top Test Accuracy", f"{best_acc_row['test_acc']:.4f}", best_acc_row["name"])
    cols[1].metric("Top Test Kappa", f"{best_kappa_row['test_kappa']:.4f}", best_kappa_row["name"])
    cols[2].metric("Total Models Evaluated", len(df))
    cols[3].metric("Model Families", len(df["type"].unique()))

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

def render_performance_section(df):
    df = df[~df["acc_mean"].isna()]
    st.header("1. Performance")
    st.subheader("A. Cross-Validation Performance (Mean Across Folds)")
    colA, colB = st.columns(2)

    with colA:
        fig = build_bar_chart(df, "name", "acc_mean", "type", "CV Mean Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0])
        st.plotly_chart(fig, width="stretch")
        st.markdown("Displays the average cross-validation accuracy across folds for each model. Higher values indicate better and more consistent predictive performance.")

    with colB:
        fig = build_bar_chart(df, "name", "kappa_mean", "type", "CV Mean Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0])
        st.plotly_chart(fig, width="stretch")
        st.markdown("Shows the mean Cohen's Kappa across CV folds per model — useful for assessing agreement beyond chance in multi-class predictions.")

    st.subheader("B. Final Holdout Test Performance")
    colA2, colB2 = st.columns(2)

    with colA2:
        fig = build_bar_chart(df, "name", "test_acc", "type", "Final Test Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0])
        st.plotly_chart(fig, width="stretch")
        st.markdown("Final holdout test accuracy for each model on the untouched test set. This reflects real-world expected performance.")

    with colB2:
        fig = build_bar_chart(df, "name", "test_kappa", "type", "Final Test Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0])
        st.plotly_chart(fig, width="stretch")
        st.markdown("Final Cohen's Kappa on the holdout test set, indicating agreement between predictions and labels beyond chance on unseen data.")

    return {
        "cv_acc": build_bar_chart(df, "name", "acc_mean", "type", "CV Mean Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0]),
        "cv_kappa": build_bar_chart(df, "name", "kappa_mean", "type", "CV Mean Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0]),
        "test_acc": build_bar_chart(df, "name", "test_acc", "type", "Final Test Accuracy by Model", HIGH_CONTRAST_PALETTE, y_range=[0.5, 1.0]),
        "test_kappa": build_bar_chart(df, "name", "test_kappa", "type", "Final Test Kappa by Model", HIGH_CONTRAST_PALETTE, y_range=[0.3, 1.0]),
    }

def render_stability_section(df):
    st.header("2. Models Stability")
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
        st.markdown("Compares cross-validation mean accuracy to final test accuracy per model to highlight generalization gaps; large differences may indicate overfitting.")

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
        st.markdown("Plots CV accuracy standard deviation versus mean to show 'risk vs reward' — models with low std and high mean are preferable.")

def render_indepth_section(df):
    st.header("3. Models Generalization")
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
        st.markdown("Boxplots of per-fold accuracies for each model. Wider distributions indicate more variability across folds, while narrow boxes show stability.")

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
        st.markdown("Shows the optimism bias: the gap between the best CV fold accuracy and the final test accuracy. Positive drops suggest over-optimistic CV estimates.")

    return fig

def render_export_section(charts):
    st.header("8. Export Images")
    st.markdown("Download high-resolution, print-ready PNGs of all dashboard charts.")

    if st.button("Export Charts"):
        with st.spinner("Saving high-resolution images... This might take a few seconds."):
            try:
                export_chart_images(charts)
                st.success(f"Success! All charts have been saved to the `{EXPORT_FOLDER}` folder in your project directory.")
            except Exception as error:
                st.error(f"Error saving images. Make sure you ran 'pip install -U kaleido' in your terminal. Error details: {error}")
