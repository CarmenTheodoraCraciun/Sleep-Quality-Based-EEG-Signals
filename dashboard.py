import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="Model Performance", layout="wide")

# --- BRIGHT ACADEMIC COLOR PALETTE ---
color_palette = [
    "#0077BB", # Deep Sky Blue 
    "#EE7733", # Vibrant Orange 
    "#009988", # Teal
    "#EE3377", # Magenta
    "#CCBB44", # Goldenrod
    "#33BBEE"  # Cyan
]

high_contrast_palette = [color_palette[0], color_palette[1], color_palette[2]]

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    df = pd.read_csv("./model_results.csv")
    df.drop(columns=['Unnamed: 0', 'path'], inplace=True, errors='ignore')

    # Safely parse fold arrays
    df['fold_accs'] = df['fold_accs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['fold_kappas'] = df['fold_kappas'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['fold_accs'] = df['fold_accs'].apply(lambda lst: [float(i) for i in lst] if isinstance(lst, list) else [])
    df['fold_kappas'] = df['fold_kappas'].apply(lambda lst: [float(i) for i in lst] if isinstance(lst, list) else [])
    
    # Safely parse ROC columns if they exist in the dataframe
    def parse_roc_col(val):
        if pd.isna(val): 
            return {}
        if isinstance(val, str):
            try:
                # Clean up numpy array string artifacts if they exist
                clean_val = val.replace('array(', '').replace(')', '')
                return ast.literal_eval(clean_val)
            except Exception:
                return {}
        return val

    for col in ['roc_fpr', 'roc_tpr', 'roc_auc']:
        if col in df.columns:
            df[col] = df[col].apply(parse_roc_col)
            
    return df

df = load_data()

# --- MAIN DASHBOARD ---
st.title("Machine Learning Systems for Sleep Quality Assessment Based on EEG Signals")
st.markdown("## Model Performance Dashboard\n\n**Author**: Carmen-Theodora Craciun")

# --- MODEL KPI METRICS ---
st.subheader("Model Leaderboard")
best_acc_row = df.loc[df['test_acc'].idxmax()]
best_kappa_row = df.loc[df['test_kappa'].idxmax()]

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Top Test Accuracy", f"{best_acc_row['test_acc']:.4f}", f"{best_acc_row['name']}")
col_m2.metric("Top Test Kappa", f"{best_kappa_row['test_kappa']:.4f}", f"{best_kappa_row['name']}")
col_m3.metric("Total Models Evaluated", len(df))
col_m4.metric("Model Families", len(df['type'].unique()))

# --- DATASET OVERVIEW METRICS ---
st.subheader("Dataset Demographics")
total_subjects = 372 
disordered_subjects = 175
disordered_pct = (disordered_subjects / total_subjects) * 100

train_count = int(total_subjects * 0.70)
val_count = int(total_subjects * 0.15)
test_count = total_subjects - train_count - val_count 

col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
col_d1.metric("Total Subjects", total_subjects, "Across 3 Databases", delta_color="off")
col_d2.metric("Clinical / Sleep Disorders", disordered_subjects, f"{disordered_pct:.1f}% of dataset", delta_color="off")
col_d3.metric("Training Set", train_count, "70% Split", delta_color="off")
col_d4.metric("Validation Set", val_count, "15% Split", delta_color="off")
col_d5.metric("Holdout Test Set", test_count, "15% Split", delta_color="off")

st.divider()

# --- SECTION 1: TOP-LINE METRICS ---
st.header("1. Performance")
colA, colB = st.columns(2)

st.subheader("A. Cross-Validation Performance (Mean Across Folds)")
colA1, colB1 = st.columns(2)

with colA1:
    fig_cv_acc = px.bar(df.sort_values('acc_mean', ascending=False), 
                  x='name', y='acc_mean', color='type', 
                  title="CV Mean Accuracy by Model", text_auto='.3f',
                  color_discrete_sequence=high_contrast_palette)
    fig_cv_acc.update_layout(yaxis_range=[0.5, 1.0], font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    fig_cv_acc.update_traces(textfont_color="black", insidetextfont=dict(color="black"), outsidetextfont=dict(color="black")) 
    st.plotly_chart(fig_cv_acc, use_container_width=True)

with colB1:
    fig_cv_kappa = px.bar(df.sort_values('kappa_mean', ascending=False), 
                  x='name', y='kappa_mean', color='type', 
                  title="CV Mean Kappa by Model", text_auto='.3f',
                  color_discrete_sequence=high_contrast_palette)
    fig_cv_kappa.update_layout(yaxis_range=[0.3, 1.0], font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    fig_cv_kappa.update_traces(textfont_color="black", insidetextfont=dict(color="black"), outsidetextfont=dict(color="black"))
    st.plotly_chart(fig_cv_kappa, use_container_width=True)

# Row 2: Final Holdout Test
st.subheader("B. Final Holdout Test Performance")
colA2, colB2 = st.columns(2)

with colA2:
    fig1 = px.bar(df.sort_values('test_acc', ascending=False), 
                  x='name', y='test_acc', color='type', 
                  title="Final Test Accuracy by Model", text_auto='.3f',
                  color_discrete_sequence=high_contrast_palette)
    fig1.update_layout(yaxis_range=[0.5, 1.0], font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    fig1.update_traces(textfont_color="black", insidetextfont=dict(color="black"), outsidetextfont=dict(color="black")) 
    st.plotly_chart(fig1, use_container_width=True)

with colB2:
    fig2 = px.bar(df.sort_values('test_kappa', ascending=False), 
                  x='name', y='test_kappa', color='type', 
                  title="Final Test Kappa by Model", text_auto='.3f',
                  color_discrete_sequence=high_contrast_palette)
    fig2.update_layout(yaxis_range=[0.3, 1.0], font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    fig2.update_traces(textfont_color="black", insidetextfont=dict(color="black"), outsidetextfont=dict(color="black"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- SECTION 2: STABILITY & OVERFITTING ---
st.header("2. Model Stability and Generalization")
colC, colD = st.columns(2)

with colC:
    melted_acc = df[['name', 'type', 'acc_mean', 'test_acc']].melt(id_vars=['name', 'type'], var_name='Metric', value_name='Score')
    fig4 = px.bar(melted_acc, x='name', y='Score', color='Metric', barmode='group',
                  title="Generalization: CV Mean vs. Holdout Test Accuracy",
                  color_discrete_sequence=[color_palette[0], color_palette[4]])
    fig4.update_layout(yaxis_range=[0.5, 1.0], font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig4, use_container_width=True)
    
with colD:
    fig5 = px.scatter(df, x='acc_std', y='acc_mean', color='type', text='name',
                      title="Risk vs Reward: CV Std Dev vs. CV Mean Accuracy",
                      color_discrete_sequence=high_contrast_palette)
    fig5.update_layout(xaxis_autorange="reversed", xaxis_title="Accuracy Std Dev (Lower is Better)", font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    fig5.update_traces(textposition='top center', marker=dict(size=12), textfont_color="black") 
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# --- SECTION 3: IN-DEPTH ANALYSIS ---
st.header("3. In-Depth Analysis")
colE, colF = st.columns(2)

with colE:
    exploded_df = df.explode('fold_accs').rename(columns={'fold_accs': 'Fold_Accuracy'})
    exploded_df['Fold_Accuracy'] = exploded_df['Fold_Accuracy'].astype(float)
    fig6 = px.box(exploded_df, x='name', y='Fold_Accuracy', color='type', points="all",
                  title="CV Variance: Fold Accuracy Distribution per Model",
                  color_discrete_sequence=high_contrast_palette)
    fig6.update_layout(font_color="black", plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig6, use_container_width=True)

with colF:
    df['drop_off'] = df['acc_best'] - df['test_acc']
    df_sorted = df.sort_values('drop_off')
    fig10 = go.Figure()
    fig10.add_trace(go.Bar(
        x=df_sorted['name'], 
        y=df_sorted['drop_off'],
        marker_color=color_palette[3] 
    ))
    fig10.update_layout(
        title="Optimism Bias: Gap Between 'Best' CV Fold and Final Test Acc",
        yaxis_title="Accuracy Drop (Best CV - Test)",
        xaxis_title="Model",
        font_color="black",
        plot_bgcolor='white', 
        paper_bgcolor='white'
    )
    st.plotly_chart(fig10, use_container_width=True)

st.divider()

# --- SECTION 4: INTERACTIVE MODEL INSPECTION ---
st.header("4. Model Inspection & Architecture")
st.markdown("Select a model to view its multi-class ROC curve and the exact Python code used to compile it.")

@st.cache_data
def load_architectures():
    try:
        with open("architectures.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

architectures_dict = load_architectures()

# Selectbox for the user
available_models = df['name'].tolist()
selected_model_arch = st.selectbox("Choose Model to Inspect:", available_models)
selected_row = df[df['name'] == selected_model_arch].iloc[0]

st.info(f"**{selected_model_arch}**\n - "
        f"Family: {selected_row['type']}\n - "
        f"Final Test Acc: {selected_row['test_acc']:.3f}")

# with col_code:
st.subheader(f"Source Code")
model_code = architectures_dict.get(selected_model_arch)

if model_code:
    st.code(model_code, language='python')
else:
    st.warning(f"Source code not found for {selected_model_arch}.")

# st.divider()

# --- SECTION 5: THESIS IMAGE EXPORT ---
st.header("5. Export Thesis Images")
st.markdown("Download high-resolution, print-ready PNGs of all dashboard charts.")

if st.button("Generate High-Res Images"):
    import os
    
    export_dir = "charts_export"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        
    with st.spinner("Saving high-resolution images... This might take a few seconds."):
        try:
            # Save the new CV Mean charts
            fig_cv_acc.write_image(f"{export_dir}/0a_cv_mean_accuracy.png", scale=4)
            fig_cv_kappa.write_image(f"{export_dir}/0b_cv_mean_kappa.png", scale=4)
            
            # Save your original charts
            fig1.write_image(f"{export_dir}/1_test_accuracy.png", scale=4)
            fig2.write_image(f"{export_dir}/2_test_kappa.png", scale=4)
            fig4.write_image(f"{export_dir}/3_generalization.png", scale=4)
            fig5.write_image(f"{export_dir}/4_pareto_front.png", scale=4)
            fig6.write_image(f"{export_dir}/5_fold_variance.png", scale=4)
            fig10.write_image(f"{export_dir}/6_optimism_bias.png", scale=4)
            
            st.success(f"Success! All charts have been saved to the `{export_dir}` folder in your project directory.")
        except Exception as e:
            st.error(f"Error saving images. Make sure you ran 'pip install -U kaleido' in your terminal. Error details: {e}")