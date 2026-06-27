import ast
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from constants import ARCHITECTURE_PATH, DATA_PATH, BASE_DIR

def parse_literal_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except Exception:
            return []
    return value if isinstance(value, list) else []

def fix_paths(df, base_dir=BASE_DIR):
    """Fix relative paths to be relative to the dashboard directory"""
    results_dir = base_dir / "results"
    
    if "cf_matrix_path" in df.columns:
        df["cf_matrix_path"] = df["cf_matrix_path"].apply(
            lambda x: str(results_dir / Path(x).name) if pd.notna(x) and x else None
        )
    
    if "hypno_path" in df.columns:
        df["hypno_path"] = df["hypno_path"].apply(
            lambda x: str(results_dir / Path(x).name) if pd.notna(x) and x else None
        )
    
    return df

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    
    df = df.drop(columns=["Unnamed: 0", "path"], errors="ignore")

    df["fold_accs"] = df["fold_accs"].apply(parse_literal_list)
    df["fold_kappas"] = df["fold_kappas"].apply(parse_literal_list)
    
    df = fix_paths(df)
    
    return df

@st.cache_data
def load_architectures(path=ARCHITECTURE_PATH):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}