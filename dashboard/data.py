# Suggested file: ./dashboard/data.py
# Data parsing and loading helpers used by the dashboard.

import ast
import json

import pandas as pd
import streamlit as st

from constants import ARCHITECTURE_PATH, DATA_PATH

# fits in ./dashboard/data.py
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


# fits in ./dashboard/data.py
def parse_roc_column(value):
    if pd.isna(value):
        return {}
    if isinstance(value, str):
        try:
            cleaned = value.replace("array(", "").replace(")", "")
            return ast.literal_eval(cleaned)
        except Exception:
            return {}
    return value


# fits in ./dashboard/data.py
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "path"], errors="ignore")

    df["fold_accs"] = df["fold_accs"].apply(parse_literal_list)
    df["fold_kappas"] = df["fold_kappas"].apply(parse_literal_list)

    for column in ["roc_fpr", "roc_tpr", "roc_auc"]:
        if column in df.columns:
            df[column] = df[column].apply(parse_roc_column)

    return df


# fits in ./dashboard/data.py
@st.cache_data
def load_architectures(path=ARCHITECTURE_PATH):
    try:
        with open(path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
