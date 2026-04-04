import pandas as pd
from scipy.stats import zscore
from scipy.signal import medfilt

# Methods for dataset

def extract_basic_temporal_features(df):
    '''
    Adds temporal context (lag/lead) strictly isolated within each subject's data. This prevents 'data leakage' between different patients.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing multiple subjects.

    Returns:
    - pd.DataFrame: The DataFrame with patient-isolated temporal context.
    '''
    df_new = df.copy()
    exclude_cols = ['Subject_ID', 'Dataset_Source', 'Label']
    cols_to_shift = [c for c in df_new.columns if c not in exclude_cols]

    def shift_group(group):
        for col in cols_to_shift:
            group[f'{col}_lag2'] = group[col].shift(2)
            group[f'{col}_lag1'] = group[col].shift(1)
            group[f'{col}_lead1'] = group[col].shift(-1)
            group[f'{col}_lead2'] = group[col].shift(-2)
        group.bfill(inplace=True)
        group.ffill(inplace=True)
        return group

    return df_new.groupby('Subject_ID', group_keys=False).apply(shift_group, include_groups=False)

def normalize_by_subject(df, feature_cols):
    '''
    Apply Z-Score normalization per patient (Subject_ID).
    Brings signals from different electrodes (Fpz, C3, C4) to the same common denominator.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - feature_cols (list): List of column names

    Returns:
    - pd.DataFrame: The normalized DataFrame.
    '''
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = df_norm.groupby('Subject_ID')[col].transform(zscore)

    df_norm[feature_cols] = df_norm[feature_cols].fillna(0)
    return df_norm

def extract_enhanced_temporal_features(df):
    '''
    Extract Ratios, Lags, Leads, Diffs and Rolling Stats using ultra-fast vectorized operations, respecting per-patient isolation.

    Parameters:
      - df (pd.DataFrame): The input DataFrame.

    Returns:
      - pd.DataFrame: The enhanced DataFrame.
    '''
    df_new = df.copy()
    exclude_cols = ['Subject_ID', 'Dataset_Source', 'Label']

    # 1. Creare Ratios (Rapoarte)
    eps = 1e-6
    df_new['Ratio_DeepSleep'] = df_new['Abs_Delta'] / (df_new['Abs_Beta'] + eps)
    df_new['Ratio_Transition'] = df_new['Abs_Theta'] / (df_new['Abs_Alpha'] + eps)

    feature_cols = [c for c in df_new.columns if c not in exclude_cols]
    grouped = df_new.groupby('Subject_ID')[feature_cols]

    # 2. Extragere Lags, Leads și Diferențe vectorizate
    lag1_df = grouped.shift(1).add_suffix('_lag1')
    lag2_df = grouped.shift(2).add_suffix('_lag2')
    lead1_df = grouped.shift(-1).add_suffix('_lead1')
    lead2_df = grouped.shift(-2).add_suffix('_lead2')
    diff_df = df_new[feature_cols].sub(grouped.shift(1)).add_suffix('_diff')

    # 3. Statistici Glisante (Rolling Stats)
    roll_mean_df = grouped.transform(lambda x: x.rolling(window=5, min_periods=1, center=True).mean()).add_suffix('_roll_mean')
    roll_std_df = grouped.transform(lambda x: x.rolling(window=5, min_periods=1, center=True).std()).add_suffix('_roll_std')

    # 4. Concatenare și curățare NaN-uri
    df_final = pd.concat([df_new, lag1_df, lag2_df, lead1_df, lead2_df, diff_df, roll_mean_df, roll_std_df], axis=1)
    df_final = df_final.groupby('Subject_ID', group_keys=False).apply(lambda g: g.bfill().ffill(), include_groups=False)

    return df_final