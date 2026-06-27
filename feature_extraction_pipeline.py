import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import skew, kurtosis, entropy, zscore

SFREQ = 100

EEG_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Sigma': (11, 16),
    'Beta': (16, 30)
}

# ============================
# 1. Artifact removal
# ============================
def remove_artifacts_raw(x, sf=100):
    cleaned = []
    for epoch in x:
        e = epoch.copy()
        e = e - np.mean(e)
        b, a = butter(2, 0.3/(sf/2), btype='high')
        e = filtfilt(b, a, e)
        thr = np.std(e) * 5
        e = np.clip(e, -thr, thr)
        cleaned.append(e)
    return np.array(cleaned)

# ============================
# 2. Spindles
# ============================
def detect_spindles_raw(x, sf=100):
    b, a = butter(4, [11/(sf/2), 16/(sf/2)], btype='band')
    spindle_events = []
    for epoch in x:
        filtered = filtfilt(b, a, epoch)
        envelope = np.abs(hilbert(filtered))
        thr = np.mean(envelope) + 2*np.std(envelope)
        above = envelope > thr
        events = []
        start = None
        for i, val in enumerate(above):
            if val and start is None:
                start = i
            if not val and start is not None:
                end = i
                duration = (end - start) / sf
                if 0.3 <= duration <= 2.0:
                    events.append((start, end))
                start = None
        spindle_events.append(events)
    return spindle_events

def spindle_features_for_epoch(events, sf=100):
    count = len(events)
    if count == 0:
        return [0, 0]
    durations = [(e[1]-e[0])/sf for e in events]
    return [count, np.mean(durations)]

# ============================
# 3. Slow waves
# ============================
def detect_slow_waves_raw(x, sf=100):
    b, a = butter(4, [0.5/(sf/2), 2/(sf/2)], btype='band')
    slow_events = []
    for epoch in x:
        filtered = filtfilt(b, a, epoch)
        zero_cross = np.where(np.diff(np.sign(filtered)))[0]
        events = []
        for i in range(len(zero_cross)-2):
            start = zero_cross[i]
            mid = zero_cross[i+1]
            end = zero_cross[i+2]
            amp = np.max(filtered[start:end]) - np.min(filtered[start:end])
            duration = (end - start) / sf
            if 0.25 <= duration <= 1.5 and amp > 50:
                events.append((start, end, amp))
        slow_events.append(events)
    return slow_events

def slow_wave_features_for_epoch(events):
    if len(events) == 0:
        return [0, 0]
    amps = [e[2] for e in events]
    return [len(events), np.mean(amps)]

# ============================
# 4. Time, PSD, Hjorth, etc.
# ============================
def extract_time_features(epoch):
    mean_val = np.mean(epoch)
    std_val = np.std(epoch)
    skew_val = skew(epoch)
    kurt_val = kurtosis(epoch)
    zero_crosses = len(np.nonzero(np.diff(epoch > 0))[0])
    return [mean_val, std_val, skew_val, kurt_val, zero_crosses]

def extract_frequency_features(freqs, psd, EEG_BANDS):
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    band_powers = []
    for fmin, fmax in EEG_BANDS.values():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_powers.append(np.sum(psd[idx]))
    total_power = np.sum(psd)
    return spec_entropy, band_powers, total_power

def extract_spectral_fine_structure(freqs, psd):
    hf_power = np.sum(psd[(freqs >= 30) & (freqs <= 45)])
    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-8))) / (np.mean(psd) + 1e-8)
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd) / np.sum(psd))
    slope = np.polyfit(freqs, psd, 1)[0]
    return [hf_power, spectral_flatness, spectral_centroid, spectral_spread, slope]

def extract_hjorth(epoch):
    dx = np.diff(epoch)
    ddx = np.diff(dx)
    var_x = np.var(epoch)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0
    complexity = mobility_dx / mobility if mobility > 0 else 0
    return [mobility, complexity]

def extract_relative_powers(band_powers, total_power):
    if total_power == 0:
        return [0] * len(band_powers)
    return [bp / total_power for bp in band_powers]

def extract_features(epoch):
    features = []
    features.extend(extract_time_features(epoch))
    features.extend(extract_hjorth(epoch))
    freqs, psd = welch(epoch, SFREQ, nperseg=SFREQ*2)
    spec_entropy, band_powers, total_power = extract_frequency_features(freqs, psd, EEG_BANDS)
    features.append(spec_entropy)
    features.extend(band_powers)
    features.append(total_power)
    features.extend(extract_relative_powers(band_powers, total_power))
    features.extend(extract_spectral_fine_structure(freqs, psd))
    return features

# ============================
# 5. Normalize & Temporal features one file
# ============================

def normalize_by_subject(df, feature_cols):
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = df_norm.groupby('Subject_ID')[col].transform(zscore)
    df_norm[feature_cols] = df_norm[feature_cols].fillna(0)
    return df_norm

def extract_enhanced_temporal_features(df):
    df_new = df.copy()
    exclude_cols = ['Subject_ID', 'Dataset_Source', 'Label']

    eps = 1e-6
    df_new['Ratio_DeepSleep'] = df_new['Abs_Delta'] / (df_new['Abs_Beta'] + eps)
    df_new['Ratio_Transition'] = df_new['Abs_Theta'] / (df_new['Abs_Alpha'] + eps)

    feature_cols = [c for c in df_new.columns if c not in exclude_cols]
    grouped = df_new.groupby('Subject_ID')[feature_cols]

    lag1_df = grouped.shift(1).add_suffix('_lag1')
    lag2_df = grouped.shift(2).add_suffix('_lag2')
    lead1_df = grouped.shift(-1).add_suffix('_lead1')
    lead2_df = grouped.shift(-2).add_suffix('_lead2')
    diff_df = df_new[feature_cols].sub(grouped.shift(1)).add_suffix('_diff')

    roll_mean_df = grouped.transform(lambda x: x.rolling(window=5, min_periods=1, center=True).mean()).add_suffix('_roll_mean')
    roll_std_df = grouped.transform(lambda x: x.rolling(window=5, min_periods=1, center=True).std()).add_suffix('_roll_std')

    df_final = pd.concat([df_new, lag1_df, lag2_df, lead1_df, lead2_df, diff_df, roll_mean_df, roll_std_df], axis=1)
    df_final = df_final.groupby('Subject_ID', group_keys=False).apply(lambda g: g.bfill().ffill(), include_groups=False)

    return df_final

    
# ============================
# 6. Process one file
# ============================
def process_single_npz(npz_path):
    data = np.load(npz_path)
    X = data["x"]      # (epochs, 3000, 1)
    y = data["y"]      # (epochs,)
    X = X.reshape(X.shape[0], -1)

    X_clean = remove_artifacts_raw(X)
    spindles = detect_spindles_raw(X_clean)
    slow_waves = detect_slow_waves_raw(X_clean)

    all_features = []
    for i in range(X_clean.shape[0]):
        feats = extract_features(X_clean[i])
        feats += spindle_features_for_epoch(spindles[i])
        feats += slow_wave_features_for_epoch(slow_waves[i])
        all_features.append(feats)

    col_names = [
        'Mean','Std','Skewness','Kurtosis','Zero_Crossings',
        'Mobility','Complexity','Spectral_Entropy',
        'Abs_Delta','Abs_Theta','Abs_Alpha','Abs_Sigma','Abs_Beta',
        'Abs_Total',
        'Rel_Delta','Rel_Theta','Rel_Alpha','Rel_Sigma','Rel_Beta',
        'Spindle_Count','Spindle_Duration',
        'SlowWave_Count','SlowWave_Amplitude',
        'HF_Power','Spectral_Flatness','Spectral_Centroid',
        'Spectral_Spread','Spectral_Slope'
    ]

    df = pd.DataFrame(all_features, columns=col_names)
    df["Label"] = y
    df["Subject_ID"] = 1
    df["Dataset_Source"] = 0
    
    exclude = ["Dataset_Source", "Label"]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    df = normalize_by_subject(df, feature_cols)

    df = extract_enhanced_temporal_features(df)
    return df
