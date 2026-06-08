# Feature Extraction

##  1. Time‑Domain Features

**Mean**
$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$

**Standard Deviation**
$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$

**Skewness**
$skew(x) = E\left[\left(\frac{x - \mu}{\sigma}\right)^3\right]$

**Kurt**
$kurt(x) = E\left[\left(\frac{x - \mu}{\sigma}\right)^4\right]$ 
  
---

##  2. Hjorth Parameters

**Mobility**

$mbt = \sqrt{\frac{\mathrm{Var}(x')}{\mathrm{Var}(x)}$

**Complexity**

$cmt = \frac{mbt(x')}{mbt(x)}$

---

##  3. Frequency‑Domain Features

**Spectral Entropy**

$P_i = \frac{PSD_i}{\sum PSD_i}$

$H = -\sum_i P_i \log_2(P_i)$

**Absolute Band Power**

$P_{band} = \sum_{f_i \in [f_{\min}, f_{\max}]} PSD(f_i)$

**Total Power**

$P_{\text{total}} = \sum_i PSD(f_i)$

---

##  4. Relative Band Power
$RP_{band} = \frac{P_{band}}{P_{total}}$

---

##  5. Spectral Fine Structure

**High‑Frequency Power**

$HF = \sum_{30 \le f_i \le 45} PSD(f_i)$

**Spectral Centroid**
$C = \frac{\sum_i f_i \cdot PSD_i}{\sum_i PSD_i}$

**Spectral Spread**
$SS = \sqrt{\frac{\sum_i (f_i - C)^2 \cdot PSD_i}{\sum_i PSD_i}}$

---

##  6. Spindle Features

**Threshold**
$tsh= \mu_{env} + 2\sigma_{env}$

**Duration**
$d= \frac{end - start}{f_s}$

**Count**
$c= n_{\text{events}}$

**Mean**
$mean_{d} = \frac{1}{n}\sum d_i$

---

##  7. Slow‑Wave Features

**Amplitude**
$A = \max(x_{i}) - \min(x_{i})$

**Duration**
$d = \frac{end - start}{f_s}$

**Mean Amplitude**
$mean_{a} = \frac{1}{n}\sum A_i$

## 8. Z-Score Normalization per Subject

$z = \frac{x - \mu_{\text{subj}}}{\sigma_{\text{subj}}}$

## 9. Temporal Features

% === Temporal Features ===

**Ratios**

$Ratio_{DeepSleep} = \frac{Abs\_\Delta}{Abs\_\Beta + \epsilon}$

$Ratio_{Transition} = \frac{Abs\_\Theta}{Abs\_\Alpha + \epsilon}$

**Lag / Lead**

$x_{lag1}(t) = x(t-1), \qquad
x_{lead1}(t) = x(t+1)$

**Differences**

$x_{diff}(t) = x(t) - x(t-1)$

**Rolling Mean / Std**

$R_{mean}(t) = \frac{1}{W}\sum_{i=t-k}^{t+k} x(i)$

$R_{sd}(t) = \sqrt{\frac{1}{W}\sum_{i=t-k}^{t+k}(x(i)-RM(t))^2}$



---

## Variables

- $x_i$ — EEG signal value at sample *i*
- $N$ — number of samples
- $mean, \mu$ — mean of the signal
- $sd, \sigma$ — standard deviation
- $\mathrm{Var}$ — variance/ deviation
- $x'$ — first derivative
- $PSD_i$ — power spectral density at frequency \(f_i\)
- $P_i$ — normalized spectral probability  