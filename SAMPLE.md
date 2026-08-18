# Smart Aromatherapy System: Feature Reference & Sample Input Batches

This guide explains all **34 Heart Rate Variability (HRV) physiological metrics** and **environmental/safety parameters** used by the system. It also provides **3 complete, ready-to-use sample batches** corresponding to different emotional and stress states.

---

## 📖 Part 1: Physiological & Environmental Metrics Explained

### 1. Time-Domain HRV Metrics
These metrics reflect beat-to-beat (R-R interval) variation over time, measuring autonomic nervous system balance:

| Feature Name | Full Name & Description | Normal Range | Stress Response |
| :--- | :--- | :--- | :--- |
| **`MEAN_RR`** | **Mean R-R Interval** (ms): Average time between consecutive heartbeats. | 600 – 1200 ms | Decreases during stress (heart beats faster). |
| **`MEDIAN_RR`** | **Median R-R Interval** (ms): Median duration between R-peaks. | 600 – 1200 ms | Decreases during stress. |
| **`SDRR`** | **Standard Deviation of R-R intervals** (ms): Overall autonomic nervous system variability. | 40 – 150 ms | Drops with sympathetic dominance. |
| **`RMSSD`** | **Root Mean Square of Successive Differences** (ms): Primary indicator of parasympathetic (vagal) calming tone. | 25 – 60 ms | **Significantly drops (< 20 ms)** during acute stress. |
| **`SDSD`** | **Standard Deviation of Successive Differences** (ms): Beat-to-beat variability of intervals. | 20 – 55 ms | Drops under stress. |
| **`SDRR_RMSSD`** | **Ratio of SDRR to RMSSD**: Balance between total and short-term variance. | 1.5 – 6.0 | Rises with high stress or sudden interruptions. |
| **`HR`** | **Heart Rate** (beats per minute): Instantaneous pulse rate. | 60 – 85 bpm | **Rises (> 85–110 bpm)** during stress and anxiety. |
| **`pNN25`** | **Percentage of successive R-R intervals differing by > 25 ms** (%). | 10 – 40% | Drops as heart rhythm stiffens under pressure. |
| **`pNN50`** | **Percentage of successive R-R intervals differing by > 50 ms** (%). | 5 – 25% | Approaches ~0% under acute stress. |

---

### 2. Poincaré & Non-Linear Complexity Metrics
Geometric and entropy measures of heart rhythm dynamics:

| Feature Name | Full Name & Description | Normal Range | Stress Response |
| :--- | :--- | :--- | :--- |
| **`SD1`** | **Poincaré Short-term Dispersion** (ms): Instantaneous beat-to-beat vagal modulation. | 15 – 45 ms | Decreases under stress. |
| **`SD2`** | **Poincaré Long-term Dispersion** (ms): Continuous autonomic regulation. | 60 – 180 ms | Changes based on acute vs sustained pressure. |
| **`sampen`** | **Sample Entropy**: Measures complexity and regularity of the heartbeat time series. | 1.5 – 2.5 | Higher irregularity under disruption/stress. |
| **`higuci`** | **Higuchi Fractal Dimension**: Complexity and self-similarity of physiological waveforms. | 1.1 – 1.4 | Shifts with physiological fatigue or arousal. |

---

### 3. Frequency-Domain (Spectral) Metrics
Frequency analysis separates sympathetic (fight-or-flight) and parasympathetic (rest-and-digest) influences:

| Feature Name | Full Name & Description | Normal Range | Stress Response |
| :--- | :--- | :--- | :--- |
| **`VLF` / `VLF_PCT`** | **Very Low Frequency Power** (0.0033 – 0.04 Hz): Thermoregulatory and hormonal influences. | 500 – 3000 ms² | Modulates during prolonged stress/fatigue. |
| **`LF` / `LF_PCT`** | **Low Frequency Power** (0.04 – 0.15 Hz): Sympathetic & parasympathetic combined activity. | 300 – 1500 ms² | Elevated during cognitive workload. |
| **`LF_NU`** | **Low Frequency Normalized Units**: Relative sympathetic engagement. | 40 – 70 nu | **Rises (> 80–95 nu)** under acute time pressure. |
| **`HF` / `HF_PCT`** | **High Frequency Power** (0.15 – 0.40 Hz): Pure parasympathetic / respiratory vagal tone. | 100 – 1000 ms² | **Suppressed** during anxiety and high stress. |
| **`HF_NU`** | **High Frequency Normalized Units**: Relative calming parasympathetic power. | 30 – 60 nu | Drops significantly (< 10 nu) during acute stress. |
| **`TP`** | **Total Power** (ms²): Total energy across all spectral frequency bands. | 1000 – 5000 ms² | Reflects overall autonomic regulatory capacity. |
| **`LF_HF`** | **Sympathovagal Ratio** ($LF / HF$): Balance between sympathetic and parasympathetic states. | 1.0 – 2.5 | **Surges (> 5.0–50.0+)** in acute fight-or-flight stress. |
| **`HF_LF`** | **Reciprocal Ratio** ($HF / LF$): Parasympathetic dominance ratio. | 0.4 – 1.0 | Drops near zero during time pressure. |

---

### 4. Relative R-R & Statistical Distribution Metrics

| Feature Name | Description |
| :--- | :--- |
| **`MEAN_REL_RR`, `MEDIAN_REL_RR`** | Relative difference between consecutive R-R intervals normalized by interval length. |
| **`SDRR_REL_RR`, `RMSSD_REL_RR`, `SDSD_REL_RR`** | Relative standard deviation and root-mean-square of relative intervals. |
| **`SDRR_RMSSD_REL_RR`** | Ratio of relative standard deviation to relative RMSSD. |
| **`KURT`, `KURT_REL_RR`** | Kurtosis of the R-R interval distribution (tailedness / outlier frequency). |
| **`SKEW`, `SKEW_REL_RR`** | Skewness of the R-R interval distribution (asymmetry in cardiac intervals). |

---

### 5. Environmental & Safety Context

| Parameter | Unit | Description & Operational Thresholds |
| :--- | :---: | :--- |
| **`Ambient Temperature`** | °C | Room temperature. Baseline is 23.0°C. If > 28°C, spray duration is reduced due to high evaporation. |
| **`Ambient Humidity`** | % | Relative humidity. Baseline is 50%. If > 70%, compressor airflow is increased to assist aerosol dispersion. |
| **`Smart Band BLE RSSI`** | dBm | Signal strength. Threshold is **$\ge -75\text{ dBm}$** (within 3 meters). If $< -75\text{ dBm}$, dispensing is **blocked** for safety. |

---

## 🧪 Part 2: Three Ready-to-Use Sample Batches

The values below can be input directly into **`python main.py`** (Interactive User Input Mode) for validation, testing, and benchmarking.

---

### 🔴 Batch 1: High Acute Stress / Time Pressure
*Profile: Sympathetic fight-or-flight spike, rapid heart rate, suppressed parasympathetic vagal tone ($RMSSD=19.3$, $LF/HF=59.1$).*

```text
MEAN_RR: 843.54
MEDIAN_RR: 844.41
SDRR: 58.50
RMSSD: 19.30
SDSD: 19.30
SDRR_RMSSD: 3.03
HR: 88.50
pNN25: 21.00
pNN50: 0.20
SD1: 13.65
SD2: 81.60
KURT: -0.45
SKEW: -0.14
MEAN_REL_RR: 0.000061
MEDIAN_REL_RR: -0.001543
SDRR_REL_RR: 0.022969
RMSSD_REL_RR: 0.011970
SDSD_REL_RR: 0.011970
SDRR_RMSSD_REL_RR: 1.92
KURT_REL_RR: -0.45
SKEW_REL_RR: -0.14
VLF: 765.52
VLF_PCT: 32.45
LF: 1566.87
LF_PCT: 66.42
LF_NU: 98.34
HF: 26.50
HF_PCT: 1.12
HF_NU: 1.66
TP: 2358.88
LF_HF: 59.13
HF_LF: 0.0169
sampen: 2.22
higuci: 1.25

Environmental & Safety Context:
  Ambient Temperature in °C: 24.5
  Ambient Humidity in %: 52.0
  Smart Band BLE RSSI in dBm: -58.0
```

#### 🎯 Expected Output for Batch 1:
- **Predicted State**: `time pressure` (Confidence: 99.0%)
- **Mood Severity Score**: `0.86 / 1.00` (High Stress)
- **Action**: `NEUTRALIZE`
- **Dispensed Fragrance**: **Slot 1: French Lavender (80%) + Slot 2: Roman Chamomile (20%)**
- **Dispense Volume**: $175.0\ \mu\text{L}$ ($11.67\text{ s}$ spray @ $49.8\text{ kPa}$)
- **Status**: `SUCCESS (Dispensed)`

---

### 🟡 Batch 2: Cognitive Interruption / Mild Agitation
*Profile: Moderate cognitive disruption, elevated standard deviation variance ($SDRR=143.9$, $SDRR/RMSSD=10.78$), moderate sympathetic arousal.*

```text
MEAN_RR: 756.71
MEDIAN_RR: 747.94
SDRR: 143.97
RMSSD: 13.36
SDSD: 13.36
SDRR_RMSSD: 10.78
HR: 82.09
pNN25: 5.93
pNN50: 0.67
SD1: 9.45
SD2: 203.38
KURT: 1.13
SKEW: 0.77
MEAN_REL_RR: 0.000310
MEDIAN_REL_RR: -0.000170
SDRR_REL_RR: 0.019649
RMSSD_REL_RR: 0.011689
SDSD_REL_RR: 0.011689
SDRR_RMSSD_REL_RR: 1.68
KURT_REL_RR: 1.13
SKEW_REL_RR: 0.77
VLF: 4750.62
VLF_PCT: 89.47
LF: 524.20
LF_PCT: 9.87
LF_NU: 93.71
HF: 35.20
HF_PCT: 0.66
HF_NU: 6.29
TP: 5310.03
LF_HF: 14.89
HF_LF: 0.0671
sampen: 1.91
higuci: 1.13

Environmental & Safety Context:
  Ambient Temperature in °C: 23.0
  Ambient Humidity in %: 55.0
  Smart Band BLE RSSI in dBm: -62.0
```

#### 🎯 Expected Output for Batch 2:
- **Predicted State**: `interruption` (Confidence: 99.0%)
- **Mood Severity Score**: `0.68 / 1.00` (Moderate Agitation)
- **Action**: `NEUTRALIZE`
- **Dispensed Fragrance**: **Slot 2: Roman Chamomile (70%) + Slot 3: Bergamot (30%)**
- **Dispense Volume**: $103.0\ \mu\text{L}$ ($6.72\text{ s}$ spray @ $43.1\text{ kPa}$)
- **Status**: `SUCCESS (Dispensed)`

---

### 🟢 Batch 3: Relaxed / Homeostatic Baseline Calm
*Profile: Low resting heart rate ($HR=64$), high parasympathetic vagal recovery ($HF=66.6$), balanced sympathovagal ratio ($LF/HF \le 2.0$).*

```text
MEAN_RR: 920.50
MEDIAN_RR: 915.20
SDRR: 74.72
RMSSD: 48.60
SDSD: 48.50
SDRR_RMSSD: 1.54
HR: 64.20
pNN25: 32.50
pNN50: 18.00
SD1: 34.30
SD2: 98.40
KURT: 1.26
SKEW: -0.70
MEAN_REL_RR: 0.000081
MEDIAN_REL_RR: -0.000951
SDRR_REL_RR: 0.017605
RMSSD_REL_RR: 0.011208
SDSD_REL_RR: 0.011208
SDRR_RMSSD_REL_RR: 1.57
KURT_REL_RR: 1.26
SKEW_REL_RR: -0.70
VLF: 1016.07
VLF_PCT: 59.82
LF: 615.91
LF_PCT: 36.26
LF_NU: 54.20
HF: 245.60
HF_PCT: 24.50
HF_NU: 45.80
TP: 1698.61
LF_HF: 1.42
HF_LF: 0.7042
sampen: 2.10
higuci: 1.24

Environmental & Safety Context:
  Ambient Temperature in °C: 22.0
  Ambient Humidity in %: 48.0
  Smart Band BLE RSSI in dBm: -55.0
```

#### 🎯 Expected Output for Batch 3:
- **Predicted State**: `no stress` (Confidence: 100.0%)
- **Mood Severity Score**: `0.15 / 1.00` (Relaxed Baseline)
- **Action**: `MAINTAIN`
- **Dispensed Fragrance**: **Slot 6: Sandalwood & Frankincense (100%)**
- **Dispense Volume**: $39.5\ \mu\text{L}$ ($2.86\text{ s}$ spray @ $30.4\text{ kPa}$)
- **Status**: `SUCCESS (Dispensed)`

---

---

## ⚡ Quick Test Shortcuts in CLI

### Shortcut A: Live Hardware & Pulse Stream (Option 1)
To test with live streaming heartbeats or real hardware (ESP32/MAX30102 on USB):
1. Run `python main.py`
2. Select **`1`** (**Connect Live Wearable Sensor**)
3. Choose **`1`** for physical USB Serial or **`2`** for real-time live pulse stream demonstration.

### Shortcut B: Preset Scenarios (Option 4)
For testing predefined physiological scenarios without manual parameter entry:
1. Run `python main.py`
2. Select **`4`** (**Interactive User Input**)
3. Select **`1`** (Preset Scenarios) $\to$ choose **`A`** (High Stress), **`B`** (Interruption), or **`C`** (Calm).
The system automatically loads all 34 verified feature values from the test set for that exact emotional condition and executes the full closed-loop pipeline!
