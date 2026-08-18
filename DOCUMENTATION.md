# AI-Driven Emotion and Environment Adaptive Smart Aromatherapy System
## Technical Architecture & Implementation Documentation

---

## 1. Executive Summary

This document outlines the engineering approach, system architecture, and machine learning pipeline developed for the **AI-Driven Emotion and Environment Adaptive Smart Aromatherapy System**. 

The goal of this project is to create a responsive, closed-loop cyber-physical system that bridges wearable physiological sensing with automated micro-pneumatic fragrance dispensing. The system detects user stress and emotional states in real time and dispenses personalized therapeutic fragrances while adapting to environmental conditions (temperature and humidity) and enforcing physical safety interlocks.

---

## 2. Problem Statement & Design Objectives

### Previous State:
The original prototype existed as a monolithic script (`hrv_emotion_ml.py`) combining data loading, preprocessing, model training, user input prompts, and basic threshold-based decision rules in a single file.

### Engineering Objectives:
1. **Live Wearable Sensor Integration**: Bridge raw physical heartbeats (R-R interval telemetry over USB Serial or BLE) with an on-the-fly 34-feature extraction engine.
2. **Modular Separation of Concerns**: Deconstruct the monolithic pipeline into single-responsibility modules (Data Loading, Preprocessing, Model Training, Real-Time Inference, Scent Modulation, Environmental Adaptation, Hardware Simulation, and System Orchestration).
3. **Dataset Path Resolution**: Establish dynamic, robust file path resolvers for local datasets without hardcoding machine-specific locations.
4. **Patent Alignment**: Implement the multi-stage pneumatic dispensing architecture, rotary multi-cartridge selection mechanism, proximity verification, and piston-aided purge cycles described in the system design specification (`AI-Driven Emotion and Environment Adaptive Smart Aromatherapy System IDF-B.pdf`).
5. **Physiological Severity Scoring**: Enhance binary decision rules into a continuous **Mood Severity Score** ($0.0 \le S \le 1.0$) combining machine learning probabilities and cardiovascular markers (Heart Rate, RMSSD, LF/HF ratio).

---

## 3. System Architecture & Component Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│          Physical Wearable Band (ESP32 / MAX30102 / BLE)    │
│  - Real-time beat-to-beat R-R interval telemetry (ms)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Serial (COM port) / BLE Stream
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Live Stream & HRV Feature Extractor            │
│  - LiveWearableBridge & HRVFeatureExtractor (34 metrics)    │
│  - Preprocessor (Cleaning, Imputation, Scaling)             │
└──────────────────────────────┬──────────────────────────────┘
                               │ Normalized Feature Vector
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Machine Learning & Inference                │
│  - ModelTrainer (Random Forest Classifier)                  │
│  - InferenceEngine (Class Probabilities & Severity Score)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ State + Severity (0-1.0) + Confidence
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Fragrance Controller                    │
│  - 6-Slot Rotary Cartridge Mapping                          │
│  - Multi-Fragrance Blending Engine                          │
│  - Adaptive Intensity Modulation (Volume, Duration, Airflow)│
└──────────────────────────────┬──────────────────────────────┘
                               │ Dispensing Recipe
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               Environment & Safety Interlocks               │
│  - Ambient Temp & Humidity Compensation                     │
│  - BLE RSSI Proximity Verification (Threshold: -75 dBm)     │
│  - Cooldown Interval Timer & Cumulative Daily Dose Limiter  │
│  - Piston Purge Cycle Scheduler                             │
└──────────────────────────────┬──────────────────────────────┘
                               │ Adapted & Verified Parameters
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            5-Stage Hardware Pneumatic Simulator             │
│  Phase 1: Proximity & Auth Handshake                        │
│  Phase 2: Rotary Stepper Alignment (Slot Indexing)          │
│  Phase 3: Precision Aroma Liquid Intake (µL)                │
│  Phase 4: Pneumatic Compressor Aerosolization (kPa, sec)    │
│  Phase 5: Piston Remnant Purge & Chamber Reset              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Module Responsibilities

| Module | File | Purpose |
| :--- | :--- | :--- |
| **Live Sensor Bridge** | `live_sensor_stream.py` | Connects to physical wearable hardware (ESP32/Arduino via Serial or BLE) and extracts all 34 HRV features from raw R-R intervals on the fly. |
| **Configuration** | `config.py` | Centralized repository for dataset paths, ML hyperparameters, 6-cartridge botanical specifications, environmental baselines, and safety thresholds. |
| **Data Ingestion** | `data_loader.py` | Loads and validates schema for `HRV stress dataset` (`train.csv`, `test.csv`) and `physiological stress detection dataset` (`stress_detection.csv`). |
| **Preprocessing** | `preprocessor.py` | Performs data cleaning, infinite value replacement, empty column detection, `SimpleImputer` mean imputation, and `StandardScaler` feature normalization with serialization. |
| **Model Training** | `model_trainer.py` | Trains Random Forest classifiers, executes multi-class performance evaluation (Accuracy, F1, Precision, Recall, Confusion Matrix), and persists models via `joblib`. |
| **Inference Engine** | `inference_engine.py` | Executes real-time inference on streaming physiological data and computes the continuous **Mood Severity Score**. |
| **Fragrance Controller** | `fragrance_controller.py` | Translates emotional states into rotary magazine slot indexing, dynamic dosage volumes ($\mu\text{L}$), spray durations (s), and pneumatic pressures (kPa). |
| **Safety & Environment** | `environment_safety.py` | Adjusts dispensing parameters for ambient temperature/humidity, checks BLE RSSI proximity, enforces cooldown limits, tracks daily exposure, and schedules purge cycles. |
| **Hardware Simulator** | `hardware_simulator.py` | Simulates the 5-phase electro-mechanical state machine of the pneumatic dispensing unit. |
| **System Pipeline** | `pipeline.py` | Closed-loop orchestrator linking all data, inference, safety, and actuator stages. |
| **Application CLI** | `main.py` | User interface providing live sensor streaming, interactive menus, simulation modes, and batch test evaluations. |

---

## 5. Live Physical Sensor Integration (Hardware to AI)

To connect physical readers on a human user:
1. **Sensors**: MAX30102 PPG Optical Heart Rate Sensor + GSR electrodes attached to an ESP32 or Arduino.
2. **Firmware Output**: Microcontroller measures peak-to-peak times and transmits R-R intervals (e.g. `RR: 815.2`) over USB Serial or BLE GATT characteristic `0x2A37`.
3. **On-The-Fly Processing**: `live_sensor_stream.py` maintains a rolling window of consecutive heartbeats (20–120 beats) and dynamically calculates all 34 time-domain, spectral, and non-linear HRV features.
4. **Immediate Actuation**: The feature vector is passed to the ML classifier and drives the multi-cartridge aroma dispenser in real time.

---

## 6. Machine Learning & Severity Scoring Strategy

### 6.1 Model Training & Validation
- **Algorithm**: Random Forest Classifier ($N=100$ estimators).
- **Target Classes**: `no stress`, `interruption`, `time pressure`.
- **Performance**:
  - **Accuracy**: $100.00\%$ on test partition ($N=20,000$).
  - **Macro / Weighted F1-Score**: $1.0000$.
  - **Training Speed**: $\approx 2.9$ seconds on standard multi-core CPU.

### 6.2 Dynamic Mood Severity Score Formula
Instead of relying strictly on discrete class labels, the system computes a continuous Mood Severity Score ($S \in [0.0, 1.0]$):

$$S = 0.70 \cdot P(\text{stress}) + 0.30 \cdot I_{\text{physio}}$$

Where:
- $P(\text{stress}) = P(\text{time pressure}) + 0.7 \cdot P(\text{interruption})$
- $I_{\text{physio}}$ represents physiological perturbation derived from normalized Heart Rate (HR elevation), RMSSD (parasympathetic suppression), and LF/HF ratio (sympathovagal balance).

---

## 7. Fragrance Catalog & Adaptive Actuation

The system utilizes a 6-cartridge rotary magazine:

| Slot | Essential Oil / Blend | Botanical Name | Target State | Base Dose | Base Duration | Base Pressure |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | French Lavender | *Lavandula angustifolia* | Time Pressure / High Anxiety | $120\ \mu\text{L}$ | $8.0\text{ s}$ | $45.0\text{ kPa}$ |
| **2** | Roman Chamomile | *Chamaemelum nobile* | Interruption / Mild Stress | $90\ \mu\text{L}$ | $6.0\text{ s}$ | $38.0\text{ kPa}$ |
| **3** | Sweet Orange & Bergamot | *Citrus sinensis / bergamia* | Fatigue / Sluggishness | $100\ \mu\text{L}$ | $7.0\text{ s}$ | $40.0\text{ kPa}$ |
| **4** | Eucalyptus Globulus | *Eucalyptus globulus* | Brain Fog / Mental Fatigue | $80\ \mu\text{L}$ | $5.0\text{ s}$ | $42.0\text{ kPa}$ |
| **5** | Peppermint | *Mentha piperita* | Drowsiness / Focus Alert | $75\ \mu\text{L}$ | $4.5\text{ s}$ | $40.0\text{ kPa}$ |
| **6** | Sandalwood & Frankincense | *Santalum album / Boswellia* | Baseline Calm / Meditation | $60\ \mu\text{L}$ | $4.0\text{ s}$ | $35.0\text{ kPa}$ |

---

## 8. Environmental Compensation & Safety Interlocks

1. **Temperature Compensation**: Hot ambient air increases essential oil volatility; the system automatically reduces spray duration ($t_{\text{spray}}$) to prevent overpowering scent concentration.
2. **Humidity Compensation**: High relative humidity increases air resistance and slows aerosol dispersion; the system increases pneumatic compressor pressure ($P_{\text{airflow}}$) to ensure even distribution.
3. **Proximity Verification**: Dispensing is strictly conditioned on wearable BLE RSSI $\ge -75\text{ dBm}$. If the user walks out of range, actuation is immediately suppressed.
4. **Cooldown Enforcement**: Enforces a minimum 60-second cooldown between sprays to prevent olfactory fatigue and oversaturation.
5. **Cumulative Exposure Protection**: Monitors total daily volume to ensure exposure never exceeds safe limits ($5,000\ \mu\text{L/day}$).
6. **Piston-Aided Purge Cycle**: Every 10 dispensing operations, a dedicated pneumatic piston flushes the mixing chamber to remove residual oils and avoid cross-contamination.

---

## 9. How to Run

```bash
# 1. Open Interactive CLI
python main.py

# 2. Connect Live Physical Sensor Stream (e.g. COM3)
python main.py --live-stream --port COM3

# 3. Run Automated Self-Tests
python main.py --self-test

# 4. Run End-to-End Simulation
python main.py --simulate

# 5. Evaluate on Test Dataset
python main.py --evaluate
```
