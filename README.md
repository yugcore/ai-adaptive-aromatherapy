# AI-Driven Emotion & Environment Adaptive Smart Aromatherapy System

A modular, closed-loop cyber-physical aromatherapy platform that monitors multimodal physiological parameters (HRV, Heart Rate, Respiration, Skin Conductance) from a wearable smart band, performs real-time machine learning stress and emotional state classification, and drives a multi-stage pneumatic fragrance dispensing mechanism with environmental compensation and physical safety interlocks.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│      Wearable Smart Band (ESP32 / MAX30102 / BLE)      │
│  - Real-time beat-to-beat (R-R interval) telemetry     │
└──────────────────────────┬─────────────────────────────┘
                           │ Raw Heartbeats / Serial Stream (live_sensor_stream.py)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Inference Engine                     │
│  - HRV Feature Extraction (34 Metrics on the fly)      │
│  - Data Preprocessing (Imputation & Scaling)           │
│  - ML Classifier (Random Forest Emotion Model)         │
│  - Continuous Mood Severity Score (0.0 - 1.0)          │
└──────────────────────────┬─────────────────────────────┘
                           │ State + Severity + Confidence
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Fragrance Controller                   │
│  - Rotary 6-Slot Cartridge Selection                   │
│  - Adaptive Intensity Modulation (Volume/Airflow/Time) │
│  - Multi-Fragrance Blending Engine                     │
└──────────────────────────┬─────────────────────────────┘
                           │ Target Recipe (Dose, Duration, Airflow)
                           ▼
┌────────────────────────────────────────────────────────┐
│            Environment & Safety Interlocks             │
│  - Ambient Temp & Humidity Compensation                │
│  - Proximity RSSI (BLE Range Threshold) & Auth Check   │
│  - Cooldown Interval & Daily Exposure Limiter          │
│  - Piston-Aided Purge Cycle Scheduler                  │
└──────────────────────────┬─────────────────────────────┘
                           │ Verified & Adapted Actuation Parameters
                           ▼
┌────────────────────────────────────────────────────────┐
│         Hardware Multi-Stage Pneumatic Dispenser       │
│  Phase 1: Proximity & Auth Handshake                   │
│  Phase 2: Rotary Magazine Stepper Indexing (6-Slot)    │
│  Phase 3: Precision Aroma Liquid Intake                │
│  Phase 4: Pressurized Pneumatic Aerosolization         │
│  Phase 5: Piston Flush & Conduit Purge                 │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| **`live_sensor_stream.py`** | Real-time hardware stream connector (USB Serial / Bluetooth) and on-the-fly 34-feature HRV extractor from live R-R heartbeat intervals. |
| **`config.py`** | Central configuration with dynamic dataset paths, model parameters, fragrance catalog (6 slots), environmental compensation coefficients, and safety thresholds. |
| **`data_loader.py`** | Data ingestion for `HRV stress dataset` (`train.csv`, `test.csv`) and `physiological stress detection dataset` (`stress_detection.csv`). |
| **`preprocessor.py`** | Data coercion, infinite value handling, empty column pruning, `SimpleImputer` and `StandardScaler` pipeline with serialization. |
| **`model_trainer.py`** | Random Forest training, multi-metric evaluation (Accuracy, F1-Score, Precision, Recall, Confusion Matrix), and `joblib` saving/loading. |
| **`inference_engine.py`** | Real-time prediction and calculation of continuous **Mood Severity Score** ($0.0 \le S \le 1.0$) combining model probabilities and cardiovascular markers. |
| **`fragrance_controller.py`** | Translates emotional state and severity score into cartridge slots, dynamic dosages ($\mu\text{L}$), spray durations (s), and pneumatic pressures (kPa). |
| **`environment_safety.py`** | Environmental compensation (ambient temperature & humidity), BLE RSSI proximity verification, cooldown timer, daily exposure limits, and purge scheduling. |
| **`hardware_simulator.py`** | Simulates the 5-phase pneumatic dispensing sequence from the patent specification. |
| **`pipeline.py`** | Closed-loop orchestrator linking all sensory and actuator subsystems. |
| **`main.py`** | Application entry point offering interactive menu (Live streaming, simulation, Colab-style input, test evaluation). |
| **`SAMPLE.md`** | Detailed feature guide explaining all 34 HRV metrics and providing 3 ready-to-use sample batches. |
| **`DOCUMENTATION.md`** | Comprehensive technical and architectural documentation. |
| **`hrv_emotion_ml.py`** | Original Google Colab monolithic script (preserved untouched). |

---

## 🌸 Fragrance Cartridge Catalog (6-Slot Rotary Magazine)

1. **Slot 1: French Lavender** (*Lavandula angustifolia*) — High acute stress, time pressure, anxiety relief.
2. **Slot 2: Roman Chamomile** (*Chamaemelum nobile*) — Mild agitation, cognitive interruption, nervous tension.
3. **Slot 3: Sweet Orange & Bergamot** (*Citrus sinensis / bergamia*) — Fatigue, mid-day slump, mood uplifting.
4. **Slot 4: Eucalyptus Globulus** (*Eucalyptus globulus*) — Cognitive overload, mental fog, respiratory clarity.
5. **Slot 5: Peppermint** (*Mentha piperita*) — Instant stimulation, alertness, focus boosting.
6. **Slot 6: Sandalwood & Frankincense** (*Santalum album / Boswellia carterii*) — Homeostasis maintenance, baseline calm.

---

## 🚀 Quickstart & Usage

### 1. Launch Interactive CLI Menu
```bash
python main.py
```
* **Option 1**: Connect Live Wearable Sensor (USB/COM Serial or Live Stream)
* **Option 2**: Train ML Model on HRV Dataset
* **Option 3**: Run End-to-End Closed-Loop Simulation (5 Scenarios)
* **Option 4**: Interactive User Input (Feature-by-Feature / Presets)
* **Option 5**: Batch Evaluate on HRV Test Dataset (`test.csv`)
* **Option 6**: Inspect Multimodal Physiological Dataset (`stress_detection.csv`)
* **Option 7**: Run System Self-Tests

### 2. Connect Physical Hardware Stream Directly
```bash
# Listen on USB COM port:
python main.py --live-stream --port COM3
```

### 3. Run Automated Self-Tests
```bash
python main.py --self-test
```

### 4. Run Full Simulation Demonstration
```bash
python main.py --simulate
```
# Trains on 100,000 samples (takes ~4-5 seconds):
```bash
python main.py --train --sample-size 100000
```

# Or train through the menu:
```bash
python main.py  # Select Option 2
```
