"""
Main Application Entrypoint for Smart Aromatherapy System.

Provides an interactive CLI and automated execution modes:
1. Connect Live Physical Wearable Sensor (USB/COM Serial or Live Stream)
2. Train ML Model on HRV Dataset
3. Run End-to-End Closed-Loop Simulation
4. Interactive User Input (Colab-style feature input)
5. Batch Evaluation on Test Dataset
6. Multi-modal Physiological Stress Dataset Inspection
7. System Self-Test Verification
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from config import DATASET_PATHS, MODEL_FILE, PREPROCESSOR_FILE, DEFAULT_ML_CONFIG
from data_loader import DatasetLoader
from preprocessor import HRVPreprocessor
from model_trainer import ModelTrainer, train_pipeline
from inference_engine import InferenceEngine
from fragrance_controller import FragranceController
from environment_safety import EnvironmentSafetyManager, EnvironmentalReadings
from hardware_simulator import HardwareSimulator
from pipeline import AromatherapySystemPipeline
from live_sensor_stream import LiveWearableBridge


def print_banner():
    print("=" * 70)
    print("   AI-DRIVEN EMOTION & ENVIRONMENT ADAPTIVE SMART AROMATHERAPY SYSTEM")
    print("   Modular Closed-Loop Physiological Sensing & Pneumatic Dispensing")
    print("=" * 70)


def connect_live_sensor(pipeline: AromatherapySystemPipeline, port: Optional[str] = None):
    """
    Connects to physical wearable hardware (ESP32/Arduino via USB Serial)
    or runs a real-time live cardiac pulse stream demonstration.
    """
    print("\n--- LIVE WEARABLE SENSOR CONNECTION ---")
    print("Options:")
    print("  1. Connect to Physical Hardware via USB Serial (ESP32 / Arduino / MAX30102)")
    print("  2. Run Real-Time Streaming Heartbeat Demo (Live Cardiac Pulse Train)")

    choice = input("\nSelect option (1 or 2, default 2): ").strip()
    bridge = LiveWearableBridge(pipeline=pipeline)

    if choice == "1":
        default_port = port or "COM3"
        user_port = input(f"Enter Serial COM Port (default '{default_port}'): ").strip()
        selected_port = user_port if user_port else default_port
        baud_str = input("Enter Baud Rate (default 115200): ").strip()
        baud = int(baud_str) if baud_str.isdigit() else 115200

        print(f"\n[Main] Starting Serial Listener on {selected_port} @ {baud}...")
        bridge.run_serial_stream(port=selected_port, baudrate=baud)
    else:
        print("\n[Main] Running Real-Time Live Cardiac Pulse Stream Simulation...")
        print("Streaming consecutive R-R beat intervals into rolling feature extractor...\n")

        # Simulate 25 realistic heartbeats transitioning into an acute stress state
        simulated_rrs = [
            840, 845, 835, 840, 830, 820, 810, 790, 770, 750,
            730, 710, 690, 670, 650, 640, 630, 625, 620, 615,
            610, 608, 605, 602, 600
        ]

        for i, beat in enumerate(simulated_rrs):
            hr = 60000.0 / beat
            print(f"  ❤️  Beat {i+1:02d}/25: R-R = {beat} ms | Instantaneous HR = {hr:.1f} bpm")
            features = bridge.add_heartbeat(beat)
            time.sleep(0.12)

        if features:
            print("\n[LiveBridge] Extracted 34 HRV Metrics from Live Heart Rhythm!")
            result = pipeline.process_cycle(features)
            result.print_summary()


def interactive_user_input(pipeline: AromatherapySystemPipeline):
    """
    Enhanced version of original Google Colab interactive feature input.
    Allows entering features manually or using preset scenarios.
    """
    print("\n--- INTERACTIVE USER PHYSIOLOGICAL INPUT ---")
    print("Options:")
    print("  1. Choose a Preset Scenario (High Stress, Mild Interruption, Baseline Calm)")
    print("  2. Enter Custom Values Feature-by-Feature (Colab Mode)")
    
    choice = input("\nSelect option (1 or 2, default 1): ").strip()
    
    loader = DatasetLoader()
    expected_features = loader.get_feature_names()

    if choice == "2":
        print(f"\nEnter values for {len(expected_features)} HRV features (press ENTER to leave empty / NaN):")
        user_dict = {}
        for col in expected_features:
            val = input(f"  {col}: ").strip()
            if val == "":
                user_dict[col] = np.nan
            else:
                try:
                    user_dict[col] = float(val)
                except ValueError:
                    user_dict[col] = np.nan
    else:
        print("\nSelect Preset Scenario:")
        print("  A. Acute Time Pressure / High Anxiety (HR=105 bpm, low RMSSD=14 ms, high LF/HF=4.8)")
        print("  B. Cognitive Interruption / Mild Stress (HR=84 bpm, moderate RMSSD=26 ms, LF/HF=2.6)")
        print("  C. Relaxed / Homeostatic Baseline (HR=64 bpm, high RMSSD=52 ms, LF/HF=1.1)")
        scen = input("Choice (A/B/C, default A): ").strip().upper()

        # Load a representative sample from test dataset
        X_test, y_test = loader.load_hrv_test(sample_size=1000)
        
        if scen == "C":
            subset = X_test[y_test == "no stress"]
        elif scen == "B":
            subset = X_test[y_test == "interruption"]
        else:
            subset = X_test[y_test == "time pressure"]

        if len(subset) > 0:
            sample_row = subset.iloc[0].to_dict()
        else:
            sample_row = {col: 0.0 for col in expected_features}

        user_dict = sample_row

    # Environment readings prompt
    print("\nEnvironmental & Safety Context:")
    temp_str = input("  Ambient Temperature in °C (default 27.5°C): ").strip()
    hum_str = input("  Ambient Humidity in % (default 65%): ").strip()
    rssi_str = input("  Smart Band BLE RSSI in dBm (default -60 dBm): ").strip()

    temp = float(temp_str) if temp_str else 27.5
    hum = float(hum_str) if hum_str else 65.0
    rssi = float(rssi_str) if rssi_str else -60.0

    env = EnvironmentalReadings(temperature_c=temp, humidity_pct=hum)

    print("\nExecuting Closed-Loop Pipeline...")
    result = pipeline.process_cycle(
        physiological_data=user_dict,
        env_readings=env,
        proximity_rssi_dbm=rssi,
    )
    result.print_summary()


def run_e2e_simulation(pipeline: AromatherapySystemPipeline):
    """
    Executes a comprehensive multi-scenario simulation demonstrating all system features:
    1. Acute Stress -> High-intensity Lavender dispensing
    2. Cognitive Interruption -> Calming Chamomile/Bergamot blend
    3. Calm Baseline -> Gentle Sandalwood maintenance
    4. Out-of-Range User -> Safety Interlock triggers
    5. Environmental Compensation -> High Temperature & Humidity adjustments
    """
    print("\n" + "#" * 70)
    print("RUNNING END-TO-END SMART AROMATHERAPY SYSTEM SIMULATION")
    print("#" * 70)

    loader = DatasetLoader()
    X_test, y_test = loader.load_hrv_test(sample_size=500)

    scenarios = [
        ("SCENARIO 1: High Acute Stress / Time Pressure", "time pressure", EnvironmentalReadings(23.0, 50.0), -58.0),
        ("SCENARIO 2: Cognitive Interruption / Agitation", "interruption", EnvironmentalReadings(24.0, 55.0), -62.0),
        ("SCENARIO 3: Homeostatic Calm Baseline", "no stress", EnvironmentalReadings(22.0, 48.0), -55.0),
        ("SCENARIO 4: Safety Interlock: User Out of Proximity Range", "time pressure", EnvironmentalReadings(23.0, 50.0), -88.0),
        ("SCENARIO 5: Environmental Adaptation: Hot & Humid Room", "time pressure", EnvironmentalReadings(32.0, 80.0), -60.0),
    ]

    for title, condition_filter, env, rssi in scenarios:
        print("\n" + "=" * 70)
        print(f">>> {title}")
        print("=" * 70)
        print(f"Environmental Context: Temp={env.temperature_c}°C, Hum={env.humidity_pct}%, BLE RSSI={rssi} dBm")

        # Find matching row
        matching = X_test[y_test == condition_filter]
        sample = matching.iloc[0].to_dict() if len(matching) > 0 else X_test.iloc[0].to_dict()

        # Reset cooldown for demo simulation
        pipeline.safety_manager.last_dispense_timestamp = 0.0

        res = pipeline.process_cycle(
            physiological_data=sample,
            env_readings=env,
            proximity_rssi_dbm=rssi,
        )
        res.print_summary()
        time.sleep(0.5)


def batch_evaluate_test_dataset(sample_size: int = 20000):
    """Evaluates the trained model across a large batch of the test dataset."""
    print(f"\n--- BATCH EVALUATION ON HRV TEST DATASET (N={sample_size:,}) ---")
    loader = DatasetLoader()
    X_test, y_test = loader.load_hrv_test(sample_size=sample_size)

    if y_test is None:
        print("Test dataset does not contain ground truth 'condition' column.")
        return

    preprocessor = HRVPreprocessor.load(PREPROCESSOR_FILE)
    trainer = ModelTrainer().load(MODEL_FILE)

    X_test_scaled = preprocessor.transform(X_test)
    trainer.evaluate(X_test_scaled, y_test, dataset_name="HRV stress dataset / test.csv")


def inspect_multimodal_dataset():
    """Inspects the secondary physiological stress dataset."""
    print("\n--- MULTIMODAL PHYSIOLOGICAL STRESS DATASET INSPECTION ---")
    loader = DatasetLoader()
    df = loader.load_physiological_stress_data()
    print("\nDataset Summary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].head(10))


def run_self_tests():
    """Runs automated unit verification across all modular components."""
    print("\n--- RUNNING SYSTEM SELF-TEST SUITE ---")
    loader = DatasetLoader()
    
    # Test 1: Data paths exist
    assert Path(DATASET_PATHS["hrv_train"]).exists(), "hrv_train missing!"
    assert Path(DATASET_PATHS["hrv_test"]).exists(), "hrv_test missing!"
    assert Path(DATASET_PATHS["physiological_stress"]).exists(), "physiological_stress missing!"
    print("  [PASS] All dataset paths verified.")

    # Test 2: Preprocessor fit & transform
    X_sample, _ = loader.load_hrv_train(sample_size=100)
    prep = HRVPreprocessor()
    transformed = prep.fit_transform(X_sample)
    assert transformed.shape[0] == 100
    assert not np.isnan(transformed).any()
    print("  [PASS] Preprocessor cleaning, imputation, and scaling verified.")

    # Test 3: Inference Engine
    engine = InferenceEngine()
    test_dict = X_sample.iloc[0].to_dict()
    inf_res = engine.predict(test_dict)
    assert inf_res.predicted_state in ["no stress", "interruption", "time pressure"]
    assert 0.0 <= inf_res.severity_score <= 1.0
    print(f"  [PASS] Inference Engine verified (State: '{inf_res.predicted_state}', Severity: {inf_res.severity_score:.2f}).")

    # Test 4: Fragrance Controller
    fc = FragranceController()
    plan = fc.generate_dispensing_plan(inf_res)
    assert plan.total_dose_ul >= 0.0
    print(f"  [PASS] Fragrance Controller verified (Action: '{plan.action}', Dose: {plan.total_dose_ul} µL).")

    # Test 5: Safety Manager & Environmental Adaptation
    sm = EnvironmentSafetyManager()
    env_hot = EnvironmentalReadings(temperature_c=30.0, humidity_pct=75.0)
    safety_res = sm.verify_and_adapt(plan, env=env_hot, proximity_rssi_dbm=-60.0)
    assert safety_res.is_safe_to_dispense == plan.requires_dispense
    print(f"  [PASS] Environmental Adaptation & Safety Manager verified.")

    # Test 6: Hardware Simulator
    hw = HardwareSimulator(verbose=False)
    hw_report = hw.execute_dispense(safety_res)
    assert hw_report.success == safety_res.is_safe_to_dispense
    print("  [PASS] Hardware 5-Stage Pneumatic Simulator verified.")

    print("\nALL SELF-TESTS PASSED SUCCESSFULLY!")


def main():
    parser = argparse.ArgumentParser(description="AI-Driven Smart Aromatherapy System CLI")
    parser.add_argument("--live-stream", action="store_true", help="Connect live wearable sensor stream")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port for live wearable sensor (e.g. COM3)")
    parser.add_argument("--train", action="store_true", help="Train model and save artifacts")
    parser.add_argument("--simulate", action="store_true", help="Run end-to-end simulation")
    parser.add_argument("--evaluate", action="store_true", help="Run batch evaluation on test dataset")
    parser.add_argument("--inspect-physio", action="store_true", help="Inspect physiological dataset")
    parser.add_argument("--self-test", action="store_true", help="Run automated unit verification")
    parser.add_argument("--sample-size", type=int, default=100000, help="Sample size for training")

    args = parser.parse_args()
    print_banner()

    pipeline = AromatherapySystemPipeline()

    if args.live_stream:
        connect_live_sensor(pipeline, port=args.port)
        return
    elif args.train:
        train_pipeline(sample_size=args.sample_size, eval_on_test=True, save_artifacts=True)
        return
    elif args.simulate:
        run_e2e_simulation(pipeline)
        return
    elif args.evaluate:
        batch_evaluate_test_dataset()
        return
    elif args.inspect_physio:
        inspect_multimodal_dataset()
        return
    elif args.self_test:
        run_self_tests()
        return

    # Interactive Menu
    while True:
        print("\n" + "-" * 50)
        print("MAIN MENU:")
        print("  1. Connect Live Wearable Sensor (USB/COM Serial or Live Stream)")
        print("  2. Train ML Model on HRV Dataset")
        print("  3. Run End-to-End Closed-Loop Simulation")
        print("  4. Interactive User Input (Feature-by-Feature / Presets)")
        print("  5. Batch Evaluate on HRV Test Dataset (test.csv)")
        print("  6. Inspect Multimodal Physiological Dataset (stress_detection.csv)")
        print("  7. Run System Self-Tests")
        print("  8. Exit")
        print("-" * 50)

        choice = input("Enter choice (1-8): ").strip()
        if choice == "1":
            connect_live_sensor(pipeline)
        elif choice == "2":
            size_str = input("Enter sample size for training (default 100,000, 0 for all): ").strip()
            size = int(size_str) if size_str.isdigit() and int(size_str) > 0 else (None if size_str == "0" else 100000)
            train_pipeline(sample_size=size, eval_on_test=True, save_artifacts=True)
            pipeline = AromatherapySystemPipeline()  # Reload with new model
        elif choice == "3":
            run_e2e_simulation(pipeline)
        elif choice == "4":
            interactive_user_input(pipeline)
        elif choice == "5":
            batch_evaluate_test_dataset()
        elif choice == "6":
            inspect_multimodal_dataset()
        elif choice == "7":
            run_self_tests()
        elif choice == "8":
            print("\nExiting Smart Aromatherapy System. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-8.")


if __name__ == "__main__":
    main()
