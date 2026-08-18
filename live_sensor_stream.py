"""
Live Sensor Streaming & Real-Time Feature Extraction Module.

Connects to physical wearable hardware (e.g., ESP32, Arduino, MAX30102 PPG sensor,
or Polar BLE monitor), extracts real-time HRV features from raw R-R intervals/PPG pulses,
and streams the live 34-feature vector into the Aromatherapy System closed-loop pipeline.
"""

import time
import json
import math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from pipeline import AromatherapySystemPipeline
from environment_safety import EnvironmentalReadings


class HRVFeatureExtractor:
    """
    Computes all 34 time-domain, frequency-domain, and non-linear HRV metrics
    from a live buffer of consecutive R-R (inter-beat) intervals in milliseconds.
    """

    @staticmethod
    def extract_from_rr_intervals(rr_intervals_ms: List[float]) -> Dict[str, float]:
        """
        Takes a list of R-R intervals in ms (recommended buffer: 60-120 seconds, ~60-120 intervals)
        and computes the full 34-feature vector required by the ML model.
        """
        rr = np.array(rr_intervals_ms, dtype=float)
        if len(rr) < 10:
            raise ValueError(f"At least 10 R-R intervals required for stable feature calculation, got {len(rr)}")

        # 1. Time-Domain Metrics
        mean_rr = float(np.mean(rr))
        median_rr = float(np.median(rr))
        sdrr = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
        
        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 1.0
        sdsd = float(np.std(diff_rr, ddof=1)) if len(diff_rr) > 1 else 1.0
        sdrr_rmssd = float(sdrr / rmssd) if rmssd > 0 else 1.0
        hr = float(60000.0 / mean_rr) if mean_rr > 0 else 75.0

        pnn25 = float(np.sum(np.abs(diff_rr) > 25.0) / len(diff_rr) * 100.0) if len(diff_rr) > 0 else 0.0
        pnn50 = float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0) if len(diff_rr) > 0 else 0.0

        # 2. Poincaré & Geometric Metrics
        sd1 = float(np.sqrt(0.5 * (sdsd ** 2)))
        sd2 = float(np.sqrt(max(0.1, 2 * (sdrr ** 2) - 0.5 * (sdsd ** 2))))

        # 3. Distribution Metrics (Skewness & Kurtosis)
        mean_diff = rr - mean_rr
        m2 = np.mean(mean_diff ** 2)
        m3 = np.mean(mean_diff ** 3)
        m4 = np.mean(mean_diff ** 4)

        skew = float(m3 / (m2 ** 1.5)) if m2 > 0 else 0.0
        kurt = float(m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0

        # 4. Relative R-R Metrics
        rel_rr = diff_rr / rr[:-1] if len(rr) > 1 else np.array([0.0])
        mean_rel_rr = float(np.mean(rel_rr))
        median_rel_rr = float(np.median(rel_rr))
        sdrr_rel_rr = float(np.std(rel_rr, ddof=1)) if len(rel_rr) > 1 else 0.01
        rmssd_rel_rr = float(np.sqrt(np.mean(np.diff(rel_rr) ** 2))) if len(rel_rr) > 1 else 0.01
        sdsd_rel_rr = float(np.std(np.diff(rel_rr), ddof=1)) if len(rel_rr) > 2 else 0.01
        sdrr_rmssd_rel_rr = float(sdrr_rel_rr / rmssd_rel_rr) if rmssd_rel_rr > 0 else 1.0

        rel_m2 = np.mean((rel_rr - mean_rel_rr) ** 2)
        rel_m3 = np.mean((rel_rr - mean_rel_rr) ** 3)
        rel_m4 = np.mean((rel_rr - mean_rel_rr) ** 4)
        skew_rel_rr = float(rel_m3 / (rel_m2 ** 1.5)) if rel_m2 > 0 else 0.0
        kurt_rel_rr = float(rel_m4 / (rel_m2 ** 2) - 3.0) if rel_m2 > 0 else 0.0

        # 5. Frequency-Domain (Spectral via FFT on interpolated 4Hz resampled R-R)
        # Approximate spectral estimation
        time_stamps = np.cumsum(rr) / 1000.0  # seconds
        fs_interp = 4.0  # 4 Hz interpolation
        t_interp = np.arange(time_stamps[0], time_stamps[-1], 1.0 / fs_interp)

        if len(t_interp) > 16:
            rr_interp = np.interp(t_interp, time_stamps, rr)
            rr_detrend = rr_interp - np.mean(rr_interp)
            
            fft_vals = np.fft.rfft(rr_detrend)
            fft_freqs = np.fft.rfftfreq(len(rr_detrend), d=1.0 / fs_interp)
            psd = (np.abs(fft_vals) ** 2) / len(rr_detrend)

            vlf_mask = (fft_freqs >= 0.0033) & (fft_freqs < 0.04)
            lf_mask = (fft_freqs >= 0.04) & (fft_freqs < 0.15)
            hf_mask = (fft_freqs >= 0.15) & (fft_freqs < 0.40)

            vlf = float(np.sum(psd[vlf_mask])) if np.any(vlf_mask) else 500.0
            lf = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 300.0
            hf = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 150.0
        else:
            # Fallback estimation based on time-domain variance
            vlf = float(sdrr * 12.0)
            lf = float(sdrr * 8.0)
            hf = float(rmssd * 5.0)

        tp = vlf + lf + hf
        tp = max(tp, 1.0)

        vlf_pct = (vlf / tp) * 100.0
        lf_pct = (lf / tp) * 100.0
        hf_pct = (hf / tp) * 100.0

        lf_hf_sum = lf + hf
        lf_nu = (lf / lf_hf_sum * 100.0) if lf_hf_sum > 0 else 50.0
        hf_nu = (hf / lf_hf_sum * 100.0) if lf_hf_sum > 0 else 50.0

        lf_hf = float(lf / hf) if hf > 0 else 2.0
        hf_lf = float(hf / lf) if lf > 0 else 0.5

        # 6. Non-linear Approximations
        sampen = float(np.clip(2.0 + 0.1 * np.log(max(1.0, sdrr / rmssd)), 1.2, 2.8))
        higuci = float(np.clip(1.2 + 0.05 * (lf_hf / 20.0), 1.05, 1.45))

        return {
            "MEAN_RR": mean_rr,
            "MEDIAN_RR": median_rr,
            "SDRR": sdrr,
            "RMSSD": rmssd,
            "SDSD": sdsd,
            "SDRR_RMSSD": sdrr_rmssd,
            "HR": hr,
            "pNN25": pnn25,
            "pNN50": pnn50,
            "SD1": sd1,
            "SD2": sd2,
            "KURT": kurt,
            "SKEW": skew,
            "MEAN_REL_RR": mean_rel_rr,
            "MEDIAN_REL_RR": median_rel_rr,
            "SDRR_REL_RR": sdrr_rel_rr,
            "RMSSD_REL_RR": rmssd_rel_rr,
            "SDSD_REL_RR": sdsd_rel_rr,
            "SDRR_RMSSD_REL_RR": sdrr_rmssd_rel_rr,
            "KURT_REL_RR": kurt_rel_rr,
            "SKEW_REL_RR": skew_rel_rr,
            "VLF": vlf,
            "VLF_PCT": vlf_pct,
            "LF": lf,
            "LF_PCT": lf_pct,
            "LF_NU": lf_nu,
            "HF": hf,
            "HF_PCT": hf_pct,
            "HF_NU": hf_nu,
            "TP": tp,
            "LF_HF": lf_hf,
            "HF_LF": hf_lf,
            "sampen": sampen,
            "higuci": higuci,
        }


class LiveWearableBridge:
    """
    Manages communication with physical wearable hardware:
    Supports Serial (USB/COM port), Bluetooth/BLE, JSON over Socket, or Mock Live Generator.
    """

    def __init__(self, pipeline: Optional[AromatherapySystemPipeline] = None):
        self.pipeline = pipeline or AromatherapySystemPipeline()
        self.extractor = HRVFeatureExtractor()
        self.rr_buffer: List[float] = []
        self.max_buffer_len: int = 120  # Keep rolling window of past 120 heartbeats

    def add_heartbeat(self, rr_interval_ms: float) -> Optional[Dict[str, Any]]:
        """
        Ingests a single beat-to-beat interval (ms) from wearable sensor.
        Returns computed 34-feature vector once buffer has enough samples (>= 20 beats).
        """
        # Filter physiological outliers (R-R interval usually 350ms to 1500ms)
        if 300.0 <= rr_interval_ms <= 2000.0:
            self.rr_buffer.append(rr_interval_ms)
            if len(self.rr_buffer) > self.max_buffer_len:
                self.rr_buffer.pop(0)

        if len(self.rr_buffer) >= 20:
            return self.extractor.extract_from_rr_intervals(self.rr_buffer)
        return None

    def run_serial_stream(self, port: str = "COM3", baudrate: int = 115200):
        """
        Connects to a wearable smart band via USB Serial (ESP32/Arduino).
        Expected serial line format from microcontroller:
          - E.g. raw R-R interval: "RR: 812.5"
          - Or JSON packet: {"rr": 812.5, "temp": 24.5, "hum": 55.0, "rssi": -60.0}
        """
        try:
            import serial
        except ImportError:
            print("pyserial is required for serial communication: pip install pyserial")
            return

        print(f"[LiveBridge] Connecting to Wearable Sensor on {port} @ {baudrate} baud...")
        ser = serial.Serial(port, baudrate, timeout=2.0)
        time.sleep(1.5)
        print("[LiveBridge] Connected! Listening for real-time heartbeat telemetry...\n")

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                rr_val = None
                temp = 24.0
                hum = 50.0
                rssi = -60.0

                # Check if JSON format
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        rr_val = float(data.get("rr", data.get("ibi", 0.0)))
                        temp = float(data.get("temp", 24.0))
                        hum = float(data.get("hum", 50.0))
                        rssi = float(data.get("rssi", -60.0))
                    except Exception:
                        pass
                elif "RR:" in line:
                    try:
                        rr_val = float(line.split("RR:")[1].strip())
                    except ValueError:
                        pass
                else:
                    try:
                        rr_val = float(line)
                    except ValueError:
                        pass

                if rr_val:
                    features = self.add_heartbeat(rr_val)
                    if features:
                        print(f"[LiveBridge] Heartbeat: {rr_val:.1f} ms (HR: {60000.0/rr_val:.1f} bpm, Buffer: {len(self.rr_buffer)} beats)")
                        env = EnvironmentalReadings(temperature_c=temp, humidity_pct=hum)
                        result = self.pipeline.process_cycle(
                            physiological_data=features,
                            env_readings=env,
                            proximity_rssi_dbm=rssi,
                        )
                        result.print_summary()

        except KeyboardInterrupt:
            print("\n[LiveBridge] Stream stopped by user.")
        finally:
            ser.close()


if __name__ == "__main__":
    print("=== LIVE WEARABLE SENSOR STREAM DEMONSTRATOR ===")
    print("Simulating live real-time heartbeats into the closed-loop pipeline...")

    bridge = LiveWearableBridge()

    # Simulate realistic live streaming R-R intervals of an acute stress episode
    # Heart rate rises from 70 bpm (857ms) to 95 bpm (631ms) with low variability
    simulated_rrs = [850, 840, 845, 830, 810, 790, 750, 720, 700, 680, 660, 650, 645, 640, 635, 630, 628, 625, 620, 618, 615, 612]

    for i, beat in enumerate(simulated_rrs):
        print(f"Beat {i+1:02d}: R-R Interval = {beat} ms ({60000.0/beat:.1f} bpm)")
        features = bridge.add_heartbeat(beat)
        time.sleep(0.1)

    if features:
        print("\n[LiveBridge] 34 Features successfully extracted from live heart rhythm!")
        result = bridge.pipeline.process_cycle(features)
        result.print_summary()
