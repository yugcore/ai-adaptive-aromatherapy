"""
Closed-Loop Pipeline Module for Smart Aromatherapy System.

Orchestrates the entire closed-loop interaction:
1. Physiological Data Ingestion & Signal Conditioning
2. Real-time Emotion & Mood Severity Inference
3. Scent Formulation & Cartridge Mapping
4. Environmental Adaptation & Safety Interlock Verification
5. Multi-Phase Hardware Dispenser Execution
"""

from dataclasses import dataclass
from typing import Dict, Any, Union, Optional
import pandas as pd

from inference_engine import InferenceEngine, InferenceResult
from fragrance_controller import FragranceController, DispensingPlan
from environment_safety import (
    EnvironmentSafetyManager,
    EnvironmentalReadings,
    SafetyVerificationResult,
)
from hardware_simulator import HardwareSimulator, HardwareExecutionReport


@dataclass
class ClosedLoopCycleResult:
    inference: InferenceResult
    plan: DispensingPlan
    safety: SafetyVerificationResult
    hardware: HardwareExecutionReport

    def print_summary(self) -> None:
        """Prints a human-readable summary of the closed-loop cycle."""
        print("+" + "-" * 66 + "+")
        print(f"| CLOSED-LOOP CYCLE SUMMARY: {self.inference.predicted_state.upper():<37} |")
        print("+" + "-" * 66 + "+")
        print(f"| Inference State:    {self.inference.predicted_state:<44} |")
        print(f"| Confidence Score:   {self.inference.confidence * 100:>5.1f}%{'':<38} |")
        print(f"| Mood Severity:      {self.inference.severity_score:>5.2f} / 1.00{'':<33} |")
        print(f"| Action Decision:    {self.plan.action.upper():<44} |")
        
        if self.plan.cartridge_doses:
            primary_dose = self.plan.cartridge_doses[0]
            print(f"| Selected Fragrance: {primary_dose.fragrance_name:<44} |")
            print(f"| Cartridge Slot:     Slot {primary_dose.slot_id:<39} |")
            print(f"| Dispense Volume:    {self.safety.adjusted_plan.total_dose_ul:>5.1f} µL{'':<37} |")
            print(f"| Spray Duration:     {self.safety.adjusted_plan.spray_duration_sec:>5.2f} s{'':<38} |")
            print(f"| Airflow Pressure:   {self.safety.adjusted_plan.airflow_pressure_kpa:>5.1f} kPa{'':<36} |")
        else:
            print(f"| Selected Fragrance: {'None (Suppressed)':<44} |")

        status_str = "SUCCESS (Dispensed)" if self.hardware.success else f"BLOCKED ({self.safety.rejection_reason})"
        print(f"| Hardware Status:    {status_str[:44]:<44} |")
        print("+" + "-" * 66 + "+")


class AromatherapySystemPipeline:
    """
    Main closed-loop system pipeline orchestrating ML inference, fragrance logic,
    safety interlocks, and mechanical dispensing.
    """

    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        fragrance_controller: Optional[FragranceController] = None,
        safety_manager: Optional[EnvironmentSafetyManager] = None,
        hardware_simulator: Optional[HardwareSimulator] = None,
        verbose: bool = True,
    ):
        self.inference_engine = inference_engine or InferenceEngine()
        self.fragrance_controller = fragrance_controller or FragranceController()
        self.safety_manager = safety_manager or EnvironmentSafetyManager()
        self.hardware_simulator = hardware_simulator or HardwareSimulator(verbose=verbose)

    def process_cycle(
        self,
        physiological_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        env_readings: Optional[EnvironmentalReadings] = None,
        device_id: str = "SMARTBAND_AGY_001",
        proximity_rssi_dbm: float = -62.0,
    ) -> ClosedLoopCycleResult:
        """
        Executes a complete closed-loop cycle from sensory input to actuator output.
        """
        # Step 1: Real-time ML Inference & Mood Severity Score
        inference_res = self.inference_engine.predict(physiological_data)

        # Step 2: Fragrance Recipe Generation & Cartridge Selection
        dispense_plan = self.fragrance_controller.generate_dispensing_plan(inference_res)

        # Step 3: Environmental Adaptation & Safety Interlocks
        safety_res = self.safety_manager.verify_and_adapt(
            plan=dispense_plan,
            env=env_readings,
            device_id=device_id,
            rssi_dbm=proximity_rssi_dbm,
        )

        # Step 4: Hardware Dispenser State Machine Execution
        hw_report = self.hardware_simulator.execute_dispense(
            safety_result=safety_res,
            device_id=device_id,
            rssi_dbm=proximity_rssi_dbm,
        )

        # Step 5: Record event in safety manager if successful
        if hw_report.success and safety_res.adjusted_plan:
            self.safety_manager.record_dispense_event(safety_res.adjusted_plan.total_dose_ul)
            if hw_report.piston_purge_executed:
                self.safety_manager.record_purge_event()

        return ClosedLoopCycleResult(
            inference=inference_res,
            plan=dispense_plan,
            safety=safety_res,
            hardware=hw_report,
        )
