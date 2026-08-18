"""
Hardware Simulator Module for Smart Aromatherapy System.

Simulates the physical 5-phase pneumatic fragrance dispensing mechanism:
- Phase 1: Proximity & Authentication Handshake
- Phase 2: Rotary Magazine Stepper Indexing & Cartridge Alignment
- Phase 3: Precision Aroma Pump Fluid Intake to Mixing Chamber
- Phase 4: Pneumatic Storage Pressurization & Micro-Aerosol Dispensing
- Phase 5: Pneumatic Piston Actuation & Residual Chamber Purge
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from fragrance_controller import DispensingPlan, CartridgeDose
from environment_safety import EnvironmentalReadings, SafetyVerificationResult


@dataclass
class HardwareExecutionReport:
    success: bool
    phases_completed: List[str]
    stepper_angle_deg: float
    total_volume_dispensed_ul: float
    airflow_pressure_applied_kpa: float
    spray_duration_sec: float
    piston_purge_executed: bool
    diagnostics: Dict[str, Any]


class HardwareSimulator:
    """
    Simulates the electro-mechanical state machine of the smart aromatherapy diffuser.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_magazine_slot: int = 1
        self.degrees_per_slot: float = 360.0 / 6.0  # 6-slot carousel = 60 deg/slot

    def log(self, phase_num: int, title: str, details: str) -> None:
        """Pretty prints hardware execution state transitions."""
        if self.verbose:
            print(f"  [HARDWARE PHASE {phase_num}] {title.upper()}: {details}")

    def execute_dispense(
        self,
        safety_result: SafetyVerificationResult,
        device_id: str = "SMARTBAND_AGY_001",
        rssi_dbm: float = -62.0,
    ) -> HardwareExecutionReport:
        """
        Runs the 5-phase mechanical dispensing sequence.
        """
        if not safety_result.is_safe_to_dispense or safety_result.adjusted_plan is None:
            if self.verbose:
                print(f"[HardwareSimulator] Execution Aborted: {safety_result.rejection_reason}")
            return HardwareExecutionReport(
                success=False,
                phases_completed=[],
                stepper_angle_deg=0.0,
                total_volume_dispensed_ul=0.0,
                airflow_pressure_applied_kpa=0.0,
                spray_duration_sec=0.0,
                piston_purge_executed=False,
                diagnostics={"rejection_reason": safety_result.rejection_reason},
            )

        plan = safety_result.adjusted_plan
        phases = []

        if self.verbose:
            print("\n" + "=" * 65)
            print(f"HARDWARE DISPENSER INITIATING: {plan.action.upper()} ({plan.target_state})")
            print("=" * 65)

        # -------------------------------------------------------------
        # PHASE 1: Proximity & Auth Handshake
        # -------------------------------------------------------------
        self.log(
            1,
            "Proximity & Auth Handshake",
            f"Device '{device_id}' validated at RSSI {rssi_dbm:.1f} dBm. Link encrypted.",
        )
        phases.append("Phase 1: Proximity Verified")

        # -------------------------------------------------------------
        # PHASE 2: Rotary Cartridge Stepper Alignment
        # -------------------------------------------------------------
        target_slot = plan.cartridge_doses[0].slot_id if plan.cartridge_doses else 1
        steps = (target_slot - self.current_magazine_slot) % 6
        target_angle = (target_slot - 1) * self.degrees_per_slot
        self.current_magazine_slot = target_slot

        self.log(
            2,
            "Rotary Magazine Indexing",
            f"Stepper rotated to Slot {target_slot} ({target_angle:.0f}°). Fragrance: '{plan.cartridge_doses[0].fragrance_name}' aligned to mixing chamber.",
        )
        phases.append(f"Phase 2: Magazine Aligned (Slot {target_slot})")

        # -------------------------------------------------------------
        # PHASE 3: Precision Aroma Pump Fluid Intake
        # -------------------------------------------------------------
        dose_breakdown = ", ".join(
            [f"{c.fragrance_name}: {c.dose_ul:.1f}µL ({c.blend_percentage:.0f}%)" for c in plan.cartridge_doses]
        )
        self.log(
            3,
            "Precision Fluid Intake",
            f"Micropump metered total {plan.total_dose_ul:.1f} µL [{dose_breakdown}] into pneumatic mixing chamber.",
        )
        phases.append(f"Phase 3: Dosed {plan.total_dose_ul:.1f} µL")

        # -------------------------------------------------------------
        # PHASE 4: Pneumatic Airflow & Aerosol Dispensing
        # -------------------------------------------------------------
        self.log(
            4,
            "Aerosolization & Dispersion",
            f"Compressor charged to {plan.airflow_pressure_kpa:.1f} kPa. Ultrasonic nozzle active for {plan.spray_duration_sec:.2f}s at {plan.nozzle_duty_cycle_pct:.0f}% duty cycle.",
        )
        phases.append(f"Phase 4: Aerosolized ({plan.spray_duration_sec:.1f}s @ {plan.airflow_pressure_kpa:.1f} kPa)")

        # -------------------------------------------------------------
        # PHASE 5: Piston Actuation & Remnant Clearance
        # -------------------------------------------------------------
        piston_purged = False
        if safety_result.triggers_purge_cycle:
            self.log(
                5,
                "Piston Purge Cycle",
                "Periodic purge scheduled: Dedicated piston actuated to expel remnants and flush chamber conduits.",
            )
            piston_purged = True
            phases.append("Phase 5: Piston Purge Completed")
        else:
            self.log(
                5,
                "Piston Reset",
                "Piston staged in home position. Chamber sealed for next actuation cycle.",
            )
            phases.append("Phase 5: Piston Staged")

        if self.verbose:
            print("=" * 65 + "\n")

        return HardwareExecutionReport(
            success=True,
            phases_completed=phases,
            stepper_angle_deg=target_angle,
            total_volume_dispensed_ul=plan.total_dose_ul,
            airflow_pressure_applied_kpa=plan.airflow_pressure_kpa,
            spray_duration_sec=plan.spray_duration_sec,
            piston_purge_executed=piston_purged,
            diagnostics={
                "target_slot": target_slot,
                "blend_count": len(plan.cartridge_doses),
                "is_multi_blend": plan.is_multi_blend,
            },
        )
