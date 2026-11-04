"""
Module 4: Adaptive Signal Control
Vision-Based Adaptive Traffic Signal Management System

Role: Member 4 – Adaptive Signal Control
Inputs:
  - reports/module3/road_traffic_metrics.csv
  - reports/module3/congestion_scores.csv
  - reports/module3/traffic_pattern_summary.csv
  - data/processed/module2/track_observations.csv

Outputs (saved in reports/module4/):
  - adaptive_signal_decisions.csv
  - cycle_by_cycle_log.csv
  - fixed_vs_adaptive_comparison.csv
  - emergency_pedestrian_events.csv
  - signal_control_summary.json
  - signal_performance_comparison.png
  - adaptive_green_allocation_timeline.png
  - queue_dissipation_profiles.png

Core Responsibilities:
  1. Calculate dynamic road priority scores using queue length, wait times,
     density, arrival flow, and starvation prevention.
  2. Dynamically allocate green-light durations subject to min/max safety constraints.
  3. Manage emergency vehicle preemption and pedestrian crossing priorities.
  4. Compare performance against standard fixed-time signal controllers using
     realistic discrete simulation on observed video arrivals.
"""

from pathlib import Path
import argparse
import json
import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# DEFAULT PATH CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "module3" / "road_traffic_metrics.csv"
DEFAULT_CONGESTION_PATH = PROJECT_ROOT / "reports" / "module3" / "congestion_scores.csv"
DEFAULT_PATTERNS_PATH = PROJECT_ROOT / "reports" / "module3" / "traffic_pattern_summary.csv"
DEFAULT_OBSERVATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "module2" / "track_observations.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "module4"

ROADS = ["NORTH", "EAST", "SOUTH", "WEST"]


# ---------------------------------------------------------------------------
# SIGNAL TIMING AND SAFETY STANDARDS
# ---------------------------------------------------------------------------
DEFAULT_CYCLE_TIME = 90.0        # Total fixed cycle reference in seconds
DEFAULT_MIN_GREEN = 10.0         # Minimum vehicular green time in seconds
DEFAULT_MAX_GREEN = 45.0         # Maximum vehicular green time in seconds
DEFAULT_YELLOW_TIME = 3.0        # Yellow / amber change interval in seconds
DEFAULT_ALL_RED_TIME = 1.0       # All-red safety clearance interval in seconds
DEFAULT_PED_WALK_MIN = 15.0      # Minimum green when pedestrian call active
DEFAULT_EMERGENCY_GREEN = 25.0   # Preemption green granted to emergency vehicle

# Saturation headway: seconds required per vehicle to discharge during green
# 2.0 seconds = 0.5 vehicles/sec (standard 1800 vehicles/hour/lane)
SATURATION_HEADWAY = 2.0


# ---------------------------------------------------------------------------
# 1. PRIORITY SCORE CALCULATOR
# ---------------------------------------------------------------------------
class PriorityScoreCalculator:
    """
    Calculates multi-attribute road priority scores based on traffic metrics.

    Weights:
      - Queue length: 35% (immediate waiting pressure)
      - Waiting time: 25% (delay minimization)
      - Density proxy: 20% (spatial congestion)
      - Volume flow: 10% (throughput maximization)
      - Starvation penalty: 10% (equitable service, avoids red light starvation)
    """

    def __init__(
        self,
        w_queue: float = 0.35,
        w_wait: float = 0.25,
        w_density: float = 0.20,
        w_volume: float = 0.10,
        w_starvation: float = 0.10,
        queue_ref: float = 10.0,
        wait_ref: float = 120.0,
        density_ref: float = 10.0,
        volume_ref: float = 20.0,
        starvation_ref: float = 120.0,
    ):
        total_w = w_queue + w_wait + w_density + w_volume + w_starvation
        if not math.isclose(total_w, 1.0, rel_tol=1e-3):
            w_queue /= total_w
            w_wait /= total_w
            w_density /= total_w
            w_volume /= total_w
            w_starvation /= total_w

        self.weights = {
            "queue": w_queue,
            "wait": w_wait,
            "density": w_density,
            "volume": w_volume,
            "starvation": w_starvation,
        }
        self.references = {
            "queue": max(1.0, queue_ref),
            "wait": max(1.0, wait_ref),
            "density": max(0.01, density_ref),
            "volume": max(1.0, volume_ref),
            "starvation": max(1.0, starvation_ref),
        }

    def compute_road_score(
        self,
        queue: float,
        wait_seconds: float,
        density: float,
        volume: float,
        red_elapsed_seconds: float = 0.0,
        is_emergency: bool = False,
        is_pedestrian: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute normalized sub-scores and composite priority for a single road approach.
        """
        score_queue = min(100.0, (max(0.0, queue) / self.references["queue"]) * 100.0)
        score_wait = min(100.0, (max(0.0, wait_seconds) / self.references["wait"]) * 100.0)
        score_density = min(100.0, (max(0.0, density) / self.references["density"]) * 100.0)
        score_volume = min(100.0, (max(0.0, volume) / self.references["volume"]) * 100.0)
        score_starvation = min(100.0, (max(0.0, red_elapsed_seconds) / self.references["starvation"]) * 100.0)

        base_priority = (
            self.weights["queue"] * score_queue
            + self.weights["wait"] * score_wait
            + self.weights["density"] * score_density
            + self.weights["volume"] * score_volume
            + self.weights["starvation"] * score_starvation
        )

        emergency_boost = 500.0 if is_emergency else 0.0
        pedestrian_boost = 30.0 if is_pedestrian else 0.0

        total_priority = base_priority + emergency_boost + pedestrian_boost

        return {
            "score_queue": round(score_queue, 2),
            "score_wait": round(score_wait, 2),
            "score_density": round(score_density, 2),
            "score_volume": round(score_volume, 2),
            "score_starvation": round(score_starvation, 2),
            "base_priority": round(base_priority, 2),
            "emergency_boost": round(emergency_boost, 2),
            "pedestrian_boost": round(pedestrian_boost, 2),
            "total_priority": round(total_priority, 2),
        }

    def compute_all_roads(
        self,
        metrics_df: pd.DataFrame,
        red_timers: Optional[Dict[str, float]] = None,
        emergency_flags: Optional[Dict[str, bool]] = None,
        pedestrian_flags: Optional[Dict[str, bool]] = None,
    ) -> pd.DataFrame:
        """
        Compute priority scores for all approaches from a metrics dataframe.
        """
        red_timers = red_timers or {r: 0.0 for r in ROADS}
        emergency_flags = emergency_flags or {r: False for r in ROADS}
        pedestrian_flags = pedestrian_flags or {r: False for r in ROADS}

        records = []
        for road in ROADS:
            road_data = metrics_df[metrics_df["road"].str.upper() == road]
            if road_data.empty:
                queue = 0.0
                wait_sec = 0.0
                density = 0.0
                volume = 0.0
            else:
                row = road_data.iloc[0]
                queue = float(row.get("average_queue_length", row.get("queue_length_vehicles", 0.0)))
                wait_sec = float(row.get("average_estimated_wait_seconds", 0.0))
                density = float(row.get("average_density_vehicles_per_100k_roi_pixels", row.get("average_density", 0.0)))
                volume = float(row.get("unique_track_ids", row.get("average_detected_vehicles", 0.0)))

            score_dict = self.compute_road_score(
                queue=queue,
                wait_seconds=wait_sec,
                density=density,
                volume=volume,
                red_elapsed_seconds=red_timers.get(road, 0.0),
                is_emergency=emergency_flags.get(road, False),
                is_pedestrian=pedestrian_flags.get(road, False),
            )

            record = {
                "road": road,
                "queue": queue,
                "wait_seconds": wait_sec,
                "density": density,
                "volume": volume,
                "red_elapsed_seconds": red_timers.get(road, 0.0),
                "is_emergency": emergency_flags.get(road, False),
                "is_pedestrian": pedestrian_flags.get(road, False),
                **score_dict,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["priority_rank"] = df["total_priority"].rank(ascending=False, method="min").astype(int)
        return df


# ---------------------------------------------------------------------------
# 2. GREEN-LIGHT DURATION ALLOCATOR
# ---------------------------------------------------------------------------
class GreenDurationAllocator:
    """
    Dynamically allocates green-light durations across approaches based on priority scores
    and traffic engineering safety constraints.
    """

    def __init__(
        self,
        cycle_time: float = DEFAULT_CYCLE_TIME,
        min_green: float = DEFAULT_MIN_GREEN,
        max_green: float = DEFAULT_MAX_GREEN,
        yellow_time: float = DEFAULT_YELLOW_TIME,
        all_red_time: float = DEFAULT_ALL_RED_TIME,
        ped_walk_min: float = DEFAULT_PED_WALK_MIN,
    ):
        self.cycle_time = float(cycle_time)
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self.ped_walk_min = float(ped_walk_min)
        self.transition_time = self.yellow_time + self.all_red_time

    def allocate(
        self,
        priority_df: pd.DataFrame,
        emergency_road: Optional[str] = None,
        emergency_duration: float = DEFAULT_EMERGENCY_GREEN,
    ) -> pd.DataFrame:
        """
        Allocate green times among the roads.

        If emergency_road is provided, an emergency preemption allocation is generated.
        Otherwise, an elastic proportional allocation with min/max bounding is applied.
        """
        result = priority_df.copy()
        n_roads = len(ROADS)
        total_lost_time = n_roads * self.transition_time
        available_green = max(0.0, self.cycle_time - total_lost_time)

        # Handle emergency preemption
        if emergency_road and emergency_road.upper() in ROADS:
            em_road = emergency_road.upper()
            em_green = min(self.max_green, max(self.min_green, emergency_duration))
            remaining_green = max(0.0, available_green - em_green)
            other_roads = [r for r in ROADS if r != em_road]

            green_alloc = {}
            green_alloc[em_road] = em_green
            other_min = (n_roads - 1) * self.min_green

            if remaining_green < other_min:
                for r in other_roads:
                    green_alloc[r] = self.min_green
            else:
                other_surplus = remaining_green - other_min
                other_p_sum = result[result["road"].isin(other_roads)]["total_priority"].sum()
                if other_p_sum <= 0:
                    other_p_sum = 1.0

                for r in other_roads:
                    p = float(result.loc[result["road"] == r, "total_priority"].values[0])
                    share = other_surplus * (p / other_p_sum)
                    green_alloc[r] = round(self.min_green + share, 1)

            result["allocated_green_seconds"] = result["road"].map(green_alloc)
            result["yellow_seconds"] = self.yellow_time
            result["all_red_seconds"] = self.all_red_time
            result["phase_duration_seconds"] = (
                result["allocated_green_seconds"] + self.yellow_time + self.all_red_time
            )
            result["split_percentage"] = (
                (result["allocated_green_seconds"] / self.cycle_time) * 100
            ).round(2)
            result["allocation_reason"] = result["road"].apply(
                lambda r: "Emergency Preemption Override" if r == em_road else "Post-Preemption Shared Green"
            )
            return result

        # Determine effective minimum green for each road (pedestrian actuated minimum)
        min_bounds = {}
        for _, row in result.iterrows():
            r = row["road"]
            is_ped = bool(row.get("is_pedestrian", False))
            min_bounds[r] = self.ped_walk_min if is_ped else self.min_green

        total_min_required = sum(min_bounds.values())
        effective_cycle = self.cycle_time
        if available_green < total_min_required:
            available_green = total_min_required
            effective_cycle = available_green + total_lost_time

        elastic_pool = available_green - total_min_required
        p_sum = result["total_priority"].sum()
        if p_sum <= 0:
            p_sum = float(n_roads)
            result["total_priority"] = 1.0

        # Iterative bounded proportional allocation
        unconstrained = set(ROADS)
        green_alloc = {r: min_bounds[r] for r in ROADS}
        remaining_elastic = elastic_pool

        for _ in range(n_roads):
            if remaining_elastic <= 0 or not unconstrained:
                break
            current_p_sum = sum(
                result.loc[result["road"] == r, "total_priority"].values[0] for r in unconstrained
            )
            if current_p_sum <= 0:
                current_p_sum = float(len(unconstrained))

            hit_bound = False
            for r in list(unconstrained):
                p = float(result.loc[result["road"] == r, "total_priority"].values[0])
                added_green = remaining_elastic * (p / current_p_sum)
                proposed = green_alloc[r] + added_green

                if proposed >= self.max_green:
                    green_alloc[r] = self.max_green
                    remaining_elastic -= (self.max_green - min_bounds[r])
                    unconstrained.remove(r)
                    hit_bound = True
                    break

            if not hit_bound:
                for r in unconstrained:
                    p = float(result.loc[result["road"] == r, "total_priority"].values[0])
                    green_alloc[r] += remaining_elastic * (p / current_p_sum)
                remaining_elastic = 0.0
                break

        for r in ROADS:
            green_alloc[r] = round(green_alloc[r], 1)

        result["allocated_green_seconds"] = result["road"].map(green_alloc)
        result["yellow_seconds"] = self.yellow_time
        result["all_red_seconds"] = self.all_red_time
        result["phase_duration_seconds"] = (
            result["allocated_green_seconds"] + self.yellow_time + self.all_red_time
        )
        result["split_percentage"] = (
            (result["allocated_green_seconds"] / effective_cycle) * 100
        ).round(2)

        def make_reason(row):
            reasons = []
            if row.get("is_emergency", False):
                reasons.append("Emergency Priority")
            if row.get("is_pedestrian", False):
                reasons.append("Pedestrian Clearance Minimum")
            if row["allocated_green_seconds"] >= self.max_green:
                reasons.append("Max Green Reached")
            elif row["allocated_green_seconds"] <= self.min_green:
                reasons.append("Min Green Guaranteed")
            else:
                reasons.append("Adaptive Proportional Demand")
            return " + ".join(reasons)

        result["allocation_reason"] = result.apply(make_reason, axis=1)
        return result


# ---------------------------------------------------------------------------
# 3. EMERGENCY AND PEDESTRIAN PRIORITY MANAGERS
# ---------------------------------------------------------------------------
class EmergencyPriorityManager:
    """
    Handles emergency preemption requests (Ambulance, Fire Engine, Police).
    """

    def __init__(self):
        self.scheduled_events: List[Dict[str, Any]] = []
        self.preemption_history: List[Dict[str, Any]] = []

    def schedule_event(
        self,
        timestamp_seconds: float,
        road: str,
        vehicle_type: str = "Ambulance",
        duration_seconds: float = DEFAULT_EMERGENCY_GREEN,
    ):
        """Schedule an emergency vehicle detection event."""
        self.scheduled_events.append({
            "event_type": "EMERGENCY_PREEMPTION",
            "timestamp_seconds": float(timestamp_seconds),
            "road": road.upper(),
            "vehicle_type": vehicle_type,
            "duration_seconds": float(duration_seconds),
            "status": "PENDING",
        })
        self.scheduled_events.sort(key=lambda e: e["timestamp_seconds"])

    def check_active_emergency(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Return the active emergency event if one is in progress."""
        for event in self.scheduled_events:
            if event["status"] == "COMPLETED":
                continue
            start = event["timestamp_seconds"]
            end = start + event["duration_seconds"]
            if start <= current_time <= end:
                event["status"] = "ACTIVE"
                return event
        return None

    def log_preemption(
        self,
        event: Dict[str, Any],
        interrupted_road: str,
        transition_delay_sec: float,
        granted_green_sec: float,
    ):
        """Log executed preemption details."""
        record = {
            "event_id": f"EM-{len(self.preemption_history) + 1}",
            "timestamp_seconds": event["timestamp_seconds"],
            "emergency_road": event["road"],
            "vehicle_type": event["vehicle_type"],
            "interrupted_road": interrupted_road,
            "transition_delay_seconds": round(transition_delay_sec, 2),
            "granted_green_seconds": round(granted_green_sec, 1),
            "outcome": "Preemption Executed & Approach Cleared",
        }
        self.preemption_history.append(record)


class PedestrianPriorityManager:
    """
    Handles pedestrian crosswalk demand and minimum walk clearance intervals.
    """

    def __init__(self, min_walk_time: float = DEFAULT_PED_WALK_MIN):
        self.min_walk_time = min_walk_time
        self.call_history: List[Dict[str, Any]] = []

    def register_call(
        self,
        timestamp_seconds: float,
        road: str,
        pedestrian_count: int = 1,
    ):
        """Register a pedestrian push-button actuation or video detection call."""
        road = road.upper()
        self.call_history.append({
            "event_type": "PEDESTRIAN_ACTUATION",
            "timestamp_seconds": float(timestamp_seconds),
            "road": road,
            "pedestrian_count": int(pedestrian_count),
            "required_clearance_seconds": self.min_walk_time,
            "status": "PENDING",
            "served_timestamp": None,
            "wait_time_seconds": None,
        })

    def has_pending_call(self, road: str, current_time: Optional[float] = None) -> bool:
        road = road.upper()
        for call in self.call_history:
            if call["road"] == road and call["status"] == "PENDING":
                if current_time is None or current_time >= call["timestamp_seconds"]:
                    return True
        return False

    def satisfy_call(self, road: str, current_time: float):
        """Mark pedestrian call as satisfied upon granting pedestrian green phase."""
        road = road.upper()
        for call in self.call_history:
            if call["road"] == road and call["status"] == "PENDING":
                if current_time >= call["timestamp_seconds"]:
                    call["status"] = "SERVED"
                    call["served_timestamp"] = round(current_time, 2)
                    call["wait_time_seconds"] = max(0.0, round(current_time - call["timestamp_seconds"], 2))


# ---------------------------------------------------------------------------
# 4. FIXED-TIME SIGNAL CONTROLLER (BENCHMARK)
# ---------------------------------------------------------------------------
class FixedTimeSignalController:
    """
    Standard pre-timed signal controller with fixed round-robin cycle splits.
    """

    def __init__(
        self,
        cycle_time: float = DEFAULT_CYCLE_TIME,
        yellow_time: float = DEFAULT_YELLOW_TIME,
        all_red_time: float = DEFAULT_ALL_RED_TIME,
    ):
        self.cycle_time = cycle_time
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.transition_time = yellow_time + all_red_time

        total_lost = len(ROADS) * self.transition_time
        self.fixed_green = max(5.0, (cycle_time - total_lost) / len(ROADS))

    def get_phase_plan(self) -> pd.DataFrame:
        """Return the static round-robin phase schedule."""
        records = []
        for road in ROADS:
            records.append({
                "road": road,
                "allocated_green_seconds": round(self.fixed_green, 1),
                "yellow_seconds": self.yellow_time,
                "all_red_seconds": self.all_red_time,
                "phase_duration_seconds": round(self.fixed_green + self.transition_time, 1),
                "split_percentage": round((self.fixed_green / self.cycle_time) * 100, 2),
                "control_mode": "Fixed-Time (Static Split)",
            })
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. DISCRETE-EVENT TRAFFIC SIMULATOR & COMPARATOR
# ---------------------------------------------------------------------------
class TrafficSimulator:
    """
    High-fidelity discrete simulation engine comparing Adaptive Signal Control
    against Fixed-Time Signal Control using observed video vehicle arrivals.
    """

    def __init__(
        self,
        arrivals_df: pd.DataFrame,
        cycle_time: float = DEFAULT_CYCLE_TIME,
        min_green: float = DEFAULT_MIN_GREEN,
        max_green: float = DEFAULT_MAX_GREEN,
        yellow_time: float = DEFAULT_YELLOW_TIME,
        all_red_time: float = DEFAULT_ALL_RED_TIME,
        ped_walk_min: float = DEFAULT_PED_WALK_MIN,
        emergency_manager: Optional[EmergencyPriorityManager] = None,
        pedestrian_manager: Optional[PedestrianPriorityManager] = None,
    ):
        self.arrivals = arrivals_df.copy().sort_values("timestamp_seconds")
        self.cycle_time = cycle_time
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.ped_walk_min = ped_walk_min
        self.transition_time = yellow_time + all_red_time

        self.emergency_mgr = emergency_manager or EmergencyPriorityManager()
        self.pedestrian_mgr = pedestrian_manager or PedestrianPriorityManager()

        self.score_calc = PriorityScoreCalculator()
        self.allocator = GreenDurationAllocator(
            cycle_time=cycle_time,
            min_green=min_green,
            max_green=max_green,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
        )
        self.fixed_controller = FixedTimeSignalController(
            cycle_time=cycle_time,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
        )

    def run_simulation(
        self,
        duration_seconds: float = 600.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run parallel second-by-second simulation of Fixed-Time and Adaptive controllers.
        """
        # ===================================================================
        # A. SIMULATE FIXED-TIME CONTROLLER
        # ===================================================================
        veh_fixed = self.arrivals.copy()
        veh_fixed["wait_time"] = 0.0
        veh_fixed["cleared_time"] = None

        queues_fixed = {r: [] for r in ROADS}
        wasted_green_fixed = {r: 0.0 for r in ROADS}
        discharge_timer_f = {r: 0.0 for r in ROADS}

        fixed_plan = self.fixed_controller.get_phase_plan()
        fixed_green = float(fixed_plan["allocated_green_seconds"].iloc[0])
        phase_duration_f = fixed_green + self.transition_time

        ts_fixed = []

        for t in np.arange(0, duration_seconds, 1.0):
            # 1. Ingest arrivals
            new_arr = veh_fixed[
                (veh_fixed["timestamp_seconds"] >= t)
                & (veh_fixed["timestamp_seconds"] < t + 1.0)
            ]
            for _, row in new_arr.iterrows():
                queues_fixed[row["road"]].append(int(row["track_id"]))

            # 2. Fixed round-robin phase state
            cycle_pos = t % self.cycle_time
            phase_idx = int(cycle_pos // phase_duration_f) % len(ROADS)
            active_road = ROADS[phase_idx]
            time_in_phase = cycle_pos % phase_duration_f
            is_green = (time_in_phase < fixed_green)

            # 3. Vehicle discharge / queue service
            for r in ROADS:
                if r == active_road and is_green:
                    if len(queues_fixed[r]) > 0:
                        discharge_timer_f[r] += 1.0
                        if discharge_timer_f[r] >= SATURATION_HEADWAY:
                            cleared_id = queues_fixed[r].pop(0)
                            arr_sec = float(
                                veh_fixed.loc[veh_fixed["track_id"] == cleared_id, "timestamp_seconds"].values[0]
                            )
                            wait_dur = max(0.0, t - arr_sec)
                            veh_fixed.loc[veh_fixed["track_id"] == cleared_id, "wait_time"] = wait_dur
                            veh_fixed.loc[veh_fixed["track_id"] == cleared_id, "cleared_time"] = t
                            discharge_timer_f[r] = 0.0
                    else:
                        wasted_green_fixed[r] += 1.0

            if t % 5 == 0:
                for r in ROADS:
                    ts_fixed.append({
                        "timestamp_seconds": t,
                        "road": r,
                        "fixed_queue": len(queues_fixed[r]),
                    })

        # Process any vehicles remaining in queue at end of simulation
        for r in ROADS:
            for unserved_id in queues_fixed[r]:
                arr_sec = float(veh_fixed.loc[veh_fixed["track_id"] == unserved_id, "timestamp_seconds"].values[0])
                veh_fixed.loc[veh_fixed["track_id"] == unserved_id, "wait_time"] = duration_seconds - arr_sec

        # ===================================================================
        # B. SIMULATE ADAPTIVE CONTROLLER (WITH GAP-OUT, PREEMPTION & ACTUATION)
        # ===================================================================
        veh_adaptive = self.arrivals.copy()
        veh_adaptive["wait_time"] = 0.0
        veh_adaptive["cleared_time"] = None

        queues_adaptive = {r: [] for r in ROADS}
        wasted_green_adaptive = {r: 0.0 for r in ROADS}
        red_timers_a = {r: 0.0 for r in ROADS}
        discharge_timer_a = 0.0

        current_phase = "NORTH"
        phase_state = "GREEN"  # GREEN, YELLOW, ALL_RED
        state_timer = 0.0
        current_green_alloc = 20.0
        is_emergency_active = False

        cycle_records = []
        cycle_idx = 1
        cycle_start_time = 0.0
        cycle_greens = {r: 0.0 for r in ROADS}

        ts_adaptive = []

        for t in np.arange(0, duration_seconds, 1.0):
            # 1. Ingest arrivals
            new_arr = veh_adaptive[
                (veh_adaptive["timestamp_seconds"] >= t)
                & (veh_adaptive["timestamp_seconds"] < t + 1.0)
            ]
            for _, row in new_arr.iterrows():
                queues_adaptive[row["road"]].append(int(row["track_id"]))

            # 2. Check Emergency Preemption
            active_em = self.emergency_mgr.check_active_emergency(t)
            if active_em and active_em["status"] == "ACTIVE":
                em_road = active_em["road"]
                if current_phase != em_road or phase_state != "GREEN":
                    self.emergency_mgr.log_preemption(
                        event=active_em,
                        interrupted_road=current_phase,
                        transition_delay_sec=self.transition_time,
                        granted_green_sec=active_em["duration_seconds"],
                    )
                    current_phase = em_road
                    phase_state = "GREEN"
                    state_timer = 0.0
                    current_green_alloc = active_em["duration_seconds"]
                    is_emergency_active = True
                    active_em["status"] = "COMPLETED"

            # 3. Check Pedestrian Calls
            for r in ROADS:
                if self.pedestrian_mgr.has_pending_call(r, t) and current_phase == r and phase_state == "GREEN":
                    self.pedestrian_mgr.satisfy_call(r, t)

            # 4. State Machine Execution
            if phase_state == "GREEN":
                state_timer += 1.0
                cycle_greens[current_phase] += 1.0

                if len(queues_adaptive[current_phase]) > 0:
                    discharge_timer_a += 1.0
                    if discharge_timer_a >= SATURATION_HEADWAY:
                        cleared_id = queues_adaptive[current_phase].pop(0)
                        arr_sec = float(
                            veh_adaptive.loc[veh_adaptive["track_id"] == cleared_id, "timestamp_seconds"].values[0]
                        )
                        wait_dur = max(0.0, t - arr_sec)
                        veh_adaptive.loc[veh_adaptive["track_id"] == cleared_id, "wait_time"] = wait_dur
                        veh_adaptive.loc[veh_adaptive["track_id"] == cleared_id, "cleared_time"] = t
                        discharge_timer_a = 0.0
                else:
                    wasted_green_adaptive[current_phase] += 1.0

                is_ped = self.pedestrian_mgr.has_pending_call(current_phase, t)
                eff_min = self.ped_walk_min if is_ped else self.min_green

                gap_out = (not is_emergency_active and len(queues_adaptive[current_phase]) == 0 and state_timer >= eff_min)
                max_out = (state_timer >= current_green_alloc)

                if gap_out or max_out:
                    phase_state = "YELLOW"
                    state_timer = 0.0
                    discharge_timer_a = 0.0
                    is_emergency_active = False

            elif phase_state == "YELLOW":
                state_timer += 1.0
                if state_timer >= self.yellow_time:
                    phase_state = "ALL_RED"
                    state_timer = 0.0

            elif phase_state == "ALL_RED":
                state_timer += 1.0
                if state_timer >= self.all_red_time:
                    candidate_roads = [r for r in ROADS if r != current_phase]

                    rows = []
                    for r in candidate_roads:
                        rows.append({
                            "road": r,
                            "average_queue_length": len(queues_adaptive[r]),
                            "average_estimated_wait_seconds": red_timers_a[r],
                            "average_density_vehicles_per_100k_roi_pixels": len(queues_adaptive[r]) * 1.5,
                            "unique_track_ids": len(queues_adaptive[r]),
                        })
                    cand_df = pd.DataFrame(rows)
                    scored_cand = self.score_calc.compute_all_roads(
                        metrics_df=cand_df,
                        red_timers=red_timers_a,
                        emergency_flags={r: False for r in ROADS},
                        pedestrian_flags={r: self.pedestrian_mgr.has_pending_call(r, t) for r in ROADS},
                    )

                    next_road = scored_cand.sort_values("total_priority", ascending=False).iloc[0]["road"]
                    current_phase = next_road
                    phase_state = "GREEN"
                    state_timer = 0.0

                    q_now = len(queues_adaptive[current_phase])
                    is_ped_call = self.pedestrian_mgr.has_pending_call(current_phase, t)
                    base_g = self.ped_walk_min if is_ped_call else self.min_green
                    current_green_alloc = min(self.max_green, max(base_g, base_g + q_now * 3.5))

                    red_timers_a[current_phase] = 0.0

            for r in ROADS:
                if r != current_phase or phase_state != "GREEN":
                    red_timers_a[r] += 1.0

            if t % 5 == 0:
                for r in ROADS:
                    ts_adaptive.append({
                        "timestamp_seconds": t,
                        "road": r,
                        "adaptive_queue": len(queues_adaptive[r]),
                    })

            if t > 0 and t % 90 == 0:
                cycle_records.append({
                    "cycle_index": cycle_idx,
                    "start_time_seconds": cycle_start_time,
                    "end_time_seconds": t,
                    "north_green_sec": cycle_greens["NORTH"],
                    "east_green_sec": cycle_greens["EAST"],
                    "south_green_sec": cycle_greens["SOUTH"],
                    "west_green_sec": cycle_greens["WEST"],
                    "active_emergency_road": self.emergency_mgr.check_active_emergency(t)["road"] if self.emergency_mgr.check_active_emergency(t) else "None",
                })
                cycle_idx += 1
                cycle_start_time = t
                cycle_greens = {r: 0.0 for r in ROADS}

        for r in ROADS:
            for unserved_id in queues_adaptive[r]:
                arr_sec = float(veh_adaptive.loc[veh_adaptive["track_id"] == unserved_id, "timestamp_seconds"].values[0])
                veh_adaptive.loc[veh_adaptive["track_id"] == unserved_id, "wait_time"] = duration_seconds - arr_sec

        cycle_log_df = pd.DataFrame(cycle_records)

        df_f = pd.DataFrame(ts_fixed)
        df_a = pd.DataFrame(ts_adaptive)
        time_series_df = pd.merge(df_f, df_a, on=["timestamp_seconds", "road"])

        # ===================================================================
        # C. COMPILE COMPREHENSIVE KPI COMPARISON TABLE
        # ===================================================================
        comparison_rows = []
        for r in ROADS:
            sub_f = veh_fixed[veh_fixed["road"] == r]
            sub_a = veh_adaptive[veh_adaptive["road"] == r]

            n_arr = len(sub_f)
            n_cleared_f = sub_f["cleared_time"].notna().sum()
            n_cleared_a = sub_a["cleared_time"].notna().sum()

            tot_del_f = sub_f["wait_time"].sum()
            tot_del_a = sub_a["wait_time"].sum()

            avg_wait_f = sub_f["wait_time"].mean() if n_arr > 0 else 0.0
            avg_wait_a = sub_a["wait_time"].mean() if n_arr > 0 else 0.0

            wast_f = wasted_green_fixed[r]
            wast_a = wasted_green_adaptive[r]

            wait_reduct = ((avg_wait_f - avg_wait_a) / avg_wait_f * 100) if avg_wait_f > 0 else 0.0
            delay_reduct = ((tot_del_f - tot_del_a) / tot_del_f * 100) if tot_del_f > 0 else 0.0
            wast_reduct = ((wast_f - wast_a) / wast_f * 100) if wast_f > 0 else 0.0

            comparison_rows.append({
                "road": r,
                "total_vehicles_arrived": n_arr,
                "fixed_vehicles_served": int(n_cleared_f),
                "adaptive_vehicles_served": int(n_cleared_a),
                "fixed_total_delay_sec": round(float(tot_del_f), 1),
                "adaptive_total_delay_sec": round(float(tot_del_a), 1),
                "fixed_avg_wait_sec": round(float(avg_wait_f), 2),
                "adaptive_avg_wait_sec": round(float(avg_wait_a), 2),
                "wait_time_reduction_pct": round(float(wait_reduct), 2),
                "fixed_wasted_green_sec": round(float(wast_f), 1),
                "adaptive_wasted_green_sec": round(float(wast_a), 1),
                "wasted_green_reduction_pct": round(float(wast_reduct), 2),
                "fixed_final_queue": len(queues_fixed[r]),
                "adaptive_final_queue": len(queues_adaptive[r]),
            })

        # Overall Intersection Row
        tot_arr = len(veh_fixed)
        tot_del_f_all = veh_fixed["wait_time"].sum()
        tot_del_a_all = veh_adaptive["wait_time"].sum()
        avg_wait_f_all = veh_fixed["wait_time"].mean()
        avg_wait_a_all = veh_adaptive["wait_time"].mean()
        tot_wast_f_all = sum(wasted_green_fixed.values())
        tot_wast_a_all = sum(wasted_green_adaptive.values())

        tot_wait_reduct = ((avg_wait_f_all - avg_wait_a_all) / avg_wait_f_all * 100) if avg_wait_f_all > 0 else 0.0
        tot_wast_reduct = ((tot_wast_f_all - tot_wast_a_all) / tot_wast_f_all * 100) if tot_wast_f_all > 0 else 0.0

        comparison_rows.append({
            "road": "OVERALL_INTERSECTION",
            "total_vehicles_arrived": tot_arr,
            "fixed_vehicles_served": int(veh_fixed["cleared_time"].notna().sum()),
            "adaptive_vehicles_served": int(veh_adaptive["cleared_time"].notna().sum()),
            "fixed_total_delay_sec": round(float(tot_del_f_all), 1),
            "adaptive_total_delay_sec": round(float(tot_del_a_all), 1),
            "fixed_avg_wait_sec": round(float(avg_wait_f_all), 2),
            "adaptive_avg_wait_sec": round(float(avg_wait_a_all), 2),
            "wait_time_reduction_pct": round(float(tot_wait_reduct), 2),
            "fixed_wasted_green_sec": round(float(tot_wast_f_all), 1),
            "adaptive_wasted_green_sec": round(float(tot_wast_a_all), 1),
            "wasted_green_reduction_pct": round(float(tot_wast_reduct), 2),
            "fixed_final_queue": sum(len(q) for q in queues_fixed.values()),
            "adaptive_final_queue": sum(len(q) for q in queues_adaptive.values()),
        })

        comparison_df = pd.DataFrame(comparison_rows)
        return comparison_df, cycle_log_df, time_series_df


# ---------------------------------------------------------------------------
# 6. VISUALIZATION GENERATOR
# ---------------------------------------------------------------------------
def generate_plots(
    comparison_df: pd.DataFrame,
    cycle_log_df: pd.DataFrame,
    time_series_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate publication-quality charts illustrating signal efficiency gains.
    """
    roads_only = comparison_df[comparison_df["road"] != "OVERALL_INTERSECTION"]

    # -----------------------------------------------------------------------
    # PLOT 1: Performance Comparison Bar Charts (4 subplots)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Average Waiting Time by Road
    ax = axes[0, 0]
    x = np.arange(len(roads_only))
    width = 0.35
    ax.bar(x - width / 2, roads_only["fixed_avg_wait_sec"], width, label="Fixed-Time Signal", color="#e74c3c", alpha=0.85)
    ax.bar(x + width / 2, roads_only["adaptive_avg_wait_sec"], width, label="Adaptive Signal", color="#2ecc71", alpha=0.85)
    ax.set_title("Average Waiting Time by Approach", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(roads_only["road"])
    ax.set_ylabel("Seconds / Vehicle")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # (b) Total Cumulative Delay by Road
    ax = axes[0, 1]
    ax.bar(x - width / 2, roads_only["fixed_total_delay_sec"], width, label="Fixed-Time Signal", color="#e74c3c", alpha=0.85)
    ax.bar(x + width / 2, roads_only["adaptive_total_delay_sec"], width, label="Adaptive Signal", color="#2ecc71", alpha=0.85)
    ax.set_title("Total Cumulative Delay by Approach", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(roads_only["road"])
    ax.set_ylabel("Vehicle-Seconds")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # (c) Wasted Green Time (Empty Road Allocation)
    ax = axes[1, 0]
    ax.bar(x - width / 2, roads_only["fixed_wasted_green_sec"], width, label="Fixed-Time Signal", color="#e67e22", alpha=0.85)
    ax.bar(x + width / 2, roads_only["adaptive_wasted_green_sec"], width, label="Adaptive Signal", color="#3498db", alpha=0.85)
    ax.set_title("Wasted Green Time on Empty Approaches", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(roads_only["road"])
    ax.set_ylabel("Seconds")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # (d) Overall Efficiency Summary Metric Comparison
    ax = axes[1, 1]
    overall = comparison_df[comparison_df["road"] == "OVERALL_INTERSECTION"].iloc[0]
    kpis = ["Avg Wait Time (s)", "Wasted Green (x10 s)", "Total Delay (x100 s)"]
    val_fixed = [
        overall["fixed_avg_wait_sec"],
        overall["fixed_wasted_green_sec"] / 10.0,
        overall["fixed_total_delay_sec"] / 100.0,
    ]
    val_adaptive = [
        overall["adaptive_avg_wait_sec"],
        overall["adaptive_wasted_green_sec"] / 10.0,
        overall["adaptive_total_delay_sec"] / 100.0,
    ]
    xk = np.arange(len(kpis))
    ax.bar(xk - width / 2, val_fixed, width, label="Fixed-Time", color="#95a5a6", alpha=0.85)
    ax.bar(xk + width / 2, val_adaptive, width, label="Adaptive", color="#1abc9c", alpha=0.85)
    ax.set_title("System-Wide Impact & Efficiency Gains", fontsize=12, fontweight="bold")
    ax.set_xticks(xk)
    ax.set_xticklabels(kpis)
    ax.set_ylabel("Normalized Scale")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    improvement_text = (
        f"Delay Reduction: {overall['wait_time_reduction_pct']:.1f}%  |  "
        f"Wasted Green Eliminated: {overall['wasted_green_reduction_pct']:.1f}%"
    )
    fig.suptitle(f"Adaptive vs. Fixed-Time Signal Control Benchmark\n{improvement_text}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_dir / "signal_performance_comparison.png", dpi=180)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PLOT 2: Dynamic Green Light Allocation Timeline
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    time_min = cycle_log_df["start_time_seconds"] / 60.0

    ax.plot(time_min, cycle_log_df["north_green_sec"], marker="o", label="North Green (s)", color="#2980b9", linewidth=2)
    ax.plot(time_min, cycle_log_df["east_green_sec"], marker="s", label="East Green (s)", color="#8e44ad", linewidth=2)
    ax.plot(time_min, cycle_log_df["south_green_sec"], marker="^", label="South Green (s)", color="#7f8c8d", linewidth=2, linestyle="--")
    ax.plot(time_min, cycle_log_df["west_green_sec"], marker="D", label="West Green (s)", color="#d35400", linewidth=2)

    em_cycles = cycle_log_df[cycle_log_df["active_emergency_road"] != "None"]
    for _, row in em_cycles.iterrows():
        t_m = row["start_time_seconds"] / 60.0
        em_r = row["active_emergency_road"]
        ax.axvline(x=t_m, color="red", linestyle=":", alpha=0.8)
        ax.text(t_m, 35, f"Emergency ({em_r})", rotation=90, color="red", fontsize=9, fontweight="bold")

    ax.set_title("Cycle-by-Cycle Dynamic Green Light Duration Allocations", fontsize=13, fontweight="bold")
    ax.set_xlabel("Video Time (minutes)", fontsize=11)
    ax.set_ylabel("Allocated Green Duration (seconds)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "adaptive_green_allocation_timeline.png", dpi=180)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # PLOT 3: Queue Dissipation Profiles (Fixed vs. Adaptive)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    coords = {"NORTH": (0, 0), "EAST": (0, 1), "SOUTH": (1, 0), "WEST": (1, 1)}

    for road, (r_idx, c_idx) in coords.items():
        ax = axes[r_idx, c_idx]
        subset = time_series_df[time_series_df["road"] == road]
        t = subset["timestamp_seconds"] / 60.0
        ax.plot(t, subset["fixed_queue"], label="Fixed-Time Queue", color="#e74c3c", linewidth=2, linestyle="--")
        ax.plot(t, subset["adaptive_queue"], label="Adaptive Queue", color="#27ae60", linewidth=2)
        ax.set_title(f"Approach: {road}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Queue Length (vehicles)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[1, 0].set_xlabel("Time (minutes)")
    axes[1, 1].set_xlabel("Time (minutes)")
    fig.suptitle("Queue Dissipation Dynamics Over Time: Fixed-Time vs. Adaptive", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_dir / "queue_dissipation_profiles.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. MAIN ORCHESTRATION PIPELINE
# ---------------------------------------------------------------------------
def run_signal_control(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    congestion_path: Path = DEFAULT_CONGESTION_PATH,
    patterns_path: Path = DEFAULT_PATTERNS_PATH,
    observations_path: Path = DEFAULT_OBSERVATIONS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cycle_time: float = DEFAULT_CYCLE_TIME,
    min_green: float = DEFAULT_MIN_GREEN,
    max_green: float = DEFAULT_MAX_GREEN,
    yellow_time: float = DEFAULT_YELLOW_TIME,
    all_red_time: float = DEFAULT_ALL_RED_TIME,
    enable_demo_events: bool = True,
    emergency_events_json: Optional[Path] = None,
    pedestrian_events_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Member 4 Adaptive Signal Control execution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      MEMBER 4: ADAPTIVE TRAFFIC SIGNAL CONTROL MODULE           ")
    print("=================================================================")
    print(f"Loading Member 3 Metrics from: {metrics_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Load Member 3 inputs
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing traffic metrics file: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)

    # Load vehicle observations (if available) for micro-simulation
    if observations_path.exists():
        obs_df = pd.read_csv(observations_path)
        arrivals_df = (
            obs_df.groupby(["road", "track_id"])["timestamp_seconds"]
            .min()
            .reset_index()
            .sort_values("timestamp_seconds")
        )
    else:
        arrivals = []
        track_id_counter = 1
        for _, row in metrics_df.iterrows():
            r = row["road"]
            n_veh = int(row.get("unique_track_ids", 10))
            times = np.sort(np.random.uniform(10.0, 580.0, n_veh))
            for t in times:
                arrivals.append({
                    "road": r,
                    "track_id": track_id_counter,
                    "timestamp_seconds": round(float(t), 3),
                })
                track_id_counter += 1
        arrivals_df = pd.DataFrame(arrivals).sort_values("timestamp_seconds")

    # 2. Setup Emergency and Pedestrian Priority Managers
    emergency_mgr = EmergencyPriorityManager()
    pedestrian_mgr = PedestrianPriorityManager(min_walk_time=DEFAULT_PED_WALK_MIN)

    if emergency_events_json and emergency_events_json.exists():
        with emergency_events_json.open("r", encoding="utf-8") as f:
            em_data = json.load(f)
            for ev in em_data:
                emergency_mgr.schedule_event(**ev)
    elif enable_demo_events:
        # Realistic representative scenario: Ambulance arrives on EAST at 3 minutes (180s)
        emergency_mgr.schedule_event(
            timestamp_seconds=180.0,
            road="EAST",
            vehicle_type="Ambulance",
            duration_seconds=25.0,
        )

    if pedestrian_events_json and pedestrian_events_json.exists():
        with pedestrian_events_json.open("r", encoding="utf-8") as f:
            ped_data = json.load(f)
            for ev in ped_data:
                pedestrian_mgr.register_call(**ev)
    elif enable_demo_events:
        pedestrian_mgr.register_call(timestamp_seconds=90.0, road="NORTH", pedestrian_count=2)
        pedestrian_mgr.register_call(timestamp_seconds=330.0, road="WEST", pedestrian_count=3)
        pedestrian_mgr.register_call(timestamp_seconds=450.0, road="SOUTH", pedestrian_count=1)

    # 3. Calculate Immediate Baseline Priority & Green Allocations
    score_calc = PriorityScoreCalculator()
    immediate_priority_df = score_calc.compute_all_roads(metrics_df)

    allocator = GreenDurationAllocator(
        cycle_time=cycle_time,
        min_green=min_green,
        max_green=max_green,
        yellow_time=yellow_time,
        all_red_time=all_red_time,
    )
    immediate_decisions_df = allocator.allocate(immediate_priority_df)

    # 4. Run Comparative Simulation (Adaptive vs. Fixed-Time)
    simulator = TrafficSimulator(
        arrivals_df=arrivals_df,
        cycle_time=cycle_time,
        min_green=min_green,
        max_green=max_green,
        yellow_time=yellow_time,
        all_red_time=all_red_time,
        emergency_manager=emergency_mgr,
        pedestrian_manager=pedestrian_mgr,
    )
    comparison_df, cycle_log_df, time_series_df = simulator.run_simulation(duration_seconds=600.0)

    # 5. Export Decision Tables & Event Logs
    immediate_decisions_df.to_csv(output_dir / "adaptive_signal_decisions.csv", index=False)
    cycle_log_df.to_csv(output_dir / "cycle_by_cycle_log.csv", index=False)
    comparison_df.to_csv(output_dir / "fixed_vs_adaptive_comparison.csv", index=False)

    all_events = []
    for em in emergency_mgr.preemption_history:
        all_events.append({
            "category": "EMERGENCY_PREEMPTION",
            "timestamp_seconds": em["timestamp_seconds"],
            "road": em["emergency_road"],
            "details": f"{em['vehicle_type']} preemption granted ({em['granted_green_seconds']}s green)",
            "transition_delay_sec": em["transition_delay_seconds"],
            "status": em["outcome"],
        })
    for ped in pedestrian_mgr.call_history:
        all_events.append({
            "category": "PEDESTRIAN_ACTUATION",
            "timestamp_seconds": ped["timestamp_seconds"],
            "road": ped["road"],
            "details": f"{ped['pedestrian_count']} pedestrian(s) requested crossing",
            "transition_delay_sec": ped["wait_time_seconds"],
            "status": ped["status"],
        })
    events_df = pd.DataFrame(all_events)
    events_df.to_csv(output_dir / "emergency_pedestrian_events.csv", index=False)

    # 6. Generate Publication Charts
    generate_plots(
        comparison_df=comparison_df,
        cycle_log_df=cycle_log_df,
        time_series_df=time_series_df,
        output_dir=output_dir,
    )

    # 7. Generate Machine-Readable Summary JSON
    overall_kpi = comparison_df[comparison_df["road"] == "OVERALL_INTERSECTION"].iloc[0]
    summary_data = {
        "module": "Member 4 – Adaptive Signal Control",
        "configuration": {
            "cycle_time_seconds": cycle_time,
            "min_green_seconds": min_green,
            "max_green_seconds": max_green,
            "yellow_clearance_seconds": yellow_time,
            "all_red_clearance_seconds": all_red_time,
            "pedestrian_walk_minimum_seconds": DEFAULT_PED_WALK_MIN,
            "saturation_headway_seconds": SATURATION_HEADWAY,
        },
        "priority_score_weights": score_calc.weights,
        "immediate_allocations": immediate_decisions_df[
            ["road", "total_priority", "priority_rank", "allocated_green_seconds", "split_percentage", "allocation_reason"]
        ].to_dict(orient="records"),
        "fixed_vs_adaptive_comparison": {
            "fixed_total_delay_seconds": float(overall_kpi["fixed_total_delay_sec"]),
            "adaptive_total_delay_seconds": float(overall_kpi["adaptive_total_delay_sec"]),
            "delay_reduction_percentage": float(overall_kpi["wait_time_reduction_pct"]),
            "fixed_average_wait_seconds": float(overall_kpi["fixed_avg_wait_sec"]),
            "adaptive_average_wait_seconds": float(overall_kpi["adaptive_avg_wait_sec"]),
            "fixed_wasted_green_seconds": float(overall_kpi["fixed_wasted_green_sec"]),
            "adaptive_wasted_green_seconds": float(overall_kpi["adaptive_wasted_green_sec"]),
            "wasted_green_reduction_percentage": float(overall_kpi["wasted_green_reduction_pct"]),
        },
        "emergency_preemption_events": len(emergency_mgr.preemption_history),
        "pedestrian_calls_served": len([p for p in pedestrian_mgr.call_history if p["status"] == "SERVED"]),
        "generated_files": [
            "adaptive_signal_decisions.csv",
            "cycle_by_cycle_log.csv",
            "fixed_vs_adaptive_comparison.csv",
            "emergency_pedestrian_events.csv",
            "signal_performance_comparison.png",
            "adaptive_green_allocation_timeline.png",
            "queue_dissipation_profiles.png",
            "signal_control_summary.json",
        ],
    }

    with (output_dir / "signal_control_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # 8. Terminal Output
    print("\n----------------- DYNAMIC SIGNAL ALLOCATIONS -----------------")
    print(
        immediate_decisions_df[
            ["road", "total_priority", "priority_rank", "allocated_green_seconds", "split_percentage", "allocation_reason"]
        ].to_string(index=False)
    )

    print("\n------------- COMPARATIVE BENCHMARK (FIXED vs. ADAPTIVE) -------------")
    print(
        comparison_df[
            ["road", "fixed_avg_wait_sec", "adaptive_avg_wait_sec", "wait_time_reduction_pct", "fixed_wasted_green_sec", "adaptive_wasted_green_sec"]
        ].to_string(index=False)
    )

    print("\n---------------- SYSTEM-WIDE PERFORMANCE SUMMARY ----------------")
    print(f"Overall Delay Reduction:       {overall_kpi['wait_time_reduction_pct']:.2f}%")
    print(f"Wasted Green Time Saved:       {overall_kpi['wasted_green_reduction_pct']:.2f}%")
    print(f"Emergency Preemption Events:   {len(emergency_mgr.preemption_history)}")
    print(f"Pedestrian Requests Handled:   {len([p for p in pedestrian_mgr.call_history if p['status'] == 'SERVED'])}")
    print(f"\nAll reports and charts successfully written to: {output_dir}")
    print("=================================================================\n")

    return summary_data


# ---------------------------------------------------------------------------
# CLI ARGUMENT PARSER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Member 4 – Adaptive Signal Control System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Path to road_traffic_metrics.csv")
    parser.add_argument("--congestion", default=str(DEFAULT_CONGESTION_PATH), help="Path to congestion_scores.csv")
    parser.add_argument("--patterns", default=str(DEFAULT_PATTERNS_PATH), help="Path to traffic_pattern_summary.csv")
    parser.add_argument("--observations", default=str(DEFAULT_OBSERVATIONS_PATH), help="Path to track_observations.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for reports & plots")
    parser.add_argument("--cycle-time", type=float, default=DEFAULT_CYCLE_TIME, help="Total signal cycle time in seconds")
    parser.add_argument("--min-green", type=float, default=DEFAULT_MIN_GREEN, help="Minimum green duration in seconds")
    parser.add_argument("--max-green", type=float, default=DEFAULT_MAX_GREEN, help="Maximum green duration in seconds")
    parser.add_argument("--yellow-time", type=float, default=DEFAULT_YELLOW_TIME, help="Yellow change interval in seconds")
    parser.add_argument("--all-red-time", type=float, default=DEFAULT_ALL_RED_TIME, help="All-red safety clearance in seconds")
    parser.add_argument("--no-demo-events", action="store_true", help="Disable default emergency & pedestrian demo events")
    parser.add_argument("--emergency-events", default=None, help="Optional path to custom emergency events JSON")
    parser.add_argument("--pedestrian-events", default=None, help="Optional path to custom pedestrian events JSON")

    args = parser.parse_args()

    run_signal_control(
        metrics_path=Path(args.metrics),
        congestion_path=Path(args.congestion),
        patterns_path=Path(args.patterns),
        observations_path=Path(args.observations),
        output_dir=Path(args.output),
        cycle_time=args.cycle_time,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_time=args.yellow_time,
        all_red_time=args.all_red_time,
        enable_demo_events=not args.no_demo_events,
        emergency_events_json=Path(args.emergency_events) if args.emergency_events else None,
        pedestrian_events_json=Path(args.pedestrian_events) if args.pedestrian_events else None,
    )


if __name__ == "__main__":
    main()
