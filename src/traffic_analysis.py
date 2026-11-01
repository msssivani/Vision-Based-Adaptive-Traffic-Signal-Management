"""
Module 3: Traffic Pattern and Congestion Analysis

Input:
  data/processed/module2/track_observations.csv
  configs/road_regions.json

Outputs (reports/module3/):
  vehicle_waiting_times.csv
  road_traffic_metrics.csv
  congestion_scores.csv
  traffic_patterns.png
  congestion_comparison.png
  traffic_analysis_summary.json

Notes:
- The input is sampled every third video frame. Waiting times are estimates.
- A gap longer than MAX_TRACK_GAP_SECONDS splits a waiting-zone visit.
- Queue length is the number of distinct tracked IDs observed in a road's
  waiting zone at a sampled frame.
- Density is an image-area proxy: detected vehicles per 100,000 ROI pixels.
  It is not a calibrated real-world vehicles/km/lane-km measurement.
- Congestion thresholds below are transparent project heuristics and can be
  adjusted after reviewing the results.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "module2" / "track_observations.csv"
ROI_JSON = PROJECT_ROOT / "configs" / "road_regions.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "module3"

ROADS = ["NORTH", "EAST", "SOUTH", "WEST"]

# Tracking observations are sampled at roughly 10 frames/second.
# Split a vehicle's waiting-zone visit if it disappears for longer than this.
MAX_TRACK_GAP_SECONDS = 1.0

# Heuristic reference values used to create a 0-100 congestion score.
# These are project thresholds, not universal traffic-engineering standards.
QUEUE_REFERENCE_VEHICLES = 10
WAIT_REFERENCE_SECONDS = 120.0
DENSITY_REFERENCE_PER_100K_PIXELS = 10.0


def polygon_area(points):
    """Return polygon area in pixels using the shoelace formula."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.nan
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def load_roi_areas(path):
    """Load road ROI areas from configs/road_regions.json."""
    if not path.exists():
        raise FileNotFoundError(f"ROI configuration not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    roads_config = config.get("roads", {})
    areas = {}
    for road in ROADS:
        key = road.lower()
        if key not in roads_config or "road_roi" not in roads_config[key]:
            raise ValueError(f"Missing road_roi for {key} in {path}")
        areas[road] = polygon_area(roads_config[key]["road_roi"])
    return areas


def load_observations(path):
    if not path.exists():
        raise FileNotFoundError(f"Tracking observations CSV not found: {path}")

    df = pd.read_csv(path)
    required = {
        "frame_index", "timestamp_seconds", "track_id", "road",
        "vehicle_class", "center_x", "center_y", "in_waiting_zone"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Tracking observations CSV is empty.")

    df = df.copy()
    df["road"] = df["road"].astype(str).str.upper()
    df = df[df["road"].isin(ROADS)].copy()
    df["timestamp_seconds"] = pd.to_numeric(df["timestamp_seconds"], errors="coerce")
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce")
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    df["in_waiting_zone"] = pd.to_numeric(df["in_waiting_zone"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["timestamp_seconds", "frame_index", "track_id"])
    df["track_id"] = df["track_id"].astype(int)
    df["frame_index"] = df["frame_index"].astype(int)
    df["in_waiting_zone"] = df["in_waiting_zone"].clip(0, 1)
    if df.empty:
        raise ValueError("No valid observations for NORTH/EAST/SOUTH/WEST were found.")
    return df.sort_values(["road", "track_id", "timestamp_seconds"])


def calculate_waiting_visits(df):
    """Estimate each vehicle's continuous waiting-zone visits."""
    waiting = df[df["in_waiting_zone"] == 1].copy()
    columns = [
        "road", "track_id", "visit_number", "start_seconds", "end_seconds",
        "estimated_wait_seconds", "observations_in_visit"
    ]
    if waiting.empty:
        return pd.DataFrame(columns=columns)

    visits = []
    for (road, track_id), group in waiting.groupby(["road", "track_id"]):
        group = group.sort_values("timestamp_seconds")
        times = group["timestamp_seconds"].to_numpy(dtype=float)
        start_idx = 0
        visit_number = 0

        for i in range(1, len(times) + 1):
            gap_break = (
                i == len(times)
                or (times[i] - times[i - 1]) > MAX_TRACK_GAP_SECONDS
            )
            if gap_break:
                segment = times[start_idx:i]
                visit_number += 1
                # A single observation cannot establish a duration.
                # For multiple observations, use first-to-last observed time.
                duration = float(segment[-1] - segment[0]) if len(segment) >= 2 else 0.0
                visits.append({
                    "road": road,
                    "track_id": int(track_id),
                    "visit_number": visit_number,
                    "start_seconds": round(float(segment[0]), 3),
                    "end_seconds": round(float(segment[-1]), 3),
                    "estimated_wait_seconds": round(duration, 3),
                    "observations_in_visit": int(len(segment)),
                })
                start_idx = i

    return pd.DataFrame(visits, columns=columns)


def calculate_frame_metrics(df, roi_areas):
    """Calculate sampled-frame queue length and image-area density proxies."""
    all_frames = sorted(df["frame_index"].unique())
    rows = []

    for frame in all_frames:
        frame_data = df[df["frame_index"] == frame]
        timestamp = float(frame_data["timestamp_seconds"].median())

        for road in ROADS:
            road_data = frame_data[frame_data["road"] == road]
            all_ids = road_data["track_id"].nunique()
            waiting_ids = road_data.loc[
                road_data["in_waiting_zone"] == 1, "track_id"
            ].nunique()
            area = roi_areas[road]
            density = (
                all_ids / area * 100000.0
                if area and np.isfinite(area) and area > 0
                else np.nan
            )
            rows.append({
                "frame_index": int(frame),
                "timestamp_seconds": round(timestamp, 3),
                "road": road,
                "detected_vehicles": int(all_ids),
                "queue_length_vehicles": int(waiting_ids),
                "roi_area_pixels": round(float(area), 2),
                "density_vehicles_per_100k_roi_pixels": (
                    round(float(density), 4) if np.isfinite(density) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def build_road_metrics(frame_metrics, waiting_visits):
    rows = []
    for road in ROADS:
        fm = frame_metrics[frame_metrics["road"] == road]
        visits = waiting_visits[waiting_visits["road"] == road]

        avg_wait = (
            float(visits.loc[visits["observations_in_visit"] >= 2,
                             "estimated_wait_seconds"].mean())
            if not visits.empty and (visits["observations_in_visit"] >= 2).any()
            else 0.0
        )
        rows.append({
            "road": road,
            "average_detected_vehicles_per_sampled_frame": round(float(fm["detected_vehicles"].mean()), 3),
            "maximum_detected_vehicles_per_sampled_frame": int(fm["detected_vehicles"].max()),
            "average_queue_length": round(float(fm["queue_length_vehicles"].mean()), 3),
            "maximum_queue_length": int(fm["queue_length_vehicles"].max()),
            "average_estimated_wait_seconds": round(avg_wait, 3),
            "average_density_vehicles_per_100k_roi_pixels": round(
                float(fm["density_vehicles_per_100k_roi_pixels"].mean()), 4
            ),
            "unique_track_ids": int(
                # Unique IDs are road-specific; same numeric ID on different roads
                # is not assumed to represent the same vehicle.
                fm["road"].nunique() * 0 + 0
            ),
        })

    metrics = pd.DataFrame(rows)
    # Fill unique ID totals directly from the source frame-level records.
    # This avoids counting a vehicle repeatedly across frames.
    return metrics


def add_unique_ids(metrics, df):
    unique_counts = df.groupby("road")["track_id"].nunique().to_dict()
    metrics["unique_track_ids"] = metrics["road"].map(unique_counts).fillna(0).astype(int)
    return metrics


def make_congestion_scores(metrics):
    result = metrics.copy()

    queue_component = (
        result["maximum_queue_length"] / QUEUE_REFERENCE_VEHICLES * 100
    ).clip(0, 100)
    wait_component = (
        result["average_estimated_wait_seconds"] / WAIT_REFERENCE_SECONDS * 100
    ).clip(0, 100)
    density_component = (
        result["average_density_vehicles_per_100k_roi_pixels"]
        / DENSITY_REFERENCE_PER_100K_PIXELS * 100
    ).clip(0, 100)

    result["congestion_score"] = (
        0.50 * queue_component
        + 0.30 * wait_component
        + 0.20 * density_component
    ).round(2)

    result["congestion_level"] = pd.cut(
        result["congestion_score"],
        bins=[-0.001, 33.333, 66.666, 100.001],
        labels=["Low", "Moderate", "High"]
    ).astype(str)

    result["score_method"] = (
        "Heuristic: 50% max queue, 30% average estimated wait, "
        "20% image-area density proxy"
    )
    return result


def make_plots(frame_metrics, congestion, output_dir):
    # Traffic pattern chart: sampled observations grouped into 30-second bins.
    pattern = frame_metrics.copy()
    pattern["time_bin_seconds"] = (
        np.floor(pattern["timestamp_seconds"] / 30) * 30
    ).astype(int)
    grouped = pattern.groupby(
        ["time_bin_seconds", "road"], as_index=False
    ).agg(
        average_detected_vehicles=("detected_vehicles", "mean"),
        average_queue_length=("queue_length_vehicles", "mean")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for road in ROADS:
        subset = grouped[grouped["road"] == road]
        ax.plot(
            subset["time_bin_seconds"] / 60,
            subset["average_detected_vehicles"],
            marker="o",
            label=road
        )
    ax.set_title("Traffic Pattern by Road (30-second bins)")
    ax.set_xlabel("Video time (minutes)")
    ax.set_ylabel("Average detected vehicles per sampled frame")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "traffic_patterns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(congestion["road"], congestion["congestion_score"])
    ax.set_title("Congestion Score by Road")
    ax.set_xlabel("Road")
    ax.set_ylabel("Heuristic congestion score (0-100)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    for bar, level in zip(bars, congestion["congestion_level"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(level),
            ha="center",
            va="bottom"
        )
    fig.tight_layout()
    fig.savefig(output_dir / "congestion_comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Module 3 traffic analysis")
    parser.add_argument("--input", default=str(INPUT_CSV), help="Path to track_observations.csv")
    parser.add_argument("--roi", default=str(ROI_JSON), help="Path to road_regions.json")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    roi_path = Path(args.roi)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_observations(input_path)
    roi_areas = load_roi_areas(roi_path)

    waiting_visits = calculate_waiting_visits(df)
    frame_metrics = calculate_frame_metrics(df, roi_areas)
    road_metrics = build_road_metrics(frame_metrics, waiting_visits)
    road_metrics = add_unique_ids(road_metrics, df)
    congestion = make_congestion_scores(road_metrics)

    # Add simple 30-second traffic-pattern aggregates.
    pattern = frame_metrics.copy()
    pattern["time_bin_start_seconds"] = (
        np.floor(pattern["timestamp_seconds"] / 30) * 30
    ).astype(int)
    pattern_summary = pattern.groupby(
        ["time_bin_start_seconds", "road"], as_index=False
    ).agg(
        average_detected_vehicles=("detected_vehicles", "mean"),
        average_queue_length=("queue_length_vehicles", "mean"),
        maximum_queue_length=("queue_length_vehicles", "max"),
        average_density=("density_vehicles_per_100k_roi_pixels", "mean")
    )

    waiting_visits.to_csv(output_dir / "vehicle_waiting_times.csv", index=False)
    road_metrics.to_csv(output_dir / "road_traffic_metrics.csv", index=False)
    congestion.to_csv(output_dir / "congestion_scores.csv", index=False)
    pattern_summary.to_csv(output_dir / "traffic_pattern_summary.csv", index=False)

    make_plots(frame_metrics, congestion, output_dir)

    summary = {
        "input_file": str(input_path),
        "observations_used": int(len(df)),
        "video_time_covered_seconds": round(float(df["timestamp_seconds"].max()), 3),
        "roads_found": sorted(df["road"].unique().tolist()),
        "unique_track_ids_by_road": {
            road: int(df.loc[df["road"] == road, "track_id"].nunique())
            for road in ROADS
        },
        "waiting_zone_observations": int(df["in_waiting_zone"].sum()),
        "waiting_time_method": (
            "First-to-last sampled waiting-zone observation per continuous visit; "
            f"visits split at gaps greater than {MAX_TRACK_GAP_SECONDS} second"
        ),
        "density_method": "Detected vehicles divided by road ROI polygon pixel area, scaled per 100,000 pixels",
        "congestion_method": (
            "Heuristic score: 50% maximum queue length, 30% average estimated "
            "waiting time, 20% average image-area density proxy"
        ),
        "limitations": [
            "Every-third-frame sampling reduces temporal precision.",
            "A vehicle can be missed between sampled frames.",
            "Waiting times are estimates based on observed waiting-zone intervals.",
            "Queue length is based on distinct tracked IDs visible in the waiting zone at sampled frames.",
            "Density is an image-area proxy, not calibrated vehicles per kilometre or lane-kilometre.",
            "Congestion thresholds are project heuristics and should be described as such."
        ],
        "congestion_by_road": congestion[
            ["road", "congestion_score", "congestion_level"]
        ].to_dict(orient="records")
    }

    with (output_dir / "traffic_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n========== MODULE 3 COMPLETED ==========")
    print(f"Observations used: {len(df)}")
    print(f"Video time covered: {df['timestamp_seconds'].max():.2f} seconds")
    print("\nCongestion by road:")
    print(congestion[["road", "congestion_score", "congestion_level"]].to_string(index=False))
    print(f"\nOutputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
